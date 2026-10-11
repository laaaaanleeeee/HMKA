
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import BGEReranker
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_builder import PromptBuilder
from src.rag.llm import GeminiLLM
from src.rag.query_rewriter import QueryRewriter
from src.rag.pipeline import RAGPipeline


QUERY_FILE = ROOT_DIR / "evaluation" / "retrieval" / "evaluation_queries.json"
OUTPUT_DIR = ROOT_DIR / "evaluation" / "rag" / "results"

OUTPUT_FILE = OUTPUT_DIR / "rag_results_gemini_fixed.json"

MODEL_NAME = "gemini-3.6-flash"
RUN_VERSION = "filter_fix_v1"
REFUSAL_TEXT = (
    "Không tìm thấy thông tin này trong tài liệu được cung cấp."
)

logging.basicConfig(level=logging.WARNING)


def build_pipeline():
    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device="cuda",
    )

    vector_store = ChromaVectorStore(
        persist_directory=str(ROOT_DIR / "data" / "vector_db" / "chroma"),
        collection_name="drug_chunks",
    )

    bm25_retriever = BM25Retriever(
        chunks_path=str(ROOT_DIR / "data" / "chunks" / "chunks.jsonl"),
    )

    hybrid_retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=True,
    )

    llm = GeminiLLM(
        model_name=MODEL_NAME,
        max_attempts=3,
    )

    query_rewriter = QueryRewriter(
        llm=llm,
        max_history_messages=6,
    )

    return RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm=llm,
        query_rewriter=query_rewriter,
        candidate_k=50,
        top_k=5,
    )


def evaluate_query(item, pipeline):
    started_at = time.perf_counter()
    result = pipeline.run(query=item["query"])
    elapsed = time.perf_counter() - started_at

    answer = result.get("answer", "") or ""
    retrieved_results = result.get("results", [])
    sources = []

    for retrieved in retrieved_results:
        metadata = retrieved.get("metadata", {})
        sources.append({
            "drug_name": metadata.get("drug_name", ""),
            "section": metadata.get("section", ""),
            "source_url": (
                metadata.get("source_url")
                or metadata.get("url", "")
            ),
            "chunk_index": metadata.get("chunk_index"),
        })

    expected_drug = item.get("expected_drug")
    expected_sections = item.get("expected_sections", [])

    def norm(value):
        return " ".join((value or "").casefold().split())

    expected_drug_norm = norm(expected_drug)

    drug_match = (
        any(
            expected_drug_norm in norm(source["drug_name"])
            or norm(source["drug_name"]) in expected_drug_norm
            for source in sources
            if source["drug_name"]
        )
        if expected_drug_norm
        else None
    )

    expected_sections_norm = [norm(section) for section in expected_sections]

    section_match = (
        any(
            expected in norm(source["section"])
            or norm(source["section"]) in expected
            for source in sources
            for expected in expected_sections_norm
            if source["section"] and expected
        )
        if expected_sections_norm
        else None
    )

    # This is a literal-string check, not a semantic evaluation.
    refusal_detected = REFUSAL_TEXT in answer

    return {
        "id": item["id"],
        "query": item["query"],
        "answerable": item["answerable"],
        "expected_drug": expected_drug,
        "expected_sections": expected_sections,
        "model": MODEL_NAME,
        "run_version": RUN_VERSION,
        "rewritten_query": result.get("rewritten_query", item["query"]),
        "answer": answer,
        "refusal_detected": refusal_detected,
        "context": result.get("context", ""),
        "sources": sources,
        "num_results": len(retrieved_results),
        "expected_drug_found": drug_match,
        "expected_section_found": section_match,
        "elapsed_seconds": round(elapsed, 3),
        "error": None,
        "answer_correct": None,
        "source_supports_answer": None,
        "review_notes": "",
    }


def save_results(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def is_quota_error(exc):
    message = str(exc).lower()
    return (
        "429" in message
        or "resource_exhausted" in message
        or "quota" in message
    )


def print_summary(rows, total_queries):
    successful = [row for row in rows if row.get("error") is None]
    answerable = [row for row in successful if row["answerable"]]
    ood = [row for row in successful if not row["answerable"]]

    def count_true(items, key):
        return sum(item.get(key) is True for item in items)

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Benchmark queries: {total_queries}")
    print(f"Completed successfully: {len(successful)}")
    print(f"Not completed: {total_queries - len(successful)}")
    print(
        "Answerable queries with expected drug found: "
        f"{count_true(answerable, 'expected_drug_found')}/"
        f"{len(answerable)}"
    )
    print(
        "Answerable queries with expected section found: "
        f"{count_true(answerable, 'expected_section_found')}/"
        f"{len(answerable)}"
    )
    print(
        f"OOD exact refusal detected: "
        f"{count_true(ood, 'refusal_detected')}/{len(ood)}"
    )
    print(
        "Note: answer correctness and source support still require review."
    )
    print(f"Results saved to: {OUTPUT_FILE}")


def main():
    if not QUERY_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file benchmark: {QUERY_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with QUERY_FILE.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    queries = benchmark["queries"]

    # Resume only results from this exact output file and run version.
    saved_rows = {}
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as file:
            previous_rows = json.load(file)

        saved_rows = {
            row["id"]: row
            for row in previous_rows
            if row.get("error") is None
            and row.get("model") == MODEL_NAME
            and row.get("run_version") == RUN_VERSION
        }

    pending = [
        item for item in queries
        if item["id"] not in saved_rows
    ]

    print(f"Total queries: {len(queries)}")
    print(f"Already completed in this run: {len(saved_rows)}")
    print(f"Remaining: {len(pending)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_FILE}")

    if pending:
        pipeline = build_pipeline()

        for index, item in enumerate(pending, start=1):
            print(
                f"\n[{index}/{len(pending)}] "
                f"{item['id']}: {item['query']}"
            )

            try:
                row = evaluate_query(item, pipeline)
                saved_rows[item["id"]] = row

                print(f"  Results: {row['num_results']}")
                print(f"  Exact refusal: {row['refusal_detected']}")
                print(f"  Time: {row['elapsed_seconds']}s")
                print(f"  Answer: {row['answer'][:250]}")

            except Exception as exc:
                traceback.print_exc()

                # Preserve successful rows; failed requests can be retried.
                save_results([
                    saved_rows[item["id"]]
                    for item in queries
                    if item["id"] in saved_rows
                ])

                if is_quota_error(exc):
                    print("Gemini quota exhausted; stopping evaluation.")
                    break

                print(f"Request failed: {exc}")
                continue

            save_results([
                saved_rows[item["id"]]
                for item in queries
                if item["id"] in saved_rows
            ])

    final_rows = [
        saved_rows[item["id"]]
        for item in queries
        if item["id"] in saved_rows
    ]

    print_summary(final_rows, len(queries))


if __name__ == "__main__":
    main()
