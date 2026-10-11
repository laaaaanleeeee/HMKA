import os

from src.embedding.embedder import BGEEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import BGEReranker
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_builder import PromptBuilder
from src.rag.llm import GeminiLLM
from src.rag.pipeline import RAGPipeline
from src.rag.query_rewriter import QueryRewriter


def create_rag_pipeline() -> RAGPipeline:
    device = os.getenv("MODEL_DEVICE", "cpu").strip().lower()
    use_fp16 = os.getenv("MODEL_USE_FP16", "false").strip().lower() == "true"

    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device=device,
    )

    vector_store = ChromaVectorStore(
        persist_directory="data/vector_db/chroma",
        collection_name="drug_chunks",
    )

    bm25_retriever = BM25Retriever(
        chunks_path="data/chunks/chunks.jsonl",
    )

    hybrid_retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        use_fp16=use_fp16,
    )

    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder()

    llm = GeminiLLM(
        model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        max_attempts=3,
    )

    query_rewriter = QueryRewriter(
        llm=llm,
        max_history_messages=6,
    )

    return RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
        query_rewriter=query_rewriter,
        candidate_k=50,
        top_k=5,
    )
