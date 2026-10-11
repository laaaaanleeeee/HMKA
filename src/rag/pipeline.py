import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _canonical_section(text: str) -> str:
    text = _normalize_text(text)
    text = text.replace("&", " ")
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RAGPipeline:
    REFUSAL_TEXT = (
        "Không tìm thấy thông tin này trong tài liệu được cung cấp."
    )

    DRUG_ALIASES = {
        "voltaren": ("voltaren",),
        "beatil": ("beatil",),
        "nebilet": ("nebilet",),
        "medsolu": ("medsolu",),
        "nexium": ("nexium",),
    }

    MEDICATION_KEYWORDS = (
        "thuốc",
        "dược",
        "hoạt chất",
        "liều",
        "liều lượng",
        "cách dùng",
        "chống chỉ định",
        "tác dụng phụ",
        "tác dụng ngoài ý muốn",
        "tương tác",
        "công dụng",
        "chỉ định",
        "điều trị",
        "uống thuốc",
        "dùng thuốc",
        "thuốc này",
        "viên nén",
        "viên nang",
        "quá liều",
        "phản ứng phụ",
        "mg",
        "mcg",
        "microgam",
        "mililit",
        "ml",
    )

    INTENT_SECTIONS = {
        "indication": (
            "công dụng",
            "chỉ định",
        ),
        "contraindication": (
            "chống chỉ định",
        ),
        "dosage": (
            "liều lượng cách dùng",
            "liều dùng",
            "liều lượng",
            "cách dùng",
            "liều",
        ),
        "side_effect": (
            "tác dụng phụ",
            "tác dụng ngoài ý muốn",
        ),
        "interaction": (
            "tương tác thuốc",
            "tương tác",
        ),
    }

    QUERY_INTENTS = (
        (
            "contraindication",
            (
                "chống chỉ định",
                "không được dùng",
                "không nên dùng",
            ),
        ),
        (
            "side_effect",
            (
                "tác dụng phụ",
                "tác dụng ngoài ý muốn",
            ),
        ),
        (
            "interaction",
            (
                "tương tác thuốc",
                "tương tác",
                "phối hợp với",
                "dùng cùng",
            ),
        ),
        (
            "dosage",
            (
                "liều dùng",
                "liều lượng",
                "cách dùng",
                "liều",
            ),
        ),
        (
            "indication",
            (
                "công dụng",
                "chỉ định",
                "điều trị bệnh gì",
                "dùng để làm gì",
                "thuốc này trị gì",
                "thuốc này chữa gì",
                "có tác dụng gì",
            ),
        ),
    )

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        context_builder,
        prompt_builder,
        llm,
        query_rewriter=None,
        candidate_k: int = 50,
        top_k: int = 5,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.candidate_k = candidate_k
        self.top_k = top_k

    @classmethod
    def is_medication_query(cls, query: str) -> bool:
        query_lower = _normalize_text(query)

        for aliases in cls.DRUG_ALIASES.values():
            for alias in aliases:
                if re.search(
                    rf"\b{re.escape(alias)}\b",
                    query_lower,
                ):
                    return True

        return any(
            keyword in query_lower
            for keyword in cls.MEDICATION_KEYWORDS
        )

    def _refusal_result(
        self,
        query: str,
        search_query: str,
    ) -> dict:
        return {
            "query": query,
            "rewritten_query": search_query,
            "answer": self.REFUSAL_TEXT,
            "results": [],
            "context": "",
        }

    def run(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> dict:
        history = history or []
        search_query = query

        if self.query_rewriter is not None and history:
            search_query = self.query_rewriter.rewrite(
                query=query,
                history=history,
            )

        logger.info(
            "RAG started: history=%s, query_length=%d",
            bool(history),
            len(search_query),
        )

        # Chặn câu hỏi rõ ràng ngoài phạm vi trước retrieval và LLM.
        if not self.is_medication_query(search_query):
            logger.info("RAG abstained: query outside medication scope")
            return self._refusal_result(query, search_query)

        candidates = self.hybrid_retriever.retrieve(
            query=search_query,
            n_results=self.candidate_k,
            candidate_k=self.candidate_k,
        )

        reranked_results = self.reranker.rerank(
            query=search_query,
            candidates=candidates,
            top_k=self.top_k,
        )

        filtered_results = self.filter_results_by_intent(
            query=search_query,
            results=reranked_results,
        )

        if not filtered_results:
            logger.info(
                "RAG abstained before LLM: candidates=%d, reranked=%d",
                len(candidates),
                len(reranked_results),
            )
            return self._refusal_result(query, search_query)

        context = self.context_builder.build(filtered_results)

        if not context or not context.strip():
            return self._refusal_result(query, search_query)

        prompt = self.prompt_builder.build(
            context=context,
            question=search_query,
        )
        answer = self.llm.generate(prompt)

        logger.info(
            "RAG completed: candidates=%d, final_results=%d",
            len(candidates),
            len(filtered_results),
        )

        return {
            "query": query,
            "rewritten_query": search_query,
            "answer": answer,
            "results": filtered_results,
            "context": context,
        }

    
    def filter_results_by_intent(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        if not results:
            return []

        query_lower = _normalize_text(query)

        mentioned_drugs = set()

        for result in results:
            drug_name = _normalize_text(
                result.get("metadata", {}).get("drug_name", "")
            )

            if not drug_name:
                continue

            if re.search(
                rf"(?<!\w){re.escape(drug_name)}(?!\w)",
                query_lower,
            ):
                mentioned_drugs.add(drug_name)

        product_results = results

        if mentioned_drugs:
            product_results = [
                result
                for result in results
                if _normalize_text(
                    result.get("metadata", {}).get("drug_name", "")
                ) in mentioned_drugs
            ]
        else:
    
            mentioned_aliases = [
                alias
                for aliases in self.DRUG_ALIASES.values()
                for alias in aliases
                if re.search(
                    rf"\b{re.escape(alias)}\b",
                    query_lower,
                )
            ]

            if mentioned_aliases:
                product_results = [
                    result
                    for result in results
                    if any(
                        re.search(
                            rf"\b{re.escape(alias)}\b",
                            _normalize_text(
                                result.get("metadata", {}).get(
                                    "drug_name", ""
                                )
                            ),
                        )
                        for alias in mentioned_aliases
                    )
                ]

                if not product_results:
                    return []

        detected_intent = None

        for intent, keywords in self.QUERY_INTENTS:
            if any(keyword in query_lower for keyword in keywords):
                detected_intent = intent
                break

        if detected_intent is None:
            return product_results

        allowed_sections = {
            _canonical_section(section)
            for section in self.INTENT_SECTIONS[detected_intent]
        }

        
        intent_results = []

        for result in product_results:
            section = _canonical_section(
                result.get("metadata", {}).get("section", "")
            )

            if section in allowed_sections:
                intent_results.append(result)
                continue

            if mentioned_drugs and any(
                section.startswith(allowed + " ")
                and any(
                    re.search(
                        rf"(?<!\w){re.escape(drug_name)}(?!\w)",
                        section,
                    )
                    for drug_name in mentioned_drugs
                )
                for allowed in allowed_sections
            ):
                intent_results.append(result)
                continue

            if not mentioned_drugs and any(
                section.startswith(allowed + " ")
                for allowed in allowed_sections
            ):
                intent_results.append(result)

        return intent_results