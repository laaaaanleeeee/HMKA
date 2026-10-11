
import re
import unicodedata

from FlagEmbedding import FlagReranker


def normalize_section(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").casefold()
    text = text.replace("&", " ")
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class BGEReranker:
    SECTION_INTENTS = {
        "dosage": [
            "liều lượng và cách dùng",
            "liều lượng cách dùng",
            "liều dùng",
            "liều lượng",
            "cách dùng",
            "liều",
            "uống",
            "dùng thuốc",
        ],
        "contraindication": [
            "chống chỉ định",
            "không được dùng",
            "không nên dùng",
        ],
        "side_effect": [
            "tác dụng phụ",
            "tác dụng ngoài ý muốn",
        ],
        "interaction": [
            "tương tác thuốc",
            "tương tác",
            "dùng cùng",
            "phối hợp với",
        ],
        "active_ingredient": [
            "thành phần hoạt chất",
            "hoạt chất",
            "thành phần",
        ],
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ):
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0 or not candidates:
            return []

        pairs = [[query, item["text"]] for item in candidates]
        scores = self.model.compute_score(pairs, normalize=False)

        if not isinstance(scores, list):
            scores = [scores]

        if len(scores) != len(candidates):
            raise ValueError(
                "Number of reranker scores does not match "
                "number of candidates."
            )

        intent = self._detect_intent(query)
        ranked_results = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            section = candidate.get("metadata", {}).get("section", "")
            reranker_score = float(score)
            section_boost = self._section_boost(section, intent)

            result["reranker_score"] = reranker_score
            result["section_boost"] = section_boost
            result["final_score"] = reranker_score + section_boost
            ranked_results.append(result)

        ranked_results.sort(
            key=lambda item: item["final_score"],
            reverse=True,
        )
        return ranked_results[:top_k]

    def _detect_intent(self, query: str) -> str | None:
        query = unicodedata.normalize("NFC", query or "").casefold()

        best_intent = None
        best_score = 0

        for intent, keywords in self.SECTION_INTENTS.items():
            score = sum(keyword in query for keyword in keywords)
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    def _section_boost(
        self,
        section: str,
        intent: str | None,
    ) -> float:
        if intent is None:
            return 0.0

        section = normalize_section(section)

        exact_sections = {
            "dosage": {
                "liều lượng cách dùng",
                "liều dùng",
                "liều lượng",
                "cách dùng",
            },
            "contraindication": {"chống chỉ định"},
            "side_effect": {
                "tác dụng phụ",
                "tác dụng ngoài ý muốn",
            },
            "interaction": {"tương tác thuốc"},
            "active_ingredient": {
                "thành phần",
                "thành phần hoạt chất",
                "hoạt chất",
            },
        }

        normalized_exact = {
            normalize_section(item)
            for item in exact_sections.get(intent, set())
        }

        if section in normalized_exact:
            return 2.0

        if intent == "contraindication" and section.startswith(
            "chống chỉ định "
        ):
            return 0.5

        if intent == "side_effect" and (
            section.startswith("tác dụng phụ ")
            or section.startswith("tác dụng ngoài ý muốn ")
        ):
            return 0.5

        if intent == "interaction" and section.startswith("tương tác "):
            return 0.5

        if intent == "dosage" and (
            "liều" in section or "cách dùng" in section
        ):
            return 0.5

        return 0.0