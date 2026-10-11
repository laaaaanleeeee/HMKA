import logging

logger = logging.getLogger(__name__)

class QueryRewriter:

    def __init__(
        self,
        llm,
        max_history_messages: int = 6,
    ) -> None:
        if max_history_messages < 1:
            raise ValueError("max_history_messages must be at least 1.")

        self.llm = llm
        self.max_history_messages = max_history_messages

    def rewrite(
        self,
        query: str,
        history: list[dict],
    ) -> str:
        if not history:
            return query

        recent_history = history[-self.max_history_messages:]

        history_lines = [
            f"{item['role']}: {item['content']}"
            for item in recent_history
            if item.get("role") in {"user", "assistant"}
            and item.get("content")
        ]

        if not history_lines:
            return query

        history_text = "\n".join(history_lines)

        prompt = f"""
            Bạn là thành phần viết lại câu hỏi cho hệ thống tra cứu thông tin thuốc.

            Nhiệm vụ:
            - Viết lại câu hỏi mới thành câu hỏi độc lập để tìm kiếm tài liệu.
            - Dùng lịch sử để giải quyết tham chiếu như "thuốc này", "nó",
            hoặc "còn tác dụng phụ thì sao".
            - Giữ nguyên ý định của người dùng.
            - Chỉ dùng thông tin có trong câu hỏi và lịch sử.
            - Không trả lời câu hỏi và không tự suy đoán tên thuốc.
            - Nếu không xác định được ngữ cảnh đáng tin cậy, giữ nguyên câu hỏi.
            - Chỉ trả về câu hỏi đã viết lại, không giải thích.

            LỊCH SỬ HỘI THOẠI:
            {history_text}

            CÂU HỎI MỚI:
            {query}

            CÂU HỎI ĐỘC LẬP:
            """.strip()

        try:
            rewritten = self.llm.generate(prompt).strip()
            rewritten = rewritten.strip('"').strip("'").strip()

            if not rewritten or len(rewritten) > 500:
                return query

            return rewritten

        except Exception:
            logger.exception(
                "Query rewriting failed; using original query."
            )
            return query
