class ContextBuilder:
    def build(self, results: list[dict]) -> str:
        context_parts = []

        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]

            drug_name = metadata.get("drug_name", "Unknown")
            section = metadata.get("section", "Unknown")
            source_url = metadata.get("source_url", "")

            text = result["text"].strip()

            lines = text.splitlines()

            while lines and not lines[0].strip():
                lines.pop(0)

            if lines and lines[0].strip() == f"Thuốc: {drug_name}":
                lines.pop(0)

            while lines and not lines[0].strip():
                lines.pop(0)

            if lines and lines[0].strip() == f"Phần: {section}":
                lines.pop(0)

            text = "\n".join(lines).strip()

            context = (
                f"[Document {index}]\n"
                f"Tên thuốc: {drug_name}\n"
                f"Section: {section}\n"
                f"URL nguồn: {source_url}\n"
                f"Nội dung:\n{text}"
            )

            context_parts.append(context)

        return "\n\n".join(context_parts)
