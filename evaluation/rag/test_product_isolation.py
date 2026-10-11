
from src.rag.pipeline import RAGPipeline


def make_result(drug_name, section):
    return {
        "text": f"Nội dung kiểm thử {drug_name} - {section}",
        "metadata": {
            "drug_name": drug_name,
            "section": section,
            "source_url": "https://example.com",
        },
    }


def main():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    results = [
        make_result("Kacerin", "Liều lượng - Cách dùng"),
        make_result("Cetirizin", "Liều dùng Cetirizin"),
        make_result("Kacerin", "Tác dụng phụ"),
    ]

    dosage_results = pipeline.filter_results_by_intent(
        "Liều dùng thuốc Kacerin như thế nào?",
        results,
    )

    assert dosage_results, "Không tìm thấy section liều dùng Kacerin."

    assert all(
        item["metadata"]["drug_name"] == "Kacerin"
        for item in dosage_results
    ), "Context chứa thông tin từ thuốc/hoạt chất khác."

    assert all(
        "liều" in item["metadata"]["section"].casefold()
        or "cách dùng" in item["metadata"]["section"].casefold()
        for item in dosage_results
    ), "Context chứa section không phù hợp."

    print("PASS: Kacerin product isolation")
    print("PASS: dosage section filtering")


if __name__ == "__main__":
    main()
