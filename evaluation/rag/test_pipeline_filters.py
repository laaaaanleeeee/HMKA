from src.rag.pipeline import RAGPipeline


def make_result(drug_name, section, text="Nội dung kiểm thử"):
    return {
        "text": text,
        "metadata": {
            "drug_name": drug_name,
            "section": section,
            "source_url": "https://example.com",
        },
    }


def main():
    pipeline = RAGPipeline.__new__(RAGPipeline)

    results = [
        make_result("Voltaren 75mg/3ml Novartis",
                    "Công dụng Voltaren 75mg/3ml Novartis"),
        make_result("Voltaren 75mg/3ml Novartis", "Chỉ định"),
        make_result("Voltaren 75mg/3ml Novartis",
                    "Liều lượng & Cách dùng"),
        make_result("Voltaren 75mg/3ml Novartis",
                    "Tác dụng ngoài ý muốn"),
        make_result("Nexium", "Công dụng"),
    ]

    indication = pipeline.filter_results_by_intent(
        "Thuốc Voltaren có công dụng gì?",
        results,
    )
    assert indication, "Không tìm thấy section công dụng."
    assert all(
        "voltaren" in item["metadata"]["drug_name"].casefold()
        for item in indication
    ), "Bộ lọc trả về nhầm thuốc."
    assert all(
        any(
            section in item["metadata"]["section"].casefold()
            for section in ("công dụng", "chỉ định")
        )
        for item in indication
    ), "Bộ lọc trả về section không liên quan."

    dosage = pipeline.filter_results_by_intent(
        "Liều dùng thuốc Voltaren như thế nào?",
        results,
    )
    assert dosage, "Không tìm thấy section liều dùng."
    assert all(
        "liều" in item["metadata"]["section"].casefold()
        or "cách dùng" in item["metadata"]["section"].casefold()
        for item in dosage
    ), "Bộ lọc trả về section không phải liều dùng."

    missing_drug = pipeline.filter_results_by_intent(
        "Công dụng thuốc Beatil là gì?",
        results,
    )
    assert not missing_drug, "Bộ lọc đã trả về thuốc khác."

    print("PASS: indication filter")
    print("PASS: dosage filter")
    print("PASS: named-drug isolation")



    
    ood_queries = [
        "Thời tiết Hà Nội hôm nay thế nào?",
        "Viết một bài thơ về mùa thu.",
        "Ai là tổng thống Mỹ?",
        "Lập lịch trình du lịch Đà Nẵng 3 ngày.",
        "25 nhân 16 bằng bao nhiêu?",
    ]

    for query in ood_queries:
        assert not pipeline.is_medication_query(query), (
            f"Câu hỏi ngoài phạm vi bị nhận diện nhầm: {query}"
        )

    medication_queries = [
        "Thuốc Voltaren có công dụng gì?",
        "Aspirin có công dụng gì?",
        "Liều dùng thuốc Nexium như thế nào?",
        "Thuốc Beatil có tác dụng phụ gì?",
    ]

    for query in medication_queries:
        assert pipeline.is_medication_query(query), (
            f"Câu hỏi thuốc bị nhận diện nhầm: {query}"
        )

    print("PASS: OOD query detection")
    print("PASS: medication query detection")

    print("All filter tests passed.")


if __name__ == "__main__":
    main()
