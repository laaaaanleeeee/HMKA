
import json
from pathlib import Path

files = [
    Path("evaluation/rag/results/rag_results.json"),
    Path("evaluation/rag/results/rag_results_ollama_qwen3_4b.json"),
]

for path in files:
    print(f"\n{'=' * 60}\nFILE: {path}")
    if not path.exists():
        print("Không tìm thấy file.")
        continue

    with path.open("r", encoding="utf-8-sig") as f:
        rows = json.load(f)

    successful = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    print(f"Tổng số dòng: {len(rows)}")
    print(f"Thành công: {len(successful)}")
    print(f"Lỗi: {len(failed)}")
    print(f"ID đã có: {[r.get('id') for r in rows]}")
    if rows:
        print("Query đầu tiên:", rows[0].get("query"))
        print("Answer đầu tiên:", rows[0].get("answer", "")[:200])
        print("Model:", rows[0].get("model"))
