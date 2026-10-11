import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.rag.llm import GeminiLLM


def main():
    llm = GeminiLLM()

    try:
        answer = llm.generate(
            "Trả lời ngắn gọn bằng tiếng Việt: Xin chào."
        )
        print("Gemini is working.")
        print("Response:", answer)
    except Exception as exc:
        print("Gemini request failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()