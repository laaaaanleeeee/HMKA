import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.api.dependencies import create_rag_pipeline


def main():
    print("Creating RAG pipeline...")

    pipeline = create_rag_pipeline()

    print("RAG pipeline created successfully.")
    print(type(pipeline).__name__)


if __name__ == "__main__":
    main()