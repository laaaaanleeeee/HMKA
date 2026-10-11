import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.database.database import Base, engine
from src.database.models import Conversation, Message


def main():
    print("Creating database tables...")

    Base.metadata.create_all(bind=engine)

    print("Database created successfully.")
    print(f"Database location: {engine.url}")


if __name__ == "__main__":
    main()