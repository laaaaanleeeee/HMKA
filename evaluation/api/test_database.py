import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.database.dependencies import get_db
from src.database.models import Conversation, Message


def main():
    db_generator = get_db()
    db = next(db_generator)

    try:
        conversations_count = db.query(Conversation).count()
        messages_count = db.query(Message).count()

        print("Database session created successfully.")
        print(f"Conversations: {conversations_count}")
        print(f"Messages: {messages_count}")
    finally:
        db_generator.close()


if __name__ == "__main__":
    main()