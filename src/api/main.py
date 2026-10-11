import logging
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from google.genai import errors as genai_errors
from sqlalchemy.orm import Session

from src.api.dependencies import create_rag_pipeline
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    ConversationHistoryResponse,
    MessageHistoryItem,
)
from src.database.dependencies import get_db
from src.database.models import Conversation, Message

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hospital Medication Knowledge Assistant",
    description="Vietnamese medication knowledge assistant powered by RAG.",
    version="1.0.0",
)

# Load AI models and retrieval components once at startup.
rag_pipeline = create_rag_pipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    try:
        conversation = None

        if request.conversation_id is not None:
            conversation = db.get(
                Conversation,
                request.conversation_id,
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

        history = []

        if conversation is not None:
            recent_messages = (
                db.query(Message)
                .filter(
                    Message.conversation_id == conversation.id
                )
                .order_by(Message.id.desc())
                .limit(6)
                .all()
            )

            recent_messages.reverse()

            history = [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in recent_messages
            ]

        # Run the RAG pipeline once.
        result = rag_pipeline.run(
            query=request.query,
            history=history,
        )

        # Deduplicate sources.
        sources = []
        seen_sources = set()

        for item in result["results"]:
            metadata = item.get("metadata", {})

            source_key = (
                metadata.get("drug_name", "Unknown"),
                metadata.get("section", "Unknown"),
                metadata.get("source_url", ""),
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append(
                Source(
                    drug_name=source_key[0],
                    section=source_key[1],
                    source_url=source_key[2],
                )
            )

        # Create a new conversation only after RAG succeeds.
        if conversation is None:
            conversation = Conversation()
            db.add(conversation)
            db.flush()

        db.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content=request.query,
                ),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=result["answer"],
                ),
            ]
        )

        conversation.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(conversation)

        return ChatResponse(
            conversation_id=conversation.id,
            answer=result["answer"],
            sources=sources,
        )

    except HTTPException:
        db.rollback()
        raise

    except genai_errors.ClientError as exc:
        db.rollback()
        logger.exception("Gemini API client error")

        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exhausted. "
                    "Please check your API quota or billing settings "
                    "and try again later."
                ),
            ) from exc

        raise HTTPException(
            status_code=502,
            detail="The Gemini API rejected the request.",
        ) from exc

    except genai_errors.ServerError as exc:
        db.rollback()
        logger.exception("Gemini API server error")

        raise HTTPException(
            status_code=503,
            detail="The Gemini API is temporarily unavailable.",
        ) from exc

    except RuntimeError as exc:
        db.rollback()
        logger.exception("LLM or RAG runtime error")

        detail = str(exc).lower()

        if "quota" in detail or "429" in detail:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exhausted. "
                    "Please try again later."
                ),
            ) from exc

        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable.",
        ) from exc

    except Exception as exc:
        db.rollback()
        logger.exception(
            "Unexpected error while processing POST /api/v1/chat"
        )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the request.",
        ) from exc


@app.get(
    "/api/v1/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
)
def get_conversation_history(
    conversation_id: int,
    db: Session = Depends(get_db),
):
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )

    return ConversationHistoryResponse(
        conversation_id=conversation.id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageHistoryItem(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )
