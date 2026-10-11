from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="User's medication-related question",
    )
    conversation_id: int | None = Field(
        default=None,
        gt=0,
        description="ID of an existing conversation, if available",
    )


class Source(BaseModel):
    drug_name: str
    section: str
    source_url: str


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[Source]

class MessageHistoryItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    created_at: datetime
    updated_at: datetime
    messages: list[MessageHistoryItem]