from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=40)

    @field_validator("messages")
    @classmethod
    def at_least_one_user_message(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if not any(m.role == MessageRole.USER for m in value):
            raise ValueError("At least one user message is required.")
        return value


class Recommendation(BaseModel):
    name: str
    url: HttpUrl
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]


class AssessmentCatalogItem(BaseModel):
    entity_id: str
    name: str
    url: str
    description: str
    duration: str | None = None
    job_levels: list[str] = []
    languages: list[str] = []
    keys: list[str] = []
    remote: str | None = None
    adaptive: str | None = None
    test_type: str


class CandidateContext(BaseModel):
    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    preference_types: List[str] = Field(default_factory=list)
    personality_required: bool = False
    leadership_required: bool = False
    client_facing_required: bool = False
