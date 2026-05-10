from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.agent import RecommendationAgent
from app.models import ChatRequest, ChatResponse, HealthResponse


logger = logging.getLogger(__name__)
router = APIRouter()


def get_agent(request: Request) -> RecommendationAgent:
    return request.app.state.agent


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    agent: RecommendationAgent = Depends(get_agent),
) -> ChatResponse:
    try:
        return agent.respond(payload.messages)
    except ValueError as exc:
        logger.warning("Validation error in /chat: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
