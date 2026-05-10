from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent import RecommendationAgent
from app.retriever import HybridRetriever
from app.routes import router


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    retriever = HybridRetriever()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing retriever resources")
        retriever.load_or_build()
        app.state.agent = RecommendationAgent(retriever=retriever)
        yield

    app = FastAPI(
        title="SHL Assessment Recommendation Agent",
        description="Stateless conversational API for SHL assessment recommendations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(router)
    return app


app = create_app()
