from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


load_dotenv()


@dataclass(frozen=True)
class Settings:
    assessments_json: Path = Path(os.getenv("ASSESSMENTS_JSON", "data/assessments.json"))
    faiss_index_dir: Path = Path(os.getenv("FAISS_INDEX_DIR", "data/faiss_index"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    keyword_weight: float = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
    top_k: int = int(os.getenv("TOP_K", "10"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        total = self.semantic_weight + self.keyword_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("SEMANTIC_WEIGHT + KEYWORD_WEIGHT must equal 1.0")


settings = Settings()
settings.validate()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
