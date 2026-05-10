from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import AssessmentCatalogItem
from app.utils import keyword_overlap_score, tokenize

logger = logging.getLogger(__name__)


IMPORTANT_TERMS = [
    "python",
    "java",
    "backend",
    "frontend",
    "cloud",
    "aws",
    "azure",
    "sql",
    "devops",
    "javascript",
    "react",
    "django",
    "flask",
    "api",
    "coding",
    "software",
]


def infer_test_type(keys: list[str]) -> str:
    """
    Infer assessment category from SHL metadata.
    """

    joined = " ".join(keys).lower()

    if "knowledge" in joined or "skills" in joined:
        return "Technical"

    if "personality" in joined or "behavior" in joined:
        return "Personality"

    if "ability" in joined or "aptitude" in joined:
        return "Cognitive"

    if "situational" in joined:
        return "Behavioral"

    return "General"


class HybridRetriever:

    def __init__(self) -> None:

        self.model: SentenceTransformer | None = None

        self._use_fallback_encoder = False

        self.index: faiss.Index | None = None

        self.items: List[AssessmentCatalogItem] = []

        self.doc_tokens: List[List[str]] = []

        self.metadata_path = (
            settings.faiss_index_dir / "metadata.json"
        )

        self.index_path = (
            settings.faiss_index_dir / "index.faiss"
        )

    def load_or_build(self) -> None:

        self._load_embedding_model()

        logger.info("Loading assessment catalog")

        self.items = self._load_catalog(
            settings.assessments_json
        )

        logger.info(
            "Loaded %s assessments",
            len(self.items),
        )

        self.doc_tokens = [
            tokenize(self._to_document(item))
            for item in self.items
        ]

        settings.faiss_index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            self.index_path.exists()
            and self.metadata_path.exists()
        ):

            logger.info("Loading FAISS index from disk")

            self.index = faiss.read_index(
                str(self.index_path)
            )

            return

        logger.info("Building FAISS index")

        self._build_index()

    def _load_embedding_model(self) -> None:

        try:

            logger.info("Loading embedding model")

            self.model = SentenceTransformer(
                settings.embedding_model
            )

            self._use_fallback_encoder = False

        except Exception as exc:

            logger.warning(
                "Could not initialize embedding model (%s). "
                "Using fallback hashing encoder.",
                exc,
            )

            self.model = None

            self._use_fallback_encoder = True

    def _build_index(self) -> None:

        texts = [
            self._to_document(item)
            for item in self.items
        ]

        vectors = self._encode(texts)

        index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        index.add(vectors)

        faiss.write_index(
            index,
            str(self.index_path),
        )

        metadata = [
            {
                "name": item.name,
                "url": item.url,
                "test_type": item.test_type,
            }
            for item in self.items
        ]

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        self.index = index

        logger.info(
            "FAISS index built successfully"
        )

    def _load_catalog(
        self,
        path: Path,
    ) -> List[AssessmentCatalogItem]:

        raw = json.loads(
            path.read_text(encoding="utf-8")
        )

        items: List[AssessmentCatalogItem] = []

        for row in raw:

            row["url"] = row.get("link", "")

            row["test_type"] = infer_test_type(
                row.get("keys", [])
            )

            items.append(
                AssessmentCatalogItem(**row)
            )

        return items

    def _to_document(
        self,
        item: AssessmentCatalogItem,
    ) -> str:

        return " | ".join([
            item.name,
            item.description,
            " ".join(item.keys),
            " ".join(item.job_levels),
            item.test_type,
            item.duration or "",
            "remote" if item.remote == "yes" else "",
            "adaptive" if item.adaptive == "yes" else "",
        ])

    def _encode(
        self,
        texts: List[str],
    ) -> np.ndarray:

        if (
            self.model
            and not self._use_fallback_encoder
        ):

            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
            )

            return np.array(
                vectors,
                dtype=np.float32,
            )

        logger.warning(
            "Using fallback hashing encoder"
        )

        dim = 384

        vectors = np.zeros(
            (len(texts), dim),
            dtype=np.float32,
        )

        for row, text in enumerate(texts):

            for token in tokenize(text):

                digest = hashlib.md5(
                    token.encode("utf-8")
                ).hexdigest()

                idx = int(digest, 16) % dim

                vectors[row, idx] += 1.0

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        return vectors / norms

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Dict[str, object] | None = None,
    ) -> List[AssessmentCatalogItem]:

        if not self.index:
            raise RuntimeError(
                "Retriever index not initialized."
            )

        k = max(
            1,
            min(
                top_k or settings.top_k,
                10,
            ),
        )

        query_vec = self._encode([query])

        semantic_scores, idxs = self.index.search(
            query_vec,
            len(self.items),
        )

        semantic_map = {
            int(doc_idx): float(score)
            for score, doc_idx in zip(
                semantic_scores[0],
                idxs[0],
                strict=True,
            )
            if doc_idx != -1
        }

        query_tokens = tokenize(query)

        query_lower = query.lower()

        scored: List[Tuple[float, int]] = []

        for i, item in enumerate(self.items):

            if (
                filters
                and not self._passes_filters(
                    item,
                    filters,
                )
            ):
                continue

            semantic = semantic_map.get(i, 0.0)

            keyword = keyword_overlap_score(
                query_tokens,
                self.doc_tokens[i],
            )

            boost = self._compute_boost(
                query_lower,
                item,
            )

            final_score = (
                settings.semantic_weight * semantic
                + settings.keyword_weight * keyword
                + boost
            )

            scored.append((final_score, i))

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            self.items[i]
            for score, i in scored[:k]
            if score > 0
        ]

    def _compute_boost(
        self,
        query: str,
        item: AssessmentCatalogItem,
    ) -> float:

        boost = 0.0

        name_lower = item.name.lower()

        desc_lower = item.description.lower()

        for term in IMPORTANT_TERMS:

            if term in query:

                if term in name_lower:
                    boost += 0.40

                if term in desc_lower:
                    boost += 0.20

        if "backend" in query:

            if (
                "backend" in name_lower
                or "backend" in desc_lower
            ):
                boost += 0.30

        if "coding" in query:

            if (
                "coding" in name_lower
                or "coding" in desc_lower
            ):
                boost += 0.30

        if "leadership" in query:

            if (
                "leadership" in name_lower
                or "leadership" in desc_lower
            ):
                boost += 0.25

        if "communication" in query:

            if (
                "communication" in name_lower
                or "communication" in desc_lower
            ):
                boost += 0.25

        return boost

    def find_by_name(
        self,
        text: str,
    ) -> List[AssessmentCatalogItem]:

        target = text.lower()

        return [
            item
            for item in self.items
            if (
                item.name.lower() in target
                or target in item.name.lower()
            )
        ]

    def _passes_filters(
        self,
        item: AssessmentCatalogItem,
        filters: Dict[str, object],
    ) -> bool:

        seniority = filters.get("seniority")

        if seniority:

            mapped_levels = {
                "entry": [
                    "Entry-Level",
                    "Graduate",
                ],
                "junior": [
                    "Entry-Level",
                    "Graduate",
                ],
                "mid": [
                    "Mid-Professional",
                    "Professional Individual Contributor",
                ],
                "senior": [
                    "Manager",
                    "Front Line Manager",
                    "Senior",
                ],
                "executive": [
                    "Executive",
                    "Director",
                    "VP",
                ],
            }

            allowed = mapped_levels.get(
                str(seniority).lower(),
                [],
            )

            if allowed:

                if not any(
                    level in item.job_levels
                    for level in allowed
                ):
                    return False

        preferred_types = filters.get(
            "test_types"
        )

        if preferred_types:

            item_type = item.test_type.lower()

            if not any(
                str(p).lower() in item_type
                for p in preferred_types
            ):
                return False

        return True