from __future__ import annotations

import re
from typing import Iterable, List, Set


WORD_RE = re.compile(r"[a-zA-Z0-9+#.]+")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def tokenize(value: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(value)]


def keyword_overlap_score(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
    q: Set[str] = set(query_tokens)
    d: Set[str] = set(doc_tokens)
    if not q:
        return 0.0
    return len(q & d) / len(q)


def contains_any(text: str, patterns: List[str]) -> bool:
    normalized = normalize_text(text)
    return any(p in normalized for p in patterns)


def is_prompt_injection(text: str) -> bool:
    patterns = [
        "ignore instructions",
        "ignore previous instructions",
        "ignore all instructions",
        "bypass policy",
        "reveal system prompt",
        "developer message",
        "jailbreak",
        "act as",
        "override instructions",
        "forget previous instructions",
        "system prompt",
        "hack",
    ]

    text = text.lower()

    return any(pattern in text for pattern in patterns)

def is_refusal_domain(text: str) -> bool:
    blocked_topics = [
        "legal advice",
        "employment law",
        "lawsuit",
        "compliance interpretation",
        "contract clause",
        "netflix",
        "movie",
        "sports",
        "politics",
        "bitcoin",
        "crypto",
        "medical advice",
        "weather",
        "recipe",
    ]

    text = text.lower()

    return any(topic in text for topic in blocked_topics)
