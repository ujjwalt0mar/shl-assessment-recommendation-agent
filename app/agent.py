from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from app.models import (
    AssessmentCatalogItem,
    CandidateContext,
    ChatMessage,
    ChatResponse,
    Recommendation,
)

from app.prompts import (
    PROMPT_INJECTION_MESSAGE,
    REFUSAL_MESSAGE,
)

from app.retriever import HybridRetriever

from app.utils import (
    is_prompt_injection,
    is_refusal_domain,
    normalize_text,
    tokenize,
)

logger = logging.getLogger(__name__)


SENIORITY_PATTERNS = {
    "entry": ["entry", "graduate", "fresher", "intern"],
    "junior": ["junior", "jr"],
    "mid": ["mid", "mid-level", "intermediate"],
    "senior": ["senior", "sr", "lead"],
    "executive": ["director", "vp", "head", "executive"],
}


TECH_KEYWORDS = [
    "java",
    "python",
    "sql",
    "javascript",
    "cloud",
    "aws",
    "azure",
    "devops",
    "backend",
    "frontend",
    "software",
    "developer",
    "engineering",
]


PERSONALITY_KEYWORDS = [
    "communication",
    "stakeholder",
    "teamwork",
    "collaboration",
    "leadership",
    "behavior",
    "personality",
    "culture",
]


@dataclass
class AgentResult:
    reply: str
    recommendations: List[AssessmentCatalogItem]
    end_of_conversation: bool


class RecommendationAgent:

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def respond(self, messages: List[ChatMessage]) -> ChatResponse:

        user_messages = [
            m.content
            for m in messages
            if m.role.value == "user"
        ]

        latest_user = user_messages[-1]

        conversation_text = "\n".join(user_messages)

        if is_prompt_injection(latest_user):

            return ChatResponse(
                reply=PROMPT_INJECTION_MESSAGE,
                recommendations=[],
                end_of_conversation=True,
            )

        if is_refusal_domain(latest_user):

            return ChatResponse(
                reply=REFUSAL_MESSAGE,
                recommendations=[],
                end_of_conversation=True,
            )

        comparisons = self._extract_comparison_targets(latest_user)

        if len(comparisons) == 2:
            return self._compare_assessments(comparisons)

        context = self._extract_context(conversation_text)

        missing = self._missing_fields(context)

        turn_count = len(user_messages)

        if missing and turn_count < 8:

            return ChatResponse(
                reply=self._next_question(missing),
                recommendations=[],
                end_of_conversation=False,
            )

        recommendations = self._retrieve_recommendations(context)

        if not recommendations:

            return ChatResponse(
                reply=(
                    "I could not find strong SHL assessment matches yet. "
                    "Please specify required technical skills or desired assessment types."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        reply = self._recommendation_reply(
            context,
            recommendations,
        )

        return ChatResponse(
            reply=reply,
            recommendations=[
                Recommendation(
                    name=item.name,
                    url=item.url,
                    test_type=item.test_type,
                )
                for item in recommendations
            ],
            end_of_conversation=turn_count >= 8,
        )

    def _extract_context(self, text: str) -> CandidateContext:

        t = normalize_text(text)

        skills = self._extract_skills(t)

        role = self._extract_role(t)

        seniority = None

        for label, variants in SENIORITY_PATTERNS.items():

            if any(v in t for v in variants):
                seniority = label
                break

        preference_types = []

        if any(k in t for k in TECH_KEYWORDS):
            preference_types.append("Technical")

        if any(k in t for k in PERSONALITY_KEYWORDS):
            preference_types.append("Personality")

        if "aptitude" in t or "reasoning" in t:
            preference_types.append("Cognitive")

        return CandidateContext(
            role=role,
            seniority=seniority,
            skills=skills,
            preference_types=sorted(set(preference_types)),
            personality_required="Personality" in preference_types,
            leadership_required="leadership" in t,
            client_facing_required="stakeholder" in t or "client" in t,
        )

    def _extract_role(self, text: str) -> str | None:

        patterns = [
            r"(java developer)",
            r"(python developer)",
            r"(software engineer)",
            r"(backend engineer)",
            r"(frontend developer)",
            r"(data analyst)",
            r"(cloud engineer)",
            r"(devops engineer)",
            r"(manager)",
        ]

        for pattern in patterns:

            match = re.search(pattern, text)

            if match:
                return match.group(1).title()

        return None

    def _extract_skills(self, text: str) -> List[str]:

        known = list(set(
            TECH_KEYWORDS + PERSONALITY_KEYWORDS
        ))

        return [
            s.title()
            for s in known
            if s in text
        ]

    def _missing_fields(self, context: CandidateContext) -> List[str]:

        missing = []

        if not context.role:
            missing.append("role")

        if not context.seniority:
            missing.append("seniority")

        return missing

    def _next_question(self, missing: List[str]) -> str:

        if "role" in missing:
            return "Which role are you hiring for?"

        if "seniority" in missing:
            return (
                "What seniority level is this role "
                "(entry, junior, mid, senior, executive)?"
            )

        return "Could you share more details about the role?"

    def _retrieve_recommendations(
        self,
        context: CandidateContext,
    ) -> List[AssessmentCatalogItem]:

        final_results = []

        seen = set()

        if "Technical" in context.preference_types:

            technical_query = (
                f"{context.role or ''} "
                f"{' '.join(context.skills)} "
                f"coding software engineering technical assessment"
            )

            technical_results = self.retriever.retrieve(
                query=technical_query,
                top_k=5,
                filters={
                    "seniority": context.seniority,
                    "test_types": ["Technical"],
                },
            )

            for item in technical_results:

                if item.name not in seen:
                    final_results.append(item)
                    seen.add(item.name)

        if "Personality" in context.preference_types:

            personality_query = (
                "communication teamwork stakeholder "
                "collaboration personality behavioral assessment"
            )

            personality_results = self.retriever.retrieve(
                query=personality_query,
                top_k=5,
                filters={
                    "seniority": context.seniority,
                    "test_types": ["Personality"],
                },
            )

            for item in personality_results:

                if item.name not in seen:
                    final_results.append(item)
                    seen.add(item.name)

        if "Cognitive" in context.preference_types:

            cognitive_query = (
                "aptitude reasoning numerical verbal cognitive assessment"
            )

            cognitive_results = self.retriever.retrieve(
                query=cognitive_query,
                top_k=3,
                filters={
                    "seniority": context.seniority,
                    "test_types": ["Cognitive"],
                },
            )

            for item in cognitive_results:

                if item.name not in seen:
                    final_results.append(item)
                    seen.add(item.name)

        if not final_results:

            fallback_query = (
                f"{context.role or ''} "
                f"{' '.join(context.skills)}"
            )

            fallback_results = self.retriever.retrieve(
                query=fallback_query,
                top_k=10,
                filters={
                    "seniority": context.seniority,
                },
            )

            final_results.extend(fallback_results)

        return final_results[:10]

    def _recommendation_reply(
        self,
        context: CandidateContext,
        recs: List[AssessmentCatalogItem],
    ) -> str:

        role = context.role or "this role"

        if context.preference_types:

            focus = ", ".join(context.preference_types)

            return (
                f"Here are recommended SHL assessments for "
                f"{role} with {focus} coverage."
            )

        return f"Here are the best SHL assessments for {role}."

    def _extract_comparison_targets(
        self,
        latest_user: str,
    ) -> List[str]:

        lower = normalize_text(latest_user)

        if "compare" not in lower and "difference" not in lower:
            return []

        names = []

        for item in self.retriever.items:

            if item.name.lower() in lower:
                names.append(item.name)

        return names[:2]

    def _compare_assessments(
        self,
        names: List[str],
    ) -> ChatResponse:

        a = next(
            (i for i in self.retriever.items if i.name == names[0]),
            None,
        )

        b = next(
            (i for i in self.retriever.items if i.name == names[1]),
            None,
        )

        if not a or not b:

            return ChatResponse(
                reply="I could not find both assessments.",
                recommendations=[],
                end_of_conversation=False,
            )

        comparison = (
            f"{a.name} focuses on {a.test_type.lower()} assessments "
            f"and is designed for {', '.join(a.job_levels[:3])}. "
            f"{b.name} focuses on {b.test_type.lower()} assessments "
            f"and is designed for {', '.join(b.job_levels[:3])}."
        )

        return ChatResponse(
            reply=comparison,
            recommendations=[
                Recommendation(
                    name=a.name,
                    url=a.url,
                    test_type=a.test_type,
                ),
                Recommendation(
                    name=b.name,
                    url=b.url,
                    test_type=b.test_type,
                ),
            ],
            end_of_conversation=False,
        )