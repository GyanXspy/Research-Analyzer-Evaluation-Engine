"""Faithfulness evaluator — measures whether the answer is grounded in context.

Performs a two-step LLM evaluation:
1. Extract factual claims from the answer.
2. Verify each claim against the provided context.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.evaluation.base import BaseEvaluator, extract_json
from app.models.schemas import EvaluationScore

logger = logging.getLogger(__name__)

_CLAIM_EXTRACTION_SYSTEM = (
    "You are a precise claim extractor. Extract all distinct factual claims "
    "from the given answer. Return ONLY a JSON object in the format:\n"
    '{"claims": ["claim 1", "claim 2", ...]}\n'
    "If there are no factual claims, return {\"claims\": []}."
)

_CLAIM_VERIFICATION_SYSTEM = (
    "You are a meticulous fact-checker. For each claim, determine whether it "
    "is supported by the provided context. Respond ONLY with a JSON object in "
    "the format:\n"
    '{"claims": [\n'
    '  {"claim": "...", "supported": true, "evidence": "..."},\n'
    '  {"claim": "...", "supported": false, "evidence": "No supporting evidence found"}\n'
    "]}"
)


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates whether the answer's factual claims are supported by context.

    Attributes:
        _llm: The language model used as a judge.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialise the faithfulness evaluator.

        Args:
            llm: A LangChain chat model instance used for evaluation.
        """
        self._llm = llm

    @property
    def metric_name(self) -> str:
        """Name of the evaluation metric."""
        return "faithfulness"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _extract_claims(self, answer: str) -> list[str]:
        """Extract factual claims from the answer using the LLM.

        Args:
            answer: The generated answer text.

        Returns:
            A list of claim strings.
        """
        messages = [
            SystemMessage(content=_CLAIM_EXTRACTION_SYSTEM),
            HumanMessage(content=f"Answer:\n{answer}"),
        ]
        response = await self._llm.ainvoke(messages)
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed.get("claims", [])
            elif isinstance(parsed, list):
                return parsed
            return []
        except Exception as exc:
            logger.warning("Claim extraction parsing failed: %s", exc)
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _verify_claims(
        self, claims: list[str], context: list[str]
    ) -> list[dict]:
        """Verify each claim against the provided context.

        Args:
            claims: List of factual claim strings.
            context: List of context passages.

        Returns:
            A list of dicts with claim, supported, and evidence keys.
        """
        combined_context = "\n---\n".join(context)
        claims_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

        messages = [
            SystemMessage(content=_CLAIM_VERIFICATION_SYSTEM),
            HumanMessage(
                content=(
                    f"Context:\n{combined_context}\n\n"
                    f"Claims to verify:\n{claims_text}"
                )
            ),
        ]
        response = await self._llm.ainvoke(messages)
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed.get("claims", [])
            elif isinstance(parsed, list):
                return parsed
            return []
        except Exception as exc:
            logger.warning("Claim verification parsing failed: %s", exc)
            return []

    async def evaluate(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate faithfulness of the answer to the context.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate.
            context: List of retrieved context passages.
            ground_truth: Not used by this evaluator (accepted for interface compat).

        Returns:
            EvaluationScore with the faithfulness metric result.
        """
        logger.info("Running faithfulness evaluation.")

        # Step 1: extract claims
        claims = await self._extract_claims(answer)
        if not claims:
            logger.info("No factual claims extracted — returning perfect score.")
            return EvaluationScore(
                metric_name=self.metric_name,
                score=1.0,
                reasoning="No factual claims found in the answer to verify.",
                details={"total_claims": 0, "supported_claims": 0, "claims": []},
            )

        # Step 2: verify claims
        verification_results = await self._verify_claims(claims, context)

        supported_count = sum(
            1 for v in verification_results if v.get("supported", False)
        )
        total_count = len(claims)
        score = supported_count / total_count if total_count > 0 else 1.0

        # Build reasoning summary
        unsupported = [
            v.get("claim", "unknown")
            for v in verification_results
            if not v.get("supported", False)
        ]
        reasoning_parts: list[str] = [
            f"{supported_count}/{total_count} claims are supported by the context."
        ]
        if unsupported:
            reasoning_parts.append(
                f"Unsupported claims: {'; '.join(unsupported)}"
            )

        logger.info("Faithfulness score: %.2f (%d/%d)", score, supported_count, total_count)

        return EvaluationScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=" ".join(reasoning_parts),
            details={
                "total_claims": total_count,
                "supported_claims": supported_count,
                "claims": verification_results,
            },
        )

