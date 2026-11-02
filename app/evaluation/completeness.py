"""Completeness evaluator — measures how thoroughly the answer covers the context.

Assesses whether the answer includes all key points available in the retrieved
context and identifies any important omissions.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.evaluation.base import BaseEvaluator, extract_json
from app.models.schemas import EvaluationScore

logger = logging.getLogger(__name__)

_COMPLETENESS_SYSTEM = (
    "You are an expert completeness evaluator for research analysis. "
    "Evaluate the completeness of the answer given the question and available "
    "context.\n\n"
    "Assess:\n"
    "1. Does the answer cover all key points available in the context?\n"
    "2. Are there important details from the context that were omitted?\n"
    "3. Is the answer comprehensive enough for a research analysis?\n\n"
    "Respond ONLY with a JSON object in the format:\n"
    "{\n"
    '  "score": <float 0-1>,\n'
    '  "reasoning": "<explanation>",\n'
    '  "covered_points": ["point 1", ...],\n'
    '  "missing_points": ["point 1", ...]\n'
    "}"
)


class CompletenessEvaluator(BaseEvaluator):
    """Evaluates the completeness of the answer relative to available context.

    Attributes:
        _llm: The language model used as a judge.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialise the completeness evaluator.

        Args:
            llm: A LangChain chat model instance used for evaluation.
        """
        self._llm = llm

    @property
    def metric_name(self) -> str:
        """Name of the evaluation metric."""
        return "completeness"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _assess_completeness(
        self, question: str, answer: str, context: list[str]
    ) -> dict:
        """Ask the LLM to assess answer completeness.

        Args:
            question: The original user question.
            answer: The generated answer text.
            context: List of context passages.

        Returns:
            A dict with score, reasoning, covered_points, and missing_points.
        """
        combined_context = "\n---\n".join(context)
        messages = [
            SystemMessage(content=_COMPLETENESS_SYSTEM),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    f"Context:\n{combined_context}"
                )
            ),
        ]
        response = await self._llm.ainvoke(messages)
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed
            return {
                "score": 0.5,
                "reasoning": str(parsed),
                "covered_points": [],
                "missing_points": [],
            }
        except Exception as exc:
            logger.warning("Completeness parsing fallback: %s", exc)
            return {
                "score": 0.0,
                "reasoning": "Failed to parse LLM response.",
                "covered_points": [],
                "missing_points": [],
            }

    async def evaluate(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate the completeness of the answer.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate.
            context: List of retrieved context passages.
            ground_truth: Not used by this evaluator (accepted for interface compat).

        Returns:
            EvaluationScore with the completeness metric result.
        """
        logger.info("Running completeness evaluation.")

        if not context:
            logger.info("No context chunks provided — cannot assess completeness.")
            return EvaluationScore(
                metric_name=self.metric_name,
                score=0.0,
                reasoning="No context chunks were provided to assess completeness against.",
                details={"covered_points": [], "missing_points": []},
            )

        result = await self._assess_completeness(question, answer, context)

        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = result.get("reasoning", "No reasoning provided.")

        logger.info("Completeness score: %.2f", score)

        return EvaluationScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=reasoning,
            details={
                "covered_points": result.get("covered_points", []),
                "missing_points": result.get("missing_points", []),
            },
        )

