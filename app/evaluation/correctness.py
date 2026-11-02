"""Correctness evaluator — compares the answer against a ground-truth reference.

Uses semantic comparison (not exact string matching) to determine how
accurate the generated answer is relative to the known-correct answer.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.evaluation.base import BaseEvaluator, extract_json
from app.models.schemas import EvaluationScore

logger = logging.getLogger(__name__)

_CORRECTNESS_SYSTEM = (
    "You are an expert answer evaluator. Compare the AI-generated answer "
    "against the ground truth answer. Consider semantic meaning, not exact "
    "wording.\n\n"
    "Respond ONLY with a JSON object in the format:\n"
    "{\n"
    '  "score": <float 0-1>,\n'
    '  "reasoning": "<explanation>",\n'
    '  "matching_points": ["point 1", ...],\n'
    '  "missing_points": ["point 1", ...],\n'
    '  "incorrect_points": ["point 1", ...]\n'
    "}"
)


class CorrectnessEvaluator(BaseEvaluator):
    """Evaluates the correctness of an answer against a ground-truth reference.

    Attributes:
        _llm: The language model used as a judge.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialise the correctness evaluator.

        Args:
            llm: A LangChain chat model instance used for evaluation.
        """
        self._llm = llm

    @property
    def metric_name(self) -> str:
        """Name of the evaluation metric."""
        return "correctness"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _compare_answers(
        self, question: str, answer: str, ground_truth: str
    ) -> dict:
        """Ask the LLM to compare the answer against ground truth.

        Args:
            question: The original user question.
            answer: The AI-generated answer.
            ground_truth: The known-correct answer.

        Returns:
            A dict with score, reasoning, matching/missing/incorrect points.
        """
        messages = [
            SystemMessage(content=_CORRECTNESS_SYSTEM),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"AI Answer: {answer}\n\n"
                    f"Ground Truth: {ground_truth}"
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
                "matching_points": [],
                "missing_points": [],
                "incorrect_points": [],
            }
        except Exception as exc:
            logger.warning("Correctness parsing fallback: %s", exc)
            return {
                "score": 0.0,
                "reasoning": "Failed to parse LLM response.",
                "matching_points": [],
                "missing_points": [],
                "incorrect_points": [],
            }

    async def evaluate(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate the correctness of the answer against ground truth.

        If no ground truth is provided, returns a score of ``-1.0`` to signal
        that this metric is not applicable.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate.
            context: List of retrieved context passages (not used directly).
            ground_truth: The known-correct answer for comparison.

        Returns:
            EvaluationScore with the correctness metric result.
        """
        if ground_truth is None:
            logger.info("No ground truth provided — skipping correctness evaluation.")
            return EvaluationScore(
                metric_name=self.metric_name,
                score=0.0,
                reasoning="No ground truth provided. Correctness cannot be assessed.",
                details={"skipped": True},
            )

        logger.info("Running correctness evaluation.")

        result = await self._compare_answers(question, answer, ground_truth)

        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = result.get("reasoning", "No reasoning provided.")

        logger.info("Correctness score: %.2f", score)

        return EvaluationScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=reasoning,
            details={
                "matching_points": result.get("matching_points", []),
                "missing_points": result.get("missing_points", []),
                "incorrect_points": result.get("incorrect_points", []),
            },
        )

