"""Answer relevance evaluator — measures how well the answer addresses the question.

Uses a rubric-based LLM evaluation to score the relevance of the generated
answer with respect to the original question.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.evaluation.base import BaseEvaluator, extract_json
from app.models.schemas import EvaluationScore

logger = logging.getLogger(__name__)

_RELEVANCE_SYSTEM = (
    "You are an expert evaluator. Assess how relevant the answer is to the "
    "question using the following rubric:\n\n"
    "Score 1.0: Answer directly and completely addresses the question\n"
    "Score 0.75: Answer mostly addresses the question with minor gaps\n"
    "Score 0.5: Answer partially addresses the question\n"
    "Score 0.25: Answer barely addresses the question\n"
    "Score 0.0: Answer is completely irrelevant\n\n"
    "Respond ONLY with a JSON object in the format:\n"
    '{"score": <float>, "reasoning": "<explanation>"}'
)


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Evaluates the relevance of an answer to the original question.

    Attributes:
        _llm: The language model used as a judge.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialise the answer relevance evaluator.

        Args:
            llm: A LangChain chat model instance used for evaluation.
        """
        self._llm = llm

    @property
    def metric_name(self) -> str:
        """Name of the evaluation metric."""
        return "answer_relevance"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _score_relevance(self, question: str, answer: str) -> dict:
        """Ask the LLM to score the answer's relevance.

        Args:
            question: The original user question.
            answer: The generated answer text.

        Returns:
            A dict with ``score`` and ``reasoning`` keys.
        """
        messages = [
            SystemMessage(content=_RELEVANCE_SYSTEM),
            HumanMessage(
                content=f"Question: {question}\nAnswer: {answer}"
            ),
        ]
        response = await self._llm.ainvoke(messages)
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed
            return {"score": 0.5, "reasoning": str(parsed)}
        except Exception as exc:
            logger.warning("Relevance parsing fallback: %s", exc)
            return {"score": 0.0, "reasoning": "Failed to parse LLM response."}

    async def evaluate(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate how relevant the answer is to the question.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate.
            context: List of retrieved context passages (not used directly).
            ground_truth: Not used by this evaluator (accepted for interface compat).

        Returns:
            EvaluationScore with the answer_relevance metric result.
        """
        logger.info("Running answer relevance evaluation.")

        result = await self._score_relevance(question, answer)

        score = float(result.get("score", 0.0))
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        reasoning = result.get("reasoning", "No reasoning provided.")

        logger.info("Answer relevance score: %.2f", score)

        return EvaluationScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=reasoning,
            details=result,
        )

