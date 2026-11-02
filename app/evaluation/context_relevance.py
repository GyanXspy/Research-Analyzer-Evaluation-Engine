"""Context relevance evaluator — measures how relevant the retrieved context is.

Evaluates each context chunk individually to determine whether it is relevant
to answering the user's question.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from app.evaluation.base import BaseEvaluator, extract_json
from app.models.schemas import EvaluationScore

logger = logging.getLogger(__name__)

_CHUNK_RELEVANCE_SYSTEM = (
    "You are a relevance assessor. Determine whether the given context chunk "
    "is relevant to answering the question.\n\n"
    "Respond ONLY with a JSON object in the format:\n"
    '{"relevant": true/false, "reason": "<brief explanation>"}'
)


class ContextRelevanceEvaluator(BaseEvaluator):
    """Evaluates the relevance of each retrieved context chunk to the question.

    Attributes:
        _llm: The language model used as a judge.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialise the context relevance evaluator.

        Args:
            llm: A LangChain chat model instance used for evaluation.
        """
        self._llm = llm

    @property
    def metric_name(self) -> str:
        """Name of the evaluation metric."""
        return "context_relevance"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _assess_chunk(self, question: str, chunk: str) -> dict:
        """Assess the relevance of a single context chunk.

        Args:
            question: The original user question.
            chunk: A single context passage.

        Returns:
            A dict with ``relevant`` (bool) and ``reason`` (str) keys.
        """
        messages = [
            SystemMessage(content=_CHUNK_RELEVANCE_SYSTEM),
            HumanMessage(
                content=f"Question: {question}\n\nContext chunk:\n{chunk}"
            ),
        ]
        response = await self._llm.ainvoke(messages)
        try:
            parsed = extract_json(response)
            if isinstance(parsed, dict):
                return parsed
            return {"relevant": False, "reason": "Non-dictionary response."}
        except Exception as exc:
            logger.warning("Chunk relevance parsing fallback: %s", exc)
            return {"relevant": False, "reason": "Failed to parse LLM response."}

    async def evaluate(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate how relevant the retrieved context chunks are.

        Args:
            question: The original user question.
            answer: The generated answer (not used directly by this evaluator).
            context: List of retrieved context passages to assess.
            ground_truth: Not used by this evaluator (accepted for interface compat).

        Returns:
            EvaluationScore with the context_relevance metric result.
        """
        logger.info("Running context relevance evaluation on %d chunks.", len(context))

        if not context:
            logger.info("No context chunks provided — returning zero score.")
            return EvaluationScore(
                metric_name=self.metric_name,
                score=0.0,
                reasoning="No context chunks were provided for evaluation.",
                details={"total_chunks": 0, "relevant_chunks": 0, "chunks": []},
            )

        import asyncio

        chunk_results = await asyncio.gather(
            *(self._assess_chunk(question, chunk) for chunk in context)
        )

        # Build per-chunk breakdown
        chunk_details: list[dict] = []
        relevant_count = 0
        for i, result in enumerate(chunk_results):
            is_relevant = result.get("relevant", False)
            if is_relevant:
                relevant_count += 1
            chunk_details.append(
                {
                    "chunk_index": i,
                    "relevant": is_relevant,
                    "reason": result.get("reason", "No reason provided."),
                    "chunk_preview": context[i][:200],
                }
            )

        total_count = len(context)
        score = relevant_count / total_count

        reasoning = (
            f"{relevant_count}/{total_count} context chunks are relevant "
            f"to the question."
        )

        logger.info("Context relevance score: %.2f (%d/%d)", score, relevant_count, total_count)

        return EvaluationScore(
            metric_name=self.metric_name,
            score=score,
            reasoning=reasoning,
            details={
                "total_chunks": total_count,
                "relevant_chunks": relevant_count,
                "chunks": chunk_details,
            },
        )

