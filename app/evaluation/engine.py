"""Evaluation engine — orchestrates all evaluation metrics concurrently.

Provides a single entry-point for running the full evaluation suite or
individual metrics against a RAG response.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.evaluation.answer_relevance import AnswerRelevanceEvaluator
from app.evaluation.completeness import CompletenessEvaluator
from app.evaluation.context_relevance import ContextRelevanceEvaluator
from app.evaluation.correctness import CorrectnessEvaluator
from app.evaluation.faithfulness import FaithfulnessEvaluator
from app.models.schemas import EvaluationRequest, EvaluationResult, EvaluationScore

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Orchestrates concurrent execution of all RAG evaluation metrics.

    Attributes:
        _settings: Application settings instance.
        _evaluators: Mapping of metric names to evaluator instances.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the evaluation engine with a judge LLM.

        Creates the judge LLM based on the configured provider and model,
        then instantiates all five evaluator components.

        Args:
            settings: Application settings with LLM and evaluation config.
        """
        self._settings = settings
        llm = self._create_judge_llm()

        self._evaluators: dict[str, Any] = {
            "faithfulness": FaithfulnessEvaluator(llm),
            "answer_relevance": AnswerRelevanceEvaluator(llm),
            "context_relevance": ContextRelevanceEvaluator(llm),
            "correctness": CorrectnessEvaluator(llm),
            "completeness": CompletenessEvaluator(llm),
        }

        logger.info(
            "EvaluationEngine initialised with model '%s' and %d evaluators.",
            getattr(settings, "eval_llm_model", "default"),
            len(self._evaluators),
        )

    def _create_judge_llm(self) -> Any:
        """Create the judge LLM instance based on application settings.

        Returns:
            A LangChain chat model instance (ChatOpenAI or ChatGoogleGenerativeAI).

        Raises:
            ValueError: If the configured LLM provider is not supported.
        """
        provider = self._settings.llm_provider.lower()
        model_name = getattr(self._settings, "eval_llm_model", "").strip()

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            if not model_name or "gemini" in model_name:
                model_name = "gpt-4o-mini"
            logger.info("Using OpenAI Judge LLM: %s", model_name)
            return ChatOpenAI(
                model=model_name,
                api_key=self._settings.openai_api_key,
                temperature=0.0,
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            if not model_name or "gpt-" in model_name or model_name in ("gemini-1.5-flash", "gemini-2.5-flash"):
                model_name = "gemini-3.5-flash-lite"
            logger.info("Using Google Judge LLM: %s", model_name)
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self._settings.google_api_key,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                "Expected 'openai' or 'google'."
            )

    async def evaluate(
        self,
        question: str | EvaluationRequest,
        answer: str | None = None,
        context: list[str] | None = None,
        ground_truth: str | None = None,
    ) -> EvaluationResult:
        """Run all evaluation metrics concurrently and aggregate the results.

        Accepts either an EvaluationRequest instance or discrete arguments.

        Args:
            question: The original user question OR an EvaluationRequest instance.
            answer: The generated answer to evaluate.
            context: List of retrieved context passages.
            ground_truth: Optional known-correct answer for the correctness metric.

        Returns:
            EvaluationResult with individual scores, overall score, and verdict.
        """
        if isinstance(question, EvaluationRequest):
            req = question
            q_text = req.question
            a_text = req.answer
            ctx_list = req.context or []
            gt_text = req.ground_truth
        else:
            q_text = question
            a_text = answer or ""
            ctx_list = context or []
            gt_text = ground_truth

        logger.info("Starting full evaluation suite for question: '%.80s...'", q_text)

        tasks = [
            evaluator.evaluate(q_text, a_text, ctx_list, gt_text)
            for evaluator in self._evaluators.values()
        ]

        results: list[EvaluationScore | Exception] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Collect valid scores, log any exceptions
        valid_scores: list[EvaluationScore] = []
        for metric_name, result in zip(self._evaluators.keys(), results):
            if isinstance(result, Exception):
                logger.error(
                    "Evaluator '%s' raised an exception: %s", metric_name, result
                )
                continue
            # Skip correctness when ground truth was not provided (marked as skipped)
            if (
                result.details
                and result.details.get("skipped", False)
            ):
                logger.info("Skipping '%s' score (no ground truth).", metric_name)
                continue
            valid_scores.append(result)

        # Calculate overall score
        if valid_scores:
            overall_score = sum(s.score for s in valid_scores) / len(valid_scores)
        else:
            overall_score = 0.0
            logger.warning("No valid evaluation scores collected.")

        # Clamp to valid range
        overall_score = max(0.0, min(1.0, round(overall_score, 4)))

        # Determine verdict based on thresholds
        if overall_score >= 0.8:
            verdict = "PASS"
        elif overall_score >= 0.5:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        logger.info(
            "Evaluation complete: overall_score=%.4f, verdict=%s (%d valid scores)",
            overall_score,
            verdict,
            len(valid_scores),
        )

        return EvaluationResult(
            scores=valid_scores,
            overall_score=overall_score,
            verdict=verdict,
            evaluated_at=datetime.now(timezone.utc),
        )

    async def evaluate_single_metric(
        self,
        metric_name: str,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Run a single evaluation metric.

        Args:
            metric_name: Name of the evaluator ('faithfulness', 'answer_relevance', etc.)
            question: The user question.
            answer: The generated answer.
            context: Retrieved context list.
            ground_truth: Optional ground truth.

        Returns:
            EvaluationScore for that metric.
        """
        if metric_name not in self._evaluators:
            raise ValueError(
                f"Unknown evaluator '{metric_name}'. "
                f"Available: {list(self._evaluators.keys())}"
            )
        return await self._evaluators[metric_name].evaluate(
            question=question,
            answer=answer,
            context=context,
            ground_truth=ground_truth,
        )
