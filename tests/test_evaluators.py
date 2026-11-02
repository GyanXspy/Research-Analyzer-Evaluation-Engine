"""Unit tests for evaluation components and the EvaluationEngine."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from app.config import Settings
from app.evaluation.base import BaseEvaluator
from app.evaluation.engine import EvaluationEngine
from app.evaluation.faithfulness import FaithfulnessEvaluator
from app.evaluation.answer_relevance import AnswerRelevanceEvaluator
from app.evaluation.context_relevance import ContextRelevanceEvaluator
from app.evaluation.correctness import CorrectnessEvaluator
from app.evaluation.completeness import CompletenessEvaluator
from app.models.schemas import EvaluationRequest, EvaluationScore


@pytest.mark.asyncio
async def test_faithfulness_evaluator_no_claims():
    mock_llm = MagicMock()
    # Mock extract claims returning empty list
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"claims": []}'))
    
    evaluator = FaithfulnessEvaluator(mock_llm)
    assert evaluator.metric_name == "faithfulness"
    
    result = await evaluator.evaluate(
        question="What is X?",
        answer="Hello world",
        context=["Context passage"],
    )
    assert result.score == 1.0
    assert result.metric_name == "faithfulness"


@pytest.mark.asyncio
async def test_correctness_evaluator_skipped_without_ground_truth():
    mock_llm = MagicMock()
    evaluator = CorrectnessEvaluator(mock_llm)
    
    result = await evaluator.evaluate(
        question="What is X?",
        answer="It is Y",
        context=["Context"],
        ground_truth=None,
    )
    assert result.metric_name == "correctness"
    assert result.details.get("skipped") is True


@pytest.mark.asyncio
async def test_evaluation_engine_aggregation():
    settings = Settings()
    engine = EvaluationEngine(settings)
    
    # Mock all internal evaluators to return fixed scores
    for name, eval_obj in engine._evaluators.items():
        score_val = 0.9 if name != "correctness" else 0.8
        eval_obj.evaluate = AsyncMock(
            return_value=EvaluationScore(
                metric_name=name,
                score=score_val,
                reasoning=f"{name} passed with high score",
                details={},
            )
        )
    
    req = EvaluationRequest(
        question="Explain transformer attention",
        answer="Attention allows weighting of relevant tokens",
        context=["Attention mechanism details"],
        ground_truth="Attention dynamically weights tokens",
    )
    
    result = await engine.evaluate(req)
    assert result.overall_score >= 0.8
    assert result.verdict == "PASS"
    assert len(result.scores) == 5
