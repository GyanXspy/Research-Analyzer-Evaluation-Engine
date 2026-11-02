"""AI Evaluation Engine package.

Provides individual evaluators for Faithfulness, Answer Relevance,
Context Relevance, Correctness, and Completeness, alongside the
EvaluationEngine orchestrator.
"""

from app.evaluation.answer_relevance import AnswerRelevanceEvaluator
from app.evaluation.base import BaseEvaluator
from app.evaluation.completeness import CompletenessEvaluator
from app.evaluation.context_relevance import ContextRelevanceEvaluator
from app.evaluation.correctness import CorrectnessEvaluator
from app.evaluation.engine import EvaluationEngine
from app.evaluation.faithfulness import FaithfulnessEvaluator

__all__ = [
    "BaseEvaluator",
    "FaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "ContextRelevanceEvaluator",
    "CorrectnessEvaluator",
    "CompletenessEvaluator",
    "EvaluationEngine",
]
