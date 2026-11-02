"""Proxy module redirecting to app.evaluation.engine for backward compatibility."""

from app.evaluation.engine import EvaluationEngine

__all__ = ["EvaluationEngine"]
