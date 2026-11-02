"""Abstract base class for evaluation metrics and shared LLM response parsing.

Provides a common interface that all evaluator implementations must follow,
alongside utilities for extracting text and JSON across multiple LLM providers.
"""

from abc import ABC, abstractmethod
import json
import logging
from typing import Any

from app.models.schemas import EvaluationScore

logger = logging.getLogger(__name__)


def extract_text(response: Any) -> str:
    """Extract string content from string, list of blocks, or AIMessage.

    Handles OpenAI, Google GenAI SDK list of blocks, and standard string payloads.
    """
    if hasattr(response, "content"):
        content = response.content
    else:
        content = response

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


def extract_json(content: Any) -> dict | list:
    """Extract and parse JSON safely from text or block content.

    Strips markdown code blocks (```json ... ```) and searches for JSON boundaries.
    """
    text = extract_text(content).strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.splitlines()
        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        if first_line.startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Search for JSON object or array bounds
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue

    raise ValueError(f"Could not parse valid JSON from content: {text[:200]}")


class BaseEvaluator(ABC):
    """Base class for all evaluation metrics.

    Subclasses must implement the ``metric_name`` property and the
    ``evaluate`` coroutine to provide a concrete evaluation strategy.
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the evaluation metric."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        answer: str,
        context: list[str],
        ground_truth: str | None = None,
    ) -> EvaluationScore:
        """Evaluate a RAG response and return a score.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate.
            context: List of retrieved context passages.
            ground_truth: Optional known-correct answer for comparison.

        Returns:
            EvaluationScore with the metric result.
        """
        ...
