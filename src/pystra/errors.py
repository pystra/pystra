"""Exceptions raised by PySTRA."""

from typing import Any

__all__ = ["PystraError", "ModelError", "AnalysisError"]


class PystraError(Exception):
    """Base class for errors raised by PySTRA."""


class ModelError(PystraError, ValueError):
    """An invalid model, distribution or input specification."""


class AnalysisError(PystraError, RuntimeError):
    """An analysis could not produce a valid result.

    ``result`` holds the failed result, when there is one, so its diagnostics
    remain available.
    """

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result
