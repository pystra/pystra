"""Exceptions raised by PySTRA."""

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

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result
