"""Custom exception classes for PromptMOO."""


class PromptMOOError(Exception):
    """Base exception for all PromptMOO errors."""


class PromptParsingError(PromptMOOError):
    """Raised when JSON or LLM response parsing fails in the optimizer."""


class LLMBatchError(PromptMOOError):
    """Raised when an LLM batch call fails."""


class InvalidBatchSizeError(PromptMOOError):
    """Raised when batch_size configuration is invalid."""


class InvalidAlgorithmContextError(PromptMOOError):
    """Raised when required keys are missing from the algorithm context."""
