"""Typed failures shared by GPT request and planner fallback layers."""

from __future__ import annotations


class GPTResponseError(RuntimeError):
    """Raised when GPT cannot produce a usable frontier assignment."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.reason = str(reason)
        self.attempts = int(attempts)
