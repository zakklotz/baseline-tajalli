"""Training loop and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .scheduler import get_cosine_warmup_scheduler

if TYPE_CHECKING:
    from .trainer import Phase1Trainer


def __getattr__(name: str) -> Any:
    if name == "Phase1Trainer":
        from .trainer import Phase1Trainer

        return Phase1Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Phase1Trainer", "get_cosine_warmup_scheduler"]
