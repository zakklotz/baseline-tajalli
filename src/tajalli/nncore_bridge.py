from __future__ import annotations

from typing import Any, Dict


def _require_nncore():
    """
    Ensure nncore is importable. This repo expects nn-core to be installed locally:
        pip install -e ./.nn-core
    """
    try:
        import nncore  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nn-core is not installed or not importable.\n\n"
            "From the tajalli repo root, run:\n"
            "  pip install -e ./.nn-core\n\n"
            "Then install tajalli:\n"
            "  pip install -e .\n"
        ) from e


def build_nncore_config(tajalli_config: Dict[str, Any]):
    """
    Placeholder: Convert a Tajalli config dict into nncore.models.TransformerConfig.

    This will be implemented in a later commit once Tajalli's config schema is
    finalized for the nn-core-backed components.
    """
    _require_nncore()

    # Late import so the error message above is the one users see.
    # from nncore.models import TransformerConfig  # type: ignore

    raise NotImplementedError(
        "Config bridge not implemented yet. "
        "This will map Tajalli configs -> nncore.models.TransformerConfig in a later commit."
    )
