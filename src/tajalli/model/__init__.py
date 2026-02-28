"""Model components for Tajallī architecture."""

from .essence import EssenceCore, EssenceCoreMatrix
from .attention import MultiHeadAttention, apply_rope, FeedForward
from .tajalli import TajalliLayer, TajalliBlock, TajalliStack
from .tajalli_model import TajalliModelPhase1, TajalliModelPhase2, TajalliModelPhase3
from .moe import PairedMoELayer, WisdomRouter, N_EXPERTS, N_PAIRS, NAME_PAIRS
from .diverse_moe import DiverseAsmaaMoE, ExpertFFN
from .lawh import LawhMemoryStore, LawhCrossAttention
from .exit_router import ExitRouter
from .barzakh import BarzakhBottleneck
from .qadr import QadrConstraints

__all__ = [
    "EssenceCore",
    "EssenceCoreMatrix",
    "MultiHeadAttention",
    "apply_rope",
    "FeedForward",
    "TajalliLayer",
    "TajalliBlock",
    "TajalliStack",
    "TajalliModelPhase1",
    "TajalliModelPhase2",
    "TajalliModelPhase3",
    "PairedMoELayer",
    "WisdomRouter",
    "N_EXPERTS",
    "N_PAIRS",
    "NAME_PAIRS",
    "DiverseAsmaaMoE",
    "ExpertFFN",
    "LawhMemoryStore",
    "LawhCrossAttention",
    "ExitRouter",
    "BarzakhBottleneck",
    "QadrConstraints",
]
