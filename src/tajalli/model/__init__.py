"""Model components for Tajallī architecture."""

from .essence import EssenceCore, EssenceCoreMatrix
from .attention import MultiHeadAttention, apply_rope, FeedForward
from .tajalli import TajalliLayer, TajalliBlock, TajalliStack
from .tajalli_block_v2 import TajalliBlockV2
from .tajalli_model import TajalliModelPhase1, TajalliModelPhase2, TajalliModelPhase3
from .moe import PairedMoELayer, WisdomRouter, N_EXPERTS, N_PAIRS, NAME_PAIRS
from .diverse_moe import DiverseAsmaaMoE, ExpertFFN
from .ayan_moe import AyanThābitahMoE, PolarizedExpert, StandardExpert
from .lawh import LawhMemoryStore, LawhCrossAttention
from .exit_router import ExitRouter
from .barzakh import BarzakhBottleneck
from .qadr import QadrConstraints
from .jamba_adapter import JambaStyleBlock, SimpleSSMLayer

__all__ = [
    "EssenceCore",
    "EssenceCoreMatrix",
    "MultiHeadAttention",
    "apply_rope",
    "FeedForward",
    "TajalliLayer",
    "TajalliBlock",
    "TajalliBlockV2",
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
    "AyanThābitahMoE",
    "PolarizedExpert",
    "StandardExpert",
    "LawhMemoryStore",
    "LawhCrossAttention",
    "ExitRouter",
    "BarzakhBottleneck",
    "QadrConstraints",
    "JambaStyleBlock",
    "SimpleSSMLayer",
]
