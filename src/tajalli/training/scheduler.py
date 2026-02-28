"""Learning rate schedulers."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine decay with linear warmup. min_lr_ratio: lr at end = base_lr * min_lr_ratio."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda)


def get_multistep_lr_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    milestones: list[int],
    gamma: float = 0.1,
) -> LambdaLR:
    """Linear warmup then step decay at milestones (multiply LR by gamma at each)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        decay = 1.0
        for m in sorted(milestones):
            if step >= m:
                decay *= gamma
        return decay

    return LambdaLR(optimizer, lr_lambda)
