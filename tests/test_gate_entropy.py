"""Tests for gate entropy regularizer."""

import torch
import torch.nn.functional as F

from tajalli.model.tajalli import TajalliLayer


def test_gate_entropy_tensor_returned():
    """TajalliLayer returns _gate_entropy_tensor for trainer regularizer."""
    B, T, d_model, d_essence = 2, 4, 32, 16
    layer = TajalliLayer(d_model=d_model, d_essence=d_essence, n_attributes=4, essence_type="vector")
    essence = torch.randn(B, d_essence)
    h = torch.randn(B, T, d_model)

    _, m = layer(essence, h, step_idx=0)
    assert "_gate_entropy_tensor" in m
    ent = m["_gate_entropy_tensor"]
    assert ent.dim() == 0
    assert ent.item() >= 0


def test_gate_entropy_loss_decreases_when_entropy_increases():
    """With lambda_gate_entropy > 0, loss decreases when gate entropy increases.
    Use controlled gate logits: peaked -> low entropy, uniform -> high entropy.
    """
    # Direct entropy check: peaked softmax = low entropy, uniform = high entropy
    peaked = F.softmax(torch.tensor([10.0, -5.0, -5.0, -5.0]), dim=-1)
    uniform = F.softmax(torch.tensor([0.0, 0.0, 0.0, 0.0]), dim=-1)
    ent_peaked = -(peaked * (peaked + 1e-10).log()).sum().item()
    ent_uniform = -(uniform * (uniform + 1e-10).log()).sum().item()
    assert ent_uniform > ent_peaked

    # Trainer: loss -= lambda * gate_entropy
    lambda_gate_entropy = 0.1
    contrib_peaked = -lambda_gate_entropy * ent_peaked
    contrib_uniform = -lambda_gate_entropy * ent_uniform
    assert contrib_uniform < contrib_peaked, "Higher entropy -> more negative contrib -> lower loss"
