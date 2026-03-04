import torch
import pytest

from tajalli.model.tajalli_model import TajalliModelPhase1
from tajalli.training.config_utils import validate_phase1_config


def _minimal_config(**overrides):
    config = {
        "run_name": "ablation-test",
        "tokenizer_name": "gpt2",
        "n_steps": 4,
        "recursive_steps": 4,
        "vocab_size": 128,
        "recurrence_mode": "tajalli",
        "attribute_gate_mode": "contextual",
    }
    config.update(overrides)
    return config


def test_validate_phase1_config_rejects_invalid_recurrence_mode():
    with pytest.raises(ValueError, match="recurrence_mode"):
        validate_phase1_config(_minimal_config(recurrence_mode="bad"))


def test_validate_phase1_config_rejects_invalid_attribute_gate_mode():
    with pytest.raises(ValueError, match="attribute_gate_mode"):
        validate_phase1_config(_minimal_config(attribute_gate_mode="bad"))


def test_phase1_plain_recursive_forward_with_optional_features():
    torch.manual_seed(0)
    model = TajalliModelPhase1(
        vocab_size=64,
        d_model=32,
        d_essence=32,
        n_heads=4,
        d_head=8,
        d_ff=96,
        n_steps=3,
        max_seq_len=16,
        recurrence_mode="plain_recursive",
        use_exit_router=True,
        use_recursive_kv_cache=True,
    )
    x = torch.randint(0, 64, (2, 8))
    logits, metrics = model(x, return_step_metrics=True)
    assert logits.shape == (2, 8, 64)
    assert metrics is not None
    assert "step_0_attribute_gate_entropy" not in metrics


def test_phase1_uniform_gates_report_uniform_gate_mean():
    torch.manual_seed(0)
    model = TajalliModelPhase1(
        vocab_size=64,
        d_model=32,
        d_essence=32,
        n_heads=4,
        d_head=8,
        d_ff=96,
        n_steps=2,
        max_seq_len=16,
        attribute_gate_mode="uniform",
    )
    x = torch.randint(0, 64, (2, 8))
    _logits, metrics = model(x, return_step_metrics=True)
    gate_mean = metrics["step_0_attribute_gate_mean"]
    expected = torch.full_like(gate_mean, 1.0 / gate_mean.numel())
    assert torch.allclose(gate_mean, expected, atol=1e-6)
