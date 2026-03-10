import json

import torch

from tajalli.data.freq_vocab import (
    labels_to_freq_rank,
    orig_to_rank_from_freq_order,
    rank_tensor_from_freq_order,
)
from tajalli.model.adaptive_output import AdaptiveOutput
from tajalli.model.tajalli_model import TajalliModelPhase1
from tajalli.training.trainer import _apply_regularization_terms


def _make_phase1_model(**overrides) -> TajalliModelPhase1:
    config = {
        "vocab_size": 64,
        "d_model": 32,
        "d_essence": 32,
        "n_heads": 4,
        "d_head": 8,
        "d_ff": 96,
        "n_steps": 2,
        "max_seq_len": 16,
        "use_exit_router": True,
    }
    config.update(overrides)
    return TajalliModelPhase1(**config)


def test_phase1_full_metrics_preserve_existing_keys():
    torch.manual_seed(0)
    model = _make_phase1_model()
    x = torch.randint(0, model.vocab_size, (2, 8))

    _logits, metrics = model(x, return_step_metrics=True)

    assert metrics is not None
    assert "step_0_attribute_gate_mean" in metrics
    assert "step_0_attribute_gate_entropy" in metrics
    assert "gate_entropy_loss" in metrics
    assert "_exit_entropy_tensor" in metrics


def test_phase1_minimal_metrics_return_only_live_tensors():
    torch.manual_seed(1)
    model = _make_phase1_model()
    x = torch.randint(0, model.vocab_size, (2, 8))

    _logits, metrics = model(x, metrics_mode="minimal")

    assert metrics is not None
    assert set(metrics) == {"gate_entropy_loss", "_exit_entropy_tensor"}
    assert metrics["gate_entropy_loss"].requires_grad
    assert metrics["_exit_entropy_tensor"].requires_grad


def test_apply_regularization_terms_supports_minimal_metrics():
    loss = torch.tensor(2.0)
    step_metrics = {
        "gate_entropy_loss": torch.tensor(0.5),
        "_exit_entropy_tensor": torch.tensor(0.25),
    }

    updated = _apply_regularization_terms(
        loss,
        step_metrics=step_metrics,
        lambda_gate_entropy=0.1,
        lambda_exit=0.2,
        grad_accum=4,
    )

    expected = 2.0 - (0.1 * 0.5) / 4 - (0.2 * 0.25) / 4
    assert torch.isclose(updated, torch.tensor(expected))


def test_labels_to_freq_rank_tensor_lookup_matches_dict():
    freq_order = [3, 0, 2, 1, 4]
    labels = torch.tensor([3, 1, -100, 4, 0, 2])
    dict_lookup = orig_to_rank_from_freq_order(freq_order)
    tensor_lookup = rank_tensor_from_freq_order(freq_order)

    expected = labels_to_freq_rank(labels, dict_lookup)
    actual = labels_to_freq_rank(labels, tensor_lookup)

    assert torch.equal(actual, expected)


def test_adaptive_output_loss_matches_forward_loss():
    torch.manual_seed(2)
    layer = AdaptiveOutput(d_model=16, vocab_size=12, cutoffs=[4, 8])
    hidden = torch.randn(10, 16)
    target = torch.tensor([0, 1, -1, 2, 3, 4, 5, 6, 7, -1])

    _output, loss = layer(hidden, target)
    loss_only = layer.loss(hidden, target)

    assert torch.allclose(loss_only, loss)
    assert torch.equal(layer.loss(hidden, torch.full_like(target, -1)), hidden.new_zeros(()))


def test_phase1_adaptive_softmax_registers_dense_rank_mapping(tmp_path):
    freq_order = [2, 0, 3, 1]
    freq_path = tmp_path / "freq.json"
    freq_path.write_text(json.dumps({"freq_order": freq_order}), encoding="utf-8")

    model = TajalliModelPhase1(
        vocab_size=4,
        d_model=16,
        d_essence=8,
        n_heads=2,
        d_head=8,
        d_ff=32,
        n_steps=2,
        max_seq_len=8,
        use_adaptive_softmax=True,
        freq_vocab_path=str(freq_path),
        adaptive_softmax_cutoffs=[2],
    )

    labels = torch.tensor([2, 1, -100, 0, 3])
    expected = labels_to_freq_rank(labels, model.orig_to_rank)
    actual = labels_to_freq_rank(labels, model.rank_mapping)

    assert torch.equal(actual, expected)
