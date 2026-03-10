import pytest
import torch
import torch.nn.functional as F

from tajalli.model.tajalli import MAX_TAJALLI_STEPS, TajalliLayer


def _legacy_forward(
    layer: TajalliLayer,
    essence: torch.Tensor,
    h_prev: torch.Tensor,
    *,
    step_idx: int,
) -> tuple[torch.Tensor, dict]:
    step_idx_clamped = min(step_idx, MAX_TAJALLI_STEPS - 1)
    if layer.hyper_net is not None:
        if layer.essence_type == "matrix":
            pooled_essence = essence.mean(dim=1)
        else:
            pooled_essence = essence.mean(dim=1) if essence.dim() == 3 else essence
        mod = layer.hyper_net(pooled_essence).view(-1, layer.n_attributes, layer.d_model)
        attributes = [h_prev * mod[:, j:j + 1, :] for j in range(layer.n_attributes)]
        gate_logits = None if layer.attribute_gate_mode == "uniform" else layer.gate(h_prev)
    elif layer.step_to_family is not None and layer.attribute_heads_family is not None:
        family = layer.step_to_family[step_idx_clamped]
        heads = layer.attribute_heads_family[family]
        if layer.essence_type == "matrix":
            attributes = [head(h_prev, essence) for head in heads]
        else:
            pooled_essence = essence.mean(dim=1) if essence.dim() == 3 else essence
            essence_expanded = pooled_essence.unsqueeze(1).expand(-1, h_prev.shape[1], -1)
            attributes = [head(essence_expanded) for head in heads]
        gate_logits = None if layer.attribute_gate_mode == "uniform" else layer.gate_family[family](h_prev)
    else:
        heads = layer.attribute_heads
        if layer.essence_type == "matrix":
            attributes = [head(h_prev, essence) for head in heads]
        else:
            pooled_essence = essence.mean(dim=1) if essence.dim() == 3 else essence
            essence_expanded = pooled_essence.unsqueeze(1).expand(-1, h_prev.shape[1], -1)
            attributes = [head(essence_expanded) for head in heads]
        gate_logits = None if layer.attribute_gate_mode == "uniform" else layer.gate(h_prev)

    if layer.attribute_gate_mode == "uniform":
        gate_weights = torch.full(
            (h_prev.shape[0], h_prev.shape[1], layer.n_attributes),
            1.0 / layer.n_attributes,
            device=h_prev.device,
            dtype=h_prev.dtype,
        )
    else:
        gate_weights = F.softmax(gate_logits, dim=-1)

    tajalli_signal = sum(
        gate * attribute
        for gate, attribute in zip(gate_weights.split(1, dim=-1), attributes)
    )
    gate_entropy = -(
        gate_weights * (gate_weights + 1e-10).log()
    ).sum(dim=-1).mean()
    alpha = torch.sigmoid(layer.alpha_per_step[step_idx_clamped])
    metrics = {
        "alpha_value": alpha.item(),
        "alpha_step_idx": step_idx_clamped,
        "attribute_gate_entropy": gate_entropy.item(),
        "attribute_gate_mean": gate_weights.mean(dim=(0, 1)).detach().cpu(),
        "_gate_entropy_tensor": gate_entropy,
    }
    return tajalli_signal, metrics


def _assert_matches_legacy(
    layer: TajalliLayer,
    essence: torch.Tensor,
    h_prev: torch.Tensor,
    *,
    step_idx: int,
) -> None:
    expected_signal, expected_metrics = _legacy_forward(layer, essence, h_prev, step_idx=step_idx)
    signal, metrics = layer(essence, h_prev, step_idx=step_idx)

    assert torch.allclose(signal, expected_signal, atol=1e-6, rtol=1e-5)
    assert metrics["alpha_value"] == pytest.approx(expected_metrics["alpha_value"])
    assert metrics["alpha_step_idx"] == expected_metrics["alpha_step_idx"]
    assert metrics["attribute_gate_entropy"] == pytest.approx(expected_metrics["attribute_gate_entropy"])
    assert torch.allclose(metrics["attribute_gate_mean"], expected_metrics["attribute_gate_mean"])
    assert torch.allclose(metrics["_gate_entropy_tensor"], expected_metrics["_gate_entropy_tensor"])


def test_vector_linear_attributes_match_legacy_loop():
    torch.manual_seed(0)
    layer = TajalliLayer(
        d_model=24,
        d_essence=16,
        n_attributes=5,
        essence_type="vector",
    )
    essence = torch.randn(2, 16)
    h_prev = torch.randn(2, 6, 24)
    _assert_matches_legacy(layer, essence, h_prev, step_idx=1)


def test_vector_mlp_family_attributes_match_legacy_loop():
    torch.manual_seed(1)
    layer = TajalliLayer(
        d_model=24,
        d_essence=16,
        n_attributes=4,
        d_attr_hidden=20,
        depth_families=2,
        family_steps=[0, 2],
        essence_type="vector",
    )
    essence = torch.randn(2, 16)
    h_prev = torch.randn(2, 5, 24)
    _assert_matches_legacy(layer, essence, h_prev, step_idx=3)


def test_matrix_attributes_match_legacy_loop():
    torch.manual_seed(2)
    layer = TajalliLayer(
        d_model=24,
        d_essence=24,
        n_attributes=4,
        essence_type="matrix",
        n_essence_rows=6,
    )
    essence = torch.randn(2, 6, 24)
    h_prev = torch.randn(2, 5, 24)
    _assert_matches_legacy(layer, essence, h_prev, step_idx=0)


def test_hypernetwork_attributes_match_legacy_loop():
    torch.manual_seed(3)
    layer = TajalliLayer(
        d_model=24,
        d_essence=16,
        n_attributes=4,
        hypernetwork_attributes=True,
        essence_type="vector",
    )
    essence = torch.randn(2, 16)
    h_prev = torch.randn(2, 5, 24)
    _assert_matches_legacy(layer, essence, h_prev, step_idx=2)
