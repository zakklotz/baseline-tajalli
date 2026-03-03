from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("torch")

from scripts.train_phase1 import preflight_phase1_config, resolve_resume_path
from tajalli.data.dataloader import (
    get_wikitext103_tokenization_kwargs,
    get_wikitext103_tokenization_meta,
)
from tajalli.data.freq_vocab import (
    build_tokenizer_metadata,
    load_freq_artifact,
    validate_freq_artifact,
)
from tajalli.training.config_utils import (
    ConfigNormalizationError,
    normalize_phase1_config,
    validate_phase1_config,
)


class FakeTokenizer:
    def __init__(self, name_or_path: str = "fake-gpt2", vocab_size: int = 8):
        if vocab_size < 4:
            raise ValueError("FakeTokenizer requires vocab_size >= 4")
        self.name_or_path = name_or_path
        self.is_fast = True
        self._vocab = {"<eos>": 0, "<unk>": 1}
        for idx in range(2, vocab_size):
            self._vocab[f"tok{idx}"] = idx
        self.eos_token = "<eos>"
        self.eos_token_id = self._vocab[self.eos_token]
        self.unk_token_id = self._vocab["<unk>"]
        self.bos_token_id = None
        self.mask_token_id = None
        self._pad_token = None
        self.pad_token_id = None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab)

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._pad_token = value
        self.pad_token_id = None if value is None else self._vocab[value]


def test_normalize_phase1_deprecated_config() -> None:
    config_path = Path(__file__).resolve().parents[2] / ".tajalli_deprecated/configs/phase1_adaptive.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config, report = normalize_phase1_config(raw, source_path=config_path)

    assert config["tokenizer_name"] == "gpt2"
    assert config["run_name"] == "phase1_adaptive"
    assert config["essence_type"] == "matrix"
    assert config["n_essence_rows"] == 64
    assert config["alpha_schedule"] == [0.25] * 8
    assert config["depth_families"] == 3
    assert config["family_steps"] == [0, 3, 6]
    assert "hypernetwork_attributes" not in config
    assert config["recursive_steps"] == 8
    assert config["n_steps"] == 8
    assert config["essence_warmup_steps"] == 2000
    assert "essence" not in config
    assert "tajalli" not in config
    assert report.alias_rewrites
    assert report.flattened_sections


def test_normalize_phase1_rejects_tokenizer_conflict() -> None:
    with pytest.raises(ConfigNormalizationError, match="tokenizer"):
        normalize_phase1_config(
            {
                "run_name": "demo",
                "n_steps": 2,
                "tokenizer": "gpt2",
                "tokenizer_name": "other-tokenizer",
            }
        )


def test_normalize_phase1_rejects_recursive_step_conflict() -> None:
    with pytest.raises(ConfigNormalizationError, match="recursive_steps"):
        normalize_phase1_config(
            {
                "run_name": "demo",
                "n_steps": 2,
                "recursive_steps": 3,
            }
        )


def test_normalize_phase1_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ConfigNormalizationError, match="Unknown keys remain under 'essence'"):
        normalize_phase1_config(
            {
                "run_name": "demo",
                "n_steps": 2,
                "essence": {"type": "matrix", "mystery": 1},
            }
        )


def test_resume_path_precedence() -> None:
    config = {"resume_from": "from-config.pt"}
    assert resolve_resume_path(config, None) == "from-config.pt"
    assert resolve_resume_path(config, "from-cli.pt") == "from-cli.pt"


def test_load_freq_artifact_supports_v0_v1_v2(tmp_path: Path) -> None:
    tok = make_fake_tokenizer()
    v0 = tmp_path / "v0.json"
    v1 = tmp_path / "v1.json"
    v2 = tmp_path / "v2.json"
    write_json(v0, {"freq_order": list(range(len(tok)))})
    write_json(
        v1,
        {
            "tokenizer": tok.name_or_path,
            "vocab_size": len(tok),
            "freq_order": list(range(len(tok))),
            "freq_counts": list(reversed(range(len(tok)))),
            "orig_to_rank": {str(idx): idx for idx in range(len(tok))},
            "split": "train",
            "total_tokens": 42,
        },
    )
    write_json(v2, make_v2_artifact(tok, path=v2))

    assert load_freq_artifact(v0).schema_version == 0
    assert load_freq_artifact(v1).schema_version == 1
    assert load_freq_artifact(v2).schema_version == 2


def test_validate_freq_artifact_rejects_bad_permutation(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    write_json(path, {"freq_order": [0, 1, 1, 3]})
    artifact = load_freq_artifact(path)

    with pytest.raises(ValueError, match="duplicate token id"):
        validate_freq_artifact(artifact, vocab_size=4)


def test_validate_freq_artifact_rejects_tokenizer_fingerprint_mismatch(tmp_path: Path) -> None:
    tok_a = make_fake_tokenizer(name="tok-a")
    tok_b = make_fake_tokenizer(name="tok-a")
    tok_b._vocab["alt-token"] = tok_b._vocab.pop("tok2")
    path = tmp_path / "artifact.json"
    write_json(path, make_v2_artifact(tok_a, path=path))
    artifact = load_freq_artifact(path)

    with pytest.raises(ValueError, match="fingerprint"):
        validate_freq_artifact(
            artifact,
            vocab_size=len(tok_a),
            tokenizer=tok_b,
            tokenizer_name=tok_b.name_or_path,
            dataset_name="wikitext",
            dataset_config="wikitext-103-v1",
            split="train",
            text_field="text",
            tokenization=get_wikitext103_tokenization_meta(tok_b),
            strict_training=True,
        )


def test_validate_freq_artifact_rejects_dataset_split_mismatch(tmp_path: Path) -> None:
    tok = make_fake_tokenizer()
    path = tmp_path / "artifact.json"
    write_json(path, make_v2_artifact(tok, path=path, dataset={"split": "validation"}))
    artifact = load_freq_artifact(path)

    with pytest.raises(ValueError, match="dataset.split"):
        validate_freq_artifact(
            artifact,
            vocab_size=len(tok),
            tokenizer=tok,
            tokenizer_name=tok.name_or_path,
            dataset_name="wikitext",
            dataset_config="wikitext-103-v1",
            split="train",
            text_field="text",
            tokenization=get_wikitext103_tokenization_meta(tok),
            strict_training=True,
        )


def test_validate_phase1_config_rejects_invalid_cutoffs() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_phase1_config(
            {
                "run_name": "demo",
                "tokenizer_name": "gpt2",
                "n_steps": 8,
                "recursive_steps": 8,
                "vocab_size": 32,
                "use_adaptive_softmax": True,
                "freq_vocab_path": "freq.json",
                "adaptive_softmax_cutoffs": [8, 8, 16],
            }
        )

    with pytest.raises(ValueError, match="vocab_size"):
        validate_phase1_config(
            {
                "run_name": "demo",
                "tokenizer_name": "gpt2",
                "n_steps": 8,
                "recursive_steps": 8,
                "vocab_size": 16,
                "use_adaptive_softmax": True,
                "freq_vocab_path": "freq.json",
                "adaptive_softmax_cutoffs": [8, 20],
            }
        )


def test_preflight_accepts_v2_artifact_and_rejects_legacy_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tok = make_fake_tokenizer()
    monkeypatch.setattr("scripts.train_phase1.get_wikitext103_tokenizer", lambda _name: tok)

    v2_path = tmp_path / "freq_v2.json"
    legacy_path = tmp_path / "freq_v0.json"
    write_json(v2_path, make_v2_artifact(tok, path=v2_path))
    write_json(legacy_path, {"freq_order": list(range(len(tok)))})

    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "run_name": "demo",
            "tokenizer_name": tok.name_or_path,
            "n_steps": 8,
            "use_adaptive_softmax": True,
            "adaptive_softmax_cutoffs": [2, 5],
            "freq_vocab_path": str(v2_path),
        },
    )

    config, report, _tokenizer, artifact = preflight_phase1_config(config_path)

    assert config["vocab_size"] == len(tok)
    assert artifact is not None
    assert artifact.schema_version == 2
    assert any("vocab_size <- tokenizer" in entry for entry in report.defaults_applied)

    write_yaml(
        config_path,
        {
            "run_name": "demo",
            "tokenizer_name": tok.name_or_path,
            "n_steps": 8,
            "use_adaptive_softmax": True,
            "adaptive_softmax_cutoffs": [2, 5],
            "freq_vocab_path": str(legacy_path),
        },
    )

    with pytest.raises(ValueError, match="schema v0"):
        preflight_phase1_config(config_path)


def test_shared_tokenization_contract_metadata_matches_kwargs() -> None:
    tok = make_fake_tokenizer()
    tokenization_meta = get_wikitext103_tokenization_meta(tok)
    kwargs = get_wikitext103_tokenization_kwargs()

    assert tokenization_meta["add_special_tokens"] == kwargs["add_special_tokens"]
    assert tokenization_meta["padding"] == kwargs["padding"]
    assert tokenization_meta["truncation"] == kwargs["truncation"]
    assert tokenization_meta["max_length"] == kwargs["max_length"]


def make_fake_tokenizer(name: str = "fake-gpt2", vocab_size: int = 8) -> FakeTokenizer:
    tok = FakeTokenizer(name_or_path=name, vocab_size=vocab_size)
    tok.pad_token = tok.eos_token
    return tok


def make_v2_artifact(
    tokenizer: FakeTokenizer,
    *,
    path: Path,
    dataset: dict[str, object] | None = None,
    tokenization: dict[str, object] | None = None,
    freq_order: list[int] | None = None,
) -> dict[str, object]:
    order = list(range(len(tokenizer))) if freq_order is None else freq_order
    counts = list(range(len(order), 0, -1))
    dataset_payload = {
        "name": "wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "text_field": "text",
    }
    if dataset is not None:
        dataset_payload.update(dataset)
    return {
        "schema_version": 2,
        "freq_order": order,
        "freq_counts": counts,
        "total_tokens": sum(counts),
        "tokenizer": build_tokenizer_metadata(tokenizer, tokenizer_name=tokenizer.name_or_path),
        "dataset": dataset_payload,
        "tokenization": tokenization or get_wikitext103_tokenization_meta(tokenizer),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
