"""Config utilities for training scripts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigNormalizationError(ValueError):
    """Raised when a Phase 1 config cannot be normalized safely."""

    def __init__(self, message: str, report: "NormalizationReport"):
        super().__init__(message)
        self.report = report


@dataclass
class NormalizationReport:
    """Record how a config was normalized before training starts."""

    source_path: str | None = None
    alias_rewrites: list[str] = field(default_factory=list)
    flattened_sections: list[str] = field(default_factory=list)
    defaults_applied: list[str] = field(default_factory=list)
    conflicts_rejected: list[str] = field(default_factory=list)

    def all_entries(self) -> list[str]:
        entries = []
        entries.extend(self.alias_rewrites)
        entries.extend(self.flattened_sections)
        entries.extend(self.defaults_applied)
        entries.extend(self.conflicts_rejected)
        return entries


def normalize_phase1_config(
    raw_config: dict[str, Any],
    source_path: str | Path | None = None,
) -> tuple[dict[str, Any], NormalizationReport]:
    """Normalize legacy/deprecated Phase 1 config keys into the flat runtime schema."""
    if not isinstance(raw_config, dict):
        raise TypeError(f"Phase 1 config must be a mapping, got {type(raw_config).__name__}")

    report = NormalizationReport(
        source_path=str(source_path) if source_path is not None else None
    )
    config = deepcopy(raw_config)

    _rewrite_alias(
        config,
        alias="tokenizer",
        canonical="tokenizer_name",
        report=report,
    )
    _rewrite_alias(
        config,
        alias="model_name",
        canonical="run_name",
        report=report,
    )

    _flatten_section(
        config,
        section_name="essence",
        field_map={
            "type": "essence_type",
            "n_rows": "n_essence_rows",
        },
        report=report,
    )
    _flatten_section(
        config,
        section_name="tajalli",
        field_map={
            "alpha_schedule": "alpha_schedule",
            "depth_families": "depth_families",
            "family_steps": "family_steps",
            "hypernetwork_attributes": "hypernetwork_attributes",
        },
        report=report,
    )

    if "tokenizer_name" not in config:
        config["tokenizer_name"] = "gpt2"
        report.defaults_applied.append("tokenizer_name <- 'gpt2'")
    if "recurrence_mode" not in config:
        config["recurrence_mode"] = "tajalli"
        report.defaults_applied.append("recurrence_mode <- 'tajalli'")
    if "attribute_gate_mode" not in config:
        config["attribute_gate_mode"] = "contextual"
        report.defaults_applied.append("attribute_gate_mode <- 'contextual'")

    _synchronize_step_aliases(config, report)
    return config, report


def validate_phase1_config(config: dict[str, Any]) -> None:
    """Validate the normalized Phase 1 training config."""
    if not isinstance(config, dict):
        raise TypeError(f"Phase 1 config must be a mapping, got {type(config).__name__}")

    n_steps = config.get("n_steps")
    recursive_steps = config.get("recursive_steps")
    if n_steps is None or recursive_steps is None:
        raise ValueError("Phase 1 config must define n_steps/recursive_steps after normalization.")
    if int(n_steps) <= 0 or int(recursive_steps) <= 0:
        raise ValueError("n_steps and recursive_steps must be positive integers.")
    if int(n_steps) != int(recursive_steps):
        raise ValueError(
            f"n_steps ({n_steps}) and recursive_steps ({recursive_steps}) must match."
        )

    tokenizer_name = config.get("tokenizer_name")
    if not tokenizer_name:
        raise ValueError("tokenizer_name is required for Phase 1 training.")

    run_name = config.get("run_name")
    if not run_name:
        raise ValueError("run_name is required for Phase 1 training.")

    recurrence_mode = str(config.get("recurrence_mode", "tajalli"))
    if recurrence_mode not in {"tajalli", "plain_recursive"}:
        raise ValueError(
            f"recurrence_mode must be one of ['tajalli', 'plain_recursive'], got {recurrence_mode!r}."
        )

    attribute_gate_mode = str(config.get("attribute_gate_mode", "contextual"))
    if attribute_gate_mode not in {"contextual", "uniform"}:
        raise ValueError(
            f"attribute_gate_mode must be one of ['contextual', 'uniform'], got {attribute_gate_mode!r}."
        )

    vocab_size = config.get("vocab_size")
    if vocab_size is not None and int(vocab_size) <= 0:
        raise ValueError("vocab_size must be a positive integer when provided.")

    if config.get("use_adaptive_softmax"):
        freq_vocab_path = config.get("freq_vocab_path")
        if not freq_vocab_path:
            raise ValueError(
                "use_adaptive_softmax=true requires freq_vocab_path in the Phase 1 config."
            )
        _validate_adaptive_softmax_cutoffs(
            config.get("adaptive_softmax_cutoffs"),
            vocab_size=config.get("vocab_size"),
        )


def apply_lawh_every_k(config: dict) -> None:
    """If lawh_every_k_steps is set, build lawh_at_steps = [0, k, 2k, ...] up to recursive_steps. Mutates config."""
    k = config.get("lawh_every_k_steps")
    if k is None:
        return
    if int(k) <= 0:
        raise ValueError("lawh_every_k_steps must be a positive integer.")
    n_steps = int(config.get("recursive_steps", config.get("n_steps", 8)))
    resolved = list(range(0, n_steps, int(k)))
    existing = config.get("lawh_at_steps")
    if existing is not None:
        existing_list = [int(step) for step in existing]
        if existing_list != resolved:
            raise ValueError(
                "lawh_every_k_steps conflicts with explicit lawh_at_steps: "
                f"computed {resolved}, found {existing_list}."
            )
        return
    config["lawh_at_steps"] = resolved


def print_normalization_report(
    report: NormalizationReport,
    title: str = "PHASE1 CONFIG NORMALIZATION",
) -> None:
    """Print a short summary of config normalization rewrites."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if report.source_path:
        print(f"  source: {report.source_path}")
    if not report.all_entries():
        print("  no rewrites applied")
        print("=" * 70 + "\n")
        return
    _print_report_section("alias_rewrites", report.alias_rewrites)
    _print_report_section("flattened_sections", report.flattened_sections)
    _print_report_section("defaults_applied", report.defaults_applied)
    _print_report_section("conflicts_rejected", report.conflicts_rejected)
    print("=" * 70 + "\n")


def print_training_config(config: dict, title: str = "TRAINING CONFIG") -> None:
    """Print all config settings in a readable format before training starts."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    priority = [
        "run_name",
        "resume_from",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "lr",
        "lr_final",
        "lr_schedule",
        "warmup_steps",
        "optimizer",
        "weight_decay",
        "grad_clip",
        "recursive_steps",
        "n_steps",
        "recurrence_mode",
        "attribute_gate_mode",
        "depth_min",
        "depth_max",
        "depth_steps",
        "variable_depth_training",
        "use_adaptive_softmax",
        "freq_vocab_path",
        "adaptive_softmax_cutoffs",
        "eval_use_memory",
        "eval_seg_len",
        "eval_mem_len",
        "d_model",
        "d_essence",
        "max_seq_len",
        "tokenizer_name",
        "vocab_size",
        "device",
        "dtype",
    ]
    seen = set()
    for key in priority:
        if key in config:
            _print_val(key, config[key])
            seen.add(key)
    for key in sorted(config.keys()):
        if key not in seen:
            _print_val(key, config[key])
    print("=" * 70 + "\n")


def _flatten_section(
    config: dict[str, Any],
    *,
    section_name: str,
    field_map: dict[str, str],
    report: NormalizationReport,
) -> None:
    section = config.pop(section_name, None)
    if section is None:
        return
    if not isinstance(section, dict):
        _raise_normalization_error(
            f"Expected '{section_name}' to be a mapping, got {type(section).__name__}.",
            report,
        )

    flattened = []
    for field_name, canonical in field_map.items():
        if field_name not in section:
            continue
        flattened.append(field_name)
        _merge_alias_value(
            config,
            alias=f"{section_name}.{field_name}",
            canonical=canonical,
            value=section.pop(field_name),
            report=report,
        )

    if flattened:
        report.flattened_sections.append(
            f"{section_name}: " + ", ".join(f"{name} -> {field_map[name]}" for name in flattened)
        )

    if section:
        unknown = ", ".join(sorted(section.keys()))
        _raise_normalization_error(
            f"Unknown keys remain under '{section_name}' after normalization: {unknown}.",
            report,
        )


def _rewrite_alias(
    config: dict[str, Any],
    *,
    alias: str,
    canonical: str,
    report: NormalizationReport,
) -> None:
    if alias not in config:
        return
    _merge_alias_value(
        config,
        alias=alias,
        canonical=canonical,
        value=config.pop(alias),
        report=report,
    )


def _merge_alias_value(
    config: dict[str, Any],
    *,
    alias: str,
    canonical: str,
    value: Any,
    report: NormalizationReport,
) -> None:
    if canonical in config:
        if config[canonical] != value:
            _raise_normalization_error(
                f"Config conflict between '{alias}'={value!r} and '{canonical}'={config[canonical]!r}.",
                report,
            )
        report.alias_rewrites.append(f"{alias} matched existing {canonical}")
        return
    config[canonical] = value
    report.alias_rewrites.append(f"{alias} -> {canonical}")


def _synchronize_step_aliases(config: dict[str, Any], report: NormalizationReport) -> None:
    n_steps = config.get("n_steps")
    recursive_steps = config.get("recursive_steps")

    if n_steps is not None and recursive_steps is not None:
        if int(n_steps) != int(recursive_steps):
            _raise_normalization_error(
                f"Config conflict between 'n_steps'={n_steps!r} and 'recursive_steps'={recursive_steps!r}.",
                report,
            )
        config["n_steps"] = int(n_steps)
        config["recursive_steps"] = int(recursive_steps)
        return

    if recursive_steps is not None:
        config["recursive_steps"] = int(recursive_steps)
        config["n_steps"] = int(recursive_steps)
        report.defaults_applied.append("n_steps <- recursive_steps")
        return

    if n_steps is not None:
        config["n_steps"] = int(n_steps)
        config["recursive_steps"] = int(n_steps)
        report.defaults_applied.append("recursive_steps <- n_steps")


def _validate_adaptive_softmax_cutoffs(
    cutoffs: Any,
    *,
    vocab_size: Any | None,
) -> None:
    if cutoffs is None:
        return
    if not isinstance(cutoffs, (list, tuple)) or not cutoffs:
        raise ValueError("adaptive_softmax_cutoffs must be a non-empty list of integers.")
    normalized = [int(cutoff) for cutoff in cutoffs]
    if any(cutoff <= 0 for cutoff in normalized):
        raise ValueError("adaptive_softmax_cutoffs must contain only positive integers.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "adaptive_softmax_cutoffs must be strictly increasing and must not contain duplicates."
        )
    if normalized != sorted(normalized):
        raise ValueError("adaptive_softmax_cutoffs must be strictly increasing.")
    if vocab_size is not None and any(cutoff >= int(vocab_size) for cutoff in normalized):
        raise ValueError(
            f"adaptive_softmax_cutoffs {normalized} must all be < vocab_size ({vocab_size})."
        )


def _print_report_section(name: str, entries: list[str]) -> None:
    if not entries:
        return
    print(f"  {name}:")
    for entry in entries:
        print(f"    - {entry}")


def _raise_normalization_error(message: str, report: NormalizationReport) -> None:
    report.conflicts_rejected.append(message)
    raise ConfigNormalizationError(message, report)


def _print_val(key: str, val: Any) -> None:
    if isinstance(val, (list, tuple)) and len(val) > 8:
        print(f"  {key}: [{val[0]}, {val[1]}, ... ({len(val)} items)]")
    elif isinstance(val, dict):
        print(f"  {key}:")
        for sub_key, sub_val in val.items():
            print(f"    {sub_key}: {sub_val}")
    else:
        print(f"  {key}: {val}")
