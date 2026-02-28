"""Config utilities for training scripts."""


def apply_lawh_every_k(config: dict) -> None:
    """If lawh_every_k_steps is set, build lawh_at_steps = [0, k, 2k, ...] up to recursive_steps. Mutates config."""
    k = config.get("lawh_every_k_steps")
    if k is None:
        return
    n_steps = config.get("recursive_steps", 8)
    config["lawh_at_steps"] = list(range(0, n_steps, k))


def print_training_config(config: dict, title: str = "TRAINING CONFIG") -> None:
    """Print all config settings in a readable format before training starts."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    # Sort keys for consistent output; put important ones first
    priority = [
        "model_name", "resume_from", "max_steps", "batch_size", "gradient_accumulation_steps",
        "lr", "lr_final", "lr_schedule", "warmup_steps", "optimizer", "weight_decay", "grad_clip",
        "recursive_steps", "depth_min", "depth_max", "depth_steps", "variable_depth_training",
        "use_adaptive_softmax", "eval_use_memory", "eval_seg_len", "eval_mem_len",
        "d_model", "d_essence", "max_seq_len", "device", "dtype",
    ]
    seen = set()
    for k in priority:
        if k in config:
            _print_val(k, config[k])
            seen.add(k)
    for k in sorted(config.keys()):
        if k not in seen:
            _print_val(k, config[k])
    print("=" * 70 + "\n")


def _print_val(key: str, val) -> None:
    if isinstance(val, (list, tuple)) and len(val) > 8:
        print(f"  {key}: [{val[0]}, {val[1]}, ... ({len(val)} items)]")
    elif isinstance(val, dict):
        print(f"  {key}:")
        for sk, sv in val.items():
            print(f"    {sk}: {sv}")
    else:
        print(f"  {key}: {val}")
