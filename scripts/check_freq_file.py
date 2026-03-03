#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tajalli.data.dataloader import (
    get_wikitext103_dataset_meta,
    get_wikitext103_tokenization_meta,
    get_wikitext103_tokenizer,
)
from tajalli.data.freq_vocab import load_freq_artifact, validate_freq_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frequency artifact for inspection or training.")
    parser.add_argument(
        "--path",
        default="data/wikitext103_token_frequencies.json",
        help="Path to the frequency artifact JSON.",
    )
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name/path expected by the run.")
    parser.add_argument("--dataset", default="wikitext", help="Expected dataset name.")
    parser.add_argument("--dataset-config", default="wikitext-103-v1", help="Expected dataset config.")
    parser.add_argument("--split", default="train", help="Expected dataset split.")
    parser.add_argument(
        "--strict-training",
        action="store_true",
        help="Require schema-v2 metadata compatibility suitable for adaptive-softmax training.",
    )
    args = parser.parse_args()

    artifact_path = Path(args.path)
    artifact = load_freq_artifact(artifact_path)
    tokenizer = get_wikitext103_tokenizer(args.tokenizer)
    tokenization_meta = get_wikitext103_tokenization_meta(tokenizer)
    dataset_meta = get_wikitext103_dataset_meta(split=args.split)
    dataset_meta["name"] = args.dataset
    dataset_meta["config"] = args.dataset_config

    validate_freq_artifact(
        artifact,
        vocab_size=int(len(tokenizer)),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        dataset_name=dataset_meta["name"],
        dataset_config=dataset_meta["config"],
        split=dataset_meta["split"],
        text_field=dataset_meta["text_field"],
        tokenization=tokenization_meta,
        strict_training=args.strict_training,
    )

    if artifact.training_compatible:
        training_status = "valid for training"
    else:
        training_status = "inspection only; rebuild with scripts/build_wt103_freq.py for training"

    print(f"[OK] {artifact_path}")
    print(f"  schema_version: {artifact.schema_version}")
    print(f"  freq_order length: {len(artifact.freq_order)}")
    print(f"  tokenizer: {args.tokenizer}")
    print(f"  dataset: {args.dataset}/{args.split}")
    print(f"  status: {training_status}")


if __name__ == "__main__":
    main()
