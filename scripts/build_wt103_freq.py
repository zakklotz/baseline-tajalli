#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from tajalli.data.dataloader import (
    BATCH_TOKENIZE,
    WIKITEXT103_TEXT_FIELD,
    get_wikitext103_dataset_meta,
    get_wikitext103_tokenization_meta,
    get_wikitext103_tokenizer,
    tokenize_wikitext103_batch,
)
from tajalli.data.freq_vocab import build_tokenizer_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a schema-v2 WikiText-103 frequency artifact.")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name/path used for training.")
    parser.add_argument(
        "--output",
        default="data/wikitext103_token_frequencies.json",
        help="Output JSON artifact path.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to count frequencies from.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = get_wikitext103_tokenizer(args.tokenizer)
    dataset_meta = get_wikitext103_dataset_meta(split=args.split)
    ds = load_dataset(
        dataset_meta["name"],
        dataset_meta["config"],
        split=dataset_meta["split"],
    )

    counter: Counter[int] = Counter()
    for start_idx in range(0, len(ds), BATCH_TOKENIZE):
        end_idx = min(start_idx + BATCH_TOKENIZE, len(ds))
        batch = ds[start_idx:end_idx]
        texts = [text for text in batch[WIKITEXT103_TEXT_FIELD] if text and text.strip()]
        if not texts:
            continue
        tokenized = tokenize_wikitext103_batch(tokenizer, texts)
        for ids in tokenized["input_ids"]:
            counter.update(ids)

    vocab_size = int(len(tokenizer))
    for token_id in range(vocab_size):
        counter.setdefault(token_id, 0)

    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    freq_order = [int(token_id) for token_id, _ in sorted_items]
    freq_counts = [int(count) for _, count in sorted_items]

    artifact = {
        "schema_version": 2,
        "freq_order": freq_order,
        "freq_counts": freq_counts,
        "total_tokens": int(sum(freq_counts)),
        "tokenizer": build_tokenizer_metadata(tokenizer, tokenizer_name=args.tokenizer),
        "dataset": dataset_meta,
        "tokenization": get_wikitext103_tokenization_meta(tokenizer),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(
        f"Saved schema-v2 frequency artifact to {output_path} "
        f"({len(freq_order)} vocab entries, split={args.split})."
    )


if __name__ == "__main__":
    main()
