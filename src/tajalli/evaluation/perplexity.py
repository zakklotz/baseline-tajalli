"""Perplexity and language modeling evaluation."""

import torch
import torch.nn as nn
from tqdm import tqdm


def compute_perplexity_with_memory(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    seg_len: int = 128,
    mem_len: int = 384,
    max_batches: int | None = None,
    n_steps: int | None = None,
    show_progress: bool = True,
) -> tuple[float, float]:
    """
    Compute perplexity with Transformer-XL style segment recurrence.
    Processes seq_len chunks in segments of seg_len; caches last mem_len hidden states.
    Returns (loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    n_batches = 0

    batch_iter = dataloader
    if show_progress:
        batch_iter = tqdm(dataloader, desc="PPL (memory)", leave=False, unit="batch")
    with torch.no_grad():
        for batch in batch_iter:
            if max_batches and n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            seq_len = input_ids.shape[1]
            if seq_len < 2:
                continue
            full_seq = torch.cat([input_ids, labels[:, -1:]], dim=1)
            L = full_seq.shape[1]
            memory_h = None
            mem_pos_start = 0
            for start in range(0, L - 1, seg_len):
                end = min(start + seg_len, L - 1)
                seg_input = full_seq[:, start : end]
                seg_labels = full_seq[:, start + 1 : end + 1]
                seg_len_cur = seg_input.shape[1]
                if seg_len_cur < 1:
                    break
                mask = torch.tril(torch.ones(seg_len_cur, seg_len_cur, device=device)).unsqueeze(0).unsqueeze(0)
                kw = dict(mask=mask, memory_h=memory_h, mem_pos_start=mem_pos_start)
                if n_steps is not None:
                    kw["n_steps"] = n_steps
                use_adaptive = getattr(model, "adaptive_output", None) is not None
                if use_adaptive:
                    kw["return_hidden"] = True
                try:
                    out = model(seg_input, **kw)
                except TypeError:
                    kw.pop("memory_h", None)
                    kw.pop("mem_pos_start", None)
                    kw.pop("return_hidden", None)
                    out = model(seg_input, mask=mask)
                    use_adaptive = False
                out_t = out[0] if isinstance(out, tuple) else out
                if use_adaptive:
                    from ..data.freq_vocab import labels_to_freq_rank
                    h = out_t
                    labels_ranked = labels_to_freq_rank(seg_labels.reshape(-1), model.orig_to_rank)
                    _, loss_t = model.adaptive_output(h.view(-1, h.size(-1)), labels_ranked)
                    n_valid = (seg_labels != -100).sum().item()
                    loss = loss_t * n_valid
                else:
                    logits = out_t
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        seg_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                n = (seg_labels != -100).sum().item()
                total_loss += loss.item()
                n_tokens += n
                memory_h = out[0] if isinstance(out, tuple) else out
                mem_pos_start = start
            n_batches += 1

    avg_loss = total_loss / max(1, n_tokens)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


def compute_perplexity(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int | None = None,
    n_steps: int | None = None,
    show_progress: bool = True,
    **model_kwargs,
) -> tuple[float, float]:
    """
    Compute validation loss and perplexity.
    Returns (loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    n_batches = 0

    batch_iter = dataloader
    if show_progress:
        batch_iter = tqdm(dataloader, desc="Perplexity", leave=False, unit="batch")
    with torch.no_grad():
        for batch in batch_iter:
            if max_batches and n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            seq_len = input_ids.shape[1]

            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
            kw = dict(mask=mask, **model_kwargs)
            if n_steps is not None:
                kw["n_steps"] = n_steps
            try:
                out = model(input_ids, **kw)
            except TypeError:
                out = model(input_ids, mask=mask)
            logits = out[0] if isinstance(out, tuple) else out

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n = (labels != -100).sum().item()
            total_loss += loss.item()
            n_tokens += n
            n_batches += 1

    avg_loss = total_loss / max(1, n_tokens)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity
