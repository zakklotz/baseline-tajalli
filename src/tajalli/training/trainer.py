"""Phase 1 and Phase 2 training loops with metrics logging."""

import math
import os
import random
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .scheduler import get_cosine_warmup_scheduler, get_multistep_lr_scheduler
from .losses import pair_orthogonality_loss, pair_coverage_loss
from ..model.diverse_moe import DiverseAsmaaMoE, RandomAssignMoE, TwoPhaseMoE


def _get_param_groups_decay_no_decay(
    named_params,
    base_lr: float,
    weight_decay: float,
    alpha_lr_scale: float = 1.0,
) -> list[dict]:
    """Split params into decay and no_decay groups. No decay for bias, *norm*, *ln*."""
    other_decay, other_no_decay, alpha_decay, alpha_no_decay = [], [], [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        is_alpha = "alpha_per_step" in n
        lr = base_lr * (alpha_lr_scale if is_alpha else 1.0)
        no_decay_param = "bias" in n or "norm" in n.lower() or "ln" in n.lower()
        if is_alpha:
            if no_decay_param:
                alpha_no_decay.append(p)
            else:
                alpha_decay.append(p)
        else:
            if no_decay_param:
                other_no_decay.append(p)
            else:
                other_decay.append(p)
    groups = []
    if other_decay:
        groups.append({"params": other_decay, "lr": base_lr, "weight_decay": weight_decay})
    if other_no_decay:
        groups.append({"params": other_no_decay, "lr": base_lr, "weight_decay": 0.0})
    if alpha_decay:
        groups.append({"params": alpha_decay, "lr": base_lr * alpha_lr_scale, "weight_decay": weight_decay})
    if alpha_no_decay:
        groups.append({"params": alpha_no_decay, "lr": base_lr * alpha_lr_scale, "weight_decay": 0.0})
    return groups


def _get_curriculum_depth_range(step: int, config: dict) -> Optional[tuple[int, int]]:
    """Return (depth_min, depth_max) for current step from depth_curriculum, or None to use default config."""
    curriculum = config.get("depth_curriculum")
    if not curriculum or not isinstance(curriculum, list):
        return None
    segment = None
    for entry in curriculum:
        end_step = entry.get("end_step", 0)
        if step <= end_step:
            segment = entry
            break
    if segment is None:
        segment = curriculum[-1]
    d_min = segment.get("depth_min")
    d_max = segment.get("depth_max")
    if d_min is not None and d_max is not None:
        return (int(d_min), int(d_max))
    return None


class Phase1Trainer:
    """Trainer for Phase 1 models with AMP, gradient accumulation, checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        model_name: str = "tajalli",
        essence_warmup_steps: Optional[int] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.essence_warmup_steps = essence_warmup_steps

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available. Training on CPU is not supported.")
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)
        torch.backends.cudnn.benchmark = True

        self.max_steps = config["max_steps"]
        self.grad_accum = config.get("gradient_accumulation_steps", 8)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.eval_every = config.get("eval_every", 1000)
        self.log_every = config.get("log_every", 100)
        self.ckpt_every = config.get("checkpoint_every", 5000)
        self.lambda_gate_entropy = config.get("lambda_gate_entropy", 0.0)
        self.lambda_exit = config.get("lambda_exit", 0.0)

        base_lr = config["lr"]
        weight_decay = config.get("weight_decay", 0.1)
        betas = tuple(config.get("betas", (0.9, 0.95)))
        eps = config.get("eps", 1e-8)
        alpha_params = [p for n, p in model.named_parameters() if "alpha_per_step" in n]
        alpha_lr_scale = 10.0 if alpha_params else 1.0
        param_groups = _get_param_groups_decay_no_decay(
            model.named_parameters(),
            base_lr=base_lr,
            weight_decay=weight_decay,
            alpha_lr_scale=alpha_lr_scale,
        )
        self.optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
        lr_schedule = config.get("lr_schedule", "cosine_with_warmup")
        warmup = config.get("warmup_steps", 500)
        if lr_schedule == "multistep":
            self.scheduler = get_multistep_lr_scheduler(
                self.optimizer,
                warmup,
                self.max_steps,
                milestones=config.get("lr_milestones", [200000, 250000]),
                gamma=config.get("lr_gamma", 0.1),
            )
        else:
            lr_final = config.get("lr_final")
            min_ratio = (lr_final / base_lr) if lr_final is not None else 0.0
            self.scheduler = get_cosine_warmup_scheduler(
                self.optimizer,
                warmup,
                self.max_steps,
                min_lr_ratio=min_ratio,
            )
        self.scaler = torch.amp.GradScaler("cuda")
        self.use_amp = True

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(log_dir, model_name))
        self.checkpoint_dir = Path(checkpoint_dir)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(
            torch.ones(seq_len, seq_len, device=device)
        ).unsqueeze(0).unsqueeze(0)

    def train_step(
        self,
        batch: dict,
        step: int,
    ) -> dict:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        seq_len = input_ids.shape[1]
        mask = self._get_causal_mask(seq_len, self.device)

        return_metrics = (
            (step + 1) % self.log_every == 0 or self.lambda_gate_entropy > 0
        ) and (
            hasattr(self.model, "tajalli_stack") or hasattr(self.model, "block")
        )

        model_kw = dict(mask=mask, return_step_metrics=return_metrics)
        depth_used = None
        if self.config.get("variable_depth_training") and hasattr(self.model, "tajalli_stack"):
            depth_warmup_steps = self.config.get("depth_warmup_steps", 0)
            if depth_warmup_steps and step < depth_warmup_steps:
                depth_used = self.config.get("depth_warmup_fixed", 6)
            else:
                curriculum_range = _get_curriculum_depth_range(step, self.config)
                if curriculum_range is not None:
                    depth_min, depth_max = curriculum_range
                    depth_used = random.randint(depth_min, depth_max)
                else:
                    depth_steps = self.config.get("depth_steps")
                    if depth_steps is not None:
                        depth_used = random.choice(depth_steps)
                    else:
                        depth_min = self.config.get("depth_min", 4)
                        depth_max = self.config.get("depth_max", 8)
                        depth_used = random.randint(depth_min, depth_max)
            model_kw["n_steps"] = depth_used

        with torch.amp.autocast("cuda"):
            use_adaptive = getattr(self.model, "adaptive_output", None) is not None
            if use_adaptive:
                model_kw["return_hidden"] = True
            try:
                out = self.model(input_ids, **model_kw)
            except TypeError:
                model_kw.pop("return_hidden", None)
                out = self.model(input_ids, mask=mask)
                return_metrics = False
                use_adaptive = False

            if isinstance(out, tuple):
                out_t, step_metrics = out[0], out[1] if return_metrics else None
            else:
                out_t, step_metrics = out, None

        # CE in fp32 for numerical stability (fp16 logits can overflow)
        if use_adaptive:
            from ..data.freq_vocab import labels_to_freq_rank
            h = out_t
            rank_src = getattr(self.model, "rank_mapping", None)
            rank_src = rank_src if rank_src is not None else self.model.orig_to_rank
            labels_ranked = labels_to_freq_rank(labels.view(-1), rank_src)
            h_f32 = h.float()
            _, loss_asm = self.model.adaptive_output(h_f32.view(-1, h.size(-1)), labels_ranked)
            loss = loss_asm / self.grad_accum
        else:
            logits = out_t
            logits_f32 = logits.float()
            loss = nn.functional.cross_entropy(
                logits_f32.view(-1, logits_f32.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / self.grad_accum

        if self.lambda_gate_entropy > 0 and step_metrics:
            gate_ent_tensor = step_metrics.get("gate_entropy_loss")
            if gate_ent_tensor is not None:
                loss = loss - (self.lambda_gate_entropy * gate_ent_tensor) / self.grad_accum
        if self.lambda_exit > 0 and step_metrics:
            exit_ent_tensor = step_metrics.get("_exit_entropy_tensor")
            if exit_ent_tensor is not None:
                loss = loss - (self.lambda_exit * exit_ent_tensor) / self.grad_accum

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % self.grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Scheduler checks optimizer._opt_called; scaler.step() may not go through the
                # patched optimizer.step(), so set it so scheduler.step() does not warn.
                setattr(self.optimizer, "_opt_called", True)
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        result = {"loss": loss.item() * self.grad_accum}
        if depth_used is not None:
            result["depth"] = depth_used
        if step_metrics:
            # Exclude live tensors from the logged metrics dict (not serializable as floats)
            result.update({
                k: v for k, v in step_metrics.items()
                if k not in ("gate_entropy_loss", "_exit_entropy_tensor")
            })
        return result

    def evaluate(self, n_steps: Optional[int] = None) -> dict:
        if self.config.get("eval_use_memory"):
            from ..evaluation.perplexity import compute_perplexity_with_memory
            seg_len = self.config.get("eval_seg_len", 128)
            mem_len = self.config.get("eval_mem_len", 384)
            loss, ppl = compute_perplexity_with_memory(
                self.model, self.val_loader, self.device,
                seg_len=seg_len, mem_len=mem_len, n_steps=n_steps,
            )
            self.model.train()
            return {"val_loss": loss, "val_perplexity": ppl}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        metrics_agg = {}
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc=f"Eval {self.model_name}",
                leave=False,
                unit="batch",
            ):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                seq_len = input_ids.shape[1]
                mask = self._get_causal_mask(seq_len, self.device)
                num_tokens = (labels != -100).sum().item()

                if hasattr(self.model, "forward"):
                    sig = self.model.forward.__code__.co_varnames
                    return_metrics = "return_step_metrics" in sig
                    n_steps_kw = "n_steps" if "n_steps" in sig else None
                    supports_return_hidden = "return_hidden" in sig
                else:
                    return_metrics = False
                    n_steps_kw = None
                    supports_return_hidden = False

                kw = {}
                if n_steps_kw and n_steps is not None:
                    kw[n_steps_kw] = n_steps
                use_adaptive = getattr(self.model, "adaptive_output", None) is not None and supports_return_hidden
                if use_adaptive:
                    kw["return_hidden"] = True

                with torch.amp.autocast("cuda"):
                    if return_metrics:
                        out, step_m = self.model(
                            input_ids, mask=mask,
                            return_step_metrics=True, **kw
                        )
                    else:
                        out = self.model(input_ids, mask=mask, **kw)
                        step_m = None
                    out_t = out[0] if isinstance(out, tuple) else out
                    if use_adaptive:
                        from ..data.freq_vocab import labels_to_freq_rank
                        h = out_t
                        rank_src = getattr(self.model, "rank_mapping", None)
                        rank_src = rank_src if rank_src is not None else self.model.orig_to_rank
                        labels_ranked = labels_to_freq_rank(labels.view(-1), rank_src)
                        h_f32 = h.float()
                        _, loss = self.model.adaptive_output(h_f32.view(-1, h.size(-1)), labels_ranked)
                    else:
                        logits = out_t
                        # CE in fp32 for numerical stability (fp16 logits can overflow)
                        logits_f32 = logits.float()
                        loss = nn.functional.cross_entropy(
                            logits_f32.view(-1, logits_f32.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                        )
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                if step_m:
                    for k, v in step_m.items():
                        metrics_agg.setdefault(k, []).append(v)

        self.model.train()
        val_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        result = {"val_loss": val_loss}
        result["val_perplexity"] = math.exp(val_loss)
        for k, v in metrics_agg.items():
            result[k] = sum(v) / len(v)
        return result

    def train(self):
        import math as _math
        self.model.train()
        global_step = 0
        best_val_ppl = float("inf")
        self.optimizer.zero_grad()
        train_iter = iter(self.train_loader)
        _gate_status = "?"
        pbar = tqdm(
            total=self.max_steps,
            desc=f"{self.model_name}",
            unit="step",
            dynamic_ncols=True,
        )

        while global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            if self.essence_warmup_steps and global_step == self.essence_warmup_steps:
                if hasattr(self.model, "essence") and hasattr(
                    self.model.essence, "freeze"
                ):
                    self.model.essence.freeze()
                    print(f"[{self.model_name}] Essence frozen at step {global_step}")

            metrics = self.train_step(batch, global_step)
            if (global_step + 1) % self.grad_accum == 0:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, global_step)

            # Update gate status every step; shown in pbar postfix
            _entropy_vals = [v for k, v in metrics.items() if k.endswith("attribute_gate_entropy") and isinstance(v, float)]
            if _entropy_vals:
                _ent = sum(_entropy_vals) / len(_entropy_vals)
                _gate_mean_vals = [v for k, v in metrics.items() if k.endswith("attribute_gate_mean")]
                _gm = _gate_mean_vals[-1] if _gate_mean_vals else None
                _n_heads = len(_gm) if _gm is not None else 7
                _max_ent = _math.log(max(_n_heads, 2))
                _pct = min(100.0, _ent / _max_ent * 100)
                _status_tag = "OK" if _pct > 50 else ("WARN" if _pct > 20 else "COLL")
                if _gm is not None:
                    _gvals = [_gm[i].item() if hasattr(_gm[i], "item") else float(_gm[i]) for i in range(len(_gm))]
                    _top = max(_gvals)
                    _top_h = _gvals.index(_top)
                    _gate_status = f"H{_top_h}={_top:.0%} ent={_ent:.2f}[{_status_tag}]"
                else:
                    _gate_status = f"ent={_ent:.2f}[{_status_tag}]"
                self.writer.add_scalar("train/gate_entropy", _ent, global_step)
                if (global_step + 1) % self.log_every == 0:
                    tqdm.write(f"  [step {global_step}] Gate: entropy={_ent:.3f}/{_max_ent:.2f} ({_pct:.0f}% diverse)  {_gate_status}")
            elif (global_step + 1) % self.log_every == 0:
                tqdm.write(f"  [step {global_step}] Gate: no data  keys={[k for k in metrics if 'gate' in k or 'attr' in k or 'entropy' in k]}")

            if (global_step + 1) % self.eval_every == 0:
                eval_n_steps = 6 if self.config.get("variable_depth_training") else None
                eval_metrics = self.evaluate(n_steps=eval_n_steps)
                for k, v in eval_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, global_step)
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.writer.add_scalar(f"val/{k}", v.item(), global_step)
                if eval_metrics["val_perplexity"] < best_val_ppl:
                    best_val_ppl = eval_metrics["val_perplexity"]
                    ckpt_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                    self._save_checkpoint(ckpt_path, global_step)
                print(
                    f"[{self.model_name}] step {global_step} "
                    f"val_ppl={eval_metrics['val_perplexity']:.2f}"
                )

            if (global_step + 1) % self.ckpt_every == 0:
                ckpt_path = self.checkpoint_dir / f"{self.model_name}_step{global_step}.pt"
                self._save_checkpoint(ckpt_path, global_step)

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{metrics.get('loss', 0):.3f}",
                best_ppl=f"{best_val_ppl:.2f}",
                gate=_gate_status,
                refresh=False,
            )

        pbar.close()
        print(f"[{self.model_name}] Training complete. Best val_ppl={best_val_ppl:.2f}")

    def _save_checkpoint(self, path: Path, step: int):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": step,
            "config": self.config,
        }
        torch.save(state, path)


class Phase2Trainer(Phase1Trainer):
    """
    Trainer for Phase 2 (MoE) model: CE + load_balance + pair orthogonality + pair coverage.
    Pair ortho/coverage ramp in after pair_loss_warmup_steps over pair_loss_ramp_steps.
    Variable-depth training; logs MoE metrics and 14x14 co-activation matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        model_name: str = "phase2_tajalli",
        **kwargs,
    ):
        super().__init__(
            model, train_loader, val_loader, config,
            log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_name=model_name,
            **kwargs,
        )
        self.pair_warmup = config.get("pair_loss_warmup_steps", 2500)
        self.pair_ramp = config.get("pair_loss_ramp_steps", 1000)
        self.lambda_ortho = config.get("lambda_ortho", 0.01)
        self.lambda_coverage = config.get("lambda_coverage", 0.01)
        self.lambda_balance = config.get("lambda_balance", 0.01)
        self.lambda_entropy = config.get("lambda_entropy", 0.0)
        self.expert_choice_routing = config.get("expert_choice_routing", False)
        # n_experts: infer from model if not in config (e.g. Phase 1 -> Phase 2)
        self.n_experts = config.get("n_experts")
        if self.n_experts is None:
            moe = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
            moe = getattr(moe, "moe_layer", None) if moe is not None else None
            self.n_experts = getattr(moe, "n_experts", 14)

    def _pair_loss_scale(self, step: int) -> float:
        if step < self.pair_warmup:
            return 0.0
        if step >= self.pair_warmup + self.pair_ramp:
            return 1.0
        return (step - self.pair_warmup) / self.pair_ramp

    def train_step(self, batch: dict, step: int) -> dict:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        seq_len = input_ids.shape[1]
        mask = self._get_causal_mask(seq_len, self.device)

        return_metrics = True
        model_kw = dict(mask=mask, return_step_metrics=return_metrics)
        if self.config.get("variable_depth_training") and hasattr(self.model, "tajalli_stack"):
            depth_warmup_steps = self.config.get("depth_warmup_steps", 0)
            if depth_warmup_steps and step < depth_warmup_steps:
                model_kw["n_steps"] = self.config.get("depth_warmup_fixed", 6)
            else:
                curriculum_range = _get_curriculum_depth_range(step, self.config)
                if curriculum_range is not None:
                    depth_min, depth_max = curriculum_range
                    model_kw["n_steps"] = random.randint(depth_min, depth_max)
                else:
                    depth_steps = self.config.get("depth_steps")
                    if depth_steps is not None:
                        model_kw["n_steps"] = random.choice(depth_steps)
                    else:
                        depth_min = self.config.get("depth_min", 4)
                        depth_max = self.config.get("depth_max", 8)
                        model_kw["n_steps"] = random.randint(depth_min, depth_max)

        use_adaptive = getattr(self.model, "adaptive_output", None) is not None
        if use_adaptive:
            model_kw["return_hidden"] = True

        with torch.amp.autocast("cuda"):
            out = self.model(input_ids, **model_kw)
            out_t, step_metrics = out[0], out[1]
            if use_adaptive:
                from ..data.freq_vocab import labels_to_freq_rank
                h = out_t
                rank_src = getattr(self.model, "rank_mapping", None)
                rank_src = rank_src if rank_src is not None else self.model.orig_to_rank
                labels_ranked = labels_to_freq_rank(labels.view(-1), rank_src)
                _, ce_loss = self.model.adaptive_output(h.view(-1, h.size(-1)), labels_ranked)
            else:
                logits = out_t
                ce_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            total_loss = ce_loss / self.grad_accum

            moe_aux = step_metrics.get("moe_aux") if step_metrics else None
            if moe_aux is not None:
                if not self.expert_choice_routing:
                    total_loss = total_loss + (self.lambda_balance * moe_aux["load_balance_loss"]) / self.grad_accum
                scale = self._pair_loss_scale(step)
                if scale > 0:
                    if "expert_outputs_subset" in moe_aux:
                        ortho = pair_orthogonality_loss(moe_aux["expert_outputs_subset"])
                        total_loss = total_loss + (scale * self.lambda_ortho * ortho) / self.grad_accum
                    cov = pair_coverage_loss(moe_aux["expert_indices"], num_experts=self.n_experts)
                    total_loss = total_loss + (scale * self.lambda_coverage * cov) / self.grad_accum

            if self.lambda_gate_entropy > 0 and step_metrics:
                gate_ent_tensor = step_metrics.get("gate_entropy_loss")
                if gate_ent_tensor is not None:
                    total_loss = total_loss - (self.lambda_gate_entropy * gate_ent_tensor) / self.grad_accum
            if getattr(self, "lambda_exit", 0) > 0 and step_metrics:
                exit_ent_tensor = step_metrics.get("_exit_entropy_tensor")
                if exit_ent_tensor is not None:
                    total_loss = total_loss - (self.lambda_exit * exit_ent_tensor) / self.grad_accum

        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (step + 1) % self.grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Scheduler checks optimizer._opt_called; scaler.step() may not go through the
                # patched optimizer.step(), so set it so scheduler.step() does not warn.
                setattr(self.optimizer, "_opt_called", True)
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        result = {"loss": ce_loss.item()}
        if step_metrics:
            for k, v in step_metrics.items():
                if k in ("moe_aux", "gate_entropy_loss", "_exit_entropy_tensor"):
                    continue
                if isinstance(v, (int, float)):
                    result[k] = v
            if moe_aux is not None:
                result["load_balance_loss"] = moe_aux["load_balance_loss"].item() if hasattr(moe_aux["load_balance_loss"], "item") else float(moe_aux["load_balance_loss"])
                if "expert_outputs_subset" in moe_aux and self._pair_loss_scale(step) > 0:
                    with torch.no_grad():
                        result["pair_ortho_loss"] = pair_orthogonality_loss(moe_aux["expert_outputs_subset"]).item()
                    result["pair_coverage_loss"] = pair_coverage_loss(moe_aux["expert_indices"], num_experts=self.n_experts).item()
                idx = moe_aux["expert_indices"]
                probs = moe_aux["router_probs"]
                B, T = probs.shape[0], probs.shape[1]
                n_exp = self.n_experts
                if idx.dim() == 3 and idx.shape[-1] == n_exp:
                    # expert-choice: idx is (B, T, n_experts) contribution mask; vectorized coactivation
                    expert_freq = idx.sum(dim=(0, 1))
                    expert_freq = expert_freq / (expert_freq.sum() + 1e-10)
                    active = (idx > 0).float()
                    coact = torch.einsum("bti,btj->ij", active, active)
                else:
                    # token-choice: idx is (B, T, top_k); vectorized expert_freq and coactivation
                    K = idx.shape[-1]
                    expert_freq = (idx.unsqueeze(-1) == torch.arange(n_exp, device=idx.device)).float().sum(dim=(0, 1, 2))
                    expert_freq = expert_freq / (B * T * K + 1e-10)
                    I = idx[:, :, 0].flatten().long()
                    J = idx[:, :, 1].flatten().long()
                    coact = torch.zeros(n_exp, n_exp, device=idx.device, dtype=torch.float32)
                    coact.view(-1).scatter_add_(0, I * n_exp + J, torch.ones(I.shape[0], device=idx.device, dtype=coact.dtype))
                    coact.view(-1).scatter_add_(0, J * n_exp + I, torch.ones(J.shape[0], device=idx.device, dtype=coact.dtype))
                result["router_entropy"] = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                result["top1_expert_share"] = expert_freq.max().item()
                result["coactivation_matrix"] = coact.cpu().float()
        return result

    def evaluate(self, n_steps: Optional[int] = None) -> dict:
        """Eval Phase 2 model; skip aggregating moe_aux (non-scalar)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        metrics_agg = {}
        last_moe_aux = None
        last_attribute_gate_mean = None
        use_adaptive = getattr(self.model, "adaptive_output", None) is not None
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Eval {self.model_name}", leave=False, unit="batch"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                seq_len = input_ids.shape[1]
                mask = self._get_causal_mask(seq_len, self.device)
                kw = dict(mask=mask, return_step_metrics=True)
                if n_steps is not None:
                    kw["n_steps"] = n_steps
                if use_adaptive:
                    kw["return_hidden"] = True
                with torch.amp.autocast("cuda"):
                    out_t, step_m = self.model(input_ids, **kw)
                    if use_adaptive:
                        from ..data.freq_vocab import labels_to_freq_rank
                        h = out_t
                        rank_src = getattr(self.model, "rank_mapping", None)
                        rank_src = rank_src if rank_src is not None else self.model.orig_to_rank
                        labels_ranked = labels_to_freq_rank(labels.view(-1), rank_src)
                        _, loss = self.model.adaptive_output(h.view(-1, h.size(-1)), labels_ranked)
                    else:
                        logits = out_t
                        # CE in fp32 for numerical stability (fp16 logits can overflow)
                        logits_f32 = logits.float()
                        loss = nn.functional.cross_entropy(
                            logits_f32.view(-1, logits_f32.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                        )
                total_loss += loss.item()
                n_batches += 1
                if step_m:
                    if n_steps is not None:
                        key = f"step_{n_steps - 1}_attribute_gate_mean"
                        if key in step_m:
                            last_attribute_gate_mean = step_m[key]
                    for k, v in step_m.items():
                        if k == "moe_aux":
                            last_moe_aux = v
                            continue
                        if isinstance(v, (int, float)):
                            metrics_agg.setdefault(k, []).append(v)
        self.model.train()
        result = {"val_loss": total_loss / max(1, n_batches)}
        result["val_perplexity"] = math.exp(result["val_loss"])
        for k, v in metrics_agg.items():
            result[k] = sum(v) / len(v)
        if last_attribute_gate_mean is not None:
            result["attribute_gate_mean"] = last_attribute_gate_mean
        if last_moe_aux is not None:
            if "archetype_weights" in last_moe_aux:
                aw = last_moe_aux["archetype_weights"]
                result["eval_archetype_mean"] = aw.mean(dim=(0, 1)).cpu().tolist()
                result["eval_archetype_weight_var_per_token"] = aw.var(dim=1).mean().item()
            if "router_probs" in last_moe_aux:
                result["eval_router_entropy"] = -(last_moe_aux["router_probs"] * (last_moe_aux["router_probs"] + 1e-10).log()).sum(dim=-1).mean().item()
        return result

    def train(self):
        self.model.train()
        global_step = 0
        best_val_ppl = float("inf")
        self.optimizer.zero_grad()
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.max_steps, desc=self.model_name, unit="step", dynamic_ncols=True)

        import math as _math
        _gate_status = "?"  # shown in pbar postfix every step

        while global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch, global_step)
            if (global_step + 1) % self.grad_accum == 0:
                for k, v in metrics.items():
                    if k == "coactivation_matrix":
                        if (global_step + 1) % (self.log_every * 10) == 0 and v is not None:
                            mat = v.float()
                            if mat.numel() > 0 and mat.max() > 0:
                                mat = mat / mat.max()
                            self.writer.add_image("moe/coactivation_14x14", mat.unsqueeze(0), global_step, dataformats="CHW")
                        continue
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, global_step)

            # Update gate status every step for pbar postfix
            entropy_vals = [v for k, v in metrics.items() if k.endswith("attribute_gate_entropy") and isinstance(v, float)]
            gate_mean_vals = [v for k, v in metrics.items() if k.endswith("attribute_gate_mean")]
            if entropy_vals:
                _entropy = sum(entropy_vals) / len(entropy_vals)
                _gate_mean = gate_mean_vals[-1] if gate_mean_vals else None
                _n_heads = len(_gate_mean) if _gate_mean is not None else 7
                _max_ent = _math.log(max(_n_heads, 2))
                _pct = min(100.0, _entropy / _max_ent * 100)
                if _gate_mean is not None:
                    _gvals = [_gate_mean[i].item() if hasattr(_gate_mean[i], "item") else float(_gate_mean[i]) for i in range(len(_gate_mean))]
                    _top = max(_gvals)
                    _top_h = _gvals.index(_top)
                    _gate_status = f"H{_top_h}={_top:.0%} ent={_entropy:.2f}({'OK' if _pct>50 else 'WARN' if _pct>20 else 'COLL'})"
                else:
                    _gate_status = f"ent={_entropy:.2f}({'OK' if _pct>50 else 'WARN' if _pct>20 else 'COLL'})"
                self.writer.add_scalar("train/gate_entropy", _entropy, global_step)
                # Detailed log every log_every steps
                if (global_step + 1) % self.log_every == 0:
                    tqdm.write(f"  [step {global_step}] Gate: entropy={_entropy:.3f}/{_max_ent:.2f} ({_pct:.0f}% diverse)  {_gate_status}")

            if (global_step + 1) % self.eval_every == 0:
                eval_metrics = self.evaluate(n_steps=6)
                for k, v in eval_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, global_step)
                if eval_metrics["val_perplexity"] < best_val_ppl:
                    best_val_ppl = eval_metrics["val_perplexity"]
                    self._save_checkpoint(self.checkpoint_dir / f"{self.model_name}_best.pt", global_step)
                pbar.write(f"[{self.model_name}] step {global_step} val_ppl={eval_metrics['val_perplexity']:.2f}")
                if "attribute_gate_mean" in eval_metrics:
                    gate_mean = eval_metrics["attribute_gate_mean"]
                    for i in range(len(gate_mean)):
                        w = gate_mean[i].item() if hasattr(gate_mean[i], "item") else float(gate_mean[i])
                        self.writer.add_scalar(f"train/attr_gate_{i}", w, global_step)
                    summary = [round(gate_mean[i].item() if hasattr(gate_mean[i], "item") else float(gate_mean[i]), 3) for i in range(len(gate_mean))]
                    pbar.write(f"Attr gate mean: {summary}")

            if (global_step + 1) % self.ckpt_every == 0:
                self._save_checkpoint(self.checkpoint_dir / f"{self.model_name}_step{global_step}.pt", global_step)

            # Log per-step alpha profile every 1000 steps
            if (global_step + 1) % 1000 == 0 and hasattr(self.model, "tajalli_stack"):
                block = getattr(self.model.tajalli_stack, "block", None)
                layer = getattr(block, "tajalli_layer", None) if block is not None else None
                if layer is not None and hasattr(layer, "alpha_per_step"):
                    with torch.no_grad():
                        alpha_vals = torch.sigmoid(layer.alpha_per_step).detach().cpu()
                    n_show = min(8, alpha_vals.shape[0])
                    profile = [round(alpha_vals[i].item(), 2) for i in range(n_show)]
                    self.writer.add_scalar("train/alpha_step_0", alpha_vals[0].item(), global_step)
                    for i in range(1, n_show):
                        self.writer.add_scalar(f"train/alpha_step_{i}", alpha_vals[i].item(), global_step)
                    pbar.write(f"Alpha profile (step 0→{n_show - 1}): {profile}")
            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{metrics.get('loss', 0):.3f}", best_ppl=f"{best_val_ppl:.2f}", gate=_gate_status, refresh=False)
        pbar.close()
        print(f"[{self.model_name}] Training complete. Best val_ppl={best_val_ppl:.2f}")


def _diversity_scale(step: int, warmup: int, ramp: int) -> float:
    """Scale from 0 to 1 over [warmup, warmup+ramp]."""
    if step < warmup:
        return 0.0
    if step >= warmup + ramp:
        return 1.0
    return (step - warmup) / ramp


class Phase2DiverseTrainer(Phase2Trainer):
    """
    Trainer for Phase 2 with DiverseAsmaaMoE: CE + entropy + ortho;
    diversity warmup/ramp; logs utilization entropy, max/min expert share every diversity_log_every.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        model_name: str = "phase2_tajalli_diverse",
        **kwargs,
    ):
        super().__init__(
            model, train_loader, val_loader, config,
            log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_name=model_name,
            **kwargs,
        )
        self.diversity_warmup = config.get("diversity_warmup_steps", 3000)
        self.diversity_ramp = config.get("diversity_ramp_steps", 1000)
        self.lambda_entropy = config.get("lambda_entropy", 0.05)
        self.lambda_ortho = config.get("lambda_ortho", 0.02)
        self.diversity_log_every = config.get("diversity_log_every", 500)

        expert_lr = config.get("expert_lr")
        moe_init = config.get("moe", {}).get("init", "copy")
        moe = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
        moe = getattr(moe, "moe_layer", None) if moe is not None else None
        if (expert_lr is not None or moe_init == "random_negated") and moe is not None and isinstance(moe, DiverseAsmaaMoE):
            expert_params = [moe.w1, moe.b1, moe.w2, moe.b2] + list(moe.router.parameters())
            expert_param_ids = {id(p) for p in expert_params}
            other_params = [p for p in self.model.parameters() if id(p) not in expert_param_ids]
            base_lr = config["lr"]
            expert_lr_val = expert_lr if expert_lr is not None else 3e-4
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": base_lr},
                    {"params": expert_params, "lr": expert_lr_val},
                ],
                weight_decay=config.get("weight_decay", 0.1),
            )
            self.scheduler = get_cosine_warmup_scheduler(
                self.optimizer,
                config.get("warmup_steps", 500),
                self.max_steps,
            )
            print(f"Phase2DiverseTrainer: two param groups (base_lr={base_lr}, expert_lr={expert_lr_val})")
        if moe is not None and isinstance(moe, TwoPhaseMoE):
            self._rescue_consecutive_below = [0] * 14
            self._rescue_steps_remaining = [0] * 14

    def _log_inference_utilization(self, batch: dict) -> None:
        """Run one batch in eval mode and print router (inference) vs uniform utilization."""
        self.model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            seq_len = input_ids.shape[1]
            mask = self._get_causal_mask(seq_len, self.device)
            _ = self.model(input_ids, mask=mask, return_step_metrics=True)
            step_metrics = _[1]
        self.model.train()
        moe_aux = step_metrics.get("moe_aux") if step_metrics else None
        if moe_aux is None or "expert_indices" not in moe_aux:
            return
        idx = moe_aux["expert_indices"]
        B, T = idx.shape[0], idx.shape[1]
        N = B * T
        expert_freq = torch.zeros(14, device=idx.device)
        for e in range(14):
            expert_freq[e] = (idx == e).float().sum()
        expert_freq = expert_freq / (N + 1e-10)
        max_s = expert_freq.max().item()
        min_s = expert_freq.min().item()
        full_s = " ".join(f"e{i}={expert_freq[i].item():.3f}" for i in range(14))
        print(f"Inference utilization (router argmax): max={max_s:.3f} min={min_s:.3f} (full: {full_s})")
        print("Training utilization is uniform by construction (~1/14 per expert).")

    def train_step(self, batch: dict, step: int) -> dict:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        seq_len = input_ids.shape[1]
        mask = self._get_causal_mask(seq_len, self.device)

        return_metrics = True
        model_kw = dict(mask=mask, return_step_metrics=return_metrics)
        n_steps = None
        if self.config.get("variable_depth_training") and hasattr(self.model, "tajalli_stack"):
            depth_warmup_steps = self.config.get("depth_warmup_steps", 0)
            if depth_warmup_steps and step < depth_warmup_steps:
                n_steps = self.config.get("depth_warmup_fixed", 6)
            else:
                curriculum_range = _get_curriculum_depth_range(step, self.config)
                if curriculum_range is not None:
                    depth_min, depth_max = curriculum_range
                    n_steps = random.randint(depth_min, depth_max)
                else:
                    depth_steps = self.config.get("depth_steps")
                    if depth_steps is not None:
                        n_steps = random.choice(depth_steps)
                    else:
                        depth_min = self.config.get("depth_min", 4)
                        depth_max = self.config.get("depth_max", 8)
                        n_steps = random.randint(depth_min, depth_max)
            model_kw["n_steps"] = n_steps
        moe = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
        moe = getattr(moe, "moe_layer", None) if moe is not None else None
        if moe is not None and not isinstance(moe, RandomAssignMoE):
            model_kw["global_step"] = step
        if isinstance(moe, TwoPhaseMoE) and hasattr(self, "_rescue_steps_remaining"):
            rescue_experts = [e for e in range(14) if self._rescue_steps_remaining[e] > 0]
            model_kw["rescue_experts"] = rescue_experts if rescue_experts else None
            self._rescue_experts_this_step = rescue_experts
        else:
            self._rescue_experts_this_step = []

        with torch.amp.autocast("cuda"):
            out = self.model(input_ids, **model_kw)
            logits, step_metrics = out[0], out[1]
            ce_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            total_loss = ce_loss / self.grad_accum

            moe_aux = step_metrics.get("moe_aux") if step_metrics else None
            if moe_aux is not None:
                scale = _diversity_scale(step, self.diversity_warmup, self.diversity_ramp)
                if isinstance(moe, RandomAssignMoE):
                    if "ortho_loss" in moe_aux and scale > 0:
                        total_loss = total_loss + (scale * self.lambda_ortho * moe_aux["ortho_loss"]) / self.grad_accum
                elif isinstance(moe, TwoPhaseMoE):
                    if "ortho_loss" in moe_aux and scale > 0:
                        total_loss = total_loss + (scale * self.lambda_ortho * moe_aux["ortho_loss"]) / self.grad_accum
                    if step >= getattr(moe, "switch_step", 10000) and "entropy_loss" in moe_aux and scale > 0:
                        total_loss = total_loss + (scale * self.lambda_entropy * moe_aux["entropy_loss"]) / self.grad_accum
                elif "entropy_loss" in moe_aux and "ortho_loss" in moe_aux:
                    total_loss = total_loss + (
                        scale * (
                            self.lambda_entropy * moe_aux["entropy_loss"]
                            + self.lambda_ortho * moe_aux["ortho_loss"]
                        )
                    ) / self.grad_accum
                elif scale > 0 and "expert_outputs_subset" in moe_aux:
                    ortho = pair_orthogonality_loss(moe_aux["expert_outputs_subset"])
                    total_loss = total_loss + (scale * self.lambda_ortho * ortho) / self.grad_accum

        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (step + 1) % self.grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Scheduler checks optimizer._opt_called; scaler.step() may not go through the
                # patched optimizer.step(), so set it so scheduler.step() does not warn.
                setattr(self.optimizer, "_opt_called", True)
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        result = {"loss": ce_loss.item()}
        if n_steps is not None:
            result["depth"] = n_steps
        if step_metrics:
            for k, v in step_metrics.items():
                if k == "moe_aux":
                    continue
                if isinstance(v, (int, float)):
                    result[k] = v
            if moe_aux is not None:
                result["load_balance_loss"] = moe_aux["load_balance_loss"].item() if hasattr(moe_aux["load_balance_loss"], "item") else float(moe_aux["load_balance_loss"])
                if "entropy_loss" in moe_aux:
                    result["entropy_loss"] = moe_aux["entropy_loss"].item()
                if "ortho_loss" in moe_aux:
                    result["ortho_loss"] = moe_aux["ortho_loss"].item()
                if "expert_outputs_subset" in moe_aux and self._pair_loss_scale(step) > 0:
                    with torch.no_grad():
                        result["pair_ortho_loss"] = pair_orthogonality_loss(moe_aux["expert_outputs_subset"]).item()
                idx = moe_aux["expert_indices"]
                probs = moe_aux["router_probs"]
                # expert_indices is (B, T, K) with K=2 (token-choice) or 14 (expert-choice); use idx for B,T
                if idx.dim() == 3:
                    B, T = idx.shape[0], idx.shape[1]
                else:
                    B, T = probs.shape[0], probs.shape[1]
                if idx.dim() == 3 and idx.shape[-1] == 14:
                    expert_freq = idx.sum(dim=(0, 1))
                    expert_freq = expert_freq / (expert_freq.sum() + 1e-10)
                    result["top1_expert_share"] = expert_freq.max().item()
                    result["min_expert_share"] = expert_freq.min().item()
                    p = expert_freq + 1e-10
                    result["utilization_entropy"] = -(p * p.log()).sum().item()
                    coact = torch.zeros(14, 14, device=idx.device)
                    for b in range(B):
                        for t in range(T):
                            contrib = (idx[b, t] > 1e-6).nonzero(as_tuple=True)[0]
                            for i in contrib:
                                for j in contrib:
                                    coact[i, j] += 1
                    result["coactivation_matrix"] = coact.cpu().float()
                    if "pair_activation_stats" in moe_aux:
                        result["pair_activation_stats"] = moe_aux["pair_activation_stats"].cpu()
                else:
                    K = idx.shape[-1]
                    expert_freq = torch.zeros(14, device=idx.device)
                    for e in range(14):
                        expert_freq[e] = (idx == e).float().sum()
                    expert_freq = expert_freq / (B * T * K + 1e-10)
                    result["top1_expert_share"] = expert_freq.max().item()
                    result["min_expert_share"] = expert_freq.min().item()
                    p = expert_freq + 1e-10
                    result["utilization_entropy"] = -(p * p.log()).sum().item()
                    coact = torch.zeros(14, 14, device=idx.device)
                    if K >= 2:
                        for b in range(B):
                            for t in range(T):
                                i, j = idx[b, t, 0].item(), idx[b, t, 1].item()
                                coact[i, j] += 1
                                coact[j, i] += 1
                    result["coactivation_matrix"] = coact.cpu().float()
                    if step == 4999 and K >= 2:
                        result["expert_freq_full"] = expert_freq.cpu().tolist()
                    if "phase" in moe_aux:
                        result["phase"] = moe_aux["phase"]
                    if step == 9999 and K == 1:
                        result["expert_freq_full"] = expert_freq.cpu().tolist()
                    if "router_weights" in moe_aux and "expert_out_norms" in moe_aux:
                        rw = moe_aux["router_weights"]
                        asn = idx.reshape(-1)
                        expert_avg_weight = torch.zeros(14, device=idx.device)
                        for e in range(14):
                            m = asn == e
                            if m.any():
                                expert_avg_weight[e] = rw[m].float().mean()
                        result["expert_avg_weight"] = expert_avg_weight.cpu()
                        result["expert_out_norms"] = moe_aux["expert_out_norms"].cpu()
                    if isinstance(moe, TwoPhaseMoE) and moe_aux.get("phase") == "B" and hasattr(self, "_rescue_consecutive_below"):
                        for e in range(14):
                            f = expert_freq[e].item()
                            if f < 0.02:
                                self._rescue_consecutive_below[e] += 1
                            else:
                                self._rescue_consecutive_below[e] = 0
                            if self._rescue_consecutive_below[e] >= 500:
                                self._rescue_steps_remaining[e] = 200
                                self._rescue_consecutive_below[e] = 0
                                print(f"Rescue routing: expert {e} triggered at step {step} (was <2% for 500 steps), routing 5% of tokens for 200 steps")
                        for e in self._rescue_experts_this_step:
                            self._rescue_steps_remaining[e] = max(0, self._rescue_steps_remaining[e] - 1)
                pr = moe_aux.get("router_probs")
                if pr is not None and pr.dim() == 2:
                    p_soft = torch.softmax(pr, dim=-1)
                    result["router_entropy"] = -(p_soft * (p_soft + 1e-10).log()).sum(dim=-1).mean().item()
        return result

    def train(self):
        self.model.train()
        global_step = 0
        best_val_ppl = float("inf")
        self.optimizer.zero_grad()
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.max_steps, desc=self.model_name, unit="step", dynamic_ncols=True)

        while global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch, global_step)
            if global_step == 0 and "top1_expert_share" in metrics and "min_expert_share" in metrics:
                print(f"Step 0 expert utilization: max={metrics['top1_expert_share']:.3f} min={metrics['min_expert_share']:.3f}")
            if metrics.get("loss", 0) > 20:
                B, S = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
                depth_s = metrics.get("depth", "?")
                print(f"Loss spike: loss={metrics['loss']:.3f} depth={depth_s} B={B} S={S}")
            if (global_step + 1) % self.grad_accum == 0:
                for k, v in metrics.items():
                    if k == "coactivation_matrix":
                        if (global_step + 1) % (self.log_every * 10) == 0 and v is not None:
                            mat = v.float()
                            if mat.numel() > 0 and mat.max() > 0:
                                mat = mat / mat.max()
                            self.writer.add_image("moe/coactivation_14x14", mat.unsqueeze(0), global_step, dataformats="CHW")
                        continue
                    if k == "pair_activation_stats" and v is not None:
                        if (global_step + 1) % self.diversity_log_every == 0 and hasattr(v, "shape"):
                            for i in range(v.shape[0]):
                                self.writer.add_scalar(f"diversity/pair_coactivation_{i}", v[i].item(), global_step)
                        continue
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"train/{k}", v, global_step)
                if "depth" in metrics and isinstance(metrics["depth"], (int, float)):
                    self.writer.add_scalar("train/depth", metrics["depth"], global_step)
                if (global_step + 1) % self.diversity_log_every == 0:
                    for key in ("utilization_entropy", "top1_expert_share", "min_expert_share"):
                        if key in metrics and isinstance(metrics[key], (int, float)):
                            self.writer.add_scalar(f"diversity/{key}", metrics[key], global_step)
                    if "archetype_weights" in metrics:
                        aw = metrics["archetype_weights"]
                        if hasattr(aw, "tolist"):
                            aw_list = aw.tolist()
                            pbar.write(f"Archetype weights: {[round(w, 3) for w in aw_list]}  (uniform=1/{len(aw_list):.0f}={1/len(aw_list):.3f})")
                            if len(aw_list) > 0:
                                pbar.write(f"Max archetype weight: {max(aw_list):.3f}  Min: {min(aw_list):.3f}")
                            for i, w in enumerate(aw_list):
                                self.writer.add_scalar(f"diversity/archetype_weight_{i}", float(w), global_step)
                if (global_step + 1) % 500 == 0 and "expert_avg_weight" in metrics and "expert_out_norms" in metrics:
                    aw = metrics["expert_avg_weight"]
                    on = metrics["expert_out_norms"]
                    for i in range(14):
                        if hasattr(aw, "numel") and aw.numel() > i:
                            self.writer.add_scalar(f"diversity/expert_avg_weight_e{i}", aw[i].item() if hasattr(aw[i], "item") else float(aw[i]), global_step)
                        if hasattr(on, "numel") and on.numel() > i:
                            self.writer.add_scalar(f"diversity/expert_out_norm_e{i}", on[i].item() if hasattr(on[i], "item") else float(on[i]), global_step)

            if (global_step + 1) % self.eval_every == 0:
                eval_metrics = self.evaluate(n_steps=6)
                for k, v in eval_metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, global_step)
                if eval_metrics["val_perplexity"] < best_val_ppl:
                    best_val_ppl = eval_metrics["val_perplexity"]
                    self._save_checkpoint(self.checkpoint_dir / f"{self.model_name}_best.pt", global_step)
                pbar.write(f"[{self.model_name}] step {global_step} val_ppl={eval_metrics['val_perplexity']:.2f}")

            if (global_step + 1) % self.ckpt_every == 0:
                self._save_checkpoint(self.checkpoint_dir / f"{self.model_name}_step{global_step}.pt", global_step)

            global_step += 1
            pbar.update(1)
            if global_step == 5000:
                moe_for_check = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
                moe_for_check = getattr(moe_for_check, "moe_layer", None) if moe_for_check is not None else None
                if isinstance(moe_for_check, RandomAssignMoE):
                    self._log_inference_utilization(batch)
                elif isinstance(moe_for_check, TwoPhaseMoE):
                    pass
                elif "top1_expert_share" in metrics and "min_expert_share" in metrics:
                    max_s = metrics["top1_expert_share"]
                    min_s = metrics["min_expert_share"]
                    full = metrics.get("expert_freq_full")
                    if full is not None:
                        full_s = " ".join(f"e{i}={full[i]:.3f}" for i in range(14))
                        print(f"Step 5000 expert utilization: max={max_s:.3f} min={min_s:.3f} (full: {full_s})")
                    else:
                        print(f"Step 5000 expert utilization: max={max_s:.3f} min={min_s:.3f}")
                    if min_s < 0.01 or max_s > 0.4:
                        print("WARNING: Utilization collapsed (min < 0.01 or max > 0.4). Stopping training.")
                        break
                    elif min_s >= 0.01 and max_s <= 0.20:
                        print("Step 5000 check passed (min >= 0.01, max <= 0.20). Continuing.")
            if global_step == 10000:
                moe_10k = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
                moe_10k = getattr(moe_10k, "moe_layer", None) if moe_10k is not None else None
                if isinstance(moe_10k, TwoPhaseMoE):
                    full = metrics.get("expert_freq_full")
                    if full is not None:
                        full_s = " ".join(f"e{i}={full[i]:.3f}" for i in range(14))
                        print(f"Step 10000 (end Phase A) training utilization (should be uniform): {full_s}")
                    self._log_inference_utilization(batch)
                    if "expert_out_norms" in metrics and hasattr(metrics["expert_out_norms"], "tolist"):
                        norms = metrics["expert_out_norms"].tolist()
                        norms_s = " ".join(f"e{i}={norms[i]:.3f}" for i in range(min(14, len(norms))))
                        print(f"Step 10000 per-expert output norms: {norms_s}")
            moe_phase = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
            moe_phase = getattr(moe_phase, "moe_layer", None) if moe_phase is not None else None
            after_transition_step = getattr(moe_phase, "switch_step", 10000) + getattr(moe_phase, "transition_steps", 2000) if isinstance(moe_phase, TwoPhaseMoE) else None
            if after_transition_step is not None and global_step == after_transition_step and "top1_expert_share" in metrics and "min_expert_share" in metrics:
                print(f"Step {global_step} (after transition) routing utilization: max={metrics['top1_expert_share']:.3f} min={metrics['min_expert_share']:.3f}")
            if metrics.get("phase") == "B" and global_step > 0 and global_step % 1000 == 0 and "top1_expert_share" in metrics and "min_expert_share" in metrics:
                max_s = metrics["top1_expert_share"]
                min_s = metrics["min_expert_share"]
                print(f"Phase B step {global_step} utilization: max={max_s:.3f} min={min_s:.3f}")
                if max_s > 0.3:
                    print("WARNING: Max expert share > 0.3 during Phase B.")
            depth_s = str(metrics.get("depth", "?"))
            postfix = {"loss": f"{metrics.get('loss', 0):.3f}", "depth": depth_s, "best_ppl": f"{best_val_ppl:.2f}"}
            if "top1_expert_share" in metrics:
                postfix["max_expert"] = f"{metrics['top1_expert_share']:.2f}"
            if "min_expert_share" in metrics:
                postfix["min_expert"] = f"{metrics['min_expert_share']:.2f}"
            if metrics.get("phase"):
                postfix["phase"] = metrics["phase"]
            pbar.set_postfix(**postfix, refresh=False)
        pbar.close()
        moe_final = getattr(getattr(self.model, "tajalli_stack", None), "block", None)
        moe_final = getattr(moe_final, "moe_layer", None) if moe_final is not None else None
        if isinstance(moe_final, (RandomAssignMoE, TwoPhaseMoE)):
            try:
                batch_end = next(iter(self.train_loader))
                self._log_inference_utilization(batch_end)
            except Exception:
                pass
        print(f"[{self.model_name}] Training complete. Best val_ppl={best_val_ppl:.2f}")


class Phase3Trainer(Phase2Trainer):
    """
    Trainer for Phase 3: CE + Phase 2 MoE losses + barzakh_reconstruction_loss.
    Logs lawh_attention_entropy, lawh_retrieval_similarity, barzakh_reconstruction_loss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        model_name: str = "phase3_tajalli",
        **kwargs,
    ):
        super().__init__(
            model, train_loader, val_loader, config,
            log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_name=model_name,
            **kwargs,
        )
        self.lambda_barzakh_recon = config.get("lambda_barzakh_recon", 0.1)

    def train_step(self, batch: dict, step: int) -> dict:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        seq_len = input_ids.shape[1]
        mask = self._get_causal_mask(seq_len, self.device)

        return_metrics = True
        model_kw = dict(mask=mask, return_step_metrics=return_metrics)
        if self.config.get("variable_depth_training") and hasattr(self.model, "tajalli_stack"):
            depth_warmup_steps = self.config.get("depth_warmup_steps", 0)
            if depth_warmup_steps and step < depth_warmup_steps:
                model_kw["n_steps"] = self.config.get("depth_warmup_fixed", 6)
            else:
                curriculum_range = _get_curriculum_depth_range(step, self.config)
                if curriculum_range is not None:
                    depth_min, depth_max = curriculum_range
                    model_kw["n_steps"] = random.randint(depth_min, depth_max)
                else:
                    depth_steps = self.config.get("depth_steps")
                    if depth_steps is not None:
                        model_kw["n_steps"] = random.choice(depth_steps)
                    else:
                        depth_min = self.config.get("depth_min", 4)
                        depth_max = self.config.get("depth_max", 8)
                        model_kw["n_steps"] = random.randint(depth_min, depth_max)

        with torch.amp.autocast("cuda"):
            out = self.model(input_ids, **model_kw)
            logits, step_metrics = out[0], out[1]
            # CE in fp32 for numerical stability (fp16 logits can overflow and yield loss > 100)
            logits_f32 = logits.float()
            ce_loss = nn.functional.cross_entropy(
                logits_f32.view(-1, logits_f32.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            total_loss = ce_loss / self.grad_accum

            moe_aux = step_metrics.get("moe_aux") if step_metrics else None
            if moe_aux is not None:
                if not self.expert_choice_routing:
                    total_loss = total_loss + (self.lambda_balance * moe_aux["load_balance_loss"]) / self.grad_accum
                scale = self._pair_loss_scale(step)
                if scale > 0:
                    if "expert_outputs_subset" in moe_aux:
                        ortho = pair_orthogonality_loss(moe_aux["expert_outputs_subset"])
                        total_loss = total_loss + (scale * self.lambda_ortho * ortho) / self.grad_accum
                    cov = pair_coverage_loss(moe_aux["expert_indices"], num_experts=self.n_experts)
                    total_loss = total_loss + (scale * self.lambda_coverage * cov) / self.grad_accum

            if step_metrics and "barzakh_reconstruction_loss" in step_metrics:
                total_loss = total_loss + (self.lambda_barzakh_recon * step_metrics["barzakh_reconstruction_loss"]) / self.grad_accum

            if self.lambda_gate_entropy > 0 and step_metrics:
                gate_ent_tensor = step_metrics.get("gate_entropy_loss")
                if gate_ent_tensor is not None:
                    total_loss = total_loss - (self.lambda_gate_entropy * gate_ent_tensor) / self.grad_accum
            if getattr(self, "lambda_exit", 0) > 0 and step_metrics:
                exit_ent_tensor = step_metrics.get("_exit_entropy_tensor")
                if exit_ent_tensor is not None:
                    total_loss = total_loss - (self.lambda_exit * exit_ent_tensor) / self.grad_accum

        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (step + 1) % self.grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Scheduler checks optimizer._opt_called; scaler.step() may not go through the
                # patched optimizer.step(), so set it so scheduler.step() does not warn.
                setattr(self.optimizer, "_opt_called", True)
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        result = {"loss": ce_loss.item()}
        if step_metrics:
            for k, v in step_metrics.items():
                if k == "moe_aux":
                    continue
                if isinstance(v, (int, float)):
                    result[k] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    result[k] = v.item()
            if "barzakh_reconstruction_loss" in step_metrics:
                br = step_metrics["barzakh_reconstruction_loss"]
                result["barzakh_reconstruction_loss"] = br.item() if isinstance(br, torch.Tensor) else br
            if "lawh_attention_entropy" in step_metrics:
                result["lawh_attention_entropy"] = step_metrics["lawh_attention_entropy"]
            if "lawh_retrieval_similarity" in step_metrics:
                result["lawh_retrieval_similarity"] = step_metrics["lawh_retrieval_similarity"]
            moe_aux = step_metrics.get("moe_aux")
            if moe_aux is not None:
                result["load_balance_loss"] = moe_aux["load_balance_loss"].item() if hasattr(moe_aux["load_balance_loss"], "item") else float(moe_aux["load_balance_loss"])
                if "expert_outputs_subset" in moe_aux and self._pair_loss_scale(step) > 0:
                    with torch.no_grad():
                        result["pair_ortho_loss"] = pair_orthogonality_loss(moe_aux["expert_outputs_subset"]).item()
                    result["pair_coverage_loss"] = pair_coverage_loss(moe_aux["expert_indices"], num_experts=self.n_experts).item()
                idx = moe_aux["expert_indices"]
                probs = moe_aux["router_probs"]
                B, T = probs.shape[0], probs.shape[1]
                n_exp = self.n_experts
                if idx.dim() == 3 and idx.shape[-1] == n_exp:
                    expert_freq = idx.sum(dim=(0, 1))
                    expert_freq = expert_freq / (expert_freq.sum() + 1e-10)
                    coact = torch.zeros(n_exp, n_exp, device=idx.device)
                    for b in range(B):
                        for t in range(T):
                            contrib = (idx[b, t] > 0).nonzero(as_tuple=True)[0]
                            for i in contrib:
                                for j in contrib:
                                    coact[i, j] += 1
                else:
                    K = idx.shape[-1]
                    expert_freq = torch.zeros(n_exp, device=idx.device)
                    for e in range(n_exp):
                        expert_freq[e] = (idx == e).float().sum()
                    expert_freq = expert_freq / (B * T * K + 1e-10)
                    coact = torch.zeros(n_exp, n_exp, device=idx.device)
                    for b in range(B):
                        for t in range(T):
                            i, j = idx[b, t, 0].item(), idx[b, t, 1].item()
                            coact[i, j] += 1
                            coact[j, i] += 1
                result["router_entropy"] = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                result["top1_expert_share"] = expert_freq.max().item()
                result["coactivation_matrix"] = coact.cpu().float()
        return result
