import csv
import io
from pathlib import Path
from types import SimpleNamespace

from tajalli.training.trainer import Phase1Trainer
import tajalli.training.trainer as trainer_mod


def test_optimizer_step_total_prefers_paper_target_optimizer_steps():
    trainer = Phase1Trainer.__new__(Phase1Trainer)
    trainer.config = {"paper_target_optimizer_steps": 50_000}
    trainer.max_steps = 1_600_000
    trainer.grad_accum = 32

    assert trainer._optimizer_step_total() == 50_000


def test_phase1_progress_bar_tracks_optimizer_steps(monkeypatch, tmp_path: Path):
    created = {}

    class FakePbar:
        def __init__(self, *, total, desc, unit, dynamic_ncols):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.dynamic_ncols = dynamic_ncols
            self.update_calls = []
            self.postfix_calls = []
            self.closed = False

        def update(self, value):
            self.update_calls.append(value)

        def set_postfix(self, **kwargs):
            self.postfix_calls.append(kwargs)

        def close(self):
            self.closed = True

    def fake_tqdm(*, total, desc, unit, dynamic_ncols):
        pbar = FakePbar(total=total, desc=desc, unit=unit, dynamic_ncols=dynamic_ncols)
        created["pbar"] = pbar
        return pbar

    monkeypatch.setattr(trainer_mod, "tqdm", fake_tqdm)

    trainer = Phase1Trainer.__new__(Phase1Trainer)
    trainer.model = SimpleNamespace(train=lambda: None)
    trainer.optimizer = SimpleNamespace(
        zero_grad=lambda: None,
        param_groups=[{"lr": 1.0e-3}],
    )
    trainer.scheduler = SimpleNamespace()
    trainer.writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None, close=lambda: None)
    trainer.train_loader = [{"input_ids": [[0]]}]
    trainer.val_loader = []
    trainer.config = {}
    trainer.model_name = "tajalli"
    trainer.max_steps = 4
    trainer.grad_accum = 2
    trainer.log_every = 10_000
    trainer.eval_every = 10_000
    trainer.ckpt_every = 10_000
    trainer.lambda_gate_entropy = 0.0
    trainer.lambda_exit = 0.0
    trainer.essence_warmup_steps = None
    trainer.checkpoint_dir = tmp_path
    trainer.run_dir = tmp_path
    trainer.train_log_f = io.StringIO()
    trainer.train_log = csv.writer(trainer.train_log_f)
    trainer.eval_log_f = io.StringIO()
    trainer.eval_log = csv.writer(trainer.eval_log_f)
    trainer._save_checkpoint = lambda *args, **kwargs: None
    trainer.train_step = lambda batch, step: {
        "loss": 1.0,
        "_tokens": 8,
        "_lr": 1.0e-3,
    }
    trainer.evaluate = lambda n_steps=None: {"val_loss": 0.0, "val_perplexity": 1.0}

    trainer.train(start_step=0, optimizer_step=0, best_val_ppl=float("inf"), tokens_total=0)

    pbar = created["pbar"]
    assert pbar.total == 2
    assert pbar.unit == "opt_step"
    assert pbar.update_calls == [1, 1]
    assert pbar.postfix_calls[-1]["accum"] == "2/2"
    assert pbar.closed is True
