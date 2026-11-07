"""Trainer: AdamW + warmup/cosine, bf16 autocast on MPS, CSV log, loss plot, checkpoint.

CLI: python -m iconsmith.train --processed data/processed --run-dir runs/x [--overfit 100]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from iconsmith.data import IconDataset, collate
from iconsmith.model import IconSmith, ModelConfig
from iconsmith.tokenizer import PAD


@dataclass
class TrainConfig:
    processed: Path
    run_dir: Path
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    overfit_n: int | None = None
    bf16: bool = True
    seed: int = 0
    log_every: int = 20
    eval_every: int = 500
    max_steps: int | None = None
    device: str = "auto"
    eval_batches: int = 50


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _to(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def lr_at(step: int, cfg: TrainConfig, total: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(
    model: IconSmith, loader: DataLoader, device: str, autocast, max_batches: int
) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with autocast():
            losses.append(model.loss(_to(batch, device)).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def _plot(log_path: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(log_path)))
    steps = [int(r["step"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    va = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, tr, label="train", lw=1)
    if va:
        ax.plot([s for s, _ in va], [v for _, v in va], label="val", marker="o", ms=3, lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def train(cfg: TrainConfig, model_cfg: ModelConfig | None = None) -> dict:
    model_cfg = model_cfg or ModelConfig()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = IconDataset(cfg.processed, "train", seed=cfg.seed)
    if cfg.overfit_n:
        train_ds = Subset(train_ds, list(range(min(cfg.overfit_n, len(train_ds)))))
        val_ds = train_ds
    else:
        val_ds = IconDataset(cfg.processed, "inset_test", seed=cfg.seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    model = IconSmith(model_cfg).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    total_steps = cfg.max_steps or cfg.epochs * len(train_loader)
    use_bf16 = cfg.bf16 and device != "cpu"

    def autocast():
        if use_bf16:
            return torch.autocast(device_type=device, dtype=torch.bfloat16)
        return torch.autocast(device_type=device, enabled=False)

    log_path = cfg.run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])
    (cfg.run_dir / "config.json").write_text(
        json.dumps({"train": {k: str(v) for k, v in asdict(cfg).items()},
                    "model": asdict(model_cfg), "device": device, "n_params": model.n_params()},
                   indent=1)
    )
    print(f"[train] device={device} params={model.n_params()/1e6:.1f}M "
          f"train={len(train_ds)} val={len(val_ds)} steps={total_steps} bf16={use_bf16}")

    step, tokens_seen, t_start = 0, 0, None
    tps = 0.0
    running, n_run = 0.0, 0
    model.train()
    done = False
    while not done:
        for batch in train_loader:
            if step >= total_steps:
                done = True
                break
            batch = _to(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, total_steps)
            try:
                with autocast():
                    loss = model.loss(batch)
            except RuntimeError as e:
                if use_bf16:
                    print(f"[train] bf16 autocast failed ({e}); falling back to fp32")
                    use_bf16 = False
                    with autocast():
                        loss = model.loss(batch)
                else:
                    raise
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            running += loss.item()
            n_run += 1
            if step == 10:
                if device == "mps":
                    torch.mps.synchronize()
                t_start, tokens_seen = time.time(), 0
            if step > 10:
                tokens_seen += int((batch["tokens"][:, 1:] != PAD).sum().item())

            if step % cfg.log_every == 0 or step == total_steps:
                if t_start is not None and step > 10:
                    if device == "mps":
                        torch.mps.synchronize()
                    tps = tokens_seen / max(1e-9, time.time() - t_start)
                val = ""
                if step % cfg.eval_every == 0 or step == total_steps:
                    val = f"{evaluate(model, val_loader, device, autocast, cfg.eval_batches):.4f}"
                tr = running / max(1, n_run)
                running, n_run = 0.0, 0
                with open(log_path, "a", newline="") as f:
                    lr_now = lr_at(step, cfg, total_steps)
                    csv.writer(f).writerow([step, f"{tr:.4f}", val, f"{lr_now:.2e}", f"{tps:.0f}"])
                print(f"[train] step {step}/{total_steps} loss {tr:.4f} val {val or '-'} "
                      f"lr {lr_at(step, cfg, total_steps):.2e} tok/s {tps:.0f}")
            if step % cfg.eval_every == 0 or step == total_steps:
                torch.save(
                    {"model_cfg": asdict(model_cfg),
                     "train_cfg": {k: str(v) for k, v in asdict(cfg).items()},
                     "state_dict": model.state_dict(), "step": step},
                    cfg.run_dir / "ckpt_last.pt",
                )
    _plot(log_path, cfg.run_dir / "loss.png")
    rows = list(csv.DictReader(open(log_path)))
    final_val = next((float(r["val_loss"]) for r in reversed(rows) if r["val_loss"]), float("nan"))
    return {"final_train_loss": float(rows[-1]["train_loss"]), "final_val_loss": final_val,
            "tokens_per_sec": tps, "steps": step}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--overfit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1408)
    a = ap.parse_args()
    cfg = TrainConfig(processed=a.processed, run_dir=a.run_dir, epochs=a.epochs,
                      batch_size=a.batch_size, lr=a.lr, warmup_steps=a.warmup_steps,
                      overfit_n=a.overfit, max_steps=a.max_steps, bf16=not a.no_bf16,
                      log_every=a.log_every, eval_every=a.eval_every, device=a.device)
    mcfg = ModelConfig(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_ff=a.d_ff)
    print(json.dumps(train(cfg, mcfg), indent=1))


if __name__ == "__main__":
    main()
