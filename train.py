"""
Training script for MiniGPT.

Usage:
    python train.py --data data/corpus.txt --epochs 10

Or configure everything in config.py and just run:
    python train.py
"""

import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import MiniGPT
from tokenizer import WordTokenizer
from dataset import load_text_file, make_dataloaders
from config import TrainingConfig


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Apple Silicon MPS")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU")
    return dev


def count_parameters(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


@torch.no_grad()
def evaluate(model: MiniGPT, loader, device: torch.device) -> float:
    """Compute average cross-entropy loss over the validation set."""
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    if len(loader) == 0:
        return float("nan")
    return total_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, loss, cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "config": cfg.__dict__,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return ckpt["epoch"]


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────

def train(cfg: TrainingConfig):
    device = get_device()

    # ── 1. Load & tokenize data ──────────────────
    print("\n── Loading data ─────────────────────────")
    text = load_text_file(cfg.data_path, max_chars=cfg.max_chars)

    tokenizer = WordTokenizer(max_vocab_size=cfg.vocab_size)
    if getattr(cfg, "finetune", False):
        # For fine-tuning, we MUST reuse the existing vocab so token IDs
        # match the embedding weights from the base checkpoint.
        tokenizer.load(cfg.vocab_path)
    else:
        tokenizer.build_vocab(text)
        tokenizer.save(cfg.vocab_path)

    token_ids = tokenizer.encode(text)
    print(f"[Tokenizer] Total tokens: {len(token_ids):,}")

    train_loader, val_loader = make_dataloaders(
        token_ids,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
    )

    # ── 2. Build model ───────────────────────────
    print("\n── Building model ───────────────────────")
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        max_seq_len=cfg.seq_len,
        dropout=cfg.dropout,
    ).to(device)

    print(f"[Model] Parameters: {count_parameters(model)}")

    # Optional: load base model weights for fine-tuning
    if getattr(cfg, "finetune", False):
        base_path = getattr(cfg, "base_checkpoint_path", "checkpoints/model_best.pt")
        if Path(base_path).exists():
            base_ckpt = torch.load(base_path, map_location="cpu")
            # Our checkpoints are saved as dicts with "model_state"
            if isinstance(base_ckpt, dict) and "model_state" in base_ckpt:
                model.load_state_dict(base_ckpt["model_state"])
            else:
                # Fallback: allow raw state_dict
                model.load_state_dict(base_ckpt)
            print(f"[Fine-tune] Loaded base model for fine-tuning from {base_path}")
        else:
            raise FileNotFoundError(f"Base checkpoint not found: {base_path}")

    # ── 3. Optimizer & scheduler ─────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    if cfg.resume and Path(cfg.checkpoint_path).exists():
        start_epoch = load_checkpoint(cfg.checkpoint_path, model, optimizer)

    # ── 4. Training ──────────────────────────────
    print(f"\n── Training for {cfg.epochs} epochs ──────────────")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                # Gradient clipping (unscale first)
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()

            # ── Progress bar ──
            if (step + 1) % cfg.log_every == 0:
                avg = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                pct = 100 * (step + 1) / len(train_loader)
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  Epoch {epoch+1}/{cfg.epochs} [{bar}] {pct:.0f}%  "
                      f"loss={avg:.4f}  lr={lr:.2e}  {elapsed:.0f}s", end="\r")

        train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"\n  Epoch {epoch+1:>2}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} (ppl={train_ppl:.1f}) | "
              f"val_loss={val_loss:.4f} (ppl={val_ppl:.1f}) | "
              f"{time.time()-t0:.1f}s")

        # ── Save best checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss, cfg,
                            cfg.checkpoint_path.replace(".pt", "_best.pt"))

        # ── Save periodic checkpoint ──
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_loss, cfg,
                            cfg.checkpoint_path)

        # ── Sample generation ──
        if (epoch + 1) % cfg.sample_every == 0:
            print("\n  ── Sample generation ──")
            _generate_sample(model, tokenizer, device, cfg.sample_prompts)

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")

    # For fine-tuning, also save a lightweight state_dict for easy loading.
    if getattr(cfg, "finetune", False):
        out_path = getattr(cfg, "finetune_out_path", "checkpoints/model_ft.pt")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"[Fine-tune] Saved fine-tuned weights → {out_path}")

    return history


def _generate_sample(model, tokenizer, device, prompts):
    model.eval()
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=40, temperature=0.8, top_k=40)
        text = tokenizer.decode(out[0].tolist())
        print(f"  Prompt: '{prompt}'\n  → {text}\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT")
    parser.add_argument("--data", default=None, help="Path to text corpus")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune from checkpoints/model_best.pt")
    parser.add_argument("--base_checkpoint", default="checkpoints/model_best.pt", help="Base checkpoint for fine-tuning")
    parser.add_argument("--ft_out", default="checkpoints/model_ft.pt", help="Output path for fine-tuned state_dict")
    args = parser.parse_args()

    cfg = TrainingConfig()
    if args.data:
        cfg.data_path = args.data
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.resume:
        cfg.resume = True

    if args.finetune:
        # Force the requested fine-tune settings
        cfg.finetune = True
        cfg.base_checkpoint_path = args.base_checkpoint
        cfg.finetune_out_path = args.ft_out
        cfg.data_path = args.data or "data/instruction.txt"
        cfg.epochs = 5 if args.epochs is None else cfg.epochs
        cfg.learning_rate = 1e-5 if args.lr is None else cfg.learning_rate

    history = train(cfg)
