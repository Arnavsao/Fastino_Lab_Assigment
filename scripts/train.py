"""
Training script for Zero-Shot Text Classification with BiEncoder.

Usage:
    python scripts/train.py                         # use default config
    python scripts/train.py --config config.yaml    # specify config
    python scripts/train.py --config config.yaml --push_to_hub  # train + push to HF Hub
"""

import argparse
import os
import sys
import time

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_dataloader
from model import BiEncoderModel


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found — training on CPU (may be slow)")
    return device


def compute_metrics(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    """Compute precision, recall, and F1 on a batch."""
    pred = (scores >= threshold).float() * mask.float()
    targets = targets * mask.float()

    tp = (pred * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    fn = ((1 - pred) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, push_to_hub: bool = False, hf_token: str = None):
    device = get_device()

    # ── Data ────────────────────────────────────────────────────────────────
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    train_loader, all_labels = get_dataloader(
        data_path=data_cfg["synthetic_data_path"],
        max_num_labels=model_cfg["max_num_labels"],
        max_num_negatives=data_cfg.get("max_negatives", 5),
        batch_size=train_cfg["batch_size"],
        split="train",
        train_split=data_cfg.get("train_split", 0.9),
    )

    val_loader, _ = get_dataloader(
        data_path=data_cfg["synthetic_data_path"],
        max_num_labels=model_cfg["max_num_labels"],
        max_num_negatives=data_cfg.get("max_negatives", 5),
        batch_size=train_cfg["batch_size"],
        split="val",
        train_split=data_cfg.get("train_split", 0.9),
    )

    print(f"📊 Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"🏷️  Label vocabulary size: {len(all_labels)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = BiEncoderModel(
        model_name=model_cfg["name"],
        max_num_labels=model_cfg["max_num_labels"],
    ).to(device)

    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    num_steps = train_cfg["num_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 100)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    # ── Checkpoints ─────────────────────────────────────────────────────────
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training ────────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    running_loss = 0.0
    best_val_f1 = 0.0
    train_iter = iter(train_loader)

    log_every = train_cfg.get("log_every", 10)
    eval_every = train_cfg.get("eval_every", 100)
    save_every = train_cfg.get("save_every", 200)
    grad_clip = train_cfg.get("gradient_clip_norm", 1.0)

    print(f"\n🏋️  Starting training for {num_steps} steps …\n")
    start_time = time.time()
    pbar = tqdm(total=num_steps, desc="Training", unit="step")

    while global_step < num_steps:
        # Cycle through the loader infinitely
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        texts = batch["texts"]
        labels = batch["labels"]
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        loss, scores, mask = model(texts, labels, targets=targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        global_step += 1
        running_loss += loss.item()

        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # ── Logging ─────────────────────────────────────────────────────────
        if global_step % log_every == 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed
            print(
                f"  Step {global_step:5d}/{num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{steps_per_sec:.1f} steps/s"
            )
            running_loss = 0.0

        # ── Validation ──────────────────────────────────────────────────────
        if global_step % eval_every == 0:
            model.eval()
            val_losses, all_metrics = [], []

            with torch.no_grad():
                for val_batch in val_loader:
                    v_texts = val_batch["texts"]
                    v_labels = val_batch["labels"]
                    v_targets = val_batch["targets"].to(device)

                    v_loss, v_scores, v_mask = model(v_texts, v_labels, targets=v_targets)
                    val_losses.append(v_loss.item())
                    all_metrics.append(compute_metrics(v_scores, v_targets, v_mask))

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
            avg_prec = sum(m["precision"] for m in all_metrics) / len(all_metrics)
            avg_rec = sum(m["recall"] for m in all_metrics) / len(all_metrics)

            print(
                f"\n  📋 Validation @ step {global_step}: "
                f"loss={avg_val_loss:.4f} | "
                f"P={avg_prec:.3f} R={avg_rec:.3f} F1={avg_f1:.3f}"
            )

            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                best_path = os.path.join(ckpt_dir, "best_model")
                model.save_pretrained(best_path)
                print(f"  💾 New best model saved (F1={best_val_f1:.3f})\n")

            model.train()

        # ── Periodic Checkpoint ─────────────────────────────────────────────
        if global_step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}")
            model.save_pretrained(ckpt_path)

    pbar.close()

    # ── Final Save ─────────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final_model")
    model.save_pretrained(final_path)
    print(f"\n✅ Training complete! Final model saved to {final_path}")
    print(f"   Best validation F1: {best_val_f1:.4f}")

    # ── Optionally push to Hub ──────────────────────────────────────────────
    if push_to_hub:
        hf_cfg = config.get("huggingface", {})
        repo_id = hf_cfg.get("model_id", "your-username/bi-encoder-zero-shot")
        private = hf_cfg.get("private", True)
        print(f"\n📤 Pushing best model to Hugging Face Hub: {repo_id} …")
        best_model = BiEncoderModel.from_pretrained(os.path.join(ckpt_dir, "best_model"))
        best_model.push_to_hub(repo_id=repo_id, private=private, token=hf_token)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiEncoder for zero-shot classification")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the best model to Hugging Face Hub after training",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (uses cached token if not provided)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, push_to_hub=args.push_to_hub, hf_token=args.hf_token)
