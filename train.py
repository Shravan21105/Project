"""
Training and validation loops for the Lung Disease CNN model.

Uses gradient accumulation to simulate large effective batch sizes
while keeping GPU micro-batches small enough for limited VRAM.
"""

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import Config


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 32,
    scaler: torch.amp.GradScaler = None,
    use_amp: bool = True,
) -> float:
    """
    Train for one epoch with gradient accumulation.

    The DataLoader yields micro-batches (e.g. 4 images).  We accumulate
    gradients over `accumulation_steps` micro-batches before doing one
    optimizer step, giving an effective batch size of
    micro_batch_size × accumulation_steps.
    """
    model.train()
    running_loss = 0.0
    num_opt_steps = 0

    optimizer.zero_grad(set_to_none=True)

    total_micro = len(loader)

    for micro_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward ────────────────────────────────────────────────
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                # Scale loss by accumulation steps so the accumulated
                # gradient is the mean over the full effective batch.
                loss = criterion(logits, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images)
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()

        running_loss += loss.item() * accumulation_steps  # unscaled for logging

        # ── Optimiser step every `accumulation_steps` micro-batches ─
        if (micro_idx + 1) % accumulation_steps == 0 or (micro_idx + 1) == total_micro:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            num_opt_steps += 1

        # ── Logging ────────────────────────────────────────────────
        if (micro_idx + 1) % (accumulation_steps * 10) == 0:
            avg_so_far = running_loss / (micro_idx + 1)
            print(f"    [Micro-batch {micro_idx+1}/{total_micro}] "
                  f"Avg Loss: {avg_so_far:.4f}  "
                  f"(opt steps: {num_opt_steps})")

    avg_loss = running_loss / max(total_micro, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    disease_labels: list,
    use_amp: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Validate on a DataLoader.

    Returns:
        avg_loss       : float
        macro_auroc    : float (macro-average across classes)
        per_class_auroc: dict mapping disease_name → AUROC
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(num_batches, 1)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # Per-class AUROC
    per_class_auroc = {}
    auroc_values = []
    for i, name in enumerate(disease_labels):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = 0.0  # all same label in this split
        per_class_auroc[name] = auc
        auroc_values.append(auc)

    macro_auroc = float(np.mean(auroc_values))
    return avg_loss, macro_auroc, per_class_auroc


def train_model(model: nn.Module, cfg: Config, train_loader, val_loader, device):
    """
    Full training loop with gradient accumulation, validation,
    LR scheduling, and checkpoint saving.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    scaler = torch.amp.GradScaler() if cfg.use_amp and device.type == "cuda" else None

    best_auroc = 0.0

    eff_bs = cfg.micro_batch_size * cfg.accumulation_steps

    print(f"\n{'='*65}")
    print(f"  Training LungDiseaseNet  |  Device: {device}")
    print(f"  Epochs: {cfg.epochs}  |  Effective batch: {eff_bs}  "
          f"(micro={cfg.micro_batch_size} × accum={cfg.accumulation_steps})")
    print(f"  LR: {cfg.learning_rate}  |  AMP: {cfg.use_amp}  "
          f"|  Grad ckpt: {cfg.gradient_checkpointing}")
    print(f"{'='*65}\n")

    for epoch in range(1, cfg.epochs + 1):
        try:
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────────
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                accumulation_steps=cfg.accumulation_steps,
                scaler=scaler, use_amp=cfg.use_amp and device.type == "cuda",
            )

            # ── Validate ───────────────────────────────────────────────────
            val_loss, val_auroc, per_class = validate(
                model, val_loader, criterion, device, cfg.disease_labels,
                use_amp=cfg.use_amp and device.type == "cuda",
            )

            scheduler.step(val_auroc)

            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"Epoch [{epoch}/{cfg.epochs}]  "
                f"Train Loss: {train_loss:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  |  "
                f"Val AUROC: {val_auroc:.4f}  |  "
                f"LR: {lr_now:.2e}  |  "
                f"Time: {elapsed:.1f}s")
        except:
            print('Terminating...\nSaving Best checkpoint')
        finally:
            # ── Checkpoint ─────────────────────────────────────────────────
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": val_auroc,
                    "val_loss": val_loss,
                }, ckpt_path)
                print(f"  ✓ Saved best model (AUROC {best_auroc:.4f}) → {ckpt_path}")

        # ── Per-class AUROC ────────────────────────────────────────────
        print("  Per-class AUROC:")
        for name, auc in per_class.items():
            print(f"    {name:25s}: {auc:.4f}")
        print()

    print(f"Training complete. Best Val AUROC: {best_auroc:.4f}")
    return best_auroc
