
"""
Evaluation utilities for the Lung Disease CNN model.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

from src.config import Config


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> None:
    """
    Run full evaluation on the test set.
    Prints per-class AUROC and classification report.
    """
    model.eval()

    all_labels = []
    all_probs = []

    print("\nRunning evaluation on test set...")

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)

        if cfg.use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
        else:
            logits = model(images)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # ── Per-class AUROC ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'Disease':25s}  {'AUROC':>8s}  {'Prevalence':>10s}")
    print(f"{'='*55}")

    auroc_values = []
    for i, name in enumerate(cfg.disease_labels):
        prevalence = all_labels[:, i].mean()
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = 0.0
        auroc_values.append(auc)
        print(f"  {name:25s}  {auc:8.4f}  {prevalence:10.4f}")

    macro_auroc = float(np.mean(auroc_values))
    print(f"{'-'*55}")
    print(f"  {'Macro AUROC':25s}  {macro_auroc:8.4f}")
    print(f"{'='*55}\n")

    # ── Classification report (threshold-based) ───────────────────────
    threshold = 0.5
    preds = (all_probs >= threshold).astype(int)

    print(f"Classification Report (threshold={threshold}):")
    print(classification_report(
        all_labels, preds,
        target_names=cfg.disease_labels,
        zero_division=0,
    ))

    return macro_auroc
