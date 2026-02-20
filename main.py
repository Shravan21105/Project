"""
Main entry point for training / evaluating the Lung Disease CNN model.

Usage:
    python -m src.main --mode train --epochs 15 --batch_size 32 --lr 1e-4
    python -m src.main --mode test --checkpoint models/checkpoints/best_model.pth
"""

import argparse
import os
import random

import numpy as np
import torch

from src.config import Config
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model
from src.model import LungDiseaseNet
from src.train import train_model


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Lung Disease CNN with Attention")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation or resume")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader num_workers (overrides config)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Build config ───────────────────────────────────────────────────
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.use_amp = False

    set_seed(cfg.seed)

    # ── Device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # ── Model ──────────────────────────────────────────────────────────
    model = LungDiseaseNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        block_channels=cfg.block_channels,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        use_checkpoint=cfg.gradient_checkpointing,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: LungDiseaseNet")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Train or Test ──────────────────────────────────────────────────
    if args.mode == "train":
        train_model(model, cfg, train_loader, val_loader, device)

        # After training, evaluate on test set with best checkpoint
        best_ckpt = os.path.join(cfg.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            print(f"\nLoading best checkpoint for test evaluation: {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            evaluate_model(model, test_loader, cfg, device)

    elif args.mode == "test":
        ckpt_path = args.checkpoint or os.path.join(cfg.checkpoint_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found at {ckpt_path}")
            print("Train the model first with: python -m src.main --mode train")
            return

        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
        print(f"  Checkpoint AUROC: {checkpoint.get('val_auroc', '?')}")

        evaluate_model(model, test_loader, cfg, device)


if __name__ == "__main__":
    main()
