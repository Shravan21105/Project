"""
Dataset and DataLoader utilities for NIH ChestX-ray14.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the NIH ChestX-ray14 dataset.

    Each sample returns:
        image  : Tensor[1, H, W]  (grayscale, normalised)
        label  : Tensor[num_classes]  (multi-hot float vector)
    """

    def __init__(
        self,
        image_names: List[str],
        labels_map: Dict[str, np.ndarray],
        image_dir: str,
        transform: transforms.Compose = None,
    ):
        self.image_names = image_names
        self.labels_map = labels_map
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load as grayscale
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels_map[img_name], dtype=torch.float32)
        return image, label


def _build_labels_map(df: pd.DataFrame, disease_labels: List[str]) -> Dict[str, np.ndarray]:
    """
    Build a mapping from image filename → multi-hot numpy vector.
    """
    labels_map: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        img_name = row["Image Index"]
        findings = str(row["Finding Labels"])
        vec = np.zeros(len(disease_labels), dtype=np.float32)
        for i, disease in enumerate(disease_labels):
            if disease in findings:
                vec[i] = 1.0
        labels_map[img_name] = vec
    return labels_map


def get_transforms(cfg: Config, is_train: bool = True) -> transforms.Compose:
    """Return image transforms for training or evaluation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # single-channel norm
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the official split files.

    The official train_val_list is further split 85/15 into train/val.
    """
    # ── Read CSV & build label map ─────────────────────────────────────
    df = pd.read_csv(cfg.csv_path)
    labels_map = _build_labels_map(df, cfg.disease_labels)

    # ── Read split files ───────────────────────────────────────────────
    with open(cfg.train_val_list, "r") as f:
        train_val_names = [line.strip() for line in f if line.strip()]
    with open(cfg.test_list, "r") as f:
        test_names = [line.strip() for line in f if line.strip()]

    # Filter to images that actually exist in raw_images dir
    existing = set(os.listdir(cfg.image_dir))
    train_val_names = [n for n in train_val_names if n in existing]
    test_names = [n for n in test_names if n in existing]

    # ── Split train_val → train / val ──────────────────────────────────
    rng = np.random.RandomState(cfg.seed)
    indices = rng.permutation(len(train_val_names))
    val_size = int(len(train_val_names) * cfg.val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_names = [train_val_names[i] for i in train_indices]
    val_names = [train_val_names[i] for i in val_indices]

    print(f"[Dataset] Train: {len(train_names)} | Val: {len(val_names)} | Test: {len(test_names)}")

    # ── Build datasets ─────────────────────────────────────────────────
    train_ds = ChestXrayDataset(train_names, labels_map, cfg.image_dir, get_transforms(cfg, is_train=True))
    val_ds = ChestXrayDataset(val_names, labels_map, cfg.image_dir, get_transforms(cfg, is_train=False))
    test_ds = ChestXrayDataset(test_names, labels_map, cfg.image_dir, get_transforms(cfg, is_train=False))

    # ── Build dataloaders ──────────────────────────────────────────────
    # Use micro_batch_size for each GPU forward pass;
    # gradient accumulation in the training loop gives the effective batch_size.
    mbs = cfg.micro_batch_size
    train_loader = DataLoader(train_ds, batch_size=mbs, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=mbs, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=mbs, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
