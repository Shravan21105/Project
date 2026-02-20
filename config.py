"""
Configuration for the Lung Disease Prediction CNN model.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    """Central configuration for training and evaluation."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path: str = ""
    image_dir: str = ""
    train_val_list: str = ""
    test_list: str = ""
    checkpoint_dir: str = ""

    # ── Image ──────────────────────────────────────────────────────────
    image_size: int = 224
    in_channels: int = 1  # grayscale chest X-rays

    # ── Model ──────────────────────────────────────────────────────────
    num_blocks: int = 6
    block_channels: Tuple[int, ...] = (64, 128, 256, 256, 512, 512)
    num_heads: int = 8  # multi-head attention heads
    dropout: float = 0.1

    # ── Training ───────────────────────────────────────────────────────
    batch_size: int = 128             # effective batch size (for LR scaling)
    micro_batch_size: int = 4         # actual GPU batch size per forward pass
    accumulation_steps: int = 32      # batch_size // micro_batch_size
    num_workers: int = 4
    epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    use_amp: bool = True              # automatic mixed precision
    gradient_checkpointing: bool = True  # trade compute for VRAM
    val_split: float = 0.15           # fraction of train_val used for validation

    # ── Disease labels (14 pathologies) ────────────────────────────────
    disease_labels: List[str] = field(default_factory=lambda: [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ])

    num_classes: int = 14

    # ── Seed ───────────────────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self):
        """Resolve paths relative to project root."""
        self.csv_path = self.csv_path or os.path.join(self.project_root, "Data_Entry_2017.csv")
        self.image_dir = self.image_dir or os.path.join(self.project_root, "data", "raw_images")
        self.train_val_list = self.train_val_list or os.path.join(self.project_root, "train_val_list.txt")
        self.test_list = self.test_list or os.path.join(self.project_root, "test_list.txt")
        self.checkpoint_dir = self.checkpoint_dir or os.path.join(self.project_root, "models", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
