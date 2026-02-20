"""
6-Block CNN with Self-Attention & Cross-Attention for Lung Disease Prediction.

Architecture Overview
─────────────────────
  Input (1×224×224)
    ↓
  Stem: Conv7×7 s2 → BN → ReLU → MaxPool
    ↓
  Block 1  (64ch)  : Conv→BN→ReLU→Conv→BN→ReLU → SelfAttn → CrossAttn(stem)
  Block 2  (128ch) : Conv→BN→ReLU→Conv→BN→ReLU → SelfAttn → CrossAttn(block1)
  Block 3  (256ch) : Conv→BN→ReLU→Conv→BN→ReLU → SelfAttn → CrossAttn(block2)
  Block 4  (256ch) : Conv→BN→ReLU→Conv→BN→ReLU → SelfAttn → CrossAttn(block3)
  Block 5  (512ch) : Conv→BN→ReLU→Conv→BN→ReLU → SelfAttn → CrossAttn(block4)
  Block 6  (512ch) : Conv→BN→ReLU→Conv→BN→ReLU → SelfAttn → CrossAttn(block5)
    ↓
  Global Average Pool → FC(512 → 14)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ═══════════════════════════════════════════════════════════════════════
#  Self-Attention
# ═══════════════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """
    Multi-head self-attention applied on spatial feature maps.

    Input shape : (B, C, H, W)
    Output shape: (B, C, H, W)

    Internally the spatial dims are flattened to a sequence of length H*W,
    attention is computed, then reshaped back.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W  # sequence length

        # Flatten spatial → (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Layer norm
        x_norm = self.norm(x_flat)

        # QKV
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual + reshape back to spatial
        out = (x_flat + out).transpose(1, 2).reshape(B, C, H, W)
        return out


# ═══════════════════════════════════════════════════════════════════════
#  Cross-Attention
# ═══════════════════════════════════════════════════════════════════════

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention between two spatial feature maps.

    Query comes from the *current* block's features.
    Key & Value come from the *previous* block's features.

    Both inputs are (B, C, H, W) but may differ in spatial size;
    we adaptively pool the context (prev) to match the query spatial dims
    and project channels if they differ.

    Output shape: same as query input (B, C_q, H_q, W_q).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, \
            f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"

        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_ctx = nn.LayerNorm(context_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query   : (B, C_q, H_q, W_q)  – current block features
            context : (B, C_ctx, H_ctx, W_ctx) – previous block features
        Returns:
            (B, C_q, H_q, W_q)
        """
        B, C_q, H_q, W_q = query.shape

        # Adaptively pool context to match query spatial dims
        context = F.adaptive_avg_pool2d(context, (H_q, W_q))

        N = H_q * W_q

        # Flatten
        q_flat = query.flatten(2).transpose(1, 2)    # (B, N, C_q)
        c_flat = context.flatten(2).transpose(1, 2)   # (B, N, C_ctx)

        q_flat = self.norm_q(q_flat)
        c_flat = self.norm_ctx(c_flat)

        # Project
        q = self.q_proj(q_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(c_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(c_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C_q)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # Residual + reshape
        out = (query.flatten(2).transpose(1, 2) + out).transpose(1, 2).reshape(B, C_q, H_q, W_q)
        return out


# ═══════════════════════════════════════════════════════════════════════
#  CNN + Attention Block
# ═══════════════════════════════════════════════════════════════════════

class CNNAttentionBlock(nn.Module):
    """
    One block of the architecture:
        Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU
        → Self-Attention → Cross-Attention(prev_features)
        + skip connection around the CNN part
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        downsample: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        stride = 2 if downsample else 1

        # ── CNN path ───────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip / projection shortcut
        self.skip = nn.Identity()
        if in_channels != out_channels or downsample:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # ── Attention ──────────────────────────────────────────────────
        self.self_attn = SelfAttention(out_channels, num_heads, dropout)
        self.cross_attn = CrossAttention(out_channels, context_channels, num_heads, dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : current input feature map
            context : feature map from the previous block (for cross-attention)
        Returns:
            output feature map (same spatial structure as after CNN downsampling)
        """
        # CNN with residual
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)

        # Attention layers (checkpointed to save VRAM)
        if self.use_checkpoint and self.training:
            out = grad_checkpoint(self.self_attn, out, use_reentrant=False)
            out = grad_checkpoint(self.cross_attn, out, context, use_reentrant=False)
        else:
            out = self.self_attn(out)
            out = self.cross_attn(out, context)

        return out


# ═══════════════════════════════════════════════════════════════════════
#  Full Model: LungDiseaseNet
# ═══════════════════════════════════════════════════════════════════════

class LungDiseaseNet(nn.Module):
    """
    6-Block CNN with Self-Attention & Cross-Attention for multi-label
    lung disease classification on chest X-rays.

    Architecture:
        Stem → Block1 → Block2 → Block3 → Block4 → Block5 → Block6
        → Global Average Pool → FC → sigmoid
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 14,
        block_channels: tuple = (64, 128, 256, 256, 512, 512),
        num_heads: int = 8,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert len(block_channels) == 6, "Must provide exactly 6 block channel sizes"

        # ── Stem ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, block_channels[0], kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(block_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # After stem:  (B, block_channels[0], H/4, W/4)  e.g. (B, 64, 56, 56)

        # ── 6 CNN+Attention Blocks ─────────────────────────────────────
        self.blocks = nn.ModuleList()
        prev_ch = block_channels[0]  # stem output channels = block_channels[0]

        for i, out_ch in enumerate(block_channels):
            # For block 0: context comes from stem (same channels as block_channels[0])
            # For block i>0: context comes from block i-1
            context_ch = block_channels[0] if i == 0 else block_channels[i - 1]

            # Downsample on blocks where channels increase
            downsample = (out_ch != prev_ch) if i > 0 else False

            self.blocks.append(
                CNNAttentionBlock(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    context_channels=context_ch,
                    num_heads=num_heads,
                    dropout=dropout,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                )
            )
            prev_ch = out_ch

        # ── Classifier head ────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(block_channels[-1], num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 224, 224)
        Returns:
            logits: (B, num_classes) — raw logits (apply sigmoid externally for probabilities)
        """
        # Stem
        stem_out = self.stem(x)  # (B, 64, 56, 56)

        # Pass through 6 blocks
        # We store each block's output so the NEXT block can use it as
        # cross-attention context.  Block 0 uses stem_out as context.
        block_out = stem_out
        block_outputs = [stem_out]  # index 0 = stem, index i+1 = block i output

        for i, block in enumerate(self.blocks):
            context = block_outputs[i]  # stem for block 0, block i-1 output for others
            block_out = block(block_out, context)
            block_outputs.append(block_out)

        # Classifier
        pooled = self.global_pool(block_out).flatten(1)  # (B, 512)
        logits = self.classifier(pooled)  # (B, 14)
        return logits


# ═══════════════════════════════════════════════════════════════════════
#  Quick sanity check
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = LungDiseaseNet(in_channels=1, num_classes=14)
    dummy = torch.randn(2, 1, 224, 224)
    out = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")
