"""
UCF-101 Action Recognition — Tensor Memory vs Baseline
=======================================================
Frames are processed sequentially; the tensor memory persists across frames,
accumulating temporal context exactly as it accumulates spatial context in images.

Dataset layout:
  data_dir/
  ├── UCF-101/                      # AVI videos organised by class
  │   ├── ApplyEyeMakeup/
  │   │   ├── v_ApplyEyeMakeup_g01_c01.avi
  │   │   └── ...
  │   └── ...
  └── ucfTrainTestlist/             # official split files
      ├── classInd.txt
      ├── trainlist01.txt  (also 02, 03)
      └── testlist01.txt   (also 02, 03)

Download:
  wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
  wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

Usage:
  python tensor_memory/video/train_video.py --data_dir /path/to/ucf101 --model baseline
  python tensor_memory/video/train_video.py --data_dir /path/to/ucf101 --model tm
  accelerate launch tensor_memory/video/train_video.py --data_dir /path/to/ucf101 --model tm
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TensorMemoryInterface, ViTBlockWithMemory, DropPath


# ---------------------------------------------------------------------------
# Pretrained frame encoder (shared across all model variants)
# ---------------------------------------------------------------------------

class ViTFrameEncoder(nn.Module):
    """
    Wraps a timm pretrained ViT as a frozen per-frame feature extractor.
    Returns [B, D] — the pooled CLS representation for each frame.
    """
    def __init__(self, model_name="vit_small_patch16_224", pretrained=True, freeze=True):
        super().__init__()
        import timm
        self.backbone  = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features
        self._frozen   = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._frozen:
            self.backbone.eval()   # keep backbone in eval regardless of parent mode
        return self

    def forward(self, x):
        """x: [B, 3, H, W]  →  [B, D]"""
        if self._frozen:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UCF101(Dataset):
    """
    UCF-101 dataset.

    Reads official trainlist/testlist split files and loads frames from
    AVI videos using decord (fast) with a cv2 fallback.
    """

    def __init__(self, data_dir, split="train", split_idx=1,
                 num_frames=8, transform=None, frames_dir=None):
        self.data_dir   = Path(data_dir)
        self.num_frames = num_frames
        self.transform  = transform
        # Pre-extracted JPEG frames dir (fast path); falls back to video decoding
        self.frames_dir = Path(frames_dir) if frames_dir else None

        self.video_root = self._find_dir(
            ["UCF-101", "UCF101", "videos"], must_exist=True, label="video"
        )
        split_root = self._find_dir(
            ["ucfTrainTestlist", "UCFTrainTestlist", "splits"],
            must_exist=False, label="split",
            check_file="classInd.txt",
        )

        self.class_to_idx = self._load_class_index(split_root)
        self.samples = self._load_split(split_root, split, split_idx)

        src = "official splits" if split_root else "auto-generated 80/20 split"
        print(f"UCF-101 [{split}/split{split_idx}] ({src}): "
              f"{len(self.samples)} clips, {len(self.class_to_idx)} classes")

    # ---- helpers -----------------------------------------------------------

    def _find_dir(self, candidates, must_exist, label, check_file=None):
        for name in candidates:
            d = self.data_dir / name
            if d.is_dir():
                if check_file is None or (d / check_file).exists():
                    return d
        # also try data_dir itself
        if check_file is None or (self.data_dir / check_file).exists():
            return self.data_dir
        if must_exist:
            raise FileNotFoundError(
                f"Could not find UCF-101 {label} directory under {self.data_dir}. "
                f"Tried: {candidates}"
            )
        return None

    def _load_class_index(self, split_root):
        if split_root and (split_root / "classInd.txt").exists():
            mapping = {}
            with open(split_root / "classInd.txt") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        idx, name = line.split()
                        mapping[name] = int(idx) - 1   # 0-indexed
            return mapping
        # Fall back: derive class list from subdirectory names
        classes = sorted(d.name for d in self.video_root.iterdir() if d.is_dir())
        return {name: i for i, name in enumerate(classes)}

    def _load_split(self, split_root, split, split_idx):
        if split_root:
            fname = ("trainlist" if split == "train" else "testlist") + f"{split_idx:02d}.txt"
            fpath = split_root / fname
            if fpath.exists():
                samples = []
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rel_path = line.split()[0]      # "ClassName/v_....avi"
                        class_name = rel_path.split("/")[0]
                        label = self.class_to_idx.get(class_name, 0)
                        samples.append((self.video_root / rel_path, label))
                return samples

        # No official splits — generate deterministic 80/20 per-class split
        return self._auto_split(split, split_idx)

    def _auto_split(self, split, split_idx, train_frac=0.8):
        import random
        rng = random.Random(42 + split_idx)
        samples = []
        for class_name, label in sorted(self.class_to_idx.items()):
            class_dir = self.video_root / class_name
            if not class_dir.is_dir():
                continue
            videos = sorted(class_dir.glob("*.avi"))
            rng.shuffle(videos)
            n_train = max(1, int(train_frac * len(videos)))
            chosen = videos[:n_train] if split == "train" else videos[n_train:]
            samples.extend((v, label) for v in chosen)
        return samples

    # ---- frame loading -----------------------------------------------------

    def _load_pyav(self, path):
        """Primary loader — PyAV via torchvision.io (suppresses deprecation warning)."""
        try:
            import warnings, torchvision.io as tvio
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vframes, _, _ = tvio.read_video(str(path), pts_unit="sec", output_format="TCHW")
            if vframes.shape[0] == 0:
                return None
            total = vframes.shape[0]
            idxs  = np.linspace(0, total - 1, self.num_frames, dtype=int)
            frames = vframes[idxs].float() / 255.0
            if self.transform:
                frames = torch.stack([
                    self.transform(transforms.ToPILImage()(frames[i]))
                    for i in range(self.num_frames)
                ])
            return frames
        except Exception:
            return None

    def _load_decord(self, path):
        try:
            import decord
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(str(path), num_threads=1)
            total = len(vr)
            if total == 0:
                return None
            idxs = np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()
            frames = vr.get_batch(idxs).permute(0, 3, 1, 2).float() / 255.0
            if self.transform:
                frames = torch.stack([
                    self.transform(transforms.ToPILImage()(frames[i]))
                    for i in range(len(idxs))
                ])
            return frames
        except Exception:
            return None

    def _load_cv2(self, path):
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None
            idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    break
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(self.transform(img) if self.transform else transforms.ToTensor()(img))
            cap.release()
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
            return torch.stack(frames[:self.num_frames])
        except Exception:
            return None

    def __len__(self):
        return len(self.samples)

    def _load_jpegs(self, video_path):
        """Load pre-extracted JPEG frames — fast path."""
        class_name = video_path.parent.name
        stem       = video_path.stem
        frame_dir  = self.frames_dir / class_name / stem
        jpegs      = sorted(frame_dir.glob("frame_*.jpg"))
        if len(jpegs) == 0:
            return None
        total = len(jpegs)
        idxs  = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i in idxs:
            img = Image.open(jpegs[i]).convert("RGB")
            frames.append(self.transform(img) if self.transform else transforms.ToTensor()(img))
        return torch.stack(frames)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.frames_dir is not None:
            frames = self._load_jpegs(path)
            if frames is not None:
                return frames, label
        for loader in (self._load_pyav, self._load_decord, self._load_cv2):
            frames = loader(path)
            if frames is not None:
                return frames, label
        raise RuntimeError(
            f"All decoders failed for: {path}\n"
            "Make sure PyAV is installed:  pip install av"
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaselineVideoViT(nn.Module):
    """
    Space-time ViT baseline. All frame tokens concatenated → joint attention.
    No persistent memory — direct comparison point for TensorMemoryVideoViT.
    """
    def __init__(self, num_classes=101, img_size=224, patch_size=16,
                 dim=384, depth=12, heads=6, num_frames=8,
                 dropout=0.1, drop_path=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.num_frames = num_frames
        self.dim = dim

        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            # ViTBlockWithMemory with use_memory=False = standard ViT block
            ViTBlockWithMemory(dim, heads, drop=dropout, attn_drop=dropout,
                               drop_path=dpr[i], use_memory=False)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.patch_embed(x.view(B * T, C, H, W))        # [B*T, D, h, w]
        x = x.flatten(2).transpose(1, 2)                     # [B*T, N, D]
        x = x.view(B, T, self.num_patches, self.dim)
        x = x + self.pos_embed.unsqueeze(1)
        x = x + self.time_embed.unsqueeze(2)
        x = x.reshape(B, T * self.num_patches, self.dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.blocks:
            x, _ = blk(x, memory_state=None, shared_memory=None)

        return self.head(self.norm(x[:, 0]))

    def get_gate_values(self):
        return {}


class TensorMemoryVideoViT(nn.Module):
    """
    Tensor Memory ViT for video (v1 architecture).

    Frames are processed one at a time. The shared tensor memory
    (h, c) persists across frames — it accumulates temporal evidence the same
    way it accumulates spatial context for images. All transformer layers
    share one TensorMemoryInterface instance (parameter-fair vs baseline).
    """
    def __init__(self, num_classes=101, img_size=224, patch_size=16,
                 dim=384, depth=12, heads=6, num_frames=8,
                 dropout=0.1, drop_path=0.1,
                 mem_channels=64, mem_shape=(8, 8, 8),
                 chunk_size=8, sigma_scale=1.0,
                 memory_every_n_layers=1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.num_frames = num_frames
        self.dim = dim
        self.mem_channels = mem_channels
        self.mem_shape = tuple(mem_shape)
        self.chunk_size = chunk_size
        self.memory_every_n_layers = memory_every_n_layers

        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_drop   = nn.Dropout(dropout)

        # ONE shared memory interface across all layers (same weights, same state)
        self.shared_memory = TensorMemoryInterface(
            embed_dim=dim,
            memory_channels=mem_channels,
            memory_shape=self.mem_shape,
            chunk_size=chunk_size,
            sigma_scale=sigma_scale,
        )

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList()
        self.layer_uses_memory = []
        for i in range(depth):
            uses_mem = (i % memory_every_n_layers == 0)
            self.blocks.append(
                ViTBlockWithMemory(dim, heads, drop=dropout, attn_drop=dropout,
                                   drop_path=dpr[i], use_memory=uses_mem)
            )
            self.layer_uses_memory.append(uses_mem)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_memory(self, B, device, dtype):
        D, H, W = self.mem_shape
        C = self.mem_channels
        h = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)
        c = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)
        return h, c

    def forward(self, x):
        """x: [B, T, C, H, W]"""
        B, T, C, H, W = x.shape
        mem = self._init_memory(B, x.device, x.dtype)

        last_cls = None
        for t in range(T):
            # embed frame t
            tokens = self.patch_embed(x[:, t]).flatten(2).transpose(1, 2)  # [B, N, D]
            tokens = tokens + self.pos_embed
            cls = self.cls_token.expand(B, -1, -1)
            tokens = self.pos_drop(torch.cat([cls, tokens], dim=1))        # [B, N+1, D]

            # pass through all layers; memory persists across frames
            for i, blk in enumerate(self.blocks):
                if self.layer_uses_memory[i]:
                    tokens, mem = blk(tokens, memory_state=mem,
                                      shared_memory=self.shared_memory,
                                      chunk_size=self.chunk_size)
                else:
                    tokens, _ = blk(tokens, memory_state=None,
                                    shared_memory=None)

            last_cls = self.norm(tokens)[:, 0]   # [B, D]

        return self.head(last_cls)

    def get_gate_values(self):
        return {
            f"layer_{i}": torch.sigmoid(blk.memory_gate).item()
            for i, blk in enumerate(self.blocks)
            if self.layer_uses_memory[i]
        }


class RegisterVideoViT(nn.Module):
    """
    Sequential frame ViT with R flat register tokens that persist across frames.

    Direct ablation of TensorMemoryVideoViT: same sequential structure and ~same
    parameter count, but memory is a flat R×D token bank (no structured grid).
    Tests whether structured tensor memory adds value beyond plain recurrent tokens.
    """
    def __init__(self, num_classes=101, img_size=224, patch_size=16,
                 dim=384, depth=12, heads=6, num_frames=8,
                 dropout=0.1, drop_path=0.1, n_registers=8):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.num_frames  = num_frames
        self.dim         = dim
        self.n_registers = n_registers

        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.registers   = nn.Parameter(torch.zeros(1, n_registers, dim))

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            ViTBlockWithMemory(dim, heads, drop=dropout, attn_drop=dropout,
                               drop_path=dpr[i], use_memory=False)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.registers,  std=0.02)

    def forward(self, x):
        """x: [B, T, C, H, W]"""
        B, T, C, H, W = x.shape
        regs = self.registers.expand(B, -1, -1)   # [B, R, D] — carry across frames

        last_cls = None
        for t in range(T):
            tokens = self.patch_embed(x[:, t]).flatten(2).transpose(1, 2)  # [B, N, D]
            tokens = tokens + self.pos_embed
            cls    = self.cls_token.expand(B, -1, -1)
            seq    = torch.cat([cls, regs, tokens], dim=1)                 # [B, 1+R+N, D]

            for blk in self.blocks:
                seq, _ = blk(seq, memory_state=None, shared_memory=None)

            regs     = seq[:, 1:1 + self.n_registers]   # updated registers → next frame
            last_cls = seq[:, 0]

        return self.head(self.norm(last_cls))

    def get_gate_values(self):
        return {}


# ---------------------------------------------------------------------------
# Pretrained model variants (shared frozen ViT backbone + temporal head)
# ---------------------------------------------------------------------------

class PretrainedBaselineVideoViT(nn.Module):
    """
    Frozen ViT-S/16 CLS per frame → small temporal transformer → head.
    No persistent state across frames — pure joint attention on all T tokens.
    """
    def __init__(self, num_classes=101, num_frames=8,
                 timm_model="vit_small_patch16_224", freeze_backbone=True,
                 depth=2, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.encoder    = ViTFrameEncoder(timm_model, pretrained=True, freeze=freeze_backbone)
        dim             = self.encoder.embed_dim
        self.num_frames = num_frames
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            ViTBlockWithMemory(dim, dim // 64, drop=dropout, attn_drop=dropout,
                               drop_path=dpr[i], use_memory=False)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.view(B * T, C, H, W)).view(B, T, -1)  # [B, T, D]
        feats = feats + self.time_embed
        cls   = self.cls_token.expand(B, -1, -1)
        seq   = torch.cat([cls, feats], dim=1)                         # [B, T+1, D]
        for blk in self.blocks:
            seq, _ = blk(seq, memory_state=None, shared_memory=None)
        return self.head(self.norm(seq[:, 0]))

    def get_gate_values(self):
        return {}


class PretrainedRegisterVideoViT(nn.Module):
    """
    Frozen ViT-S/16 CLS per frame + persistent register tokens across frames.
    """
    def __init__(self, num_classes=101, num_frames=8,
                 timm_model="vit_small_patch16_224", freeze_backbone=True,
                 depth=2, dropout=0.1, drop_path=0.1, n_registers=8):
        super().__init__()
        self.encoder     = ViTFrameEncoder(timm_model, pretrained=True, freeze=freeze_backbone)
        dim              = self.encoder.embed_dim
        self.num_frames  = num_frames
        self.n_registers = n_registers
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.registers   = nn.Parameter(torch.zeros(1, n_registers, dim))

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            ViTBlockWithMemory(dim, dim // 64, drop=dropout, attn_drop=dropout,
                               drop_path=dpr[i], use_memory=False)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.registers, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        regs     = self.registers.expand(B, -1, -1)   # [B, R, D]
        last_cls = None
        for t in range(T):
            feat = self.encoder(x[:, t]).unsqueeze(1)  # [B, 1, D]
            cls  = self.cls_token.expand(B, -1, -1)
            seq  = torch.cat([cls, regs, feat], dim=1) # [B, 1+R+1, D]
            for blk in self.blocks:
                seq, _ = blk(seq, memory_state=None, shared_memory=None)
            regs     = seq[:, 1:1 + self.n_registers]
            last_cls = seq[:, 0]
        return self.head(self.norm(last_cls))

    def get_gate_values(self):
        return {}


class PretrainedTensorMemoryVideoViT(nn.Module):
    """
    Frozen ViT-S/16 CLS per frame + tensor memory across frames.
    """
    def __init__(self, num_classes=101, num_frames=8,
                 timm_model="vit_small_patch16_224", freeze_backbone=True,
                 depth=2, dropout=0.1, drop_path=0.1,
                 mem_channels=32, mem_shape=(4, 4, 4), chunk_size=1,
                 sigma_scale=1.0, memory_every_n_layers=1):
        super().__init__()
        self.encoder    = ViTFrameEncoder(timm_model, pretrained=True, freeze=freeze_backbone)
        dim             = self.encoder.embed_dim
        self.num_frames = num_frames
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))
        self.chunk_size = chunk_size

        self.shared_memory = TensorMemoryInterface(
            embed_dim=dim, memory_channels=mem_channels,
            memory_shape=mem_shape, chunk_size=chunk_size, sigma_scale=sigma_scale,
        )

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList()
        self.layer_uses_memory = []
        for i in range(depth):
            uses_mem = (i % memory_every_n_layers == 0)
            self.blocks.append(
                ViTBlockWithMemory(dim, dim // 64, drop=dropout, attn_drop=dropout,
                                   drop_path=dpr[i], use_memory=uses_mem)
            )
            self.layer_uses_memory.append(uses_mem)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        C, (D, H, W) = mem_channels, mem_shape
        self._mem_init_shape = (C, D, H, W)

    def _init_memory(self, B, device, dtype):
        C, D, H, W = self._mem_init_shape
        return (torch.zeros(B, C, D, H, W, device=device, dtype=dtype),
                torch.zeros(B, C, D, H, W, device=device, dtype=dtype))

    def forward(self, x):
        B, T, C, H, W = x.shape
        mem      = self._init_memory(B, x.device, x.dtype)
        last_cls = None
        for t in range(T):
            feat   = self.encoder(x[:, t]).unsqueeze(1)  # [B, 1, D]
            cls    = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, feat], dim=1)        # [B, 2, D]
            for i, blk in enumerate(self.blocks):
                if self.layer_uses_memory[i]:
                    tokens, mem = blk(tokens, memory_state=mem,
                                      shared_memory=self.shared_memory,
                                      chunk_size=self.chunk_size)
                else:
                    tokens, _ = blk(tokens, memory_state=None, shared_memory=None)
            last_cls = self.norm(tokens)[:, 0]
        return self.head(last_cls)

    def get_gate_values(self):
        return {
            f"layer_{i}": torch.sigmoid(blk.memory_gate).item()
            for i, blk in enumerate(self.blocks)
            if self.layer_uses_memory[i]
        }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, accelerator):
    model.eval()
    total_loss = total_top1 = total_top5 = total_n = 0

    for frames, labels in loader:
        logits = model(frames)
        loss   = criterion(logits, labels)

        logits, labels, loss = accelerator.gather_for_metrics(
            (logits, labels, loss.unsqueeze(0))
        )
        total_loss += loss.mean().item()
        total_top1 += (logits.argmax(-1) == labels).sum().item()
        total_top5 += (logits.topk(5, -1).indices == labels.unsqueeze(1)).any(1).sum().item()
        total_n    += labels.size(0)

    model.train()
    return {
        "val/loss": total_loss / max(len(loader), 1),
        "val/top1": 100.0 * total_top1 / max(total_n, 1),
        "val/top5": 100.0 * total_top5 / max(total_n, 1),
    }


def save_checkpoint(accelerator, model, optimizer, scheduler,
                    epoch, best_top1, args, is_best=False):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt = {
        "epoch":     epoch,
        "model":     accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_top1": best_top1,
        "args":      vars(args),
    }
    torch.save(ckpt, os.path.join(args.checkpoint_dir, f"{args.model}_latest.pt"))
    if is_best:
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f"{args.model}_best.pt"))
        print(f"  Saved best (top-1={best_top1:.2f}%)")
    if (epoch + 1) % args.save_every == 0:
        torch.save(ckpt, os.path.join(
            args.checkpoint_dir, f"{args.model}_ep{epoch:03d}.pt"))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

ALL_MODELS = ["baseline", "registers", "tm"]

def parse_args():
    p = argparse.ArgumentParser("UCF-101 Tensor Memory")

    p.add_argument("--data_dir",  required=True)
    p.add_argument("--split_idx", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--num_frames",type=int, default=8)

    p.add_argument("--model",    default="tm", choices=ALL_MODELS)
    p.add_argument("--run_all",  action="store_true",
                   help="Train all models in --models and print comparison table")
    p.add_argument("--models",   default=",".join(ALL_MODELS),
                   help="Comma-separated model list for --run_all")

    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--patch_size", type=int,   default=16)
    p.add_argument("--d_model",    type=int,   default=384)
    p.add_argument("--n_layer",    type=int,   default=12)
    p.add_argument("--n_head",     type=int,   default=6)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--drop_path",  type=float, default=0.2)

    p.add_argument("--mem_channels",         type=int,   default=64)
    p.add_argument("--mem_shape",            type=int, nargs=3, default=[8, 8, 8])
    p.add_argument("--chunk_size",           type=int,   default=8)
    p.add_argument("--sigma_scale",          type=float, default=1.0)
    p.add_argument("--memory_every_n_layers",type=int,   default=1)
    p.add_argument("--n_registers",          type=int,   default=8,
                   help="Register tokens for the registers baseline")

    # Pretrained backbone options
    p.add_argument("--pretrained",       action="store_true",
                   help="Use timm pretrained ViT backbone instead of training from scratch")
    p.add_argument("--timm_model",       default="vit_small_patch16_224",
                   help="timm model name for pretrained backbone")
    p.add_argument("--freeze_backbone",  action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze pretrained backbone weights (only train temporal head)")
    p.add_argument("--temporal_depth",   type=int, default=4,
                   help="Number of transformer layers in temporal head (pretrained mode)")

    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=0.05)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--clip_grad",       type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    p.add_argument("--frames_dir",   default=None,
                   help="Path to pre-extracted JPEG frames (fast loader). "
                        "Generate with: python extract_frames.py")
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--checkpoint_dir", default="checkpoints_ucf101")
    p.add_argument("--results_dir",    default="results")
    p.add_argument("--resume",         default=None)
    p.add_argument("--save_every",     type=int, default=5)
    p.add_argument("--seed",           type=int, default=42)

    p.add_argument("--wandb",    action="store_true")
    p.add_argument("--project",  default="ucf101-tensor-memory")
    p.add_argument("--run_name", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_video_model(args):
    if getattr(args, "pretrained", False):
        pt_common = dict(
            num_classes=101,
            num_frames=args.num_frames,
            timm_model=args.timm_model,
            freeze_backbone=args.freeze_backbone,
            depth=args.temporal_depth,
            dropout=args.dropout,
            drop_path=args.drop_path,
        )
        if args.model == "baseline":
            return PretrainedBaselineVideoViT(**pt_common)
        elif args.model == "registers":
            return PretrainedRegisterVideoViT(**pt_common, n_registers=args.n_registers)
        elif args.model == "tm":
            return PretrainedTensorMemoryVideoViT(
                **pt_common,
                mem_channels=args.mem_channels,
                mem_shape=tuple(args.mem_shape),
                chunk_size=args.chunk_size,
                sigma_scale=args.sigma_scale,
                memory_every_n_layers=args.memory_every_n_layers,
            )

    common = dict(
        num_classes=101,
        img_size=args.img_size, patch_size=args.patch_size,
        dim=args.d_model, depth=args.n_layer, heads=args.n_head,
        num_frames=args.num_frames,
        dropout=args.dropout, drop_path=args.drop_path,
    )
    if args.model == "baseline":
        return BaselineVideoViT(**common)
    elif args.model == "registers":
        return RegisterVideoViT(**common, n_registers=args.n_registers)
    elif args.model == "tm":
        return TensorMemoryVideoViT(
            **common,
            mem_channels=args.mem_channels,
            mem_shape=tuple(args.mem_shape),
            chunk_size=args.chunk_size,
            sigma_scale=args.sigma_scale,
            memory_every_n_layers=args.memory_every_n_layers,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from {ALL_MODELS}")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def _diagnose(args, accelerator, train_loader):
    """Print one-time diagnostics on data and encoder output."""
    if not accelerator.is_main_process:
        return
    frames, labels = next(iter(train_loader))
    frames = frames.float()
    print(f"\n[diag] frames  shape={tuple(frames.shape)}  "
          f"min={frames.min():.3f}  max={frames.max():.3f}  "
          f"nonzero={frames.count_nonzero().item()}/{frames.numel()}")
    print(f"[diag] labels  unique={labels.unique().numel()}/101  "
          f"sample={labels[:8].tolist()}")

    if getattr(args, "pretrained", False):
        import timm
        backbone = timm.create_model(args.timm_model, pretrained=False, num_classes=0)
        backbone.eval()
        with torch.no_grad():
            feat = backbone(frames[0, :1].cpu().float())
        print(f"[diag] encoder output  shape={tuple(feat.shape)}  "
              f"mean={feat.mean():.4f}  std={feat.std():.4f}")
    print()


def run_single(args, accelerator, train_loader, val_loader):
    """Train one model. Returns dict with best_top1, best_top5, params, train_time_s."""
    model     = build_video_model(args)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if accelerator.is_main_process:
        print("=" * 60)
        print(f"  model={args.model.upper()}  params={n_params/1e6:.2f}M")
        if getattr(args, "pretrained", False):
            frozen = "frozen" if args.freeze_backbone else "fine-tuned"
            print(f"  backbone={args.timm_model} ({frozen})  temporal_depth={args.temporal_depth}  T={args.num_frames}")
        else:
            print(f"  d={args.d_model}  L={args.n_layer}  H={args.n_head}  T={args.num_frames}")
        if args.model == "tm":
            print(f"  Memory: shape={args.mem_shape}  C={args.mem_channels}  "
                  f"chunk={args.chunk_size}  every={args.memory_every_n_layers}L")
        elif args.model == "registers":
            print(f"  Registers: R={args.n_registers}")
        print("=" * 60)

    if args.model == "baseline":
        _diagnose(args, accelerator, train_loader)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    spe = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=spe * args.warmup_epochs,
        num_training_steps=spe * args.epochs,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    start_epoch = 0
    best_top1 = best_top5 = 0.0
    if args.resume:
        ckpt_path = args.resume if args.resume != "auto" else \
            os.path.join(args.checkpoint_dir, f"{args.model}_latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_top1   = ckpt.get("best_top1", 0.0)
            if accelerator.is_main_process:
                print(f"  Resumed from epoch {ckpt['epoch']}, best={best_top1:.2f}%")

    t_start = time.time()
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_loss = ep_correct = ep_n = 0

        bar = tqdm(train_loader, desc=f"Ep {epoch:03d} [{args.model}]",
                   disable=not accelerator.is_main_process,
                   dynamic_ncols=True, leave=False)

        for frames, labels in bar:
            logits = model(frames)
            loss   = criterion(logits, labels)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            ep_loss    += loss.item()
            ep_correct += (logits.argmax(-1) == labels).sum().item()
            ep_n       += labels.size(0)

            if accelerator.is_main_process:
                gv = accelerator.unwrap_model(model).get_gate_values()
                gs = f"  gate={sum(gv.values())/len(gv):.3f}" if gv else ""
                bar.set_postfix_str(f"loss={loss.item():.4f}  "
                                    f"top1={100.*ep_correct/max(ep_n,1):.1f}%{gs}")
                if args.wandb:
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/top1": 100. * ep_correct / max(ep_n, 1),
                        "train/lr":   scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }, step=global_step)

        val_metrics = evaluate(model, val_loader, criterion, accelerator)
        accelerator.wait_for_everyone()

        is_best = val_metrics["val/top1"] > best_top1
        if is_best:
            best_top1 = val_metrics["val/top1"]
            best_top5 = val_metrics["val/top5"]

        if accelerator.is_main_process:
            print(f"Ep {epoch:03d} | top1={val_metrics['val/top1']:.2f}%  "
                  f"top5={val_metrics['val/top5']:.2f}%  "
                  f"loss={val_metrics['val/loss']:.4f}  best={best_top1:.2f}%")
            if args.wandb:
                accelerator.log(val_metrics, step=global_step)

        save_checkpoint(accelerator, model, optimizer, scheduler,
                        epoch, best_top1, args, is_best=is_best)

    return {
        "model":          args.model,
        "params":         n_params,
        "best_top1":      best_top1,
        "best_top5":      best_top5,
        "train_time_s":   time.time() - t_start,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb" if (args.wandb and not args.run_all) else None,
    )

    if args.wandb and not args.run_all and accelerator.is_main_process:
        run_name = args.run_name or (f"{args.model}_f{args.num_frames}_d{args.d_model}"
                                     f"_L{args.n_layer}_split{args.split_idx}")
        accelerator.init_trackers(
            project_name=args.project, config=vars(args),
            init_kwargs={"wandb": {"name": run_name}},
        )

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = UCF101(args.data_dir, split="train", split_idx=args.split_idx,
                      num_frames=args.num_frames, transform=train_tf,
                      frames_dir=args.frames_dir)
    val_ds   = UCF101(args.data_dir, split="test",  split_idx=args.split_idx,
                      num_frames=args.num_frames, transform=val_tf,
                      frames_dir=args.frames_dir)

    nw = args.num_workers
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    if accelerator.is_main_process:
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    models_to_run = ([m.strip() for m in args.models.split(",")]
                     if args.run_all else [args.model])
    all_results = {}

    for model_name in models_to_run:
        args.model = model_name
        if accelerator.is_main_process:
            print(f"\n{'-'*60}\n  Running: {model_name.upper()}\n{'-'*60}")
        result = run_single(args, accelerator, train_loader, val_loader)
        all_results[model_name] = result

        if accelerator.is_main_process:
            os.makedirs(args.results_dir, exist_ok=True)
            with open(os.path.join(args.results_dir, f"ucf101_{model_name}.json"), "w") as f:
                json.dump(result, f, indent=2)

    # ── Comparison table ─────────────────────────────────────────────────────
    if accelerator.is_main_process and len(models_to_run) > 1:
        print(f"\n{'='*60}")
        print("  UCF-101 ACTION RECOGNITION — COMPARISON")
        print(f"{'='*60}")
        hdr = f"{'Model':12s}  {'Params':>8s}  {'Top-1':>8s}  {'Top-5':>8s}  {'Time':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for mn, r in all_results.items():
            print(f"{mn:12s}  {r['params']/1e6:>7.2f}M  "
                  f"{r['best_top1']:>7.2f}%  {r['best_top5']:>7.2f}%  "
                  f"{r['train_time_s']/60:>6.1f}m")
        out = os.path.join(args.results_dir, "ucf101_comparison.json")
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results → {out}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
