#!/usr/bin/env python3
"""
MAE-style image patch reconstruction on CUB-200-2011.
Compares: base ViT vs Tensor Memory ViT.

Usage:
    accelerate launch tensor_memory/image/train_image.py --model base          --data_dir /data/cub200
    accelerate launch tensor_memory/image/train_image.py --model tm   --data_dir /data/cub200
    python tensor_memory/image/train_image.py --run_all --data_dir /data/cub200
"""

import os, sys, time, argparse, json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TensorMemoryInterface, ViTBlockWithMemory, DropPath

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')  # full TF32 coverage on Blackwell

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


# ── Patch utilities ───────────────────────────────────────────────────────────

def random_masking(x: torch.Tensor, mask_ratio: float):
    """x: [B,N,D] → x_vis [B,N_vis,D], mask [B,N] (1=masked), ids_restore [B,N]"""
    B, N, D = x.shape
    n_keep = max(1, int(N * (1 - mask_ratio)))
    ids_shuffle = torch.rand(B, N, device=x.device).argsort(dim=1)
    ids_restore  = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    x_vis = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :n_keep] = 0
    mask = mask.gather(1, ids_restore)
    return x_vis, mask, ids_restore


def patchify(imgs: torch.Tensor, p: int) -> torch.Tensor:
    B, C, H, W = imgs.shape
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = torch.einsum('bchpwq->bhwpqc', x)
    return x.reshape(B, h * w, p * p * C)


def unpatchify(x: torch.Tensor, p: int, sz: int) -> torch.Tensor:
    h = w = sz // p
    B = x.shape[0]
    x = x.reshape(B, h, w, p, p, 3)
    x = torch.einsum('bhwpqc->bchpwq', x)
    return x.reshape(B, 3, h * p, w * p)


def denorm(t: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(_MEAN, device=t.device).view(1, 3, 1, 1)
    s = torch.tensor(_STD,  device=t.device).view(1, 3, 1, 1)
    return (t * s + m).clamp(0, 1)


# ── Shared building blocks ────────────────────────────────────────────────────

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


class Block(nn.Module):
    def __init__(self, dim, heads, ratio=4.0, drop=0.0, dp=0.0, window=None):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.dp     = DropPath(dp) if dp > 0 else nn.Identity()
        self.norm2  = nn.LayerNorm(dim)
        self.window = window
        h = int(dim * ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        n = self.norm1(x)
        attn_mask = None
        if self.window is not None:
            S = x.shape[1]
            r = torch.arange(S, device=x.device)
            dist = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
            attn_mask = torch.where(dist <= self.window,
                                    torch.zeros(S, S, device=x.device),
                                    torch.full((S, S), float('-inf'), device=x.device))
        y, _ = self.attn(n, n, n, need_weights=False, attn_mask=attn_mask)
        x = x + self.dp(y)
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


# ── Encoders (MAE-style: encode visible patches only) ─────────────────────────

class BaseReconEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=512, depth=6, heads=8,
                 drop=0.0, dp=0.1):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim         = dim

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        dpr = [x.item() for x in torch.linspace(0, dp, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, drop=drop, dp=dpr[i]) for i in range(depth)])
        self.norm   = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, imgs, mask_ratio):
        B = imgs.shape[0]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # [B,N,D]
        x = x + self.pos_embed[:, 1:]                          # add positional embed before masking
        x_vis, mask, ids_restore = random_masking(x, mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        return self.norm(x_vis), mask, ids_restore

    def get_gate_values(self): return {}


class LocalReconEncoder(BaseReconEncoder):
    """BaseReconEncoder with sliding-window attention (ablation: does global range matter?)."""
    def __init__(self, *args, window=16, **kwargs):
        super().__init__(*args, **kwargs)
        depth = len(self.blocks)
        dp    = kwargs.get("dp", 0.1)
        dim   = kwargs.get("dim", args[2] if len(args) > 2 else 512)
        heads = kwargs.get("heads", args[3] if len(args) > 3 else 8)
        drop  = kwargs.get("drop", 0.0)
        dpr   = [x.item() for x in torch.linspace(0, dp, depth)]
        self.blocks = nn.ModuleList(
            [Block(dim, heads, drop=drop, dp=dpr[i], window=window) for i in range(depth)]
        )


class RegisterReconEncoder(BaseReconEncoder):
    """BaseReconEncoder + R learnable register tokens (ablation: structured tensor memory vs flat tokens?)."""
    def __init__(self, *args, n_registers=8, **kwargs):
        super().__init__(*args, **kwargs)
        dim = kwargs.get("dim", args[2] if len(args) > 2 else 512)
        self.registers = nn.Parameter(torch.zeros(1, n_registers, dim))
        nn.init.trunc_normal_(self.registers, std=0.02)

    def forward(self, imgs, mask_ratio):
        B = imgs.shape[0]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:]
        x_vis, mask, ids_restore = random_masking(x, mask_ratio)
        cls  = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        regs = self.registers.expand(B, -1, -1)
        x_vis = torch.cat([cls, regs, x_vis], dim=1)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        n_reg = self.registers.shape[1]
        x_vis = torch.cat([x_vis[:, :1], x_vis[:, 1 + n_reg:]], dim=1)  # strip registers
        return self.norm(x_vis), mask, ids_restore


class TensorMemoryReconEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=512, depth=6, heads=8,
                 drop=0.0, dp=0.1,
                 mem_channels=32, mem_shape=(4, 4, 4), chunk_size=16,
                 sigma_scale=1.0, memory_every_n_layers=1):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim         = dim
        self.mem_channels = mem_channels
        self.mem_shape   = tuple(mem_shape)
        self.chunk_size  = chunk_size

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        self.shared_memory = TensorMemoryInterface(
            embed_dim=dim, memory_channels=mem_channels,
            memory_shape=self.mem_shape, chunk_size=chunk_size, sigma_scale=sigma_scale,
        )

        dpr = [x.item() for x in torch.linspace(0, dp, depth)]
        self.blocks      = nn.ModuleList()
        self.uses_memory = []
        for i in range(depth):
            um = (i % memory_every_n_layers == 0)
            self.blocks.append(
                ViTBlockWithMemory(dim=dim, num_heads=heads, mlp_ratio=4., drop=drop,
                                   attn_drop=drop, drop_path=dpr[i], use_memory=um)
            )
            self.uses_memory.append(um)

        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def _init_mem(self, B, dev, dtype):
        D, H, W = self.mem_shape
        h = torch.zeros(B, self.mem_channels, D, H, W, device=dev, dtype=dtype)
        c = torch.zeros(B, self.mem_channels, D, H, W, device=dev, dtype=dtype)
        return h, c

    def forward(self, imgs, mask_ratio):
        B = imgs.shape[0]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:]
        x_vis, mask, ids_restore = random_masking(x, mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)
        x_vis = torch.cat([cls, x_vis], dim=1)

        mem = self._init_mem(B, x_vis.device, x_vis.dtype)
        for i, blk in enumerate(self.blocks):
            if self.uses_memory[i]:
                x_vis, mem = blk(x_vis, mem, shared_memory=self.shared_memory,
                                 chunk_size=self.chunk_size)
            else:
                x_vis, _ = blk(x_vis, None, shared_memory=None)

        return self.norm(x_vis), mask, ids_restore

    def get_gate_values(self):
        return {f"layer_{i}": torch.sigmoid(blk.memory_gate).item()
                for i, blk in enumerate(self.blocks) if self.uses_memory[i]}


# ── Shared MAE decoder ────────────────────────────────────────────────────────

class ReconDecoder(nn.Module):
    def __init__(self, enc_dim, dec_dim=256, depth=4, heads=4, num_patches=196, patch_size=16):
        super().__init__()
        self.dec_dim     = dec_dim
        self.num_patches = num_patches

        self.embed      = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, dec_dim))
        self.blocks     = nn.ModuleList([Block(dec_dim, heads) for _ in range(depth)])
        self.norm       = nn.LayerNorm(dec_dim)
        self.pred       = nn.Linear(dec_dim, patch_size * patch_size * 3)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, x_enc, ids_restore):
        B = x_enc.shape[0]
        x = self.embed(x_enc)                                          # [B, N_vis+1, dec_dim]
        n_mask = ids_restore.shape[1] - (x.shape[1] - 1)
        mt = self.mask_token.expand(B, n_mask, -1)
        x_ = torch.cat([x[:, 1:], mt], dim=1)                         # [B, N, dec_dim]
        x_ = x_.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_dim))
        x  = torch.cat([x[:, :1], x_], dim=1)                         # [B, N+1, dec_dim]
        x  = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x)[:, 1:])                          # [B, N, P²·3]


# ── Full reconstruction model ─────────────────────────────────────────────────

class PatchReconModel(nn.Module):
    def __init__(self, encoder, decoder, patch_size=16, img_size=224):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.patch_size = patch_size
        self.img_size   = img_size

    def forward(self, imgs, mask_ratio=0.75):
        x_enc, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred   = self.decoder(x_enc, ids_restore)                      # [B,N,P²·3]
        target = patchify(imgs, self.patch_size)

        # per-patch normalization (MAE paper §3.1)
        mean = target.mean(-1, keepdim=True)
        std  = (target.var(-1, keepdim=True) + 1e-6).sqrt()
        target_n = (target - mean) / std

        loss = ((pred - target_n) ** 2).mean(-1)                       # [B,N]
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss, pred, mask, target

    @torch.no_grad()
    def reconstruct(self, imgs, mask_ratio=0.75):
        """Return composite image [B,3,H,W] in ImageNet-norm space and mask [B,N]."""
        x_enc, mask, ids_restore = self.encoder(imgs, mask_ratio)
        pred   = self.decoder(x_enc, ids_restore)
        target = patchify(imgs, self.patch_size)
        mean   = target.mean(-1, keepdim=True)
        std    = (target.var(-1, keepdim=True) + 1e-6).sqrt()
        pred_img = pred * std + mean                                    # denorm to ImageNet-norm space
        mask_e   = mask.unsqueeze(-1).expand_as(target)
        composite = target * (1 - mask_e) + pred_img * mask_e
        return unpatchify(composite, self.patch_size, self.img_size), mask

    def get_gate_values(self):
        return self.encoder.get_gate_values()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """pred/target: [B,3,H,W] in [0,1]"""
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
    return (-10 * torch.log10(mse + 1e-8)).mean().item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, k: int = 7) -> float:
    """Approximate SSIM via avg-pool local statistics. pred/target: [B,3,H,W] in [0,1]."""
    C1, C2 = 0.01**2, 0.03**2
    p = k // 2

    def pool(t): return F.avg_pool2d(t, k, stride=1, padding=p)

    mu1, mu2 = pool(pred), pool(target)
    s1  = pool(pred ** 2)        - mu1 ** 2
    s2  = pool(target ** 2)      - mu2 ** 2
    s12 = pool(pred * target)    - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return (num / (den + 1e-8)).mean().item()


# ── Dataset ───────────────────────────────────────────────────────────────────

class CUB200Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        assert split in ("train", "test")
        self.transform = transform
        cub = root if os.path.basename(os.path.normpath(root)) == "CUB_200_2011" \
              else os.path.join(root, "CUB_200_2011")
        imgs_dir   = os.path.join(cub, "images")
        imgs_txt   = os.path.join(cub, "images.txt")
        labels_txt = os.path.join(cub, "image_class_labels.txt")
        split_txt  = os.path.join(cub, "train_test_split.txt")
        for p in [imgs_dir, imgs_txt, labels_txt, split_txt]:
            if not os.path.exists(p): raise FileNotFoundError(f"Missing: {p}")
        id2path = {int(a): os.path.join(imgs_dir, b)
                   for line in open(imgs_txt) for a, b in [line.split()]}
        id2lbl  = {int(a): int(b) - 1 for line in open(labels_txt) for a, b in [line.split()]}
        id2tr   = {int(a): int(b)     for line in open(split_txt)  for a, b in [line.split()]}
        want = 1 if split == "train" else 0
        self.samples = [(id2path[i], id2lbl[i]) for i in sorted(id2path) if id2tr[i] == want]
        if not self.samples: raise RuntimeError(f"No samples for split={split}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataloaders(args, accelerator):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(int(args.img_size / 0.875)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    try:
        train_ds = CUB200Dataset(args.data_dir, "train", train_tfm)
        val_ds   = CUB200Dataset(args.data_dir, "test",  val_tfm)
        if accelerator.is_main_process:
            print(f"CUB-200: train={len(train_ds)}  val={len(val_ds)}")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[WARN] CUB-200 load failed ({e}), falling back to FakeData")
        train_ds = datasets.FakeData(5000, (3, args.img_size, args.img_size), 200, train_tfm)
        val_ds   = datasets.FakeData(1000, (3, args.img_size, args.img_size), 200, val_tfm)

    nw = args.num_workers
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
              prefetch_factor=4 if nw > 0 else None)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,  drop_last=True,  **kw)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False, drop_last=False, **kw)
    return train_dl, val_dl


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, accelerator, epoch, args):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=f"Ep {epoch:03d} [train]", disable=not accelerator.is_main_process,
               dynamic_ncols=True, leave=False)
    for step, (imgs, _) in enumerate(bar):
        imgs = imgs.to(memory_format=torch.channels_last)
        with accelerator.accumulate(model):
            loss, _, _, _ = model(imgs, args.mask_ratio)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_val = accelerator.gather(loss.detach()).mean().item()
        total += loss_val
        n += 1

        if accelerator.is_main_process:
            gv = accelerator.unwrap_model(model).get_gate_values()
            gs = f"  gate={sum(gv.values())/len(gv):.3f}" if gv else ""
            bar.set_postfix_str(f"loss={loss_val:.4f}{gs}")

    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, accelerator, args):
    """Single pass over val set; evaluates all mask ratios per batch."""
    model.eval()
    um  = accelerator.unwrap_model(model)
    mrs = args.eval_mask_ratios
    acc = {mr: {"mse": 0., "psnr": 0., "ssim": 0., "n": 0} for mr in mrs}

    bar = tqdm(loader, desc="eval", disable=not accelerator.is_main_process,
               dynamic_ncols=True, leave=False)
    for imgs, _ in bar:
        imgs_g = accelerator.gather(imgs)
        for mr in mrs:
            recon, _ = um.reconstruct(imgs, mr)
            recon_g  = accelerator.gather(recon)
            if accelerator.is_main_process:
                imgs_01  = denorm(imgs_g)
                recon_01 = denorm(recon_g)
                acc[mr]["mse"]  += F.mse_loss(recon_01, imgs_01).item()
                acc[mr]["psnr"] += compute_psnr(recon_01, imgs_01)
                acc[mr]["ssim"] += compute_ssim(recon_01, imgs_01)
                acc[mr]["n"]    += 1

    model.train()
    return {
        mr: {k: v / max(1, acc[mr]["n"]) for k, v in acc[mr].items() if k != "n"}
        for mr in mrs
    }


# ── Model factory ─────────────────────────────────────────────────────────────

ALL_MODELS = ["base", "local", "registers", "tm"]

def build_model(args):
    num_patches = (args.img_size // args.patch_size) ** 2
    decoder = ReconDecoder(
        enc_dim=args.d_model, dec_dim=args.dec_dim, depth=args.dec_depth,
        heads=args.dec_heads, num_patches=num_patches, patch_size=args.patch_size,
    )
    enc_kw = dict(
        img_size=args.img_size, patch_size=args.patch_size,
        dim=args.d_model, depth=args.n_layer, heads=args.n_head,
        drop=args.dropout, dp=args.drop_path,
    )
    if args.model == "base":
        enc = BaseReconEncoder(**enc_kw)
    elif args.model == "local":
        enc = LocalReconEncoder(**enc_kw, window=args.local_window)
    elif args.model == "registers":
        enc = RegisterReconEncoder(**enc_kw, n_registers=args.n_registers)
    elif args.model == "tm":
        enc = TensorMemoryReconEncoder(
            **enc_kw,
            mem_channels=args.mem_channels, mem_shape=tuple(args.mem_shape),
            chunk_size=args.chunk_size, sigma_scale=args.sigma_scale,
            memory_every_n_layers=args.memory_every_n_layers,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from {ALL_MODELS}")
    return PatchReconModel(enc, decoder, patch_size=args.patch_size, img_size=args.img_size)


# ── Presets ───────────────────────────────────────────────────────────────────

# small  → ~10M params
# base   → ~45M params
# large  → ~115M params
PRESETS = {
    "small": dict(
        d_model=192, n_layer=4,  n_head=3,
        dec_dim=96,  dec_depth=2, dec_heads=3,
        mem_channels=16, mem_shape=[4, 4, 4], memory_every_n_layers=2,
    ),
    "base": dict(
        d_model=384, n_layer=6,  n_head=6,
        dec_dim=256, dec_depth=4, dec_heads=6,
        mem_channels=32, mem_shape=[6, 6, 6], memory_every_n_layers=2,
    ),
    "large": dict(
        d_model=768, n_layer=12, n_head=12,
        dec_dim=512, dec_depth=6, dec_heads=8,
        mem_channels=64, mem_shape=[8, 8, 8], memory_every_n_layers=1,
    ),
}


# ── Argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    # Pre-scan argv for --preset so we can set arch defaults before full parse
    _preset_name = "small"
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--preset" and i + 1 < len(sys.argv):
            _preset_name = sys.argv[i + 1]
    arch = PRESETS.get(_preset_name, PRESETS["small"])

    p = argparse.ArgumentParser("CUB-200 Patch Reconstruction: base vs tensor memory")
    p.add_argument("--model",    default="tm", choices=ALL_MODELS)
    p.add_argument("--models",   default=",".join(ALL_MODELS),
                   help="Comma-separated model list for --run_all")
    p.add_argument("--run_all",  action="store_true", help="Train all models in --models, print comparison table")
    p.add_argument("--data_dir", default="/data/cub200")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--preset",   default="small", choices=list(PRESETS),
                   help="Architecture preset (small/base/large); individual arch flags override it")

    # Architecture (defaults come from preset; individual flags override)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--d_model",    type=int, default=arch["d_model"])
    p.add_argument("--n_layer",    type=int, default=arch["n_layer"])
    p.add_argument("--n_head",     type=int, default=arch["n_head"])
    # Decoder
    p.add_argument("--dec_dim",   type=int, default=arch["dec_dim"])
    p.add_argument("--dec_depth", type=int, default=arch["dec_depth"])
    p.add_argument("--dec_heads", type=int, default=arch["dec_heads"])

    # Tensor memory
    p.add_argument("--mem_channels",          type=int,   default=arch["mem_channels"])
    p.add_argument("--mem_shape",             type=int,   nargs="+", default=arch["mem_shape"])
    p.add_argument("--chunk_size",            type=int,   default=16)
    p.add_argument("--sigma_scale",           type=float, default=1.0)
    p.add_argument("--memory_every_n_layers", type=int,   default=arch["memory_every_n_layers"])
    p.add_argument("--n_registers",           type=int,   default=8,
                   help="Number of register tokens for the registers baseline")
    p.add_argument("--local_window",          type=int,   default=16,
                   help="Attention window half-width for the local baseline")

    # Training
    p.add_argument("--mask_ratio",       type=float, default=0.75)
    p.add_argument("--eval_mask_ratios", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch_size",       type=int,   default=256)
    p.add_argument("--accum_steps",      type=int,   default=1)
    p.add_argument("--lr",               type=float, default=1.5e-4)
    p.add_argument("--weight_decay",     type=float, default=0.05)
    p.add_argument("--warmup_epochs",    type=int,   default=10)
    p.add_argument("--clip_grad",        type=float, default=1.0)
    p.add_argument("--dropout",          type=float, default=0.0)
    p.add_argument("--drop_path",        type=float, default=0.1)
    p.add_argument("--val_freq",         type=int,   default=10)

    # System
    p.add_argument("--dtype",       default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--compile",     action="store_true", help="torch.compile the model")
    p.add_argument("--save_dir",    default="./checkpoints_recon")
    p.add_argument("--results_dir", default="./results")

    # Resume
    p.add_argument("--resume",      action="store_true", help="Resume from latest or best checkpoint")
    p.add_argument("--resume_ckpt", default="",          help="Explicit checkpoint path (overrides auto-detect)")
    p.add_argument("--start_epoch", type=int, default=-1, help="Epoch already completed (for _best.pt resume without full state)")

    # W&B (single-model runs only)
    p.add_argument("--wandb",   action="store_true")
    p.add_argument("--project", default="tensor-memory-recon-icml")

    return p.parse_args()


# ── Single-model training run ─────────────────────────────────────────────────

def run_experiment(args, accelerator):
    set_seed(args.seed)
    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    train_dl, val_dl = build_dataloaders(args, accelerator)
    model = build_model(args)

    total_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print("=" * 64)
        print(f"  model={args.model}  preset={args.preset}  params={total_params/1e6:.2f}M")
        print(f"  enc: D={args.d_model} L={args.n_layer} H={args.n_head}")
        print(f"  dec: D={args.dec_dim} L={args.dec_depth}  mask_ratio={args.mask_ratio}")
        if args.model == "tm":
            mp = sum(p.numel() for n, p in model.named_parameters()
                     if "shared_memory" in n or "ln_memory" in n or "memory_gate" in n)
            print(f"  memory params: {mp/1e6:.3f}M ({100*mp/total_params:.1f}%)")
        print("=" * 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    spe = max(1, len(train_dl) // args.accum_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=spe * args.warmup_epochs,
        num_training_steps=spe * args.epochs,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    latest_ckpt = os.path.join(args.save_dir, f"recon_{args.model}_latest.pt")
    best_ckpt   = os.path.join(args.save_dir, f"recon_{args.model}_best.pt")
    start_epoch  = 0
    best_psnr    = 0.0
    best_results = {}
    resume_data  = None

    _ckpt_path = (
        args.resume_ckpt if args.resume_ckpt else
        latest_ckpt      if (args.resume and os.path.exists(latest_ckpt)) else
        best_ckpt        if (args.resume and os.path.exists(best_ckpt))   else ""
    )
    if _ckpt_path and os.path.exists(_ckpt_path):
        resume_data = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(resume_data, dict) and "model" in resume_data:
            model.load_state_dict(resume_data["model"])
            start_epoch  = resume_data.get("epoch", -1) + 1
            best_psnr    = resume_data.get("best_psnr", 0.)
            best_results = resume_data.get("best_results", {})
        else:
            model.load_state_dict(resume_data)  # plain weights dict (_best.pt)
            resume_data  = None                 # no optimizer/scheduler state
            start_epoch  = (args.start_epoch + 1) if args.start_epoch >= 0 else 0
        if accelerator.is_main_process:
            print(f"  → resuming from epoch {start_epoch}  (best PSNR={best_psnr:.2f}dB)")
    # ─────────────────────────────────────────────────────────────────────────

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            if accelerator.is_main_process:
                print("  torch.compile enabled")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"  [WARN] torch.compile skipped: {e}")

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    if resume_data is not None and "optimizer" in resume_data:
        optimizer.load_state_dict(resume_data["optimizer"])
        scheduler.load_state_dict(resume_data["scheduler"])
    elif start_epoch > 0:
        # Fast-forward scheduler to match the resumed epoch (no saved state)
        for _ in range(start_epoch * spe):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler,
                                     accelerator, epoch, args)

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_dl, accelerator, args)

            if accelerator.is_main_process:
                parts = []
                for mr in sorted(metrics):
                    v = metrics[mr]
                    parts.append(f"mr={mr:.0%}: PSNR={v['psnr']:.2f}dB SSIM={v['ssim']:.4f}")
                print(f"Ep {epoch:03d} | loss {train_loss:.4f} | {' | '.join(parts)} | {time.time()-t0:.1f}s")

                if args.wandb:
                    log = {"epoch": epoch, "train/loss": train_loss}
                    for mr, v in metrics.items():
                        tag = f"mr{int(mr*100)}"
                        log[f"val/psnr_{tag}"] = v["psnr"]
                        log[f"val/ssim_{tag}"] = v["ssim"]
                        log[f"val/mse_{tag}"]  = v["mse"]
                    accelerator.log(log)

                psnr_train_mr = metrics.get(args.mask_ratio, metrics.get(0.75, {})).get("psnr", 0.0)
                if psnr_train_mr > best_psnr:
                    best_psnr    = psnr_train_mr
                    best_results = {float(k): v for k, v in metrics.items()}
                    torch.save(accelerator.unwrap_model(model).state_dict(), best_ckpt)
                    print(f"  → saved best checkpoint  ({best_ckpt})")

                # Always persist full training state so any run can be resumed
                torch.save({
                    "epoch":        epoch,
                    "model":        accelerator.unwrap_model(model).state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "scheduler":    scheduler.state_dict(),
                    "best_psnr":    best_psnr,
                    "best_results": best_results,
                }, latest_ckpt)
                print(f"  → saved latest checkpoint  (ep {epoch})")

    # Save per-model results JSON
    if accelerator.is_main_process:
        out = {
            "model":        args.model,
            "params":       total_params,
            "best_results": best_results,
            "config": {
                "d_model": args.d_model, "n_layer": args.n_layer, "n_head": args.n_head,
                "mask_ratio": args.mask_ratio, "epochs": args.epochs,
            },
        }
        path = os.path.join(args.results_dir, f"recon_{args.model}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → results saved  ({path})")

    return best_results, total_params


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    mp   = args.dtype if args.dtype in ("fp16", "bf16") else "no"
    accelerator = Accelerator(
        mixed_precision=mp,
        gradient_accumulation_steps=args.accum_steps,
        log_with="wandb" if (args.wandb and not args.run_all) else None,
    )

    if args.wandb and not args.run_all and accelerator.is_main_process:
        accelerator.init_trackers(
            args.project, config=vars(args),
            init_kwargs={"wandb": {"name": f"recon-{args.model}-cub200"}},
        )

    models_to_run = [m.strip() for m in args.models.split(",")] if args.run_all else [args.model]
    all_results   = {}

    for model_name in models_to_run:
        args.model = model_name
        json_path  = os.path.join(args.results_dir, f"recon_{model_name}.json")
        if args.resume and os.path.exists(json_path):
            if accelerator.is_main_process:
                print(f"\n{'━'*64}\n  Skipping {model_name.upper()} (results exist: {json_path})\n{'━'*64}")
            with open(json_path) as f:
                saved = json.load(f)
            all_results[model_name] = {
                "params":  saved["params"],
                "metrics": {float(k): v for k, v in saved["best_results"].items()},
            }
            continue
        if accelerator.is_main_process:
            print(f"\n{'━'*64}\n  Running: {model_name.upper()}\n{'━'*64}")
        results, n_params = run_experiment(args, accelerator)
        all_results[model_name] = {"params": n_params, "metrics": results}

    if accelerator.is_main_process and len(models_to_run) > 1:
        mrs = sorted(args.eval_mask_ratios)
        print(f"\n{'='*72}")
        print("  PATCH RECONSTRUCTION — CUB-200-2011")
        print(f"{'='*72}")
        # Header
        hdr = f"{'Model':10s}  {'Params':>8s}"
        for mr in mrs:
            hdr += f"  {'PSNR@'+str(int(mr*100))+'%':>10s}  {'SSIM@'+str(int(mr*100))+'%':>10s}"
        print(hdr)
        print("-" * len(hdr))
        for mn, res in all_results.items():
            row = f"{mn:10s}  {res['params']/1e6:>7.2f}M"
            for mr in mrs:
                v = res["metrics"].get(mr, res["metrics"].get(str(mr), {}))
                row += f"  {v.get('psnr', 0):>10.2f}  {v.get('ssim', 0):>10.4f}"
            print(row)

        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, "patch_recon_cub200_comparison.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nFull results → {out_path}")

    if args.wandb and not args.run_all:
        accelerator.end_training()


if __name__ == "__main__":
    main()
