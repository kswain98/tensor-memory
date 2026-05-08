#!/usr/bin/env python3
"""
Publication-quality visualization of tensor memory at inference time.
Hooks non-invasively into tensor_memory_scan and FactorizedConv3d —
no changes to model code required.

Works with:
  • Image patch reconstruction  (PatchReconModel  from train_image.py)
  • Video action recognition     (TensorMemoryVideoViT  from train_video.py)

Figures produced per run
────────────────────────
  1  memory_energy_overlay  — ||h||₂ projected to 2-D, blended onto the input
  2  write_trajectory_3d    — 3-D scatter of write coords, coloured by time
  3  memory_slices          — mean projections along each grid axis (D/H/W)
  4  sigma_curve            — per-chunk write spread σ over the sequence
  5  gate_maps              — LSTM input / forget / output gate averages
  [video]
  6  temporal_memory        — memory-energy map at each decoded frame

Usage
─────
  # Image (patch reconstruction checkpoint)
  python tensor_memory/image/visualize.py \\
      --mode image \\
      --checkpoint ./checkpoints_recon/recon_tm_best.pt \\
      --input ./CUB_200_2011/images/001.Black_footed_Albatross/pic.jpg \\
      --output ./viz_output \\
      --d_model 192 --n_layer 4 --n_head 3 \\
      --mem_channels 16 --mem_shape 4 4 4

  # Video (UCF-101 checkpoint — args auto-loaded from ckpt)
  python tensor_memory/image/visualize.py \\
      --mode video \\
      --checkpoint ./checkpoints_ucf101/tm_best.pt \\
      --input ./UCF-101/Basketball/v_Basketball_g01_c01.avi \\
      --output ./viz_output
"""

from __future__ import annotations
import os, sys, argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (needed for 3-D projection)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "video"))
sys.path.insert(0, str(_HERE.parent))

import model as _st
from model import TensorMemoryInterface, FactorizedConv3d

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  NON-INVASIVE MEMORY HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryCapture:
    """
    Context manager that patches tensor_memory_scan.

    When capture_evolution=True, the FIRST call is run step-by-step (same
    math, no duplication) so we capture h after every `evo_chunk_size` tokens.
    Subsequent calls use the original fast path.
    """
    def __init__(self, capture_evolution: bool = False, evo_chunk_size: int = 16):
        self.calls:        List[Dict]          = []
        self.evolution_hs: List[torch.Tensor]  = []   # [1,C,D,H,W] per evo step
        self._capture_evo  = capture_evolution
        self._evo_chunk    = evo_chunk_size
        self._original     = None

    def __enter__(self):
        self._original = _st.tensor_memory_scan
        _calls    = self.calls
        _evo      = self.evolution_hs
        _orig     = self._original
        _do_evo   = self._capture_evo
        _evo_chk  = self._evo_chunk

        def _patched(h_init, c_init, read_coords_seq, content_seq,
                     mu_seq, sigma_seq, gate_conv, grid_cache):

            if _do_evo and len(_evo) == 0:
                # ── step-by-step first call — captures intermediate h states ──
                from model import efficient_gaussian_write as _egw
                S   = read_coords_seq.shape[1]
                h, c = h_init, c_init
                _evo.append(h[0:1].detach().cpu().float())  # step 0: empty
                outputs_list = []

                for t in range(S):
                    B, C_mem = h.shape[0], h.shape[1]
                    # READ
                    rc_t     = read_coords_seq[:, t].view(B, 1, 1, 1, 3)
                    mem_vec  = F.grid_sample(h, rc_t, align_corners=True,
                                             padding_mode="border").view(B, C_mem)
                    outputs_list.append(mem_vec)
                    # WRITE + CONV
                    content_vol, _ = _egw(mu_seq[:, t], sigma_seq[:, t],
                                          grid_cache, content_seq[:, t])
                    gates  = gate_conv(torch.cat([content_vol, h], dim=1))
                    gi, gf, go, gg = gates.chunk(4, dim=1)
                    c = torch.sigmoid(gf) * c + torch.sigmoid(gi) * torch.tanh(gg)
                    h = torch.sigmoid(go) * torch.tanh(c)
                    # save after every chunk
                    if (t + 1) % _evo_chk == 0 or t == S - 1:
                        _evo.append(h[0:1].detach().cpu().float())

                output_stack = torch.stack(outputs_list, dim=1)
                h_next, c_next = h, c
                out = (output_stack, h_next, c_next)
            else:
                out = _orig(h_init, c_init, read_coords_seq, content_seq,
                            mu_seq, sigma_seq, gate_conv, grid_cache)

            output_stack, h_next, c_next = out
            _calls.append({
                "h_init":       h_init[0:1].detach().cpu().float(),
                "h_next":       h_next[0:1].detach().cpu().float(),
                "c_next":       c_next[0:1].detach().cpu().float(),
                "read_coords":  read_coords_seq[0:1].detach().cpu().float(),
                "write_coords": mu_seq[0:1].detach().cpu().float(),
                "sigma":        sigma_seq[0:1].detach().cpu().float(),
                "content":      content_seq[0:1].detach().cpu().float(),
            })
            return out

        _st.tensor_memory_scan = _patched
        return self

    def __exit__(self, *_):
        _st.tensor_memory_scan = self._original

    @property
    def final_h(self) -> Optional[torch.Tensor]:
        return self.calls[-1]["h_next"] if self.calls else None

    @property
    def all_write_coords(self) -> torch.Tensor:
        return torch.cat([c["write_coords"] for c in self.calls], dim=1)

    @property
    def all_read_coords(self) -> torch.Tensor:
        return torch.cat([c["read_coords"] for c in self.calls], dim=1)

    @property
    def all_sigma(self) -> torch.Tensor:
        return torch.cat([c["sigma"] for c in self.calls], dim=1)


class GateConvHook:
    """Registers forward hooks on all FactorizedConv3d modules to capture gates."""

    def __init__(self, model: nn.Module):
        self.calls: List[Dict] = []
        self._handles = []
        for mod in model.modules():
            if isinstance(mod, FactorizedConv3d):
                self._handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, module, inp, output):
        # output: [B, 4C, D, H, W]  (4 gate channels from pointwise conv)
        g = output.detach().cpu().float()
        C = g.shape[1] // 4
        self.calls.append({
            "i": torch.sigmoid(g[:, 0*C:1*C]).mean(1),  # [B, D, H, W]
            "f": torch.sigmoid(g[:, 1*C:2*C]).mean(1),
            "o": torch.sigmoid(g[:, 2*C:3*C]).mean(1),
        })

    def remove(self):
        for h in self._handles:
            h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_model(ckpt_path: str, args) -> nn.Module:
    from train_image import (
        TensorMemoryReconEncoder, ReconDecoder, PatchReconModel,
    )
    num_patches = (args.img_size // args.patch_size) ** 2
    enc = TensorMemoryReconEncoder(
        img_size=args.img_size, patch_size=args.patch_size,
        dim=args.d_model, depth=args.n_layer, heads=args.n_head,
        mem_channels=args.mem_channels, mem_shape=tuple(args.mem_shape),
        chunk_size=args.chunk_size, sigma_scale=1.0,
        memory_every_n_layers=args.memory_every_n_layers,
    )
    dec = ReconDecoder(
        enc_dim=args.d_model, dec_dim=args.dec_dim, depth=args.dec_depth,
        heads=args.dec_heads, num_patches=num_patches, patch_size=args.patch_size,
    )
    model = PatchReconModel(enc, dec, patch_size=args.patch_size, img_size=args.img_size)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    return model.eval()


def load_video_model(ckpt_path: str) -> tuple:
    """Returns (model, saved_args_dict)."""
    from train_video import TensorMemoryVideoViT
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    model = TensorMemoryVideoViT(
        num_classes=101,
        img_size=saved.get("img_size", 224),
        patch_size=saved.get("patch_size", 16),
        dim=saved.get("d_model", 384),
        depth=saved.get("n_layer", 12),
        heads=saved.get("n_head", 6),
        num_frames=saved.get("num_frames", 8),
        dropout=0.0, drop_path=0.0,
        mem_channels=saved.get("mem_channels", 64),
        mem_shape=tuple(saved.get("mem_shape", [8, 8, 8])),
        chunk_size=saved.get("chunk_size", 8),
        sigma_scale=saved.get("sigma_scale", 1.0),
        memory_every_n_layers=saved.get("memory_every_n_layers", 1),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), saved


# ═══════════════════════════════════════════════════════════════════════════════
# §3  INPUT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_tensor(path: str, img_size: int = 224) -> tuple:
    """Returns (tensor [1,3,H,W], pil_image)."""
    pil = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return tfm(pil).unsqueeze(0), pil.resize((img_size, img_size))


def load_video_tensor(path: str, num_frames: int = 8, img_size: int = 224) -> tuple:
    """Returns (tensor [1,T,3,H,W], list of PIL frames)."""
    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    frames_pil = []
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path), num_threads=1)
        idxs = np.linspace(0, len(vr) - 1, num_frames, dtype=int).tolist()
        raw = vr.get_batch(idxs).permute(0, 3, 1, 2).float() / 255.0
        for i in range(num_frames):
            frames_pil.append(transforms.ToPILImage()(raw[i]))
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total - 1, num_frames, dtype=int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        while len(frames_pil) < num_frames:
            frames_pil.append(frames_pil[-1])

    tensors = torch.stack([tfm(f) for f in frames_pil]).unsqueeze(0)  # [1,T,3,H,W]
    pil_frames = [f.resize((img_size, img_size)) for f in frames_pil]
    return tensors, pil_frames


def denorm(t: torch.Tensor) -> np.ndarray:
    m = torch.tensor(_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(_STD).view(1, 3, 1, 1)
    return ((t * s + m).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy())


# ═══════════════════════════════════════════════════════════════════════════════
# §4  PLOT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

_CMAP_ENERGY = "plasma"
_CMAP_GATE   = "RdBu_r"
_DPI         = 150

def _save(fig, path):
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path}")


def memory_energy_2d(h: torch.Tensor) -> np.ndarray:
    """h [B,C,D,H,W] → [H,W] mean energy map (mean over C, max over D)."""
    energy = h[0].norm(dim=0)                    # [D, H, W]
    return energy.max(dim=0).values.numpy()       # [H, W]


def upsample_heatmap(hmap: np.ndarray, target_hw: tuple) -> np.ndarray:
    t = torch.tensor(hmap).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=target_hw, mode="bicubic", align_corners=False)
    out = up.squeeze().numpy()
    return (out - out.min()) / (out.max() - out.min() + 1e-8)


# ─── Figure 1: memory energy overlay ─────────────────────────────────────────

def plot_memory_energy_overlay(
    h: torch.Tensor,
    img_np: np.ndarray,
    output_path: str,
    title: str = "Memory Energy",
):
    H, W = img_np.shape[:2]
    raw_map = memory_energy_2d(h)
    hmap = upsample_heatmap(raw_map, (H, W))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(hmap, cmap=_CMAP_ENERGY)
    axes[1].set_title("Memory Energy (projected)")
    axes[1].axis("off")
    # Overlay
    axes[2].imshow(img_np)
    im = axes[2].imshow(hmap, cmap=_CMAP_ENERGY, alpha=0.55)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    D, H_m, W_m = h.shape[2:]
    fig.suptitle(f"{title}  (memory grid {D}×{H_m}×{W_m})", fontsize=11, y=1.01)
    _save(fig, output_path)


# ─── Figure 2: 3-D write trajectory ──────────────────────────────────────────

def _style_3d_ax(ax):
    """Clean 3-D axis: keep bounding box, remove grid/fill."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#cccccc")
    ax.grid(False)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=6, pad=0, colors="#999999")
    for line in (ax.xaxis.line, ax.yaxis.line, ax.zaxis.line):
        line.set_color("#cccccc")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


def plot_write_trajectory_3d(
    write_coords: torch.Tensor,
    read_coords:  torch.Tensor,
    output_path:  str,
    sigma:        Optional[torch.Tensor] = None,
    img_np:       Optional[np.ndarray]  = None,
    mask:         Optional[torch.Tensor] = None,
    patch_size:   int = 16,
    img_size:     int = 224,
    chunk_size:   int = 16,
):
    _PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
                "#F4A261", "#8338EC", "#06D6A0", "#FB5607"]

    # ── Compute full chunk count, drop partial last chunk ─────────────────────
    n_side    = img_size // patch_size
    n_patches = n_side * n_side
    visible   = (~mask.bool()).numpy().flatten()[:n_patches] if mask is not None \
                else np.ones(n_patches, dtype=bool)
    vis_idx   = np.where(visible)[0]
    T         = max(len(vis_idx) // chunk_size, 1)   # full chunks only

    wc     = write_coords[0].numpy()[:T]
    rc     = read_coords[0].numpy()[:T]
    colors = [_PALETTE[t % len(_PALETTE)] for t in range(T)]

    if sigma is not None:
        sv    = sigma[0, :T, 0].numpy()
        sn    = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
        wsz   = 180 + sn * 320
    else:
        wsz   = np.full(T, 220)

    # ── Publication rcParams ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
    })

    fig = plt.figure(figsize=(14, 5.2), dpi=180, facecolor="white")
    gs  = gridspec.GridSpec(
        1, 3, figure=fig, wspace=0.10,
        left=0.02, right=0.98, top=0.84, bottom=0.12,
    )

    # ── (a) Input image ───────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.imshow(img_np)

    for pidx in range(n_patches):       # grey masked patches
        if mask is not None and mask.flatten()[pidx]:
            r, c = pidx // n_side, pidx % n_side
            ax_a.add_patch(plt.Rectangle(
                (c * patch_size, r * patch_size), patch_size, patch_size,
                linewidth=0, facecolor="#555555", alpha=0.65))

    for pos, pidx in enumerate(vis_idx[:T * chunk_size]):   # coloured chunks
        r, c = pidx // n_side, pidx % n_side
        ax_a.add_patch(plt.Rectangle(
            (c * patch_size, r * patch_size), patch_size, patch_size,
            linewidth=0.4, edgecolor="white",
            facecolor=colors[pos // chunk_size], alpha=0.70))

    from matplotlib.patches import Patch
    leg_h = [Patch(facecolor=colors[i], edgecolor="none", label=f"Chunk {i}")
             for i in range(T)]
    leg_h += [Patch(facecolor="#555555", edgecolor="none", label="Masked")]
    ax_a.legend(handles=leg_h, ncol=T + 1, loc="upper center",
                bbox_to_anchor=(0.5, -0.04), fontsize=7.5,
                frameon=False, handlelength=1.2, handletextpad=0.4,
                columnspacing=0.7)
    ax_a.set_title("(a)  Input — visible patches by processing chunk")
    ax_a.axis("off")

    # ── Shared 3-D panel helper ───────────────────────────────────────────────
    def _panel_3d(ax, coords, sizes, label, subtitle):
        # Faint connecting line
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                color="#bbbbbb", lw=1.2, alpha=0.7, zorder=1)
        # Dots
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors, s=sizes, marker="o", depthshade=False,
                   edgecolors="white", linewidths=0.8, alpha=0.95, zorder=5)
        # Chunk-index labels offset slightly so they don't sit on the dot
        for t in range(T):
            ax.text(coords[t, 0] + 0.08, coords[t, 1] + 0.08, coords[t, 2] + 0.08,
                    str(t), fontsize=8, fontweight="bold",
                    color=colors[t], zorder=10)
        _style_3d_ax(ax)
        ax.set_title(f"{label}\n{subtitle}", pad=6)

    _panel_3d(fig.add_subplot(gs[1], projection="3d"),
              wc, wsz,
              "(b)  Memory Writes",
              "dot size ∝ write spread σ (large = broad, small = precise)")

    _panel_3d(fig.add_subplot(gs[2], projection="3d"),
              rc, np.full(T, 160),
              "(c)  Memory Reads",
              "location sampled from memory before processing each chunk")

    fig.text(0.5, 0.01,
             "Chunk colors are consistent across (a), (b), (c).  "
             "The model learns to write different patch groups "
             "to distinct regions of the 3-D memory volume.",
             ha="center", fontsize=8, color="#444444")

    _save(fig, output_path)


# ─── Figure 3: memory slices (axis projections) ───────────────────────────────

def plot_memory_slices(h: torch.Tensor, output_path: str, title: str = ""):
    """h [B,C,D,H,W] → D/H/W-axis mean projections."""
    energy = h[0].norm(dim=0)          # [D, H, W]
    proj_D = energy.mean(0).numpy()    # [H, W] — collapse depth
    proj_H = energy.mean(1).numpy()    # [D, W] — collapse height
    proj_W = energy.mean(2).numpy()    # [D, H] — collapse width

    D, H, W = h.shape[2:]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle(f"Memory Volume Projections  {D}×{H}×{W}  {title}", fontsize=11, y=1.02)
    labels = [
        ("Mean over D\n(H × W view)", proj_D),
        ("Mean over H\n(D × W view)", proj_H),
        ("Mean over W\n(D × H view)", proj_W),
    ]
    for ax, (lbl, proj) in zip(axes, labels):
        im = ax.imshow(proj, cmap=_CMAP_ENERGY, origin="lower", aspect="auto")
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, output_path)


# ─── Figure 4: sigma curve ────────────────────────────────────────────────────

def plot_sigma_curve(sigma: torch.Tensor, output_path: str):
    """sigma [B, T, 1] → line plot."""
    s = sigma[0, :, 0].numpy()
    T = len(s)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(T), s, "o-", color="#e15759", linewidth=1.5, markersize=4)
    ax.fill_between(range(T), 0, s, alpha=0.15, color="#e15759")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("σ (write spread)")
    ax.set_title("Write Spread σ over Sequence\n(larger = wider Gaussian footprint)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, output_path)


# ─── Figure 5: gate maps ──────────────────────────────────────────────────────

def plot_gate_maps(gate_calls: List[Dict], output_path: str):
    """Visualise average LSTM gate activations across all FactorizedConv3d calls."""
    if not gate_calls:
        print("  [skip] no gate captures")
        return
    # Average over all calls
    keys = ["i", "f", "o"]
    names = ["Input gate (i)", "Forget gate (f)", "Output gate (o)"]
    avg = {k: torch.stack([c[k] for c in gate_calls]).mean(0)[0] for k in keys}
    # avg[k] shape: [D, H, W]

    fig, axes = plt.subplots(len(keys), 3, figsize=(11, 3.5 * len(keys)))
    for row, (k, name) in enumerate(zip(keys, names)):
        energy = avg[k]  # [D, H, W]
        projs = [
            (energy.mean(0).numpy(), "Mean over D"),
            (energy.mean(1).numpy(), "Mean over H"),
            (energy.mean(2).numpy(), "Mean over W"),
        ]
        # Auto-scale per gate so spatial patterns are always visible
        vmin = min(p.min() for p, _ in projs)
        vmax = max(p.max() for p, _ in projs)
        vpad = max((vmax - vmin) * 0.05, 1e-6)
        vmin -= vpad; vmax += vpad
        for col, (proj, lbl) in enumerate(projs):
            ax = axes[row, col] if len(keys) > 1 else axes[col]
            im = ax.imshow(proj, cmap=_CMAP_GATE, vmin=vmin, vmax=vmax,
                           origin="lower", aspect="auto")
            ax.set_title(f"{name}\n({lbl})", fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("LSTM Gate Activations (averaged over sequence)", fontsize=11)
    _save(fig, output_path)


# ─── Figure 6: memory state evolution (image) ────────────────────────────────

def plot_memory_evolution(
    evolution_hs: List[torch.Tensor],   # [1,C,D,H,W] per step, first = initial
    img_np:       np.ndarray,
    vis_idx:      np.ndarray,           # visible patch indices (raster order)
    chunk_size:   int,
    patch_size:   int,
    img_size:     int,
    output_path:  str,
):
    _PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
                "#F4A261", "#8338EC", "#06D6A0", "#FB5607"]

    # evolution_hs: [initial, after_step1, after_step2, …]
    # each step = chunk_size tokens → show one column per full chunk + initial
    n_full = len(vis_idx) // chunk_size
    frames  = [evolution_hs[0]]                      # initial
    for i in range(1, n_full + 1):
        idx = min(i, len(evolution_hs) - 1)
        frames.append(evolution_hs[idx])
    n_cols = len(frames)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.0),
                              dpi=180, facecolor="white")
    fig.subplots_adjust(hspace=0.04, wspace=0.03,
                        left=0.06, right=0.91, top=0.88, bottom=0.06)

    n_side = img_size // patch_size

    for col, h in enumerate(frames):
        chunk_done = col - 1   # -1 = none done yet

        # ── top row: image with cumulative chunk highlights ───────────────────
        ax_t = axes[0, col]
        ax_t.imshow(img_np)
        n_highlighted = min((chunk_done + 1) * chunk_size, len(vis_idx))
        for pos in range(n_highlighted):
            ci   = pos // chunk_size
            pidx = vis_idx[pos]
            r, c = pidx // n_side, pidx % n_side
            ax_t.add_patch(plt.Rectangle(
                (c * patch_size, r * patch_size), patch_size, patch_size,
                linewidth=0.3, edgecolor="white",
                facecolor=_PALETTE[ci % len(_PALETTE)], alpha=0.72))
        ax_t.axis("off")
        if col == 0:
            ax_t.set_title("Initial\n(no patches\nprocessed)", fontsize=8, pad=4)
        else:
            ax_t.set_title(f"After\nchunk {chunk_done}",
                           fontsize=8, pad=4,
                           color=_PALETTE[chunk_done % len(_PALETTE)],
                           fontweight="bold")

        # ── bottom row: memory energy heatmap ────────────────────────────────
        ax_b = axes[1, col]
        energy = h[0].norm(dim=0)
        proj   = energy.max(dim=0).values.numpy()
        hmap   = upsample_heatmap(proj, (img_size, img_size))
        ax_b.imshow(img_np, alpha=0.25)
        im = ax_b.imshow(hmap, cmap="plasma", vmin=0, vmax=1, alpha=0.85)
        ax_b.axis("off")

    # Shared colorbar
    cax = fig.add_axes([0.925, 0.06, 0.012, 0.38])
    fig.colorbar(im, cax=cax)
    cax.tick_params(labelsize=7)
    cax.set_ylabel("Memory energy  ‖h‖₂", fontsize=7)

    # Row labels on the left
    fig.text(0.015, 0.73, "Patches\nprocessed", fontsize=8,
             va="center", ha="center", rotation=90, color="#444")
    fig.text(0.015, 0.32, "Memory\nenergy", fontsize=8,
             va="center", ha="center", rotation=90, color="#444")

    fig.suptitle("Memory State Evolution — 3-D memory builds up as chunks are processed",
                 fontsize=10, fontweight="bold", y=0.95)
    _save(fig, output_path)


# ─── Figure 7 (video): temporal memory evolution ──────────────────────────────

def plot_temporal_memory(
    frame_memories: List[torch.Tensor],   # list of h [B,C,D,H,W], one per frame
    pil_frames: List[Image.Image],
    output_path: str,
):
    T = len(frame_memories)
    W_fig = min(T, 8)
    fig, axes = plt.subplots(2, W_fig, figsize=(2.2 * W_fig, 5))

    for t in range(W_fig):
        # top row: frame
        axes[0, t].imshow(pil_frames[t])
        axes[0, t].set_title(f"Frame {t}", fontsize=8)
        axes[0, t].axis("off")

        # bottom row: memory energy
        hmap = memory_energy_2d(frame_memories[t])
        H_f, W_f = np.array(pil_frames[t]).shape[:2]
        hmap_up = upsample_heatmap(hmap, (H_f, W_f))
        ax = axes[1, t]
        ax.imshow(np.array(pil_frames[t]))
        ax.imshow(hmap_up, cmap=_CMAP_ENERGY, alpha=0.6)
        ax.set_title(f"Memory energy t={t}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Temporal Memory Evolution", fontsize=11)
    _save(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# §5  INFERENCE RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_image(args, out_dir: Path):
    print("Loading image model …")
    model = load_image_model(args.checkpoint, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img_t, pil_img = load_image_tensor(args.input, args.img_size)
    img_t = img_t.to(device)
    img_np = np.array(pil_img)

    cap = MemoryCapture(capture_evolution=True, evo_chunk_size=args.chunk_size)
    gate_hook = GateConvHook(model)

    with cap:
        loss, pred, mask, target = model(img_t, mask_ratio=args.mask_ratio)

    gate_hook.remove()

    h_final = cap.final_h
    wc      = cap.all_write_coords
    rc      = cap.all_read_coords
    sigma   = cap.all_sigma

    # Visible patch indices (for evolution figure)
    n_side    = args.img_size // args.patch_size
    n_patches = n_side * n_side
    _mask_flat = mask[0].cpu().bool().numpy().flatten()[:n_patches]
    vis_idx    = np.where(~_mask_flat)[0]

    print(f"  Captured {len(cap.calls)} memory call(s),  "
          f"{wc.shape[1]} write steps,  {len(cap.evolution_hs)} evo frames,  "
          f"h shape={list(h_final.shape)}")

    plot_memory_energy_overlay(h_final, img_np,
                               str(out_dir / "memory_energy_overlay.png"),
                               title="Image Patch Reconstruction")
    plot_write_trajectory_3d(wc, rc,
                             str(out_dir / "write_trajectory_3d.png"),
                             sigma=sigma,
                             img_np=img_np,
                             mask=mask[0].cpu(),
                             patch_size=args.patch_size,
                             img_size=args.img_size,
                             chunk_size=args.chunk_size)
    plot_memory_slices(h_final,
                       str(out_dir / "memory_slices.png"),
                       title="(image)")
    plot_sigma_curve(sigma,
                     str(out_dir / "sigma_curve.png"))
    plot_gate_maps(gate_hook.calls,
                   str(out_dir / "gate_maps.png"))

    if cap.evolution_hs:
        plot_memory_evolution(
            cap.evolution_hs, img_np, vis_idx,
            chunk_size=args.chunk_size,
            patch_size=args.patch_size,
            img_size=args.img_size,
            output_path=str(out_dir / "memory_evolution.png"),
        )

    # Bonus: reconstruction comparison
    from train_image import unpatchify, patchify
    target_patches = patchify(img_t, args.patch_size)
    mean_p = target_patches.mean(-1, keepdim=True)
    std_p  = (target_patches.var(-1, keepdim=True) + 1e-6).sqrt()
    pred_pixels = pred * std_p + mean_p
    mask_e  = mask.unsqueeze(-1).expand_as(target_patches)
    composite = target_patches * (1 - mask_e) + pred_pixels * mask_e
    recon_img = unpatchify(composite, args.patch_size, args.img_size)
    masked_img = unpatchify(target_patches * (1 - mask_e), args.patch_size, args.img_size)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(denorm(masked_img.cpu())); axes[1].set_title("Masked input"); axes[1].axis("off")
    axes[2].imshow(denorm(recon_img.cpu())); axes[2].set_title("Reconstruction"); axes[2].axis("off")
    fig.suptitle(f"MAE Reconstruction  (mask_ratio={args.mask_ratio:.0%})", fontsize=11)
    _save(fig, str(out_dir / "reconstruction.png"))

    print(f"\nAll figures written to {out_dir}/")


@torch.no_grad()
def run_video(args, out_dir: Path):
    print("Loading video model …")
    model, saved_args = load_video_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_frames = saved_args.get("num_frames", args.num_frames)
    img_size   = saved_args.get("img_size", 224)
    vid_t, pil_frames = load_video_tensor(args.input, num_frames, img_size)
    vid_t = vid_t.to(device)

    # For temporal evolution, we capture memory after each frame
    # by hooking TensorMemoryVideoViT.forward directly
    frame_memories: List[torch.Tensor] = []

    def _frame_hook(module, inp, output):
        pass   # not used; we capture via MemoryCapture per frame

    # Patch TensorMemoryVideoViT to yield memory state per frame
    from train_video import TensorMemoryVideoViT
    _orig_forward = TensorMemoryVideoViT.forward

    def _instrumented_forward(self_, x_):
        B, T, C, H, W = x_.shape
        mem = self_._init_memory(B, x_.device, x_.dtype)
        for t in range(T):
            tokens = self_.patch_embed(x_[:, t]).flatten(2).transpose(1, 2)
            tokens = tokens + self_.pos_embed
            cls    = self_.cls_token.expand(B, -1, -1)
            tokens = self_.pos_drop(torch.cat([cls, tokens], dim=1))
            for i, blk in enumerate(self_.blocks):
                if self_.layer_uses_memory[i]:
                    tokens, mem = blk(tokens, memory_state=mem,
                                      shared_memory=self_.shared_memory,
                                      chunk_size=self_.chunk_size)
                else:
                    tokens, _ = blk(tokens, memory_state=None, shared_memory=None)
            frame_memories.append(mem[0].unsqueeze(0).detach().cpu().float())  # h
        last_cls = self_.norm(tokens)[:, 0]
        return self_.head(last_cls)

    TensorMemoryVideoViT.forward = _instrumented_forward

    cap = MemoryCapture()
    gate_hook = GateConvHook(model)

    with cap:
        logits = model(vid_t)

    TensorMemoryVideoViT.forward = _orig_forward
    gate_hook.remove()

    pred_class = logits.argmax(-1).item()
    h_final = cap.final_h
    wc = cap.all_write_coords
    rc = cap.all_read_coords
    sigma = cap.all_sigma

    print(f"  Predicted class: {pred_class}")
    print(f"  Captured {len(cap.calls)} memory calls,  "
          f"{wc.shape[1]} write steps,  h shape={list(h_final.shape)}")

    plot_memory_energy_overlay(h_final, np.array(pil_frames[-1]),
                               str(out_dir / "memory_energy_overlay.png"),
                               title=f"Video — predicted class {pred_class}")
    plot_write_trajectory_3d(wc, rc,
                             str(out_dir / "write_trajectory_3d.png"),
                             sigma=sigma)
    plot_memory_slices(h_final,
                       str(out_dir / "memory_slices.png"),
                       title="(final frame)")
    plot_sigma_curve(sigma,
                     str(out_dir / "sigma_curve.png"))
    plot_gate_maps(gate_hook.calls,
                   str(out_dir / "gate_maps.png"))

    if frame_memories:
        plot_temporal_memory(frame_memories, pil_frames,
                             str(out_dir / "temporal_memory.png"))

    print(f"\nAll figures written to {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  ARGPARSE + ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Tensor Memory Inference Visualizer")
    p.add_argument("--mode",       default="image", choices=["image", "video"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input",      required=True, help="Image file or video clip")
    p.add_argument("--output",     default="./viz_output")
    p.add_argument("--mask_ratio", type=float, default=0.75,
                   help="[image] MAE mask ratio for visualization")
    p.add_argument("--num_frames", type=int, default=8,
                   help="[video] frames to decode")

    # Image model arch (only needed if --mode image)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--d_model",    type=int, default=192)
    p.add_argument("--n_layer",    type=int, default=4)
    p.add_argument("--n_head",     type=int, default=3)
    p.add_argument("--dec_dim",    type=int, default=96)
    p.add_argument("--dec_depth",  type=int, default=2)
    p.add_argument("--dec_heads",  type=int, default=3)
    p.add_argument("--mem_channels",          type=int,   default=16)
    p.add_argument("--mem_shape",             type=int, nargs="+", default=[4, 4, 4])
    p.add_argument("--chunk_size",            type=int,   default=16)
    p.add_argument("--memory_every_n_layers", type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "image":
        run_image(args, out_dir)
    else:
        run_video(args, out_dir)


if __name__ == "__main__":
    main()
