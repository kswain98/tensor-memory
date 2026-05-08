"""Toy 1: Occlusion and object permanence — paper §4.1.

Synthetic 16x16 frames. A single ball moves with constant integer velocity;
a static rectangular occluder hides part of the frame. The model sees T frames
and predicts the ball's quadrant at the *final* frame (4-class classification).

We sweep occlusion length: for L consecutive middle frames the ball is hidden
by the occluder (the occluder always covers it, regardless of position).
Tensor Memory should hold up better at larger L because it can keep the ball's
trajectory in its persistent memory state.

Run:
    python tensor_memory/toy/occlusion.py
    python tensor_memory/toy/occlusion.py --methods base,tensor --occlusion_lens 2,6
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import make_backbone, count_params


def gen_batch(batch, T, occ_len, frame_size=16, device='cuda'):
    """Returns (frames [B, T, 1, H, W], target_quadrant [B] in {0..3}).

    Ball: 2x2 pixels at (x, y) starting from a random position with a random
    integer velocity in {-1, 0, +1}^2. Wraps around the frame.
    Occluder: rectangle covering rows in [r0, r0+occ_h) and cols in [c0, c0+occ_w).
    During the middle `occ_len` frames the ball pixels are zeroed out (hidden).
    """
    H = W = frame_size
    px = torch.randint(0, W, (batch,), device=device)
    py = torch.randint(0, H, (batch,), device=device)
    vx = torch.randint(-1, 2, (batch,), device=device)
    vy = torch.randint(-1, 2, (batch,), device=device)
    # Avoid zero velocity (stationary ball is degenerate)
    nonzero = (vx == 0) & (vy == 0)
    vx[nonzero] = 1

    frames = torch.zeros(batch, T, 1, H, W, device=device)
    occ_start = max(0, (T - occ_len) // 2)
    occ_end = occ_start + occ_len
    for t in range(T):
        cx = (px + vx * t) % W
        cy = (py + vy * t) % H
        if t < occ_start or t >= occ_end:
            for dy in range(2):
                for dx in range(2):
                    frames[torch.arange(batch), t, 0, (cy + dy) % H, (cx + dx) % W] = 1.0

    # Final position quadrant
    cx_final = (px + vx * (T - 1)) % W
    cy_final = (py + vy * (T - 1)) % H
    quad = (cx_final >= W // 2).long() + 2 * (cy_final >= H // 2).long()
    return frames, quad


class OcclusionModel(nn.Module):
    """Per-frame patch tokens fed sequentially into the backbone."""

    def __init__(self, backbone, d=128, layers=3, heads=4, T=8, frame_size=16, patch=4, **bb_kwargs):
        super().__init__()
        self.T = T
        self.frame_size = frame_size
        self.patch = patch
        self.tokens_per_frame = (frame_size // patch) ** 2
        self.patch_embed = nn.Conv2d(1, d, kernel_size=patch, stride=patch)
        self.tok_pos = nn.Parameter(torch.zeros(1, self.tokens_per_frame, d))
        self.frame_pos = nn.Parameter(torch.zeros(1, T, 1, d))
        self.backbone = make_backbone(backbone, d, layers, heads, **bb_kwargs)
        self.head = nn.Linear(d, 4)

    def forward(self, frames):
        B, T, _, H, W = frames.shape
        x = frames.view(B * T, 1, H, W)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B*T, P, d]
        x = x + self.tok_pos
        x = x.view(B, T, self.tokens_per_frame, -1)
        x = x + self.frame_pos
        x = x.view(B, T * self.tokens_per_frame, -1)
        S = x.size(1)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        h = self.backbone(x, mask=mask)
        # Predict from the final-frame mean
        last = h[:, -self.tokens_per_frame:, :].mean(dim=1)
        return self.head(last)


def train_one(method, occ_len, args, device):
    torch.manual_seed(args.seed)
    model = OcclusionModel(
        method, d=args.d, layers=args.layers, heads=args.heads,
        T=args.T, frame_size=args.frame_size, patch=args.patch,
        mem_channels=args.mem_channels, mem_shape=tuple(args.mem_shape),
        chunk_size=1, n_slots=args.n_slots, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.time()
    for step in range(args.steps):
        frames, target = gen_batch(args.batch, args.T, occ_len, args.frame_size, device)
        logits = model(frames)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            frames, target = gen_batch(args.batch, args.T, occ_len, args.frame_size, device)
            pred = model(frames).argmax(-1)
            correct += (pred == target).sum().item()
            total += target.numel()
    res = {
        'method': method,
        'occlusion_len': occ_len,
        'final_acc': correct / total,
        'params': count_params(model),
        'wall_seconds': round(time.time() - t0, 1),
    }
    if hasattr(model.backbone, 'get_gates'):
        res['final_gates'] = model.backbone.get_gates()
    return res


def main():
    p = argparse.ArgumentParser("Toy 1: Occlusion / object permanence")
    p.add_argument("--methods", default="base,base_wide,slots,tensor")
    p.add_argument("--occlusion_lens", default="0,2,4,6")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--frame_size", type=int, default=16)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mem_channels", type=int, default=16)
    p.add_argument("--mem_shape", type=int, nargs=3, default=[6, 6, 6])
    p.add_argument("--n_slots", type=int, default=8)
    p.add_argument("--outdir", default="./results_toy1")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    occs = [int(x) for x in args.occlusion_lens.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Toy 1: Occlusion | device={device}")
    print(f"Methods: {methods}, occlusion lens: {occs}, T={args.T}, frame={args.frame_size}\n")

    results = []
    for m in methods:
        for L in occs:
            r = train_one(m, L, args, device)
            print(f"  [{m:<10}] occ_len={L}  acc={r['final_acc']:.3f}  ({r['wall_seconds']}s)")
            results.append(r)

    out = os.path.join(args.outdir, "toy1_occlusion.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")

    print("\n=== Accuracy vs occlusion length ===")
    print("  method      " + "  ".join(f"L={L}" for L in occs))
    for m in methods:
        row = f"  {m:<10}  "
        for L in occs:
            r = next(x for x in results if x['method'] == m and x['occlusion_len'] == L)
            row += f"{r['final_acc']:.3f}  "
        print(row)


if __name__ == '__main__':
    main()
