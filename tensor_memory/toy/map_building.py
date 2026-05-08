"""Toy 2: Long-horizon map building — paper §4.1.

There is a hidden 8x8 binary occupancy grid. At each of T steps the model
sees a small (2x2) patch of it at a random (row, col) window position
(observation token = patch values + window position). At the end, a query
token asks about a specific cell (qrow, qcol); the model must predict whether
that cell is occupied. We sweep T (the horizon).

Tensor Memory should hold accuracy as T grows; vanilla attention has to scan
through the full observation history every time.

Run:
    python tensor_memory/toy/map_building.py
    python tensor_memory/toy/map_building.py --horizons 16,32,64
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import make_backbone, count_params


def gen_batch(batch, T, grid_size=8, patch=2, device='cuda'):
    """Returns (obs_tokens [B, T+1, p*p+3], target [B] in {0, 1}).

    Per-step observation features: flattened patch (patch*patch) + (row, col, 0).
    Query features (last token): zeros for patch + (qrow, qcol, 1) marker bit so the
    model can distinguish observation from query. A linear projection in the model
    maps these features to model-d.
    """
    H = W = grid_size
    p = patch
    grid = torch.randint(0, 2, (batch, H, W), device=device).float()

    # Vectorised patch extraction via unfold.
    # patches_all[b, r, c] is the p*p patch at (r:r+p, c:c+p).
    patches_all = grid.unfold(1, p, 1).unfold(2, p, 1)        # [B, H-p+1, W-p+1, p, p]
    r0 = torch.randint(0, H - p + 1, (batch, T), device=device)
    c0 = torch.randint(0, W - p + 1, (batch, T), device=device)
    bi = torch.arange(batch, device=device).unsqueeze(1).expand(batch, T)
    obs_patches = patches_all[bi, r0, c0].reshape(batch, T, p * p)

    obs_pos = torch.zeros(batch, T, 3, device=device)
    obs_pos[..., 0] = r0.float() / H
    obs_pos[..., 1] = c0.float() / W
    # obs_pos[..., 2] stays 0 (observation marker)

    qr = torch.randint(0, H, (batch,), device=device)
    qc = torch.randint(0, W, (batch,), device=device)
    target = grid[torch.arange(batch), qr, qc].long()
    q_patch = torch.zeros(batch, 1, p * p, device=device)
    q_pos = torch.stack([qr.float() / H, qc.float() / W, torch.ones(batch, device=device)], dim=-1).unsqueeze(1)

    feats = torch.cat([
        torch.cat([obs_patches, obs_pos], dim=-1),
        torch.cat([q_patch, q_pos], dim=-1),
    ], dim=1)  # [B, T+1, p*p+3]
    return feats, target


class MapModel(nn.Module):
    def __init__(self, backbone, d=128, layers=3, heads=4, T_max=128, patch=2, **bb_kwargs):
        super().__init__()
        self.in_dim = patch * patch + 3
        self.embed = nn.Linear(self.in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, T_max + 1, d))
        self.backbone = make_backbone(backbone, d, layers, heads, **bb_kwargs)
        self.head = nn.Linear(d, 2)

    def forward(self, feats):
        B, S, _ = feats.shape
        x = self.embed(feats) + self.pos[:, :S]
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        h = self.backbone(x, mask=mask)
        return self.head(h[:, -1])  # query token's prediction


def train_one(method, T, args, device):
    torch.manual_seed(args.seed)
    model = MapModel(
        method, d=args.d, layers=args.layers, heads=args.heads,
        T_max=max(args.horizons_list), patch=args.patch,
        mem_channels=args.mem_channels, mem_shape=tuple(args.mem_shape),
        chunk_size=1, n_slots=args.n_slots, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.time()
    for step in range(args.steps):
        feats, target = gen_batch(args.batch, T, args.grid_size, args.patch, device)
        logits = model(feats)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            feats, target = gen_batch(args.batch, T, args.grid_size, args.patch, device)
            pred = model(feats).argmax(-1)
            correct += (pred == target).sum().item()
            total += target.numel()
    res = {
        'method': method,
        'horizon': T,
        'final_acc': correct / total,
        'params': count_params(model),
        'wall_seconds': round(time.time() - t0, 1),
    }
    if hasattr(model.backbone, 'get_gates'):
        res['final_gates'] = model.backbone.get_gates()
    return res


def main():
    p = argparse.ArgumentParser("Toy 2: Map building")
    p.add_argument("--methods", default="base,base_wide,slots,tensor")
    p.add_argument("--horizons", default="16,32,64")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--grid_size", type=int, default=8)
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mem_channels", type=int, default=16)
    p.add_argument("--mem_shape", type=int, nargs=3, default=[6, 6, 6])
    p.add_argument("--n_slots", type=int, default=8)
    p.add_argument("--outdir", default="./results_toy2")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    args.horizons_list = [int(x) for x in args.horizons.split(',')]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Toy 2: Map building | device={device}")
    print(f"Methods: {methods}, horizons: {args.horizons_list}, grid={args.grid_size}^2, patch={args.patch}^2\n")

    results = []
    for m in methods:
        for T in args.horizons_list:
            r = train_one(m, T, args, device)
            print(f"  [{m:<10}] T={T:<3d}  acc={r['final_acc']:.3f}  ({r['wall_seconds']}s)")
            results.append(r)

    out = os.path.join(args.outdir, "toy2_map_building.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")

    print("\n=== Accuracy vs horizon ===")
    print("  method      " + "  ".join(f"T={T}" for T in args.horizons_list))
    for m in methods:
        row = f"  {m:<10}  "
        for T in args.horizons_list:
            r = next(x for x in results if x['method'] == m and x['horizon'] == T)
            row += f"{r['final_acc']:.3f}  "
        print(row)


if __name__ == '__main__':
    main()
