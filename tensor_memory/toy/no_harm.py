"""Toy 4: No-harm control — paper §4.1.

A short fully-observed task where persistent state should not matter.
Tensor Memory should match the baseline and its learned gates should stay
near their initialization (sigma(0) = 0.5).

Task: copy task with one-token lookback.
    Input:  s_1 s_2 ... s_T  (each s_t in [0, V))
    Target: s_t = (s_{t-1} + 1) mod V    for t >= 1, s_0 random

Each prediction depends only on the immediately preceding token, so a
small Transformer should solve it with high accuracy without using any
persistent memory. We report (accuracy, learned gate values).
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import make_backbone, count_params


def gen_batch(batch, seq_len, vocab, device):
    s0 = torch.randint(0, vocab, (batch,), device=device)
    seq = [s0]
    for _ in range(seq_len - 1):
        seq.append((seq[-1] + 1) % vocab)
    x = torch.stack(seq, dim=1)
    # Target: x shifted left by 1; last position has no target (use ignore_index)
    target = torch.cat([x[:, 1:], torch.full((batch, 1), -100, device=device, dtype=x.dtype)], dim=1)
    return x, target


class NoHarmModel(nn.Module):
    def __init__(self, backbone, d=64, layers=2, heads=4, vocab=16, seq_len=32, **bb_kwargs):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        self.backbone = make_backbone(backbone, d, layers, heads, **bb_kwargs)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, S = x.shape
        h = self.tok(x) + self.pos[:, :S]
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        h = self.backbone(h, mask=mask)
        return self.head(h)


def train_one(method, args, device):
    torch.manual_seed(args.seed)
    model = NoHarmModel(
        method, d=args.d, layers=args.layers, heads=args.heads,
        vocab=args.vocab, seq_len=args.seq_len,
        mem_channels=args.mem_channels, mem_shape=tuple(args.mem_shape),
        chunk_size=1, n_slots=args.n_slots, dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.time()
    for step in range(args.steps):
        x, y = gen_batch(args.batch, args.seq_len, args.vocab, device)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = gen_batch(args.batch, args.seq_len, args.vocab, device)
            pred = model(x).argmax(-1)
            valid = y != -100
            correct += (pred[valid] == y[valid]).sum().item()
            total += valid.sum().item()
    acc = correct / total

    res = {
        'method': method,
        'final_acc': acc,
        'params': count_params(model),
        'wall_seconds': round(time.time() - t0, 1),
    }
    if hasattr(model.backbone, 'get_gates'):
        res['final_gates'] = model.backbone.get_gates()
        res['mean_gate'] = sum(res['final_gates']) / len(res['final_gates'])
    return res


def main():
    p = argparse.ArgumentParser("Toy 4: No-harm control")
    p.add_argument("--methods", default="base,base_wide,slots,tensor")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--vocab", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mem_channels", type=int, default=16)
    p.add_argument("--mem_shape", type=int, nargs=3, default=[6, 6, 6])
    p.add_argument("--n_slots", type=int, default=8)
    p.add_argument("--outdir", default="./results_toy4")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Toy 4: No-harm control | device={device}")
    print(f"Methods: {methods}, seq_len={args.seq_len}, vocab={args.vocab}, steps={args.steps}\n")

    results = []
    for m in methods:
        r = train_one(m, args, device)
        extra = ""
        if 'mean_gate' in r:
            extra = f"  mean_gate={r['mean_gate']:.3f}  gates={[round(g,2) for g in r['final_gates']]}"
        print(f"  [{m:<10}] acc={r['final_acc']:.3f}  params={r['params']:,}  ({r['wall_seconds']}s){extra}")
        results.append(r)

    out = os.path.join(args.outdir, "toy4_no_harm.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")
    print("Tensor Memory passes if (a) acc within ~1pp of base and (b) mean gate near 0.5 (its init).")


if __name__ == '__main__':
    main()
