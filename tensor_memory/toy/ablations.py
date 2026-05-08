"""Toy ablations — paper §4.1.

We ablate the design choices inside TensorMemoryInterface, all on the
coordinate-binding task (Toy 3) at one fixed difficulty (n_writes=20,
noise_sigma=0.05), so each ablation produces a single number we can compare
against the default Tensor Memory.

Ablations:
    1. write_mode  = gaussian | hard            (Gaussian-weighted vs nearest-cell write)
    2. share_coords = true | false              (one coord head vs two)
    3. factorized  = on | off                   (factorized 3D conv vs 1x1x1 conv only)
    4. mem_shape   = (4,4,4) | (6,6,6) | (8,8,8) (memory resolution)
    5. chunk_size  = 1 | 2 | 4                  (chunking)

Each variant trains a fresh model and reports final binding accuracy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import (
    FactorizedConv3d,
    TensorMemoryEmbedder,
    efficient_gaussian_write,
)
from backbones import TransformerBlock, count_params
from coord_binding import gen_batch, CoordBindingModel


# =====================================================================
# Ablatable Tensor Memory
# =====================================================================

class AblatableTensorMemory(nn.Module):
    """TensorMemoryInterface with toggles for each ablation."""

    def __init__(self, embed_dim, memory_channels, memory_shape, chunk_size,
                 write_mode='gaussian', share_coords=True, factorized=True, sigma_scale=1.0):
        super().__init__()
        assert write_mode in ('gaussian', 'hard')
        self.memory_shape = tuple(memory_shape)
        self.memory_channels = memory_channels
        self.chunk_size = chunk_size
        self.write_mode = write_mode
        self.share_coords = share_coords
        self.factorized_on = factorized

        self.write_proj = nn.Linear(chunk_size * embed_dim, embed_dim)
        self.writer = TensorMemoryEmbedder(embed_dim, memory_channels, sigma_scale=sigma_scale)
        self.coord_net = nn.Linear(embed_dim, 3)
        if not share_coords:
            # Separate write coord head; read still uses self.coord_net
            self.coord_net_write = nn.Linear(embed_dim, 3)

        if factorized:
            self.gate_conv = FactorizedConv3d(2 * memory_channels, 4 * memory_channels)
        else:
            # No spatial mixing: per-cell gates via 1x1x1 conv only
            self.gate_conv = nn.Conv3d(2 * memory_channels, 4 * memory_channels, kernel_size=1)

        self.out_proj = nn.Linear(memory_channels, embed_dim)

        D, H, W = self.memory_shape
        z = torch.linspace(-1, 1, D); y = torch.linspace(-1, 1, H); x = torch.linspace(-1, 1, W)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing='ij')
        self.register_buffer('grid_cache', torch.stack([gx, gy, gz], dim=0).unsqueeze(0))

    def init_state(self, batch_size, device, dtype):
        D, H, W = self.memory_shape
        C = self.memory_channels
        h = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        return h, c

    def _write_volume(self, mu, sigma, content):
        """Either Gaussian-weighted broadcast write, or hard nearest-cell write."""
        if self.write_mode == 'gaussian':
            cv, _ = efficient_gaussian_write(mu, sigma, self.grid_cache, content)
            return cv
        # 'hard': place content at nearest cell index (still differentiable wrt content)
        D, H, W = self.memory_shape
        # mu is in [-1, 1]; convert to integer indices
        ix = ((mu[:, 0] + 1) * 0.5 * (W - 1)).round().long().clamp(0, W - 1)
        iy = ((mu[:, 1] + 1) * 0.5 * (H - 1)).round().long().clamp(0, H - 1)
        iz = ((mu[:, 2] + 1) * 0.5 * (D - 1)).round().long().clamp(0, D - 1)
        B, C = content.shape
        cv = torch.zeros(B, C, D, H, W, device=content.device, dtype=content.dtype)
        cv[torch.arange(B), :, iz, iy, ix] = content
        return cv

    def _scan(self, h, c, read_coords, content_seq, mu_seq, sigma_seq):
        B, S, _ = read_coords.shape
        outputs = []
        for t in range(S):
            rc = read_coords[:, t].view(B, 1, 1, 1, 3)
            mem_vec = F.grid_sample(h, rc, align_corners=True, padding_mode='border').view(B, self.memory_channels)
            outputs.append(mem_vec)
            cv = self._write_volume(mu_seq[:, t], sigma_seq[:, t], content_seq[:, t])
            combined = torch.cat([cv, h], dim=1)
            gates = self.gate_conv(combined)
            i, f, o, g = gates.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        return torch.stack(outputs, dim=1), h, c

    def forward(self, x, prev_state, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        B, N, D_emb = x.shape
        pad = (chunk_size - (N % chunk_size)) % chunk_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        N_pad = x.size(1)
        nc = N_pad // chunk_size
        xg = x.view(B, nc, chunk_size, D_emb)

        x_read = xg[:, :, 0, :]
        x_flat = xg.flatten(2)
        x_write = self.write_proj(x_flat)

        read_coords = torch.tanh(self.coord_net(x_read))
        if self.share_coords:
            write_coords = torch.tanh(self.coord_net(x_write))
        else:
            write_coords = torch.tanh(self.coord_net_write(x_write))
        content, sigma = self.writer(x_write)

        h0, c0 = prev_state
        out_stack, h_next, c_next = self._scan(h0, c0, read_coords, content, write_coords, sigma)

        out = out_stack.repeat_interleave(chunk_size, dim=1)
        if pad:
            out = out[:, :N, :]
        return self.out_proj(out), (h_next, c_next)


# =====================================================================
# Backbone wrapper that uses AblatableTensorMemory
# =====================================================================

class AblatableTensorBackbone(nn.Module):
    def __init__(self, d, layers, heads, mem_channels=16, mem_shape=(6, 6, 6),
                 chunk_size=1, dropout=0.1, write_mode='gaussian',
                 share_coords=True, factorized=True, sigma_scale=1.0):
        super().__init__()
        self.token_blocks = nn.ModuleList(
            [TransformerBlock(d, heads, dropout) for _ in range(layers)]
        )
        self.memory = AblatableTensorMemory(
            embed_dim=d, memory_channels=mem_channels, memory_shape=mem_shape,
            chunk_size=chunk_size, write_mode=write_mode,
            share_coords=share_coords, factorized=factorized, sigma_scale=sigma_scale,
        )
        self.mem_lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.0])) for _ in range(layers)]
        )
        self.mem_drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)
        self.chunk_size = chunk_size

    def forward(self, x, mask=None):
        B = x.size(0)
        mem = self.memory.init_state(B, x.device, x.dtype)
        for i, blk in enumerate(self.token_blocks):
            x = blk(x, mask)
            m, mem = self.memory(self.mem_lns[i](x), mem, chunk_size=self.chunk_size)
            x = x + torch.sigmoid(self.gates[i]) * self.mem_drop(m)
        return self.ln(x)

    def get_gates(self):
        return [torch.sigmoid(g).item() for g in self.gates]


class AblatableModel(nn.Module):
    """CoordBindingModel but with an explicitly configured AblatableTensorBackbone."""

    def __init__(self, d=64, layers=2, heads=4, vocab=32, **bb_kwargs):
        super().__init__()
        self.type_emb = nn.Embedding(2, d)
        self.coord_proj = nn.Linear(3, d)
        self.value_emb = nn.Embedding(vocab, d)
        self.backbone = AblatableTensorBackbone(d, layers, heads, **bb_kwargs)
        self.head = nn.Linear(d, vocab)

    def forward(self, write_k, write_v, query_k):
        B, W = write_v.shape
        Q = query_k.size(1)
        dev = write_k.device
        wt = self.type_emb(torch.zeros(B, W, dtype=torch.long, device=dev))
        qt = self.type_emb(torch.ones(B, Q, dtype=torch.long, device=dev))
        write_toks = wt + self.coord_proj(write_k) + self.value_emb(write_v)
        query_toks = qt + self.coord_proj(query_k)
        x = torch.cat([write_toks, query_toks], dim=1)
        S = x.size(1)
        mask = torch.triu(torch.ones(S, S, device=dev), diagonal=1).bool()
        h = self.backbone(x, mask=mask)
        return self.head(h[:, W:, :])


# =====================================================================
# Train + eval one ablation config
# =====================================================================

def train_one(label, args, device, **bb_kwargs):
    torch.manual_seed(args.seed)
    model = AblatableModel(d=args.d, layers=args.layers, heads=args.heads,
                           vocab=args.vocab, **bb_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.time()
    for _ in range(args.steps):
        wk, wv, qk, tv = gen_batch(args.batch, args.n_writes, args.queries,
                                   args.vocab, args.noise_sigma, device)
        logits = model(wk, wv, qk)
        loss = F.cross_entropy(logits.flatten(0, 1), tv.flatten())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            wk, wv, qk, tv = gen_batch(args.batch, args.n_writes, args.queries,
                                       args.vocab, args.noise_sigma, device)
            pred = model(wk, wv, qk).argmax(-1)
            correct += (pred == tv).sum().item()
            total += tv.numel()
    return {
        'label': label,
        'final_acc': correct / total,
        'params': count_params(model),
        'wall_seconds': round(time.time() - t0, 1),
        'config': bb_kwargs,
        'final_gates': model.backbone.get_gates(),
    }


def main():
    p = argparse.ArgumentParser("Toy ablations on coord binding")
    p.add_argument("--n_writes", type=int, default=20)
    p.add_argument("--noise_sigma", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--outdir", default="./results_toy_ablations")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Toy ablations on coord binding (W={args.n_writes}, σ={args.noise_sigma}) | device={device}\n")

    # Ablation list: (label, kwargs)
    runs = [
        ("default (gauss/share/factorized/6^3/cs=1)",
            dict(write_mode='gaussian', share_coords=True, factorized=True, mem_shape=(6, 6, 6), chunk_size=1)),

        ("write=hard",
            dict(write_mode='hard', share_coords=True, factorized=True, mem_shape=(6, 6, 6), chunk_size=1)),

        ("coords=separate",
            dict(write_mode='gaussian', share_coords=False, factorized=True, mem_shape=(6, 6, 6), chunk_size=1)),

        ("factorized=off",
            dict(write_mode='gaussian', share_coords=True, factorized=False, mem_shape=(6, 6, 6), chunk_size=1)),

        ("mem_shape=4^3",
            dict(write_mode='gaussian', share_coords=True, factorized=True, mem_shape=(4, 4, 4), chunk_size=1)),

        ("mem_shape=8^3",
            dict(write_mode='gaussian', share_coords=True, factorized=True, mem_shape=(8, 8, 8), chunk_size=1)),

        ("chunk_size=2",
            dict(write_mode='gaussian', share_coords=True, factorized=True, mem_shape=(6, 6, 6), chunk_size=2)),

        ("chunk_size=4",
            dict(write_mode='gaussian', share_coords=True, factorized=True, mem_shape=(6, 6, 6), chunk_size=4)),
    ]

    results = []
    for label, cfg in runs:
        r = train_one(label, args, device, mem_channels=16, dropout=0.1, **cfg)
        gates = [round(g, 2) for g in r['final_gates']]
        print(f"  {label:<40s}  acc={r['final_acc']:.3f}  params={r['params']:,}  ({r['wall_seconds']}s)  gates={gates}")
        results.append(r)

    out = os.path.join(args.outdir, "toy_ablations.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")


if __name__ == '__main__':
    main()
