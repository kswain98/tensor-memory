"""Toy 3: Coordinate Binding — paper §4.1.

Setup:
    Each example is a stream of WRITE tokens followed by QUERY tokens.
    - WRITE(k, v): k ∈ [-1, 1]^3 (continuous coordinate), v ∈ {0, ..., V-1}
    - QUERY(k_noisy):  predict the value of the *nearest* written key.

We sweep number of writes and Gaussian noise applied to query coordinates.
Tensor Memory should retain accuracy as #writes grows because it stores
information at continuous coordinates by construction; attention-based
baselines are forced to do soft KV lookup over a long token list.

Run:
    python tensor_memory/toy/coord_binding.py
    python tensor_memory/toy/coord_binding.py --methods base,tensor --n_writes 10,50 --steps 2000
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import make_backbone, count_params


WRITE_TOK, QUERY_TOK = 0, 1


def gen_batch(batch_size, n_writes, n_queries, vocab, noise_sigma, device):
    """Generate (write_k, write_v, query_k, target_v).

    Targets are the value of the *nearest* write key to the (noisy) query —
    this gives a deterministic best answer even at high noise.
    """
    write_k = torch.rand(batch_size, n_writes, 3, device=device) * 2 - 1
    write_v = torch.randint(0, vocab, (batch_size, n_writes), device=device)

    qidx = torch.randint(0, n_writes, (batch_size, n_queries), device=device)
    chosen_k = torch.gather(write_k, 1, qidx.unsqueeze(-1).expand(-1, -1, 3))
    query_k = (chosen_k + torch.randn_like(chosen_k) * noise_sigma).clamp(-1, 1)

    dists = torch.cdist(query_k, write_k)
    nearest = dists.argmin(dim=-1)
    target_v = torch.gather(write_v, 1, nearest)
    return write_k, write_v, query_k, target_v


class CoordBindingModel(nn.Module):
    """Tokens are WRITE(k,v) then QUERY(k). Backbone runs over the joint stream;
    the head is applied only to the query positions to predict v."""

    def __init__(self, backbone, d=64, layers=2, heads=4, vocab=32, **bb_kwargs):
        super().__init__()
        self.d = d
        self.vocab = vocab
        self.type_emb = nn.Embedding(2, d)
        self.coord_proj = nn.Linear(3, d)
        self.value_emb = nn.Embedding(vocab, d)
        self.backbone = make_backbone(backbone, d, layers, heads, **bb_kwargs)
        self.head = nn.Linear(d, vocab)

    def forward(self, write_k, write_v, query_k):
        B, W = write_v.shape
        Q = query_k.size(1)
        dev = write_k.device

        wt = self.type_emb(torch.zeros(B, W, dtype=torch.long, device=dev))
        wc = self.coord_proj(write_k)
        wv = self.value_emb(write_v)
        write_toks = wt + wc + wv

        qt = self.type_emb(torch.ones(B, Q, dtype=torch.long, device=dev))
        qc = self.coord_proj(query_k)
        query_toks = qt + qc

        x = torch.cat([write_toks, query_toks], dim=1)
        S = x.size(1)
        mask = torch.triu(torch.ones(S, S, device=dev), diagonal=1).bool()
        h = self.backbone(x, mask=mask)
        return self.head(h[:, W:, :])


def train_one(method, n_writes, noise_sigma, args, device):
    torch.manual_seed(args.seed)
    model = CoordBindingModel(
        method, d=args.d, layers=args.layers, heads=args.heads, vocab=args.vocab,
        mem_channels=args.mem_channels, mem_shape=tuple(args.mem_shape),
        chunk_size=1, n_slots=args.n_slots, dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = count_params(model)
    n_queries = args.queries

    train_curve = []
    t0 = time.time()
    for step in range(args.steps):
        wk, wv, qk, tv = gen_batch(args.batch, n_writes, n_queries, args.vocab, noise_sigma, device)
        logits = model(wk, wv, qk)
        loss = F.cross_entropy(logits.flatten(0, 1), tv.flatten())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == tv).float().mean().item()
            train_curve.append({'step': step, 'loss': loss.item(), 'acc': acc})

    # Final eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            wk, wv, qk, tv = gen_batch(args.batch, n_writes, n_queries, args.vocab, noise_sigma, device)
            pred = model(wk, wv, qk).argmax(-1)
            correct += (pred == tv).sum().item()
            total += tv.numel()
    final_acc = correct / total

    res = {
        'method': method,
        'n_writes': n_writes,
        'noise_sigma': noise_sigma,
        'final_acc': final_acc,
        'params': n_params,
        'wall_seconds': round(time.time() - t0, 1),
        'train_curve': train_curve,
    }
    if hasattr(model.backbone, 'get_gates'):
        res['final_gates'] = model.backbone.get_gates()
    return res


def main():
    p = argparse.ArgumentParser("Toy 3: Coordinate Binding")
    p.add_argument("--methods", default="base,base_wide,slots,tensor")
    p.add_argument("--n_writes", default="5,10,20,50")
    p.add_argument("--noise_sigmas", default="0.0,0.05,0.1")
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
    p.add_argument("--mem_channels", type=int, default=16)
    p.add_argument("--mem_shape", type=int, nargs=3, default=[6, 6, 6])
    p.add_argument("--n_slots", type=int, default=8)
    p.add_argument("--outdir", default="./results_toy3")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    writes = [int(x) for x in args.n_writes.split(',')]
    noises = [float(x) for x in args.noise_sigmas.split(',')]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Toy 3: Coordinate Binding | device={device}")
    print(f"Methods : {methods}")
    print(f"Writes  : {writes}")
    print(f"Noise σ : {noises}")
    print(f"Steps   : {args.steps}, batch={args.batch}, queries/example={args.queries}")
    print()

    results = []
    for m in methods:
        for nw in writes:
            for ns in noises:
                r = train_one(m, nw, ns, args, device)
                gate_str = ""
                if 'final_gates' in r:
                    gate_str = f"  gates={[round(g,2) for g in r['final_gates']]}"
                print(f"  [{m:<10}] W={nw:<2d} σ={ns:.2f}  acc={r['final_acc']:.3f}  ({r['wall_seconds']}s){gate_str}")
                results.append(r)

    out = os.path.join(args.outdir, "toy3_coord_binding.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out}")

    # Console summary table: rows methods x noise, cols writes
    print("\n=== Final accuracy ===")
    header = "  method      σ      " + "  ".join(f"W={w:<3d}" for w in writes)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in methods:
        for ns in noises:
            row = f"  {m:<10}  {ns:.2f}  "
            for nw in writes:
                r = next(x for x in results if x['method'] == m and x['n_writes'] == nw and x['noise_sigma'] == ns)
                row += f"{r['final_acc']:.3f}   "
            print(row)


if __name__ == '__main__':
    main()
