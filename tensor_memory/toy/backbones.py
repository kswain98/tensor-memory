"""Backbones used by all four toy diagnostics.

Each backbone takes [B, N, d] tokens (already embedded) and an optional causal
mask, and returns [B, N, d] processed tokens. Toy scripts add their own
embedding layers and readout heads.

Available backbones (paper §4.1 baselines):
    base        - Standard pre-norm Transformer
    base_wide   - Same depth/heads but wider MLP, parameter-matched to tensor
    slots       - Transformer + N learnable global slots (read/write via attention)
    tensor      - Transformer + Tensor Memory (ours)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import TensorMemoryInterface


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: Attn -> MLP, both with residual."""

    def __init__(self, d, heads, dropout=0.1, mlp_mult=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        h = int(d * mlp_mult)
        self.mlp = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.mlp(self.ln2(x))
        return x


class BaseBackbone(nn.Module):
    def __init__(self, d, layers, heads, dropout=0.1, mlp_mult=4.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, heads, dropout, mlp_mult) for _ in range(layers)]
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln(x)


class SlotsBackbone(nn.Module):
    """Transformer + persistent learnable slots.

    Each layer:
        1. Token self-attention + MLP (standard transformer block).
        2. Tokens read from slots via cross-attention.
        3. Slots updated by attending back to tokens.

    `n_slots` slots persist across layers within one forward pass.
    """

    def __init__(self, d, layers, heads, n_slots=8, dropout=0.1, mlp_mult=4.0):
        super().__init__()
        self.n_slots = n_slots
        self.slot_init = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)

        self.token_blocks = nn.ModuleList(
            [TransformerBlock(d, heads, dropout, mlp_mult) for _ in range(layers)]
        )
        self.read_attn = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(layers)]
        )
        self.write_attn = nn.ModuleList(
            [nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in range(layers)]
        )
        self.ln_q_tokens = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_kv_slots = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_q_slots = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_kv_tokens = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B = x.size(0)
        slots = self.slot_init.expand(B, -1, -1).contiguous()
        for i, blk in enumerate(self.token_blocks):
            x = blk(x, mask)
            # Tokens read slots
            q = self.ln_q_tokens[i](x)
            kv = self.ln_kv_slots[i](slots)
            r, _ = self.read_attn[i](q, kv, kv, need_weights=False)
            x = x + self.drop(r)
            # Slots updated by tokens
            q = self.ln_q_slots[i](slots)
            kv = self.ln_kv_tokens[i](x)
            u, _ = self.write_attn[i](q, kv, kv, need_weights=False)
            slots = slots + self.drop(u)
        return self.ln(x)


class TensorBackbone(nn.Module):
    """Transformer + Tensor Memory (ours).

    A single TensorMemoryInterface is shared across layers; the (h, c) state
    is threaded through the stack. Each layer reads a memory readout, fuses it
    via a learned scalar gate (initialized so sigma(gate) = 0.5).
    """

    def __init__(self, d, layers, heads, mem_channels=16, mem_shape=(6, 6, 6),
                 chunk_size=1, dropout=0.1, sigma_scale=1.0, mlp_mult=4.0):
        super().__init__()
        self.token_blocks = nn.ModuleList(
            [TransformerBlock(d, heads, dropout, mlp_mult) for _ in range(layers)]
        )
        self.memory = TensorMemoryInterface(
            embed_dim=d, memory_channels=mem_channels, memory_shape=mem_shape,
            chunk_size=chunk_size, sigma_scale=sigma_scale,
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


def make_backbone(name, d, layers, heads, **kwargs):
    """Factory. `kwargs` forwarded to the chosen backbone.

    For 'base_wide', the MLP multiplier is bumped so the parameter count
    roughly matches 'tensor' at the default tensor-memory size.
    """
    dropout = kwargs.get('dropout', 0.1)
    if name == 'base':
        return BaseBackbone(d, layers, heads, dropout=dropout)
    if name == 'base_wide':
        # ~+5-10% params vs base, comparable to tensor's overhead
        return BaseBackbone(d, layers, heads, dropout=dropout, mlp_mult=4.5)
    if name == 'slots':
        return SlotsBackbone(d, layers, heads, n_slots=kwargs.get('n_slots', 8), dropout=dropout)
    if name == 'tensor':
        return TensorBackbone(
            d, layers, heads,
            mem_channels=kwargs.get('mem_channels', 16),
            mem_shape=tuple(kwargs.get('mem_shape', (6, 6, 6))),
            chunk_size=kwargs.get('chunk_size', 1),
            sigma_scale=kwargs.get('sigma_scale', 1.0),
            dropout=dropout,
        )
    raise ValueError(f"Unknown backbone: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
