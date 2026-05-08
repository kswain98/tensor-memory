#!/usr/bin/env python3
# V2-B: Fast stream processor writes to a tensor memory.
# Transformer operates ONLY on learned latent tokens that interact with memory.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class FactorizedConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_d = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1),
                                padding=(1, 0, 0), groups=in_channels)
        self.conv_h = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1),
                                padding=(0, 1, 0), groups=in_channels)
        self.conv_w = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3),
                                padding=(0, 0, 1), groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self._init_identity()

    def _init_identity(self):
        for m in [self.conv_d, self.conv_h, self.conv_w]:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)
            if m.kernel_size == (3, 1, 1):
                m.weight.data[:, 0, 1, 0, 0] = 1.0
            elif m.kernel_size == (1, 3, 1):
                m.weight.data[:, 0, 0, 1, 0] = 1.0
            elif m.kernel_size == (1, 1, 3):
                m.weight.data[:, 0, 0, 0, 1] = 1.0
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_d(x)
        x = self.conv_h(x)
        x = self.conv_w(x)
        return self.pointwise(x)


def gaussian_splat(mu, sigma, grid_cache, content):
    B = mu.shape[0]
    C = content.shape[1]
    mu_view = mu.view(B, 3, 1, 1, 1)
    sigma_view = sigma.view(B, 1, 1, 1, 1)
    diff = grid_cache - mu_view
    dist_sq = diff.pow(2).sum(dim=1, keepdim=True)
    mask = torch.exp(-dist_sq / (2.0 * sigma_view.pow(2) + 1e-6))
    return content.view(B, C, 1, 1, 1) * mask


def memory_read(h, coords):
    B, Cmem = h.shape[0], h.shape[1]
    rc = coords.view(B, 1, 1, 1, 3)
    out = F.grid_sample(h, rc, align_corners=True, padding_mode="border")
    return out.view(B, Cmem)


def memory_read_sequence(h, coords_seq):
    B, Cmem = h.shape[0], h.shape[1]
    S = coords_seq.shape[1]
    rc = coords_seq.view(B, S, 1, 1, 3)
    out = F.grid_sample(h, rc, align_corners=True, padding_mode="border")
    out = out.squeeze(-1).squeeze(-1)
    return out.transpose(1, 2).contiguous()


def memory_step(h, c, read_coord, write_coord, content, sigma, gate_conv, grid_cache):
    read_vec = memory_read(h, read_coord)
    content_vol = gaussian_splat(write_coord, sigma, grid_cache, content)
    combined = torch.cat([content_vol, h], dim=1)
    gates = gate_conv(combined)
    i, f, o, g = gates.chunk(4, dim=1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    o = torch.sigmoid(o)
    g = torch.tanh(g)
    c_next = f * c + i * g
    h_next = o * torch.tanh(c_next)
    return read_vec, h_next, c_next


def memory_scan(h, c, read_coords, write_coords, content_seq, sigma_seq,
                gate_conv, grid_cache, block_size=16):
    B, S, _ = read_coords.shape
    reads = []
    for t0 in range(0, S, block_size):
        t1 = min(t0 + block_size, S)
        for t in range(t0, t1):
            rv, h, c = memory_step(
                h, c, read_coords[:, t], write_coords[:, t],
                content_seq[:, t], sigma_seq[:, t], gate_conv, grid_cache)
            reads.append(rv)
    reads = torch.stack(reads, dim=1)
    return reads, h, c


class TensorMemoryEmbedder(nn.Module):
    def __init__(self, d_model, mem_channels, sigma_scale=1.0, initial_sigma_bias=1.0):
        super().__init__()
        self.sigma_scale = sigma_scale
        self.content_net = nn.Linear(d_model, mem_channels)
        self.sigma_net = nn.Linear(d_model, 1)
        nn.init.constant_(self.sigma_net.bias, initial_sigma_bias)

    def forward(self, x):
        content = self.content_net(x)
        sigma = (F.softplus(self.sigma_net(x)) + 1e-4) * self.sigma_scale
        return content, sigma


class MemoryController(nn.Module):
    def __init__(self, d_model, mem_channels, sigma_scale=1.0):
        super().__init__()
        self.coord_net = nn.Linear(d_model, 3)
        self.writer = TensorMemoryEmbedder(d_model, mem_channels, sigma_scale=sigma_scale)

    def forward(self, x):
        coords = torch.tanh(self.coord_net(x))
        content, sigma = self.writer(x)
        return coords, coords, content, sigma


class SDPAAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, causal=False):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class LatentMemoryInteraction(nn.Module):
    def __init__(self, d_model, mem_channels, dropout=0.0):
        super().__init__()
        self.out_proj = nn.Linear(mem_channels, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, latents, mem_state, controller, gate_conv, grid_cache, block_size=16):
        h, c = mem_state
        read_coords, write_coords, content, sigma = controller(latents)
        mem_reads, h, c = memory_scan(
            h, c, read_coords=read_coords, write_coords=write_coords,
            content_seq=content, sigma_seq=sigma, gate_conv=gate_conv,
            grid_cache=grid_cache, block_size=block_size)
        mem_out = self.drop(self.out_proj(mem_reads))
        return mem_out, (h, c)


class LatentTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mem_channels, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SDPAAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln_mem = nn.LayerNorm(d_model)
        self.mem = LatentMemoryInteraction(d_model, mem_channels, dropout=dropout)
        self.mem_gate = nn.Parameter(torch.tensor([-2.0]))
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, latents, mem_state, controller, gate_conv, grid_cache):
        latents = latents + self.drop1(self.attn(self.ln1(latents), causal=False))
        mem_out, mem_state = self.mem(
            self.ln_mem(latents), mem_state, controller, gate_conv, grid_cache)
        g = torch.sigmoid(self.mem_gate)
        latents = latents + g * mem_out
        latents = latents + self.mlp(self.ln2(latents))
        return latents, mem_state


class FastStreamGRU(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward_sequence(self, x):
        y, _ = self.gru(x)
        return self.drop(y)

    def step(self, x_t, h_prev):
        h = self.cell(x_t, h_prev)
        return self.drop(h)


class TensorMemoryAugmentedTransformerV2B(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, depth, num_latents,
                 mem_channels, mem_shape, max_len=2048, dropout=0.1,
                 sigma_scale=1.0, reasoner_every_n_steps=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mem_channels = mem_channels
        self.mem_shape = tuple(mem_shape)
        self.max_len = max_len
        self.reasoner_every_n_steps = max(1, int(reasoner_every_n_steps))

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.fast = FastStreamGRU(d_model, dropout=dropout)
        self.fast_ln = nn.LayerNorm(d_model)
        self.controller = MemoryController(d_model, mem_channels, sigma_scale=sigma_scale)
        self.gate_conv = FactorizedConv3d(2 * mem_channels, 4 * mem_channels)

        Dz, Hy, Wx = self.mem_shape
        z = torch.linspace(-1, 1, steps=Dz)
        y = torch.linspace(-1, 1, steps=Hy)
        x = torch.linspace(-1, 1, steps=Wx)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        grid_cache = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
        self.register_buffer("grid_cache", grid_cache)

        self.latent_tokens = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.latent_ln0 = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            LatentTransformerBlock(d_model, nhead, mem_channels, dropout=dropout)
            for _ in range(depth)])
        self.latent_ln = nn.LayerNorm(d_model)
        self.readout_proj = nn.Linear(mem_channels, d_model)
        self.readout_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        nn.init.zeros_(self.pos_embed)

    def init_memory(self, batch_size, device, dtype):
        Dz, Hy, Wx = self.mem_shape
        C = self.mem_channels
        h = torch.zeros(batch_size, C, Dz, Hy, Wx, device=device, dtype=dtype)
        c = torch.zeros(batch_size, C, Dz, Hy, Wx, device=device, dtype=dtype)
        return h, c

    def forward(self, input_ids, mem_state=None, run_reasoner=True):
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embed[:, :S, :]
        fast_feat = self.fast_ln(self.fast.forward_sequence(x))
        read_coords, write_coords, content_seq, sigma_seq = self.controller(fast_feat)

        if mem_state is None:
            mem_state = self.init_memory(B, input_ids.device, fast_feat.dtype)
        h, c = mem_state
        _, h, c = memory_scan(
            h, c, read_coords=read_coords, write_coords=write_coords,
            content_seq=content_seq, sigma_seq=sigma_seq,
            gate_conv=self.gate_conv, grid_cache=self.grid_cache, block_size=16)
        mem_state = (h, c)

        if run_reasoner:
            latents = self.latent_ln0(self.latent_tokens.expand(B, -1, -1))
            for blk in self.blocks:
                latents, mem_state = blk(
                    latents, mem_state, self.controller, self.gate_conv, self.grid_cache)
            latents = self.latent_ln(latents)

        h, _ = mem_state
        mem_vecs = memory_read_sequence(h, read_coords)
        y = self.readout_ln(self.readout_proj(mem_vecs))
        logits = self.lm_head(y)
        return logits, mem_state