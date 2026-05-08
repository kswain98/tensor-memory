#!/usr/bin/env python3
# V2-A: Fast token encoder (GRU as stand-in for SSM) processes the input stream,
#       then the Transformer ONLY operates on memory-derived chunk tokens.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


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


def efficient_gaussian_write(mu, sigma, grid_cache, content):
    B = mu.shape[0]
    C = content.shape[1]
    mu_view = mu.view(B, 3, 1, 1, 1)
    sigma_view = sigma.view(B, 1, 1, 1, 1)
    diff = grid_cache - mu_view
    dist_sq = diff.pow(2).sum(dim=1, keepdim=True)
    mask = torch.exp(-dist_sq / (2.0 * sigma_view.pow(2) + 1e-6))
    content_vol = content.view(B, C, 1, 1, 1) * mask
    return content_vol, mask


def tensor_memory_scan(h_init, c_init, read_coords_seq, content_seq,
                           mu_seq, sigma_seq, gate_conv, grid_cache):
    B, S, _ = read_coords_seq.shape
    C, D, H, W = h_init.shape[1:]
    outputs = []
    h = h_init
    c = c_init
    BLOCK_SIZE = 16
    for t0 in range(0, S, BLOCK_SIZE):
        t1 = min(t0 + BLOCK_SIZE, S)
        for t in range(t0, t1):
            rc = read_coords_seq[:, t].view(B, 1, 1, 1, 3)
            mem_vec = F.grid_sample(h, rc, align_corners=True, padding_mode="border").view(B, C)
            outputs.append(mem_vec)
            content_vol, _ = efficient_gaussian_write(
                mu_seq[:, t], sigma_seq[:, t], grid_cache, content_seq[:, t])
            combined = torch.cat([content_vol, h], dim=1)
            gates = gate_conv(combined)
            i, f, o, g = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
    outputs = torch.stack(outputs, dim=1)
    return outputs, h, c


class TensorMemoryEmbedder(nn.Module):
    def __init__(self, embed_dim, memory_channels, initial_sigma_bias=1.0, sigma_scale=1.0):
        super().__init__()
        self.sigma_scale = sigma_scale
        self.content_net = nn.Linear(embed_dim, memory_channels)
        self.sigma_net = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.sigma_net.bias, initial_sigma_bias)

    def forward(self, x):
        content = self.content_net(x)
        raw_sigma = F.softplus(self.sigma_net(x)) + 1e-4
        sigma = raw_sigma * self.sigma_scale
        return content, sigma


class TensorMemoryInterface(nn.Module):
    def __init__(self, embed_dim, memory_channels, memory_shape, chunk_size, sigma_scale=1.0):
        super().__init__()
        self.memory_shape = tuple(memory_shape)
        self.memory_channels = memory_channels
        self.chunk_size = chunk_size
        self.write_proj = nn.Linear(chunk_size * embed_dim, embed_dim)
        self.writer = TensorMemoryEmbedder(embed_dim, memory_channels, sigma_scale=sigma_scale)
        self.coord_net = nn.Linear(embed_dim, 3)
        self.gate_conv = FactorizedConv3d(2 * memory_channels, 4 * memory_channels)
        self.out_proj = nn.Linear(memory_channels, embed_dim)

        D, H, W = self.memory_shape
        z = torch.linspace(-1, 1, steps=D)
        y = torch.linspace(-1, 1, steps=H)
        x = torch.linspace(-1, 1, steps=W)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
        self.register_buffer("grid_cache", grid)

    def init_state(self, batch_size, device, dtype):
        D, H, W = self.memory_shape
        C = self.memory_channels
        h = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        return h, c

    def _chunk(self, x, chunk_size):
        B, N, D = x.shape
        pad_len = (chunk_size - (N % chunk_size)) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        Np = x.shape[1]
        num_chunks = Np // chunk_size
        x_grouped = x.view(B, num_chunks, chunk_size, D)
        return x_grouped, pad_len, N, num_chunks

    def forward_chunks(self, x, prev_state, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        x_grouped, pad_len, N, num_chunks = self._chunk(x, chunk_size)
        B, Nc, Cs, D = x_grouped.shape
        x_read_input = x_grouped[:, :, 0, :]
        x_flat = x_grouped.flatten(2)
        x_write_input = self.write_proj(x_flat)
        h, c = prev_state
        read_coords = torch.tanh(self.coord_net(x_read_input))
        write_coords = torch.tanh(self.coord_net(x_write_input))
        content, sigma = self.writer(x_write_input)
        chunk_mem, h_next, c_next = tensor_memory_scan(
            h, c, read_coords, content, write_coords, sigma,
            self.gate_conv, self.grid_cache)
        chunk_out = self.out_proj(chunk_mem)
        return chunk_out, (h_next, c_next), pad_len

    def forward(self, x, prev_state, chunk_size=None):
        chunk_out, next_state, pad_len = self.forward_chunks(x, prev_state, chunk_size=chunk_size)
        B, Nc, D = chunk_out.shape
        Cs = self.chunk_size if chunk_size is None else chunk_size
        tok_out = chunk_out.unsqueeze(2).expand(-1, -1, Cs, -1).reshape(B, Nc * Cs, D)
        if pad_len > 0:
            tok_out = tok_out[:, :-pad_len, :]
        return tok_out, next_state


class FlashSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class MemoryTokenTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FlashSelfAttention(d_model, nhead, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, z):
        z = z + self.drop(self.attn(self.ln1(z)))
        z = z + self.mlp(self.ln2(z))
        return z


class FastTokenEncoderGRU(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_size=d_model, hidden_size=d_model,
                          num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.drop(y)


class TensorMemoryAugmentedTransformerV2A(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_mem_layers, mem_channels,
                 mem_shape, max_len=1024, dropout=0.1, chunk_size=16, sigma_scale=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.fast = FastTokenEncoderGRU(d_model, dropout=dropout)
        self.fast_ln = nn.LayerNorm(d_model)
        self.memory = TensorMemoryInterface(
            embed_dim=d_model, memory_channels=mem_channels,
            memory_shape=mem_shape, chunk_size=chunk_size, sigma_scale=sigma_scale)
        self.mem_in_ln = nn.LayerNorm(d_model)
        self.mem_blocks = nn.ModuleList([
            MemoryTokenTransformerBlock(d_model, nhead, dropout=dropout)
            for _ in range(num_mem_layers)])
        self.mem_out_ln = nn.LayerNorm(d_model)
        self.fuse_gate = nn.Parameter(torch.tensor([-2.0]))
        self.fuse_ln = nn.LayerNorm(d_model)
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        nn.init.zeros_(self.pos_encoder)

    def init_memory(self, batch_size, device, dtype):
        return self.memory.init_state(batch_size, device, dtype)

    def forward(self, x_ids, memory_state=None):
        B, S = x_ids.shape
        x = self.embedding(x_ids) + self.pos_encoder[:, :S, :]
        h_tok = self.fast_ln(self.fast(x))
        if memory_state is None:
            memory_state = self.init_memory(B, x_ids.device, h_tok.dtype)
        m_chunk, memory_state, pad_len = self.memory.forward_chunks(
            self.mem_in_ln(h_tok), memory_state, chunk_size=self.chunk_size)
        z = m_chunk
        for blk in self.mem_blocks:
            z = blk(z)
        z = self.mem_out_ln(z)
        Nc = z.shape[1]
        Cs = self.chunk_size
        z_up = z.unsqueeze(2).expand(-1, -1, Cs, -1).reshape(B, Nc * Cs, self.d_model)
        if pad_len > 0:
            z_up = z_up[:, :-pad_len, :]
        z_up = z_up[:, :S, :]
        g = torch.sigmoid(self.fuse_gate)
        y = h_tok + g * z_up
        y = self.fuse_ln(y)
        logits = self.head(self.final_ln(y))
        return logits, memory_state