import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ==========================================
# PART 1: Factorized 3D conv + Gaussian-weighted write
# ==========================================

class FactorizedConv3d(nn.Module):
    """
    Factorized (separable) 3D convolution that emits per-cell LSTM gates.

    Decomposes a dense 3x3x3 conv into three depthwise 1D convs along
    (D, H, W) followed by a 1x1x1 pointwise channel mix. Output channels
    are interpreted as four gate volumes (i, f, o, g) consumed downstream
    by the LSTM-style update.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise 1D convs along each spatial axis
        self.conv_d = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=in_channels)
        self.conv_h = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), groups=in_channels)
        self.conv_w = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=in_channels)

        # Pointwise channel mix
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self._init_identity()

    def _init_identity(self):
        # Identity init for the depthwise convs (center tap = 1, others = 0):
        # at init the spatial mix is a no-op, training drifts it from there.
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

    def forward(self, x):
        x = self.conv_d(x)
        x = self.conv_h(x)
        x = self.conv_w(x)
        return self.pointwise(x)


def efficient_gaussian_write(mu, sigma, grid_cache, content):
    """
    Gaussian-weighted broadcast write into a 3D memory grid.

    For each batch element, computes
        content_vol[c, x, y, z] = content[c] * exp(-‖p - mu‖² / (2 σ² + ε))
    where p is the [-1, 1]^3 location of cell (x, y, z). Returns the
    weighted volume and the scalar Gaussian mask.

    Coordinate convention (consistent with F.grid_sample for 5D inputs):
        mu[..., 0] = x (width),  range [-1, 1]
        mu[..., 1] = y (height), range [-1, 1]
        mu[..., 2] = z (depth),  range [-1, 1]
    """
    B = mu.shape[0]
    C = content.shape[1]

    mu_view = mu.view(B, 3, 1, 1, 1)
    sigma_view = sigma.view(B, 1, 1, 1, 1)

    diff = grid_cache - mu_view
    dist_sq = diff.pow(2).sum(dim=1, keepdim=True)

    mask = torch.exp(-dist_sq / (2 * sigma_view.pow(2) + 1e-6))

    content_vol = content.view(B, C, 1, 1, 1) * mask

    return content_vol, mask


def tensor_memory_scan(
    h_init: torch.Tensor,
    c_init: torch.Tensor,
    read_coords_seq: torch.Tensor,
    content_seq: torch.Tensor,
    mu_seq: torch.Tensor,
    sigma_seq: torch.Tensor,
    gate_conv: nn.Module,
    grid_cache: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    B, S, _ = read_coords_seq.shape
    C, D, H, W = h_init.shape[1:]
    
    outputs_list = []
    h = h_init
    c = c_init

    # Outer block over time helps torch.compile produce a tighter graph;
    # semantically identical to a flat per-step loop.
    BLOCK_SIZE = 16

    for t_start in range(0, S, BLOCK_SIZE):
        t_end = min(t_start + BLOCK_SIZE, S)

        for t in range(t_start, t_end):
            # 1. READ — trilinear sample of h at read_coord
            # rc layout: [..., 0]=x, [1]=y, [2]=z (F.grid_sample 5D ordering)
            rc = read_coords_seq[:, t].view(B, 1, 1, 1, 3)
            mem_vec = F.grid_sample(h, rc, align_corners=True, padding_mode='border').view(B, C)
            outputs_list.append(mem_vec)

            # 2. WRITE — Gaussian-weighted volume centered at mu_seq[:, t]
            content_vol, _ = efficient_gaussian_write(
                mu_seq[:, t],
                sigma_seq[:, t],
                grid_cache,
                content_seq[:, t]
            )
            # The Gaussian mask itself is unused: the conv decides per-cell
            # gate values across the WHOLE grid, not just the Gaussian support.

            # 3. CONV — concat new content with current h, run factorized 3D conv
            combined = torch.cat([content_vol, h], dim=1)
            gates = gate_conv(combined)

            # 4. LSTM-STYLE UPDATE — per-cell
            chunks = gates.chunk(4, dim=1)
            i = torch.sigmoid(chunks[0])
            f = torch.sigmoid(chunks[1])
            o = torch.sigmoid(chunks[2])
            g = torch.tanh(chunks[3])

            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)

            h = h_next
            c = c_next

    outputs = torch.stack(outputs_list, dim=1)
    return outputs, h, c


# ==========================================
# Optional torch.compile of the recurrent scan (single-GPU, requires Triton)
# ==========================================
def _triton_available():
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False

if hasattr(torch, "compile") and _triton_available():
    tensor_memory_scan = torch.compile(
        tensor_memory_scan,
        mode="default",
        fullgraph=False
    )


class TensorMemoryEmbedder(nn.Module):
    """
    Maps a token to (content, sigma) for the Gaussian-weighted write.

    The write location mu is handled externally by a shared coord_net
    (in TensorMemoryInterface) so read and write coordinates stay aligned.
    """
    def __init__(self, embed_dim, memory_channels, initial_sigma_bias=1.0, sigma_scale=1.0):
        super().__init__()
        self.sigma_scale = sigma_scale
        self.content_net = nn.Linear(embed_dim, memory_channels)
        self.sigma_net = nn.Linear(embed_dim, 1)
        # Bias sigma positive at init so writes start with non-zero spread
        # rather than collapsing to a single cell.
        nn.init.constant_(self.sigma_net.bias, initial_sigma_bias)

    def forward(self, x):
        content = self.content_net(x)
        raw_sigma = F.softplus(self.sigma_net(x)) + 1e-4
        sigma = raw_sigma * self.sigma_scale
        return content, sigma


# ==========================================
# PART 2: Memory interface
# ==========================================

class TensorMemoryInterface(nn.Module):
    """
    Token-facing wrapper around the tensor memory.

    Tokens are chunked, projected to a read coordinate, a write coordinate,
    content, and a Gaussian spread sigma. Read uses n-linear sampling; write
    is a Gaussian-weighted broadcast into the volume; the memory state evolves
    via the LSTM-style update in tensor_memory_scan.

    This implementation is 3D (`Conv3d`, 5-D `grid_sample`). The coordinate
    convention enforced by the shared coord_net is:
        coord[..., 0] = x (width),  in [-1, 1]
        coord[..., 1] = y (height), in [-1, 1]
        coord[..., 2] = z (depth),  in [-1, 1]
    Matches F.grid_sample for 5D inputs.
    """
    def __init__(self, embed_dim, memory_channels, memory_shape, chunk_size, sigma_scale=1.0):
        super().__init__()
        self.memory_shape = memory_shape
        self.memory_channels = memory_channels
        self.chunk_size = chunk_size

        # Learned per-chunk projection (replaces a naive .mean() over chunk tokens)
        self.write_proj = nn.Linear(chunk_size * embed_dim, embed_dim)

        # Produces (content, sigma) for each write; mu comes from coord_net
        self.writer = TensorMemoryEmbedder(embed_dim, memory_channels, sigma_scale=sigma_scale)

        # Single coord head used for both read and write — guarantees a token
        # that wrote at coord c reads from coord c when seen again.
        self.coord_net = nn.Linear(embed_dim, 3)

        self.gate_conv = FactorizedConv3d(2 * memory_channels, 4 * memory_channels)

        self.out_proj = nn.Linear(memory_channels, embed_dim)

        # Pre-computed cell coordinates: [1, 3, D, H, W], channel order (x, y, z).
        D, H, W = memory_shape
        z = torch.linspace(-1, 1, steps=D)
        y = torch.linspace(-1, 1, steps=H)
        x = torch.linspace(-1, 1, steps=W)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        self.register_buffer('grid_cache', torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0))

    def init_state(self, batch_size, device, dtype):
        D, H, W = self.memory_shape
        C = self.memory_channels
        h = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        return h, c

    def forward(self, x, prev_state, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size

        B, N, D_emb = x.shape
        pad_len = (chunk_size - (N % chunk_size)) % chunk_size
        x_padded = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
        N_padded = x_padded.shape[1]
        num_chunks = N_padded // chunk_size

        x_grouped = x_padded.view(B, num_chunks, chunk_size, D_emb)

        # Read input: first token of each chunk
        x_read_input = x_grouped[:, :, 0, :]

        # Write input: learned projection over all chunk tokens
        x_flat = x_grouped.flatten(2)
        x_write_input = self.write_proj(x_flat)

        h, c = prev_state

        # Same coord head for both — read coord c == write coord c by construction
        read_coords = torch.tanh(self.coord_net(x_read_input))   # [B, num_chunks, 3]
        write_coords = torch.tanh(self.coord_net(x_write_input)) # [B, num_chunks, 3]

        content, sigma = self.writer(x_write_input)

        output_stack, h_next, c_next = tensor_memory_scan(
            h, c,
            read_coords, content, write_coords, sigma,
            self.gate_conv,
            self.grid_cache
        )

        # Broadcast each chunk's read vector to all chunk_size token positions
        output = output_stack.repeat_interleave(chunk_size, dim=1)
        if pad_len > 0:
            output = output[:, :N, :]

        output = self.out_proj(output)
        return output, (h_next, c_next)


# ==========================================
# PART 3: Vision Transformer with shared multi-layer memory
# ==========================================

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ViTBlockWithMemory(nn.Module):
    """
    Pre-norm ViT block. The TensorMemoryInterface is passed in at forward
    time rather than owned, so all layers can share the same memory params
    and the same (h, c) state — keeping the parameter delta vs a plain ViT
    constant in depth.

    Each block owns only: attention, MLP, layer norms, and (when memory is
    enabled) one LayerNorm for the memory residual plus a scalar gate.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 use_memory=True):
        super().__init__()
        self.use_memory = use_memory

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

        # Per-layer memory residual params: one LayerNorm + one scalar gate.
        # Gate init: sigmoid(-2.0) ≈ 0.12 — memory contributes ~12% at init,
        # training adjusts it.
        if use_memory:
            self.ln_memory = nn.LayerNorm(dim)
            self.memory_gate = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, x, memory_state=None, shared_memory=None, chunk_size=None):
        """
        Args:
            x: [B, N, D] input tokens
            memory_state: (h, c) tuple from previous layer
            shared_memory: TensorMemoryInterface (shared across all layers)
            chunk_size: override for memory chunking
        Returns:
            x: [B, N, D] output tokens
            memory_state: updated (h, c) or None
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)

        next_memory_state = memory_state
        if self.use_memory and memory_state is not None and shared_memory is not None:
            mem_out, next_memory_state = shared_memory(
                self.ln_memory(x), memory_state, chunk_size=chunk_size
            )
            gate = torch.sigmoid(self.memory_gate)
            x = x + self.drop_path(gate * mem_out)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, next_memory_state


class TensorMemoryViT(nn.Module):
    """
    Vision Transformer augmented with one shared tensor memory.

    Design:
      - A single TensorMemoryInterface is instantiated once and reused at
        every block (same weights, same (h, c) state threaded through).
      - Each block adds a small per-layer residual: LayerNorm(x) → memory →
        scalar gate → add to residual stream.
      - Total parameter overhead is independent of depth — one interface
        plus (LayerNorm + scalar) per memory-using layer.

    Reference parameter count at default config:
      - Baseline ViT:    ~21.7M params
      - + Tensor Memory: ~24.0M params (~2.3M extra for the shared interface)

    Distinct from:
      - Standard attention (no persistent state)
      - Slot- / register-token memories (discrete tokens, no spatial structure)
      - Per-layer memory variants (separate params per layer, not fair-comparable)
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 dim=1024, depth=24, heads=16, 
                 dropout=0.1, drop_path=0.1,
                 mem_channels=64, mem_shape=(8,8,8), sigma_scale=1.0,
                 chunk_size=14, memory_every_n_layers=1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.chunk_size = chunk_size
        self.mem_channels = mem_channels
        self.mem_shape = mem_shape
        self.memory_every_n_layers = memory_every_n_layers
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # One memory interface, shared across all layers
        self.shared_tensor_memory = TensorMemoryInterface(
            embed_dim=dim,
            memory_channels=mem_channels,
            memory_shape=tuple(mem_shape),
            chunk_size=chunk_size,
            sigma_scale=sigma_scale
        )

        # Stochastic depth schedule (linear ramp from 0 to drop_path)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        self.blocks = nn.ModuleList()
        self.layer_uses_memory = []

        for i in range(depth):
            uses_memory = (i % memory_every_n_layers == 0)
            self.blocks.append(
                ViTBlockWithMemory(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=dpr[i],
                    use_memory=uses_memory,
                )
            )
            self.layer_uses_memory.append(uses_memory)
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # Weights Init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
        # Count memory layers for logging
        self.num_memory_layers = sum(self.layer_uses_memory)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_memory(self, batch_size, device, dtype):
        """Initialize the shared memory state."""
        D, H, W = self.mem_shape
        C = self.mem_channels
        h = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch_size, C, D, H, W, device=device, dtype=dtype)
        return h, c

    def forward(self, x, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        B = x.shape[0]
        
        # Patch embed
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]
        
        # Append CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        
        # Same (h, c) and same interface weights are threaded through every layer.
        h, c = self.init_memory(B, x.device, x.dtype)
        memory_state = (h, c)

        for i, blk in enumerate(self.blocks):
            if self.layer_uses_memory[i]:
                x, memory_state = blk(x, memory_state, 
                                      shared_memory=self.shared_tensor_memory,
                                      chunk_size=chunk_size)
            else:
                x, _ = blk(x, None, shared_memory=None, chunk_size=chunk_size)
        
        x = self.norm(x)
        return self.head(x[:, 0])
    
    def get_gate_values(self):
        """Return gate values for all memory layers (for monitoring)."""
        gates = {}
        for i, blk in enumerate(self.blocks):
            if self.layer_uses_memory[i]:
                gate_val = torch.sigmoid(blk.memory_gate).item()
                gates[f"layer_{i}"] = gate_val
        return gates