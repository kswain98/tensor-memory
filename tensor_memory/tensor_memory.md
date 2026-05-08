# Tensor Memory — Design

This document describes the mechanism implemented in [model.py](model.py). It is a
reading guide for the file: read it alongside the source and the section markers
will line up.

## What it is

A fixed-size tensor memory module that augments a Transformer with a
structured, coordinate-addressed read/write store. Tokens are projected to
continuous coordinates in `[-1, 1]^n`, written into the grid via a
Gaussian-weighted broadcast, and read back with n-linear sampling. State
evolves between writes through a factorized n-D convolution that emits
LSTM-style gates per cell.

**Dimensionality.** The reference implementation in [`model.py`](model.py)
uses `n = 3` (a `D × H × W` grid). Nothing about the method requires 3D —
only the conv rank (`Conv3d` → `Conv1d`/`Conv2d`/etc.), the `grid_sample`
rank, and the `grid_cache` shape would need to change. The choice of 3D
here is a particular instantiation, not a defining feature.

This is *not* 3D Gaussian Splatting in the rendering sense — there are no
anisotropic Gaussians, no rasterization, no learned opacities. The "Gaussian"
here is a scalar isotropic weight applied to a single content vector when
broadcasting it into the volume.

The intended inductive bias is complementary to attention:

| Attention | Tensor Memory |
|---|---|
| Soft, content-addressed lookup over a token list | Continuous coordinate addressing into a fixed n-D grid |
| O(N²) per layer in tokens | O(N) per layer in tokens; cost is in grid updates, independent of N |
| No persistent state | Persistent `(h, c)` state across the sequence and across layers |

The store has fixed capacity `C × D × H × W` regardless of sequence length.

## Memory state

Two tensors of shape `[B, C, D, H, W]`:

- `h` — hidden state (the readable contents of the grid)
- `c` — cell state (LSTM-style accumulator, not read directly)

`init_state` ([model.py:236](model.py:236)) returns zeros.

## Coordinate convention

A single convention is enforced everywhere so reads and writes line up with
`F.grid_sample`'s expected ordering for 5D tensors:

```
coord[..., 0] = x  (width)   in [-1, 1]
coord[..., 1] = y  (height)  in [-1, 1]
coord[..., 2] = z  (depth)   in [-1, 1]
```

The grid cache registered as a buffer ([model.py:234](model.py:234)) is built
in this order: `torch.stack([grid_x, grid_y, grid_z])`. Both the Gaussian write
and the trilinear read consume coordinates in this `(x, y, z)` form.

## Per-step recurrence

[model.py:85](model.py:85) implements one step of the recurrence for a single
token position `t`. Given `(h, c)` and the token's `(read_coord, write_coord,
content, sigma)`:

1. **Read.** `mem_vec = F.grid_sample(h, read_coord)` → `[B, C]`. Trilinear,
   border-padded, differentiable in both `h` and `read_coord`.
2. **Write.** Gaussian-weighted broadcast of `content` centered at
   `write_coord` with spread `sigma`, producing a `[B, C, D, H, W]` volume:
   ```
   content_vol[b, :, x, y, z] = content[b] · exp(-‖p - mu‖² / (2 σ² + ε))
   ```
   where `p` is the cell's `[-1, 1]^3` location.
3. **Conv.** Concatenate the content volume with `h` along the channel axis
   and pass through the `FactorizedConv3d` 3D conv. It outputs `4·C`
   channels — four gate volumes `(i, f, o, g)`.
4. **Update.** Standard LSTM update applied per cell:
   ```
   c ← f ⊙ c + i ⊙ g
   h ← o ⊙ tanh(c)
   ```

The scan loops `t = 0 … S-1` with a manual outer block (`BLOCK_SIZE = 16`,
[model.py:105](model.py:105)) that helps `torch.compile` produce a tighter
graph; semantically it is identical to a flat per-step loop.

The whole function is optionally wrapped in `torch.compile` if Triton is
available ([model.py:158](model.py:158)).

## Factorized 3D conv (FactorizedConv3d)

[model.py:10](model.py:10) `FactorizedConv3d` decomposes a dense `3×3×3` 3D
conv into three depthwise 1D convs along D, H, W followed by a `1×1×1`
pointwise mix:

```
in_channels = 2 · C   (content_vol concat with h)
out_channels = 4 · C  (four LSTM gate volumes)

x → conv_d (3,1,1) groups=in   # 3-tap depthwise mix along depth
  → conv_h (1,3,1) groups=in   # 3-tap depthwise mix along height
  → conv_w (1,1,3) groups=in   # 3-tap depthwise mix along width
  → pointwise (1,1,1)          # mix channels, project to 4C
```

Init choices in `_init_identity` ([model.py:31](model.py:31)):

- The depthwise convs are initialized as identity (center weight = 1, others
  = 0). At init the spatial component is a no-op, so the module starts as a
  pure pointwise channel mixer.
- The pointwise weights get standard Xavier init; bias = 0.

This keeps the gate distributions well-behaved at init: 3-tap spatial mixing
across neighbors is introduced gradually as training begins rather than
dominating from step 1.

Cost vs. a dense `3³` 3D conv: parameter count drops from `27·C_in·C_out` to
`9·C_in + C_in·C_out`; arithmetic complexity drops similarly.

## Gaussian write

[model.py:55](model.py:55) `efficient_gaussian_write`. Given:

- `mu` : `[B, 3]` — write center in `[-1, 1]^3`
- `sigma` : `[B, 1]` — isotropic spread
- `content` : `[B, C]` — what to write
- `grid_cache` : `[1, 3, D, H, W]` — pre-computed cell coordinates

Returns a content volume `[B, C, D, H, W]` and a Gaussian mask `[B, 1, D, H,
W]`. The mask is returned for diagnostics; the recurrent scan ignores it
because the conv decides per-cell gate values across the *whole* grid
(not just the local Gaussian support).

`TensorMemoryEmbedder` ([model.py:166](model.py:166)) produces `(content, sigma)`
from a token via two linears. `sigma` is `softplus(·) + ε` and biased
positive at init (`initial_sigma_bias=1.0`) to avoid collapse to a point
write before the model has anything to spread.

## Token interface

`TensorMemoryInterface.forward` ([model.py:243](model.py:243)) is the public
entry point used by ViT blocks. Given input tokens `[B, N, D_emb]` and a
prior `(h, c)` state:

1. **Chunk.** Right-pad `N` to a multiple of `chunk_size`, then reshape into
   `[B, num_chunks, chunk_size, D_emb]`. `chunk_size` controls the granularity
   of the recurrence — one memory step per chunk, not per token. Larger
   chunk sizes amortize the per-step overhead at the cost of coarser
   addressing.
2. **Read input.** First token of each chunk → `[B, num_chunks, D_emb]`.
3. **Write input.** All `chunk_size · D_emb` flattened features per chunk →
   `[B, num_chunks, chunk_size · D_emb]` → `write_proj` linear →
   `[B, num_chunks, D_emb]`. A learned projection (not `.mean()`).
4. **Coordinates.** A single `coord_net: D_emb → 3` produces both read and
   write coordinates, run on the read input and write input respectively, then
   passed through `tanh` to land in `[-1, 1]^3`.
5. **Content / sigma.** `TensorMemoryEmbedder` consumes the write input.
6. **Recurrent scan** over the `num_chunks` axis returns the per-chunk read
   vectors `[B, num_chunks, C]` and the new `(h, c)`.
7. **Broadcast.** Each chunk's read vector is broadcast across the
   `chunk_size` original token positions, then trimmed to `N`. This
   `out_proj`-projected output is what the consumer adds to its residual
   stream.

### Why share `coord_net` between read and write

Reads and writes reference the same conceptual locations in the volume — the
"slot" you wrote to is the slot you should later read from. Using one
projection enforces this alignment by construction: a token mapped to write
coord `c` will, if presented again, also map to read coord `c`. Two separate
heads can only learn this consistency through training pressure and rarely do
exactly.

## ViT integration

[model.py:306](model.py:306) `ViTBlockWithMemory` is a standard pre-norm
transformer block (attention + MLP) with one extra step between attention and
MLP: the block calls a *shared* `TensorMemoryInterface` it does not own,
applies a per-layer LayerNorm to the input, and adds `sigmoid(memory_gate) ·
mem_out` to the residual.

[model.py:369](model.py:369) `TensorMemoryViT` instantiates **one** memory
interface and passes it to every block. The same `(h, c)` is threaded through
the layer stack — each layer reads from / writes to the same store. This is
deliberate, for two reasons:

1. **Parameter parity.** A baseline ViT compared against this model is only
   bigger by one shared interface (~`mem_channels · D_emb` projection params
   plus a single `FactorizedConv3d` instance), not by `depth ×` that.
2. **Memory persistence as a layer-spanning channel.** The store accumulates
   evidence across layers, not just across token positions within one layer.

Per-layer gates start at `sigmoid(-2.0) ≈ 0.12` ([model.py:339](model.py:339))
so memory contributes a small fraction at init; training adjusts it.
`memory_every_n_layers` lets you skip memory in some layers — disabled layers
still run attention and MLP, they just don't read or write.

## Hyperparameter intuition

| Knob | Effect |
|---|---|
| `mem_channels` (`C`) | Width of each cell. More channels = richer per-location codes, quadratic cost in pointwise conv. |
| `mem_shape` (`D, H, W`) | Spatial resolution. Doubling each dim costs 8× the volume. |
| `chunk_size` | Tokens per memory step. Higher = faster but coarser; the read result is broadcast back to all `chunk_size` positions identically. |
| `sigma_scale` | Multiplier on the writer's softplus output. Caps how spread-out a single write can be relative to the grid. |
| `memory_every_n_layers` | Sparsity of layers that touch memory. `1` = every layer. |

## What the model does not do

- It does not learn the grid resolution; `mem_shape` is fixed.
- It does not address cells by index — only by continuous `[-1, 1]^3`
  coordinate, with a soft Gaussian write footprint.
- It does not use attention internally. The only "lookup" is `grid_sample`,
  and the only spatial mixing is the 3-tap depthwise 3D conv.
- It does not implement 3D Gaussian Splatting (Kerbl et al. 2023).
  No anisotropic Gaussians, no rasterizer, no learned opacity — just an
  isotropic-Gaussian-weighted broadcast write into a fixed memory grid.
