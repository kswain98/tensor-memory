# Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers

<p align="center">
  <a href="#">Project Page</a> &nbsp;·&nbsp;
  <a href="#">Paper</a> &nbsp;·&nbsp;
  <a href="#">arXiv</a>
</p>

A transformer / ViT augmented with a **fixed-size memory**. Tokens project to continuous `[-1, 1]ⁿ` coordinates; writes are Gaussian-weighted broadcasts into the grid, reads are n-linear samples. The store has fixed capacity (`C` channels × grid resolution) regardless of sequence length, evolves through LSTM-style updates driven by a factorized n-D conv, and is shared across all transformer layers — so memory persists across both token positions and depth.

The reference implementation here uses **n = 3** (a `D × H × W` grid), but the design is dimension-agnostic — only the conv rank and `grid_sample` rank need to change.

The mechanism is described in [`tensor_memory/tensor_memory.md`](tensor_memory/tensor_memory.md). The reference implementation is in [`tensor_memory/model.py`](tensor_memory/model.py).

## Results

<!--
Drop the figures into ./assets/ and replace this block with:

<p align="center">
  <img src="assets/lm_perplexity.png"  alt="WikiText-2 perplexity"  width="48%">
  <img src="assets/toy_summary.png"    alt="Toy diagnostics"        width="48%">
</p>
-->

_Figures coming soon._

## Install

```
pip install -r requirements.txt
```

## Quick start

Run the fastest toy diagnostic end-to-end on synthetic data — all four backbones, ~1 minute on a single GPU:

```
$ python tensor_memory/toy/no_harm.py --methods base,base_wide,slots,tensor --steps 200
```

That's the whole loop: generate data, train, eval. The same scripts run all the toy baselines below — only the `--methods` flag changes.

## Language modeling

Train and benchmark all eight methods on WikiText-2 with the default config:

```
$ python tensor_memory/text/train_text.py --dataset wikitext2
```

Single method or subset:

```
$ python tensor_memory/text/train_text.py --dataset wikitext2 --method tensor
$ python tensor_memory/text/train_text.py --dataset wikitext2 --method base,tensor
```

Same model code, same hyperparameters, different long-context mechanism:

| `--method`     | mechanism                                                       | KV memory       |
|----------------|-----------------------------------------------------------------|-----------------|
| `base`         | Full causal attention Transformer                               | O(L)            |
| `base_sln`     | Base + post-attention LayerNorm (SubLN)                         | O(L)            |
| `local`        | Sliding-window attention (W=64)                                 | O(W)            |
| `xl`           | Transformer-XL — segment recurrence + relative position bias    | O(W + M)        |
| `linear`       | Linear attention (Performer-style ELU+1 kernel)                 | O(1)            |
| `tensor`       | **Base + Tensor Memory (this work)**                            | O(C·D·H·W)      |
| `tensor_sln`   | Tensor + SubLN ablation                                         | O(C·D·H·W)      |
| `tensor_local` | Tensor + windowed attention ablation                            | O(W + C·D·H·W)  |

Char-level Shakespeare is also supported:

```
$ python tensor_memory/text/train_text.py --dataset shakespeare
```

Pass `--benchmark` to also report prefill throughput and peak GPU memory. Pass `--wandb` to log to Weights & Biases.

## Image

MAE-style patch reconstruction on CUB-200-2011:

```
$ accelerate launch tensor_memory/image/train_image.py --model tm --data_dir /path/to/cub200
```

[`tensor_memory/image/visualize.py`](tensor_memory/image/visualize.py) renders memory-state figures from a trained checkpoint.

## Video

Action recognition on UCF-101 (decode frames once with [`extract_frames.py`](tensor_memory/video/extract_frames.py)):

```
$ accelerate launch tensor_memory/video/train_video.py --data_dir /path/to/ucf101 --model tm
```

## Toy diagnostics

Five synthetic diagnostics covering occlusion, long-horizon spatial reasoning, coordinate binding, no-harm gate stability, and design ablations — together they probe the paper's claims one at a time:

```
$ python tensor_memory/toy/<name>.py
```

where `<name>` ∈ {`occlusion`, `map_building`, `coord_binding`, `no_harm`, `ablations`}. Each takes about 5 minutes on a single GPU.

## Datasets

- **Toys**: synthetic, generated on the fly — no preparation step.
- **WikiText-2 / Shakespeare**: auto-downloaded on first run (via HuggingFace `datasets` or a `wget`/`curl` fallback).
- **CUB-200-2011 / UCF-101**: download manually from the respective project pages.

## Configuration

Every script declares its config at the top of the file, overridable via `--key value` on the command line. The headline knobs:

| flag                       | default                                  | meaning                                                |
|----------------------------|------------------------------------------|--------------------------------------------------------|
| `--mem_channels`           | `16` (toy) / `32` (img/vid/text)         | channels per cell `C`                                  |
| `--mem_shape`              | `6 6 6`                                  | grid resolution `(D, H, W)`                            |
| `--chunk_size`             | `1` (toy) / `16` (img/vid/text)          | tokens per memory step                                 |
| `--sigma_scale`            | `1.0`                                    | cap on the writer's spread σ                           |
| `--memory_every_n_layers`  | `1`                                      | sparsity of layers that touch memory                   |
| `--methods` / `--method` / `--model` | varies                         | which backbone to train                                |
| `--steps` / `--epochs`     | varies                                   | training length                                        |
| `--seed`                   | `42`                                     | random seed                                            |

The full list lives at the top of each train script.

## Citation

```bibtex
@misc{swain2026tensormemory,
      title={Tensor Memory: Fixed-Size Recurrent State for Long-Horizon Transformers},
      author={Kabir Swain and Sijie Han and Daniel Karl I. Weidele and Mauro Martino and Antonio Torralba},
      year={2026},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/XXXX.XXXXX},
}
```

## Acknowledgements

We would like to thank Manel Baradad ([@mbaradad](https://github.com/mbaradad)) and Minyoung Huh ([@minyoungg](https://github.com/minyoungg)) for their helpful advice and discussion.
