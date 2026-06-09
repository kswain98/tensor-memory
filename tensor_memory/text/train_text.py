#!/usr/bin/env python3
"""
Language modeling experiments for Tensor Memory.

Datasets:
  - wikitext2    : Word-level WikiText-2 (pre-tokenized) — standard small LM benchmark

Baselines (paper Table 1):
  - base         : Full causal attention Transformer
  - base_sln     : Base + post-attention LayerNorm ("Base + SLN" in the paper)
  - local        : Sliding window attention (window=64) + absolute position embeddings
  - local_rope   : Sliding window attention (window=64) + RoPE (LLaMA/Mistral style)
  - xl           : Transformer-XL (segment recurrence, relative position bias)
  - linear       : Linear attention (ELU+1 kernel, simplified Performer)

Ours:
  - tensor       : Tensor Memory — add-on module on full-attention Transformer
                   (fixed-size memory with factorized 3D conv update, applied at every layer)
  - tensor_sln   : Ablation — tensor + SubLN
  - tensor_local : Ablation — tensor with windowed attention instead of full

NOTE: linear is a simplified re-implementation (ELU+1 kernel) capturing the
  core architectural principle. It is NOT the official Performer FAVOR+
  implementation. Comparisons reflect architectural principles, not
  engineering-optimized throughput.

Usage:
  # WikiText-2 run (default config)
  python tensor_memory/text/train_text.py --dataset wikitext2

  # Single method for debugging
  python tensor_memory/text/train_text.py --dataset wikitext2 --method tensor --seq_lens 512

  # Add-on ablation: base vs base+memory
  python tensor_memory/text/train_text.py --dataset wikitext2 --method base,tensor --seq_lens 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import json
import time
import zipfile
import argparse
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Weights & Biases (optional — graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def wandb_enabled() -> bool:
    """Check if wandb is both available and was requested."""
    return WANDB_AVAILABLE and wandb is not None and wandb.run is not None

# ---------------------------------------------------------------------------
# §1  DATASET LOADERS
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, '?') for i in ids)


class WordTokenizer:
    """Simple word-level tokenizer with vocabulary cap."""
    def __init__(self, text: str, max_vocab: int = 30000):
        from collections import Counter
        words = text.split()
        counts = Counter(words)
        
        self.special = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        most_common = counts.most_common(max_vocab - len(self.special))
        
        self.stoi = dict(self.special)
        for w, _ in most_common:
            self.stoi[w] = len(self.stoi)
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        self.unk_id = self.special['<unk>']
        self.eos_id = self.special['<eos>']
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, self.unk_id) for w in text.split()]
    
    def decode(self, ids: List[int]) -> str:
        return ' '.join(self.itos.get(i, '<unk>') for i in ids)


def load_shakespeare(data_dir: str = "./data") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    """Load Shakespeare char-level data from local Folger corpus. Split: 80/10/10.

    Tries (in order):
      1. Local Folger corpus: shakespeare-dataset-main/text/ relative to repo root
      2. data_dir/shakespeare-dataset-main/text/
      3. Fallback download of tiny-shakespeare from Karpathy's char-rnn repo
    """
    # Locate the Folger corpus (37 plays, ~5.4 MB)
    _script_root = Path(__file__).resolve().parent.parent.parent
    _candidates = [
        _script_root / "shakespeare-dataset-main" / "text",
        Path(data_dir) / "shakespeare-dataset-main" / "text",
        Path("shakespeare-dataset-main") / "text",
    ]
    folger_dir = next((d for d in _candidates if d.is_dir()), None)

    if folger_dir is not None:
        txt_files = sorted(folger_dir.glob("*.txt"))
        texts = [f.read_text(encoding="utf-8", errors="replace") for f in txt_files]
        text = "\n\n".join(texts)
        print(f"  Loaded Folger Shakespeare: {len(txt_files)} plays, {len(text):,} chars  ({folger_dir})")
    else:
        # Fallback: download tiny-shakespeare
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "shakespeare.txt")
        if not os.path.exists(path):
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"  Folger corpus not found — downloading tiny-Shakespeare from {url}...")
            urllib.request.urlretrieve(url, path)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"  Loaded tiny-Shakespeare: {len(text):,} chars")

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n_train = int(0.8 * len(data))
    n_val   = int(0.1 * len(data))

    train_data = data[:n_train]
    val_data   = data[n_train:n_train + n_val]
    test_data  = data[n_train + n_val:]

    print(f"  vocab={tok.vocab_size}  train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}")
    return train_data, val_data, test_data, tok.vocab_size, "char"


def load_enwik8(data_dir: str = "./data") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    """Load enwik8 char-level data. Standard split: 90M/5M/5M."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "enwik8")
    
    if not os.path.exists(path):
        zip_path = os.path.join(data_dir, "enwik8.zip")
        if not os.path.exists(zip_path):
            import urllib.request
            url = "http://mattmahoney.net/dc/enwik8.zip"
            print(f"  Downloading enwik8 from {url}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                print(f"  Primary download failed ({e}), trying alternative...")
                url = "https://data.deepai.org/enwik8.zip"
                urllib.request.urlretrieve(url, zip_path)
        
        print("  Extracting enwik8...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
    
    with open(path, 'rb') as f:
        raw = f.read()
    
    data = torch.tensor(list(raw), dtype=torch.long)
    train_data = data[:90_000_000]
    val_data = data[90_000_000:95_000_000]
    test_data = data[95_000_000:100_000_000]
    vocab_size = 256
    
    print(f"  enwik8: {len(data):,} bytes, train={len(train_data):,}, "
          f"val={len(val_data):,}, test={len(test_data):,}")
    return train_data, val_data, test_data, vocab_size, "char"


def load_wikitext(version: str = "2", data_dir: str = "./data") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    """Load WikiText-2 or WikiText-103 word-level data (pre-tokenized)."""
    os.makedirs(data_dir, exist_ok=True)
    
    train_text = val_text = test_text = None
    try:
        from datasets import load_dataset
        # Use pre-tokenized version (standard for published results)
        name = "wikitext-2-v1" if version == "2" else "wikitext-103-v1"
        print(f"  Loading WikiText-{version} (pre-tokenized) via HuggingFace datasets...")
        ds = load_dataset("wikitext", name, cache_dir=data_dir)
        train_text = "\n".join(ds["train"]["text"])
        val_text = "\n".join(ds["validation"]["text"])
        test_text = "\n".join(ds["test"]["text"])
    except ImportError:
        print("  HuggingFace datasets not available, downloading raw files...")
        train_text, val_text, test_text = _download_wikitext_raw(version, data_dir)
    except Exception as e:
        print(f"  HuggingFace download failed ({e}), trying raw files...")
        train_text, val_text, test_text = _download_wikitext_raw(version, data_dir)
    
    max_vocab = 30000 if version == "2" else 50000
    tok = WordTokenizer(train_text, max_vocab=max_vocab)
    
    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    test_data = torch.tensor(tok.encode(test_text), dtype=torch.long)
    
    print(f"  WikiText-{version}: train={len(train_data):,}, val={len(val_data):,}, "
          f"test={len(test_data):,}, vocab={tok.vocab_size}")
    return train_data, val_data, test_data, tok.vocab_size, "word"


def _download_wikitext_raw(version: str, data_dir: str) -> Tuple[str, str, str]:
    """Fallback: load WikiText from local files or download via wget/curl."""
    wt_dir = os.path.join(data_dir, f"wikitext-{version}")
    
    # Check if files already exist (e.g. pre-downloaded by bash script)
    all_exist = all(
        os.path.exists(os.path.join(wt_dir, f"wiki.{s}.tokens"))
        for s in ["train", "valid", "test"]
    )
    
    if not all_exist:
        os.makedirs(wt_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, f"wikitext-{version}-v1.zip")
        s3_url = f"https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-{version}-v1.zip"
        
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            # Python's urlretrieve doesn't follow 301 redirects properly,
            # so use wget or curl (which do) via subprocess
            import subprocess
            downloaded = False
            for cmd in [
                ["wget", "-q", "--no-check-certificate", "-O", zip_path, s3_url],
                ["curl", "-sL", "-o", zip_path, s3_url],
            ]:
                try:
                    print(f"    Trying: {' '.join(cmd[:3])}... {s3_url}")
                    subprocess.check_call(cmd, timeout=120)
                    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000:
                        downloaded = True
                        break
                except Exception as e:
                    print(f"    Failed: {e}")
            
            if not downloaded:
                raise RuntimeError(
                    f"Could not download WikiText-{version}.\n"
                    f"Please either:\n"
                    f"  1. pip install datasets\n"
                    f"  2. Manually: wget -O {zip_path} {s3_url} && unzip {zip_path} -d {data_dir}\n"
                )
        
        print(f"    Extracting wikitext-{version}-v1.zip...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
    
    texts = {}
    for split in ["train", "valid", "test"]:
        path = os.path.join(wt_dir, f"wiki.{split}.tokens")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path} — ls {wt_dir}: {os.listdir(wt_dir)}")
        with open(path) as f:
            texts[split] = f.read()
    
    return texts["train"], texts["valid"], texts["test"]


def load_dataset_by_name(name: str, data_dir: str = "./data"):
    """Unified dataset loader. Returns (train, val, test, vocab_size, level)."""
    loaders = {
        "wikitext2":   lambda: load_wikitext("2", data_dir),
        "shakespeare": lambda: load_shakespeare(data_dir),
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")

    return loaders[name]()


# ---------------------------------------------------------------------------
# §2  BATCHING
# ---------------------------------------------------------------------------

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    """Sample random contiguous chunks for language modeling."""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Data too short ({len(data)}) for seq_len={seq_len}")
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


class SequentialBatchIterator:
    """
    Yields consecutive, non-overlapping segments for Transformer-XL training.

    Splits data into `batch_size` parallel streams (like standard LM batching),
    then yields seq_len-sized chunks that advance sequentially. This ensures
    cached memory from segment N is relevant to segment N+1.

    Usage:
        it = SequentialBatchIterator(train_data, batch_size=32, seq_len=256, device=device)
        for step in range(steps):
            x, y = it.next_batch()
    """
    def __init__(self, data: torch.Tensor, batch_size: int, seq_len: int,
                 device: torch.device):
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size

        # Truncate data to fit evenly into batch_size streams
        n_tokens = (len(data) - 1) // batch_size * batch_size
        self.data = data[:n_tokens + 1]  # +1 for targets
        self.stream_len = n_tokens // batch_size

        # Reshape into [batch_size, stream_len] parallel streams
        self.streams = self.data[:n_tokens].view(batch_size, self.stream_len)
        self.targets = self.data[1:n_tokens + 1].view(batch_size, self.stream_len)

        self.cursor = 0
        self.wrapped = False

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x, y) of shape [batch_size, seq_len].
        Wraps around when data is exhausted. Sets self.wrapped = True on wrap.
        """
        self.wrapped = False
        if self.cursor + self.seq_len > self.stream_len:
            self.cursor = 0  # wrap around
            self.wrapped = True

        x = self.streams[:, self.cursor:self.cursor + self.seq_len].to(self.device)
        y = self.targets[:, self.cursor:self.cursor + self.seq_len].to(self.device)
        self.cursor += self.seq_len
        return x, y

    def reset(self):
        self.cursor = 0


# ---------------------------------------------------------------------------
# §3  MODEL DEFINITIONS
# ---------------------------------------------------------------------------

class MiniTransformer(nn.Module):
    """Standard causal Transformer baseline.

    Optional `sub_ln` adds a post-attention LayerNorm (the "SubLN" variant used
    in modern LLMs). Reported in the paper as "Base + SLN".

    When sub_ln=True:  Pre-LN → Attn → SubLN → Residual  (matches modern LLMs)
    When sub_ln=False: Pre-LN → Attn → Residual           (classic pre-norm transformer)
    """
    def __init__(self, vocab, d=128, heads=4, layers=4, seq_len=256, 
                 window=None, dropout=0.1, tie_weights=True, sub_ln=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        self.blocks = nn.ModuleList([self._block(d, heads, dropout, sub_ln) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if tie_weights:
            self.head.weight = self.tok.weight
        self.window = window
        self.d = d
        self.heads = heads
        self.seq_len = seq_len
        self.sub_ln = sub_ln
        
        # Dropout for attention output residual and embeddings
        self.resid_drop = nn.Dropout(dropout)
        self.embed_drop = nn.Dropout(dropout)
        
    def _block(self, d, heads, dropout, sub_ln=False):
        blk = nn.ModuleDict({
            'ln1': nn.LayerNorm(d),
            'attn': nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True),
            'ln2': nn.LayerNorm(d),
            'mlp': nn.Sequential(
                nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(4*d, d), nn.Dropout(dropout)
            )
        })
        if sub_ln:
            # Post-attention LayerNorm (SubLN, as in modern LLMs).
            blk['attn_ln'] = nn.LayerNorm(d)
        return blk
    
    def _get_mask(self, S, device):
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        if self.window:
            # Vectorized: mask out positions more than window steps back
            row_idx = torch.arange(S, device=device).unsqueeze(1)
            col_idx = torch.arange(S, device=device).unsqueeze(0)
            too_far = (row_idx - col_idx) >= self.window
            mask = mask | too_far
        return mask
    
    def forward(self, x):
        B, S = x.shape
        h = self.embed_drop(self.tok(x) + self.pos[:, :S])
        mask = self._get_mask(S, x.device)
        
        for blk in self.blocks:
            h2 = blk['ln1'](h)
            h2, _ = blk['attn'](h2, h2, h2, attn_mask=mask, need_weights=False)
            if 'attn_ln' in blk:
                h2 = blk['attn_ln'](h2)
            h = h + self.resid_drop(h2)
            h = h + blk['mlp'](blk['ln2'](h))
        
        return self.head(self.ln(h))


class TensorTransformer(MiniTransformer):
    """Transformer + Tensor Memory v1 (ours).
    
    Pure add-on module: takes a standard causal Transformer and adds a gated
    tensor memory residual at each layer. The fixed-size memory provides a
    complementary inductive bias to attention — structured coordinate-based
    read/write instead of soft content-based lookup.
    
    Key design choices for the add-on story:
      - Base transformer is UNCHANGED (full causal attention, no window restriction)
      - Fresh memory each forward call (within-sequence enrichment)
      - Trained with same random batching as base (no special training regime)
      - Gate initialized at 0.0 (sigmoid=0.5) so memory contributes from the start
    """
    def __init__(self, vocab, d=128, heads=4, layers=4, seq_len=256, 
                 window=None, mem_channels=32, mem_shape=(8,8,8), 
                 chunk_size=16, dropout=0.1, sub_ln=False, **kwargs):
        # window=None: full attention by default (pure add-on, don't cripple base)
        super().__init__(vocab, d, heads, layers, seq_len, window, dropout, sub_ln=sub_ln)
        from model import TensorMemoryInterface
        self.memory = TensorMemoryInterface(d, mem_channels, mem_shape, chunk_size=chunk_size)
        self.chunk_size = chunk_size
        
        self.mem_lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor([0.0]))  # sigmoid(0)=0.5, memory active from start
            for _ in range(layers)
        ])
        self.mem_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, S = x.shape
        h = self.embed_drop(self.tok(x) + self.pos[:, :S])
        mask = self._get_mask(S, x.device)
        
        # Fresh memory each call — within-sequence processing only
        mem = self.memory.init_state(B, h.device, h.dtype)
        
        for i, blk in enumerate(self.blocks):
            h2 = blk['ln1'](h)
            h2, _ = blk['attn'](h2, h2, h2, attn_mask=mask, need_weights=False)
            if 'attn_ln' in blk:
                h2 = blk['attn_ln'](h2)
            h = h + self.resid_drop(h2)
            h = h + blk['mlp'](blk['ln2'](h))
            
            m_out, mem = self.memory(self.mem_lns[i](h), mem)
            h = h + torch.sigmoid(self.gates[i]) * self.mem_drop(m_out)
        
        return self.head(self.ln(h))
    
    def get_gates(self):
        return [torch.sigmoid(g).item() for g in self.gates]


# ---------------------------------------------------------------------------
# §3b  REMAINING BASELINES
# ---------------------------------------------------------------------------

class TransformerXL(nn.Module):
    """
    Transformer-XL with segment-level recurrence and relative position bias.

    Key design (matches Dai et al., 2019):
      - NO absolute positional embeddings — uses learned relative position bias
        added to attention logits so cached memory tokens get correct distances.
      - Pre-norm on query only; cached memory stores raw hidden states so they
        are NOT double-normalised when reused as keys/values.
      - Memory is detached between segments (no BPTT across segments).

    IMPORTANT: Requires sequential (not random) batching to benefit from
    segment recurrence. See SequentialBatchIterator.
    
    NOTE: This model already had correct residual dropout (self.resid_drop)
    and embedding dropout in the original. No fix needed here.
    """
    def __init__(self, vocab, d=128, heads=4, layers=4, seq_len=256,
                 window=None, mem_len=256, dropout=0.1, **kwargs):
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.num_layers = layers
        self.mem_len = mem_len
        self.seq_len = seq_len

        self.tok = nn.Embedding(vocab, d)
        # No absolute pos embed — relative bias handles positions

        self.blocks = nn.ModuleList()
        self.qkv_projs = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d),
                'ln2': nn.LayerNorm(d),
                'mlp': nn.Sequential(
                    nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(4 * d, d), nn.Dropout(dropout),
                ),
            }))
            self.qkv_projs.append(nn.Linear(d, 3 * d, bias=False))
            self.out_projs.append(nn.Linear(d, d, bias=False))

        # Learned relative position bias per head: covers distances
        # from 0 to mem_len + seq_len.  rel_bias[head, distance].
        max_dist = mem_len + seq_len
        self.rel_bias = nn.Parameter(torch.zeros(heads, max_dist))
        nn.init.normal_(self.rel_bias, std=0.02)

        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # FIX: Add embedding dropout (was missing)
        self.embed_drop = nn.Dropout(dropout)
        self._memories = None

    def init_memory(self, batch_size, device):
        """Zero-init per-layer memory caches."""
        self._memories = [
            torch.zeros(batch_size, 0, self.d, device=device)
            for _ in range(self.num_layers)
        ]

    def _rel_attn(self, q, k, v, layer_idx, S, M, device):
        """
        Relative-position-biased attention via SDPA (fused softmax + matmul).
          q: [B, H, S, Dh]   (current segment queries)
          k: [B, H, M+S, Dh] (memory + current keys)
          v: [B, H, M+S, Dh]
        """
        total = M + S
        qi_pos = torch.arange(M, M + S, device=device)        # [S]
        kj_pos = torch.arange(0, total, device=device)         # [M+S]
        dist = (qi_pos.unsqueeze(1) - kj_pos.unsqueeze(0)).clamp(0, self.rel_bias.size(1) - 1)

        # [1, H, S, M+S] additive bias — causal -inf baked in
        bias = self.rel_bias[:, dist].unsqueeze(0)             # [1, H, S, M+S]
        causal = torch.arange(total, device=device).unsqueeze(0) > \
                 (torch.arange(S, device=device) + M).unsqueeze(1)  # [S, M+S]
        bias = bias.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        scale = self.head_dim ** -0.5
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)

    def forward(self, x, use_memory=True):
        B, S = x.shape
        # FIX: Apply embedding dropout
        h = self.embed_drop(self.tok(x))  # [B, S, D]  — no absolute pos embed
        new_memories = []

        for i, blk in enumerate(self.blocks):
            # --- cache raw hidden states BEFORE layernorm ---
            new_memories.append(h.detach())

            # Memory: raw (un-normalised) hidden states from previous segment
            if use_memory and self._memories is not None and self._memories[i].size(1) > 0:
                mem = self._memories[i]
                M = mem.size(1)
            else:
                mem = None
                M = 0

            # Pre-norm on current segment only
            h_normed = blk['ln1'](h)

            # Project Q from current, K/V from [mem ; current]
            # Mem is raw (not normalised) — avoids double-norm
            if mem is not None:
                mem_normed = blk['ln1'](mem)
                kv_input = torch.cat([mem_normed, h_normed], dim=1)
            else:
                kv_input = h_normed

            qkv_q = self.qkv_projs[i](h_normed)
            qkv_kv = self.qkv_projs[i](kv_input)

            q = qkv_q[:, :, :self.d]
            k = qkv_kv[:, :, self.d:2*self.d]
            v = qkv_kv[:, :, 2*self.d:]

            q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(B, M + S, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(B, M + S, self.heads, self.head_dim).transpose(1, 2)

            attn_out = self._rel_attn(q, k, v, i, S, M, x.device)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d)
            attn_out = self.resid_drop(self.out_projs[i](attn_out))

            h = h + attn_out
            h = h + blk['mlp'](blk['ln2'](h))

        # Update memory caches (truncate to mem_len)
        if use_memory:
            updated = []
            for i, new_m in enumerate(new_memories):
                if self._memories is not None and self._memories[i].size(1) > 0:
                    combined = torch.cat([self._memories[i], new_m], dim=1)
                else:
                    combined = new_m
                updated.append(combined[:, -self.mem_len:].detach())
            self._memories = updated

        return self.head(self.ln(h))


class LinearAttentionTransformer(MiniTransformer):
    """Linear attention (Performer-style ELU kernel) — O(n) complexity."""
    def __init__(self, vocab, d=128, heads=4, layers=4, seq_len=256, dropout=0.1, sub_ln=False, **kwargs):
        super().__init__(vocab, d, heads, layers, seq_len, None, dropout, sub_ln=sub_ln)
        self.head_dim = d // heads
        self.attn_drop = nn.Dropout(dropout)
    
    def _elu_feature_map(self, x):
        return F.elu(x) + 1
    
    def forward(self, x):
        B, S = x.shape
        h = self.embed_drop(self.tok(x) + self.pos[:, :S])
        
        for blk in self.blocks:
            h2 = blk['ln1'](h)
            
            qkv = blk['attn'].in_proj_weight
            bias = blk['attn'].in_proj_bias
            proj = F.linear(h2, qkv, bias)
            q, k, v = proj.chunk(3, dim=-1)
            
            q = q.view(B, S, self.heads, self.head_dim)
            k = k.view(B, S, self.heads, self.head_dim)
            v = v.view(B, S, self.heads, self.head_dim)
            
            q = self._elu_feature_map(q)
            k = self._elu_feature_map(k)
            
            kv = torch.einsum('bshd,bshm->bshdm', k, v)
            kv = torch.cumsum(kv, dim=1)
            k_sum = torch.cumsum(k, dim=1)
            
            num = torch.einsum('bshd,bshdm->bshm', q, kv)
            den = torch.einsum('bshd,bshd->bsh', q, k_sum).unsqueeze(-1) + 1e-6
            
            out = (num / den).reshape(B, S, -1)
            out = blk['attn'].out_proj(out)
            if 'attn_ln' in blk:
                out = blk['attn_ln'](out)
            h = h + self.resid_drop(out)
            h = h + blk['mlp'](blk['ln2'](h))
        
        return self.head(self.ln(h))


class RoPETransformer(MiniTransformer):
    """Sliding-window Transformer with Rotary Position Embeddings (RoPE).

    Replaces absolute position embeddings with RoPE applied to Q and K,
    matching the position encoding used in LLaMA / Mistral.  Supports the
    same sliding-window mask as `local` so the two are directly comparable.
    Uses F.scaled_dot_product_attention (FlashAttention when is_causal=True).
    """
    def __init__(self, vocab, d=128, heads=4, layers=4, seq_len=256,
                 window=None, dropout=0.1, tie_weights=True, sub_ln=False, **kwargs):
        super().__init__(vocab, d, heads, layers, seq_len, window, dropout, tie_weights, sub_ln)
        del self.pos                      # no absolute position embedding
        self.head_dim = d // heads
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, freqs)     # [seq_len, head_dim//2]
        self.register_buffer('rope_cos', freqs.cos())
        self.register_buffer('rope_sin', freqs.sin())

    def _apply_rope(self, x, S):
        """x: [B, H, S, Dh] — rotate Q or K with RoPE frequencies."""
        cos = self.rope_cos[:S]           # [S, Dh//2]
        sin = self.rope_sin[:S]
        x1, x2 = x[..., :self.head_dim // 2], x[..., self.head_dim // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        B, S = x.shape
        h = self.embed_drop(self.tok(x))  # no absolute pos embed

        if self.window is None:
            sdpa_mask, sdpa_causal = None, True
        else:
            bool_mask = self._get_mask(S, x.device)   # [S, S] True=blocked
            sdpa_mask = torch.zeros(S, S, device=x.device, dtype=torch.float32)
            sdpa_mask.masked_fill_(bool_mask, float('-inf'))
            sdpa_causal = False

        for blk in self.blocks:
            h2 = blk['ln1'](h)
            proj = F.linear(h2, blk['attn'].in_proj_weight, blk['attn'].in_proj_bias)
            q, k, v = proj.chunk(3, dim=-1)

            q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)

            q = self._apply_rope(q, S)
            k = self._apply_rope(k, S)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, is_causal=sdpa_causal,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d)
            attn_out = blk['attn'].out_proj(attn_out)

            if 'attn_ln' in blk:
                attn_out = blk['attn_ln'](attn_out)
            h = h + self.resid_drop(attn_out)
            h = h + blk['mlp'](blk['ln2'](h))

        return self.head(self.ln(h))


# ---------------------------------------------------------------------------
# §4  MODEL FACTORY
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(method: str, vocab: int, d: int, heads: int, layers: int, 
                seq_len: int, window: int, dropout: float = 0.1, **kwargs) -> nn.Module:
    """Build a model by name with consistent hyperparameters."""
    sub_ln = kwargs.get('sub_ln', False)
    common = dict(vocab=vocab, d=d, heads=heads, layers=layers, seq_len=seq_len, dropout=dropout)
    
    if method == 'base':
        return MiniTransformer(**common, window=None, sub_ln=sub_ln)
    elif method == 'base_sln':
        # Base + SubLN (post-attention LayerNorm) — appears as "Base + SLN" in the paper.
        return MiniTransformer(**common, window=None, sub_ln=True)
    elif method == 'local':
        return MiniTransformer(**common, window=window, sub_ln=sub_ln)
    elif method == 'local_rope':
        return RoPETransformer(**common, window=window, sub_ln=sub_ln)
    elif method == 'xl':
        # xl_mem_len: fixed cache size so XL doesn't gain context capacity as seq_len grows.
        # Defaults to seq_len when running a single length; caller should pass base_seq for sweeps.
        xl_mem_len = kwargs.get('xl_mem_len', seq_len)
        return TransformerXL(**common, mem_len=xl_mem_len)
    elif method == 'linear':
        return LinearAttentionTransformer(**common, sub_ln=sub_ln)
    elif method == 'tensor':
        # Pure add-on: full attention base + tensor memory module
        return TensorTransformer(
            **common, window=None, sub_ln=sub_ln,
            mem_channels=kwargs.get('mem_channels', 32),
            mem_shape=kwargs.get('mem_shape', (8, 8, 8)),
            chunk_size=kwargs.get('chunk_size', 16),
        )
    elif method == 'tensor_sln':
        # Tensor + SubLN ablation
        return TensorTransformer(
            **common, window=None, sub_ln=True,
            mem_channels=kwargs.get('mem_channels', 32),
            mem_shape=kwargs.get('mem_shape', (8, 8, 8)),
            chunk_size=kwargs.get('chunk_size', 16),
        )
    elif method == 'tensor_local':
        # Ablation: windowed attention + tensor memory
        return TensorTransformer(
            **common, window=window, sub_ln=sub_ln,
            mem_channels=kwargs.get('mem_channels', 32),
            mem_shape=kwargs.get('mem_shape', (8, 8, 8)),
            chunk_size=kwargs.get('chunk_size', 16),
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# §5  TRAINING & EVALUATION
# ---------------------------------------------------------------------------

def train_eval(
    name: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    seq_len: int,
    steps: int = 3000,
    batch_size: int = 64,
    lr: float = 3e-4,
    log_interval: int = 500,
    eval_batches: int = 50,
    warmup_steps: int = 0,
    lr_schedule: str = "flat",
    wandb_log_every: int = 10,
    patience: int = 10,
    use_amp: bool = False,
) -> Dict:
    """Train and evaluate a model with early stopping on validation PPL.
    
    Reports test metrics from the best validation checkpoint.
    Uses sequential non-overlapping evaluation (standard LM protocol).
    
    Args:
        patience: Stop training if val PPL doesn't improve for this many evaluations.
                  Set to 0 to disable early stopping.
    """
    model = model.to(device)
    n_params = count_parameters(model)

    _amp_on = use_amp and device.type == 'cuda'
    amp_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if _amp_on else contextlib.nullcontext()
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    if lr_schedule == "cosine":
        def _schedule(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule)
    else:
        def _schedule(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule)
    
    is_xl = hasattr(model, 'init_memory') and callable(getattr(model, 'init_memory', None))
    vocab_out = model.head.out_features if hasattr(model.head, 'out_features') else model.head.weight.shape[0]
    
    # XL needs sequential batching so cached memory is relevant
    if is_xl:
        train_iter = SequentialBatchIterator(train_data, batch_size, seq_len, device)
        model.init_memory(batch_size, device)
    
    train_losses = []
    val_ppls = []
    best_val_ppl = float('inf')
    best_state_dict = None
    best_step = 0
    no_improve_count = 0
    t_start = time.time()
    
    pbar = tqdm(range(1, steps + 1), desc=name, unit="step", dynamic_ncols=True) if TQDM_AVAILABLE else None
    for step in (pbar if pbar is not None else range(1, steps + 1)):
        model.train()
        if is_xl:
            x, y = train_iter.next_batch()
            if train_iter.wrapped:
                model.init_memory(batch_size, device)
        else:
            x, y = get_batch(train_data, batch_size, seq_len, device)

        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_out), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()

        train_loss = loss.item()
        train_losses.append(train_loss)
        
        if wandb_enabled() and step % wandb_log_every == 0:
            log_dict = {
                "train/loss": train_loss,
                "train/ppl": math.exp(min(train_loss, 20)),
                "train/bpc": train_loss / math.log(2),
                "train/grad_norm": grad_norm,
                "train/lr": scheduler.get_last_lr()[0],
                "step": step,
            }
            if hasattr(model, 'get_gates'):
                gates = model.get_gates()
                log_dict["gates/mean"] = sum(gates) / len(gates)
                for gi, gv in enumerate(gates):
                    log_dict[f"gates/layer_{gi}"] = gv
            wandb.log(log_dict, step=step)
        
        if step % log_interval == 0 or step == steps:
            # Save training memory state, eval gets its own
            _saved_memory = None
            if is_xl:
                if hasattr(model, '_memories') and model._memories is not None:
                    # TransformerXL
                    _saved_memory = ('xl', [m.clone() for m in model._memories])
                elif hasattr(model, '_memory_state') and model._memory_state is not None:
                    # TensorTransformer
                    _saved_memory = ('tensor', (model._memory_state[0].clone(), 
                                                model._memory_state[1].clone()))
            
            # Sequential non-overlapping eval (capped for speed during training)
            val_result = evaluate_full(
                model, val_data, device, seq_len, batch_size,
                max_batches=eval_batches,
            )
            val_ppl = val_result['ppl']
            val_bpc = val_result['bpc']
            avg_val_loss = val_result['loss']
            
            # Restore training memory state
            if _saved_memory is not None:
                if _saved_memory[0] == 'xl':
                    model._memories = _saved_memory[1]
                elif _saved_memory[0] == 'tensor':
                    model._memory_state = _saved_memory[1]
            
            val_ppls.append({'step': step, 'ppl': val_ppl})
            
            # Track best checkpoint
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_step = step
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if wandb_enabled():
                wandb.log({
                    "val/ppl": val_ppl, "val/bpc": val_bpc,
                    "val/loss": avg_val_loss, "val/best_ppl": best_val_ppl,
                    "val/best_step": best_step,
                    "step": step,
                }, step=step)
            
            gates_str = ""
            if hasattr(model, 'get_gates'):
                g = model.get_gates()
                gates_str = f" gates={sum(g)/len(g):.3f}"

            elapsed = time.time() - t_start
            if pbar is not None:
                postfix = {"val_ppl": f"{val_ppl:.2f}", "best": f"{best_val_ppl:.2f}@{best_step}"}
                if gates_str:
                    postfix["gates"] = f"{sum(g)/len(g):.3f}"
                pbar.set_postfix(postfix)
            else:
                print(f"  [{name}] step {step}/{steps}: val_ppl={val_ppl:.2f} "
                      f"best={best_val_ppl:.2f}@{best_step}{gates_str} ({elapsed:.0f}s)")

            # Early stopping
            if patience > 0 and no_improve_count >= patience:
                print(f"  [{name}] Early stopping at step {step} "
                      f"(no improvement for {patience} evals, best@{best_step})")
                break
    
    if pbar is not None:
        pbar.close()

    # Restore best checkpoint for final evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model = model.to(device)
        print(f"  [{name}] Restored best checkpoint from step {best_step} (val_ppl={best_val_ppl:.2f})")
    
    # Final full evaluation on validation set (entire split, no cap)
    final_val = evaluate_full(model, val_data, device, seq_len, batch_size, max_batches=0)
    final_ppl = final_val['ppl']
    bpc = final_val['bpc']
    elapsed = time.time() - t_start
    
    result = {
        'method': name,
        'final_ppl': final_ppl,
        'best_ppl': best_val_ppl,
        'best_step': best_step,
        'bpc': bpc,
        'params': n_params,
        'train_time_s': elapsed,
        'val_curve': val_ppls,
    }
    
    if hasattr(model, 'get_gates'):
        result['final_gates'] = model.get_gates()
    
    if wandb_enabled():
        wandb.run.summary["final_ppl"] = final_ppl
        wandb.run.summary["best_ppl"] = best_val_ppl
        wandb.run.summary["best_step"] = best_step
        wandb.run.summary["final_bpc"] = bpc
        wandb.run.summary["params"] = n_params
        wandb.run.summary["train_time_s"] = elapsed
        if hasattr(model, 'get_gates'):
            wandb.run.summary["final_gate_mean"] = sum(result['final_gates']) / len(result['final_gates'])
    
    return result


# ---------------------------------------------------------------------------
# §6  THROUGHPUT / MEMORY BENCHMARKING
# ---------------------------------------------------------------------------

def benchmark_throughput(model: nn.Module, vocab: int, seq_len: int, 
                        batch_size: int, device: torch.device, 
                        n_iters: int = 50) -> Dict:
    """Measure forward throughput and peak GPU memory."""
    model = model.to(device).eval()
    
    is_xl = hasattr(model, 'init_memory') and callable(getattr(model, 'init_memory', None))
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    
    if is_xl:
        model.init_memory(batch_size, device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    if is_xl:
        model.init_memory(batch_size, device)
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - t0
    tokens_per_sec = (batch_size * seq_len * n_iters) / elapsed
    
    peak_mem_mb = 0
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return {
        'tokens_per_sec': tokens_per_sec,
        'peak_mem_mb': peak_mem_mb,
        'time_per_batch_ms': (elapsed / n_iters) * 1000,
    }


def log_benchmark_to_wandb(bench: Dict):
    if wandb_enabled():
        wandb.run.summary["bench/tokens_per_sec"] = bench['tokens_per_sec']
        wandb.run.summary["bench/peak_mem_mb"] = bench['peak_mem_mb']
        wandb.run.summary["bench/time_per_batch_ms"] = bench['time_per_batch_ms']


def evaluate_full(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    max_batches: int = 0,
) -> Dict:
    """
    Evaluate a model on the ENTIRE data split using sequential non-overlapping
    segments. This is the standard evaluation protocol for language modeling
    (Merity 2018, Dai 2019, Gu 2024).
    
    For XL models: uses sequential batching with memory carry-over and reset on wrap.
    For all other models: uses sequential non-overlapping batching (no random sampling).
    
    Args:
        max_batches: If > 0, cap evaluation at this many batches (for mid-training speed).
                     0 = evaluate entire split.
    Returns dict with loss, ppl, bpc.
    """
    model.eval()
    is_xl = hasattr(model, 'init_memory') and callable(getattr(model, 'init_memory', None))
    vocab_out = model.head.out_features if hasattr(model.head, 'out_features') else model.head.weight.shape[0]

    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        
        eval_iter = SequentialBatchIterator(data, batch_size, seq_len, device)
        if is_xl:
            model.init_memory(batch_size, device)
        
        n_batches = 0
        while True:
            sx, sy = eval_iter.next_batch()
            if eval_iter.wrapped:
                if is_xl:
                    model.init_memory(batch_size, device)
                break  # Went through entire dataset
            
            sl = F.cross_entropy(model(sx).view(-1, vocab_out), sy.view(-1), reduction='sum')
            total_loss += sl.item()
            total_tokens += sy.numel()
            n_batches += 1
            
            if max_batches > 0 and n_batches >= max_batches:
                break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        'loss': avg_loss,
        'ppl': math.exp(min(avg_loss, 20)),
        'bpc': avg_loss / math.log(2),
        'tokens_evaluated': total_tokens,
    }


# ---------------------------------------------------------------------------
# §7  MAIN EXPERIMENT RUNNER
# ---------------------------------------------------------------------------

ALL_METHODS = ['base', 'base_sln', 'local', 'local_rope', 'xl', 'linear', 'tensor', 'tensor_sln', 'tensor_local']
ALL_DATASETS = ['wikitext2', 'shakespeare']

DATASET_DEFAULTS = {
    # Word-level, ~2M tokens, vocab ~30k
    'wikitext2': {
        'seq_lens':     [512],
        'steps':        10000,   # reduced from 40k; use --steps 40000 for full run
        'batch_size':   32,
        'd_model':      384,
        'heads':        8,
        'layers':       8,
        'warmup_steps': 1000,
        'dropout':      0.2,
    },
    # Char-level, ~5.4M chars (Folger complete works), vocab ~90
    'shakespeare': {
        'seq_lens':     [512],
        'steps':        10000,   # reduced from 20k; use --steps 20000 for full run
        'batch_size':   64,
        'd_model':      256,
        'heads':        8,
        'layers':       6,
        'warmup_steps': 500,
        'dropout':      0.1,
    },
}


def run_experiment(args):
    """Run a complete experiment for one dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    datasets_to_run = ALL_DATASETS if args.dataset == 'all' else [args.dataset]
    all_results = {}
    
    for ds_name in datasets_to_run:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#'*70}")
        
        train_data, val_data, test_data, vocab, level = load_dataset_by_name(ds_name, args.data_dir)

        if args.data_fraction < 1.0:
            n = max(4096, int(len(train_data) * args.data_fraction))
            train_data = train_data[:n]
            print(f"  data_fraction={args.data_fraction:.0%}: using {len(train_data):,} train tokens")

        defaults = DATASET_DEFAULTS.get(ds_name, DATASET_DEFAULTS['wikitext2'])
        
        seq_lens = [int(x) for x in args.seq_lens.split(",")] if args.seq_lens else defaults['seq_lens']
        steps = args.steps or defaults['steps']
        base_batch = args.batch or defaults['batch_size']
        d_model = args.d_model or defaults['d_model']
        heads = args.heads or defaults['heads']
        layers = args.layers or defaults['layers']
        window = args.window
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else defaults.get('warmup_steps', 500)
        dropout = args.dropout if args.dropout is not None else defaults.get('dropout', 0.1)

        # Fairness invariants across seq_lens:
        #   1. Constant token budget per step: batch_size × seq_len = base_batch × base_seq
        #   2. Linear LR scaling: lr ∝ batch_size (linear scaling rule — keeps lr-per-sample fixed)
        #   3. Fixed XL mem_len: XL cache size = base_seq throughout, so it doesn't gain
        #      context capacity as seq_len grows (base_seq chosen = shortest seq in sweep).
        base_seq = min(seq_lens)
        tokens_per_batch = base_batch * base_seq
        xl_mem_len = args.xl_mem_len if args.xl_mem_len else base_seq

        # Support comma-separated method list
        if args.method:
            methods = [m.strip() for m in args.method.split(",")]
        else:
            methods = ALL_METHODS

        print(f"  Config: d={d_model}, heads={heads}, layers={layers}, "
              f"base_batch={base_batch} (tokens/batch={tokens_per_batch}), "
              f"steps={steps}, schedule={args.schedule}, "
              f"warmup={warmup_steps}, dropout={dropout}")
        print(f"  Seq lengths: {seq_lens}")
        print(f"  Methods: {methods}")

        ds_results = []

        seq_lens_iter = tqdm(seq_lens, desc=f"{ds_name} seq_lens", unit="seq") if TQDM_AVAILABLE else seq_lens
        for seq_len in seq_lens_iter:
            # Scale batch size to maintain constant token budget; minimum 1
            batch_size = max(1, tokens_per_batch // seq_len)
            # Linear LR scaling: lr ∝ batch_size so lr-per-sample stays constant
            effective_lr = args.lr * batch_size / base_batch
            print(f"\n{'='*60}")
            print(f"  Sequence Length: {seq_len}, Batch: {batch_size}, "
                  f"LR: {effective_lr:.2e}, XL mem_len: {xl_mem_len}")
            print(f"{'='*60}")
            
            row = {
                'dataset': ds_name, 'level': level, 'seq_len': seq_len,
                'batch_size': batch_size, 'effective_lr': effective_lr,
                'xl_mem_len': xl_mem_len, 'window': window,
                'd_model': d_model, 'heads': heads, 'layers': layers,
            }
            
            methods_iter = tqdm(methods, desc=f"seq{seq_len} methods", unit="method", leave=False) if TQDM_AVAILABLE else methods
            for method in methods_iter:
                torch.manual_seed(args.seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed(args.seed)
                
                wb_run = None
                if WANDB_AVAILABLE and args.wandb:
                    run_name = f"{ds_name}/{method}/seq{seq_len}"
                    wb_config = {
                        "dataset": ds_name, "method": method, "seq_len": seq_len,
                        "window": window, "d_model": d_model, "heads": heads,
                        "layers": layers, "batch_size": batch_size, "steps": steps,
                        "lr": effective_lr, "base_lr": args.lr, "lr_schedule": args.schedule,
                        "xl_mem_len": xl_mem_len,
                        "warmup_steps": warmup_steps, "seed": args.seed,
                        "dropout": dropout,
                        "mem_channels": args.mem_channels,
                        "mem_shape": list(args.mem_shape),
                        "chunk_size": args.chunk_size,
                        "num_latents": args.num_latents, "level": level,
                    }
                    wb_run = wandb.init(
                        project=args.wandb_project, entity=args.wandb_entity,
                        group=f"{ds_name}/seq{seq_len}", name=run_name,
                        tags=[ds_name, method, f"seq{seq_len}", level],
                        config=wb_config, reinit=True,
                    )
                
                try:
                    model = build_model(
                        method, vocab, d_model, heads, layers, seq_len, window,
                        dropout=dropout,
                        mem_channels=args.mem_channels,
                        mem_shape=tuple(args.mem_shape),
                        chunk_size=args.chunk_size,
                        num_latents=args.num_latents,
                        xl_mem_len=xl_mem_len,
                    )
                    n_params = count_parameters(model)
                    # For Tensor variants, break out backbone vs memory params for transparency.
                    extra_str = ""
                    if hasattr(model, 'memory'):
                        mem_params = count_parameters(model.memory)
                        gate_params = sum(p.numel() for p in model.gates)
                        ln_params = count_parameters(model.mem_lns)
                        backbone_params = n_params - mem_params - gate_params - ln_params
                        extra_str = (f" [backbone={backbone_params:,} "
                                     f"+ mem={mem_params + gate_params + ln_params:,}]")
                    print(f"\n  --- {method} ({n_params:,} params{extra_str}) ---")

                    if wandb_enabled():
                        wandb.config.update({"params": n_params}, allow_val_change=True)

                    if getattr(args, 'compile', False):
                        model = torch.compile(model)

                    result = train_eval(
                        method, model, train_data, val_data, device,
                        seq_len=seq_len, steps=steps, batch_size=batch_size,
                        lr=effective_lr, log_interval=args.log_interval,
                        eval_batches=args.eval_batches,
                        warmup_steps=warmup_steps, lr_schedule=args.schedule,
                        patience=args.patience,
                        use_amp=getattr(args, 'amp', False),
                    )
                    result['lr'] = effective_lr
                    
                    # Test set evaluation on best checkpoint (already restored in train_eval)
                    test_result = evaluate_full(
                        model, test_data, device, seq_len, batch_size,
                        max_batches=0,  # full test set
                    )
                    result['test_ppl'] = test_result['ppl']
                    result['test_bpc'] = test_result['bpc']
                    result['test_loss'] = test_result['loss']
                    
                    if wandb_enabled():
                        wandb.run.summary["test/ppl"] = test_result['ppl']
                        wandb.run.summary["test/bpc"] = test_result['bpc']
                        wandb.run.summary["test/loss"] = test_result['loss']
                    
                    if args.benchmark:
                        bench = benchmark_throughput(model, vocab, seq_len, batch_size, device)
                        result.update(bench)
                        log_benchmark_to_wandb(bench)
                    
                    row[method] = result
                    save_run(result, ds_name, method, seq_len, args.outdir)
                    print(f"  ✓ {method}: val_PPL={result['final_ppl']:.2f} "
                          f"test_PPL={result['test_ppl']:.2f} "
                          f"BPC={result['bpc']:.3f} "
                          f"best@step{result['best_step']} "
                          f"params={result['params']:,}")

                except Exception as e:
                    print(f"  ✗ {method} FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    row[method] = {'error': str(e)}
                    if wandb_enabled():
                        wandb.run.summary["error"] = str(e)
                
                if wb_run is not None:
                    wb_run.finish()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            ds_results.append(row)
        
        all_results[ds_name] = ds_results
    
    return all_results


def print_summary(all_results: Dict, methods: List[str]):
    """Print formatted summary tables."""
    col_w = max(14, max((len(m) + 2) for m in methods) if methods else 14)

    def _table(title, key, fmt=".2f"):
        for ds_name, ds_results in all_results.items():
            print(f"\n{'='*120}")
            print(f"  {ds_name.upper()} — {title}")
            print(f"{'='*120}")
            header = f"  {'SeqLen':<10}"
            for m in methods:
                header += f"{m:<{col_w}}"
            print(header)
            print(f"  {'-'*110}")
            for row in ds_results:
                line = f"  {row['seq_len']:<10}"
                for m in methods:
                    if m in row and isinstance(row[m], dict) and key in row[m]:
                        line += f"{row[m][key]:<{col_w}{fmt}}"
                    else:
                        line += f"{'—':<{col_w}}"
                print(line)

    # Validation PPL
    _table("Validation Perplexity (↓ better)", "final_ppl")

    # Test PPL
    _table("Test Perplexity (↓ better)", "test_ppl")

    # BPC for char-level datasets
    for ds_name, ds_results in all_results.items():
        if ds_results and ds_results[0].get('level') == 'char':
            print(f"\n{'='*120}")
            print(f"  {ds_name.upper()} — Bits Per Character, Validation (↓ better)")
            print(f"{'='*120}")
            header = f"  {'SeqLen':<10}"
            for m in methods:
                header += f"{m:<{col_w}}"
            print(header)
            print(f"  {'-'*110}")
            for row in ds_results:
                line = f"  {row['seq_len']:<10}"
                for m in methods:
                    if m in row and isinstance(row[m], dict) and 'bpc' in row[m]:
                        line += f"{row[m]['bpc']:<{col_w}.3f}"
                    else:
                        line += f"{'—':<{col_w}}"
                print(line)

            print(f"\n  {ds_name.upper()} — Bits Per Character, Test (↓ better)")
            print(f"  {'-'*110}")
            header = f"  {'SeqLen':<10}"
            for m in methods:
                header += f"{m:<{col_w}}"
            print(header)
            print(f"  {'-'*110}")
            for row in ds_results:
                line = f"  {row['seq_len']:<10}"
                for m in methods:
                    if m in row and isinstance(row[m], dict) and 'test_bpc' in row[m]:
                        line += f"{row[m]['test_bpc']:<{col_w}.3f}"
                    else:
                        line += f"{'—':<{col_w}}"
                print(line)

    # Parameter count table
    print(f"\n{'='*120}")
    print(f"  PARAMETER COUNTS (for reproducibility & fairness)")
    print(f"{'='*120}")
    header = f"  {'Dataset':<14}{'Config':<24}"
    for m in methods:
        header += f"{m:<{col_w}}"
    print(header)
    print(f"  {'-'*110}")
    for ds_name, ds_results in all_results.items():
        if ds_results:
            row = ds_results[0]
            config_str = f"d={row.get('d_model','?')},L={row.get('layers','?')}"
            line = f"  {ds_name:<14}{config_str:<24}"
            for m in methods:
                if m in row and isinstance(row[m], dict) and 'params' in row[m]:
                    p = row[m]['params']
                    if p >= 1_000_000:
                        line += f"{p/1e6:.2f}M{'':<{col_w-7}}"
                    elif p >= 1_000:
                        line += f"{p/1e3:.1f}K{'':<{col_w-6}}"
                    else:
                        line += f"{p:<{col_w}}"
                else:
                    line += f"{'—':<{col_w}}"
            print(line)


def save_run(result: Dict, ds_name: str, method: str, seq_len: int, outdir: str):
    """Save a single completed run immediately so progress survives interruption."""
    os.makedirs(outdir, exist_ok=True)

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return round(obj, 4)
        return obj

    outfile = os.path.join(outdir, f"{ds_name}_{method}_seq{seq_len}.json")
    with open(outfile, "w") as f:
        json.dump(clean(result), f, indent=2)


def save_results(all_results: Dict, outdir: str):
    """Save results to JSON."""
    os.makedirs(outdir, exist_ok=True)
    
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return round(obj, 4)
        return obj
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for ds_name, ds_results in all_results.items():
        outfile = os.path.join(outdir, f"lm_results_{ds_name}_{timestamp}.json")
        with open(outfile, "w") as f:
            json.dump(clean(ds_results), f, indent=2)
        print(f"  Saved: {outfile}")
    
    combined_file = os.path.join(outdir, f"lm_results_all_{timestamp}.json")
    with open(combined_file, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"  Saved: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Language modeling experiments for Tensor Memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # WikiText-2 run (default config)
  python tensor_memory/text/train_text.py --dataset wikitext2

  # Single method for debugging
  python tensor_memory/text/train_text.py --dataset wikitext2 --method tensor --seq_lens 512

  # Add-on ablation: base vs base+memory
  python tensor_memory/text/train_text.py --dataset wikitext2 --method base,tensor --seq_lens 512

  # Window ablation
  python tensor_memory/text/train_text.py --dataset wikitext2 --method base,local,tensor,tensor_local --seq_lens 512

  # Run everything (all methods × all datasets)
  python tensor_memory/text/train_text.py --dataset all --benchmark
        """
    )
    
    parser.add_argument("--dataset", default="wikitext2",
                       choices=ALL_DATASETS + ['all'])
    parser.add_argument("--data_dir", default="./data")
    
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    
    parser.add_argument("--seq_lens", default=None,
                       help="Comma-separated sequence lengths")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--xl_mem_len", type=int, default=None,
                       help="Fixed XL cache size in tokens. Defaults to the shortest seq_len "
                            "in the sweep so XL's effective context doesn't grow as seq_len increases.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=None,
                       help="Dropout rate (default: dataset-specific)")
    parser.add_argument("--schedule", default="cosine", choices=["flat", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Warmup steps (default: dataset-specific)")
    parser.add_argument("--seed", type=int, default=42)
    
    # Method selection — now supports comma-separated list
    parser.add_argument("--method", default=None,
                       help=f"Method(s), comma-separated. Options: {ALL_METHODS}")
    
    # Tensor Memory specifics
    parser.add_argument("--mem_channels", type=int, default=32)
    parser.add_argument("--mem_shape", type=int, nargs=3, default=[8, 8, 8])
    parser.add_argument("--chunk_size", type=int, default=16,
                       help="Chunk size for tensor memory (default: 16). Higher = faster but coarser.")
    parser.add_argument("--num_latents", type=int, default=64,
                       help="Number of latent tokens for V2-B (default: 64)")
    
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=50,
                       help="Batches per mid-training eval (default: 50). Full eval always runs at end.")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (evals without improvement). 0=disabled.")
    parser.add_argument("--amp", action="store_true",
                       help="Enable bfloat16 automatic mixed precision (2-4x faster on Ampere+).")
    parser.add_argument("--compile", action="store_true",
                       help="Apply torch.compile to each model before training (~30s overhead, then faster).")
    parser.add_argument("--quick", action="store_true",
                       help="Shortcut for fast iteration: d=128, heads=4, layers=4, steps=5000. "
                            "Individual flags still override.")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                       help="Use only this fraction of training data (e.g. 0.1 for 10%%). "
                            "Useful for quick ablations; val/test sets are always full.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--outdir", default="./results")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="tensor-memory-lm")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    if args.no_wandb:
        args.wandb = False
    if args.wandb and not WANDB_AVAILABLE:
        print("  WARNING: --wandb requested but wandb not installed.")
        args.wandb = False
    if args.wandb:
        os.environ.setdefault("WANDB_START_METHOD", "thread")
    
    print(f"Tensor Memory — LM Experiments")
    print(f"{'='*60}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Schedule:  {args.schedule} (warmup={args.warmup_steps or 'dataset default'})")
    print(f"  Seed:      {args.seed}")
    print(f"  Device:    {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  wandb:     {'ON → ' + args.wandb_project if args.wandb else 'OFF'}")
    print(f"  Time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --quick: small model for fast iteration (individual flags still override)
    if args.quick:
        if args.d_model is None:
            args.d_model = 128
        if args.heads is None:
            args.heads = 4
        if args.layers is None:
            args.layers = 4
        if args.steps is None:
            args.steps = 5000
        print(f"  --quick: d={args.d_model}, heads={args.heads}, layers={args.layers}, steps={args.steps}")

    torch.manual_seed(args.seed)
    # TF32 is enabled by default on Ampere+ but be explicit for older PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_results = run_experiment(args)
    
    if args.method:
        methods = [m.strip() for m in args.method.split(",")]
    else:
        methods = ALL_METHODS
    print_summary(all_results, methods)
    save_results(all_results, args.outdir)
    
    print(f"\n{'='*60}")
    print(f"  Done! Results saved to {args.outdir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()