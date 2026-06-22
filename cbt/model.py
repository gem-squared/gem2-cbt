"""Tiny GPT-style Transformer with optional semantic-pixel injection + boundary head.

Configs used in v1 experiments:
  baseline_lm : use_pixel=False, use_boundary=False  (pure LM)
  cbt_textonly: use_pixel=False, use_boundary=True   (boundary head from text only)
  cbt_v0      : use_pixel=True,  use_boundary=True    (level+contract injection + head)

v2 ablation ladder uses independent use_level / use_contract flags:
  text_only      : use_level=False, use_contract=False, use_boundary=True
  text_level     : use_level=True,  use_contract=False, use_boundary=True
  text_contract  : use_level=False, use_contract=True,  use_boundary=True
  cbt_v0         : use_level=True,  use_contract=True,  use_boundary=True
  random_contract: same arch as cbt_v0, contract IDs randomized at data level
  contract_only  : use_level=False, use_contract=True,  use_boundary=True, text zeroed

use_pixel=True is a backward-compat alias for use_level=True AND use_contract=True.
Attention is identical in all configs (v0 does NOT modify attention; that is v1).
"""
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CBTConfig:
    vocab_size: int
    block_size: int = 192
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 96
    dropout: float = 0.1
    n_levels: int = 3
    n_contracts: int = 8
    use_level: bool = False    # inject level embedding independently
    use_contract: bool = False # inject contract embedding independently
    use_boundary: bool = False
    use_pixel: bool = False    # DEPRECATED: sets use_level=use_contract=True

    def __post_init__(self):
        if self.use_pixel:
            self.use_level = True
            self.use_contract = True


class CausalSelfAttention(nn.Module):
    def __init__(self, c: CBTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head = c.n_head
        self.n_embd = c.n_embd
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.attn_drop = nn.Dropout(c.dropout)
        self.resid_drop = nn.Dropout(c.dropout)
        self.register_buffer(
            "causal",
            torch.tril(torch.ones(c.block_size, c.block_size)).view(1, 1, c.block_size, c.block_size),
        )

    def forward(self, x, key_pad=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.causal[:, :, :T, :T] == 0, float("-inf"))
        if key_pad is not None:  # (B,T) 1=real 0=pad -> mask out pad keys
            att = att.masked_fill(key_pad[:, None, None, :] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, c: CBTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )

    def forward(self, x, key_pad=None):
        x = x + self.attn(self.ln1(x), key_pad=key_pad)
        x = x + self.mlp(self.ln2(x))
        return x


class CBT(nn.Module):
    def __init__(self, c: CBTConfig):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        if c.use_level:
            self.level_emb = nn.Embedding(c.n_levels, c.n_embd)
        if c.use_contract:
            self.contract_emb = nn.Embedding(c.n_contracts, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.use_boundary:
            self.boundary_head = nn.Sequential(
                nn.Linear(c.n_embd, c.n_embd), nn.GELU(), nn.Linear(c.n_embd, 2)
            )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attn_pad=None, level=None, contract=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        if self.c.use_level:
            x = x + self.level_emb(level)[:, None, :]
        if self.c.use_contract:
            x = x + self.contract_emb(contract)[:, None, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, key_pad=attn_pad)
        x = self.ln_f(x)
        lm_logits = self.lm_head(x)
        bnd_logits = None
        if self.c.use_boundary:
            if attn_pad is not None:
                # last real token: it has attended to the whole sequence (causal)
                last = attn_pad.sum(1).clamp(min=1).long() - 1   # (B,)
                pooled = x[torch.arange(B, device=x.device), last]
            else:
                pooled = x[:, -1, :]
            bnd_logits = self.boundary_head(pooled)
        return lm_logits, bnd_logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
