"""Load JSONL examples into padded tensors for LM + boundary heads."""
import json
import torch
from torch.utils.data import Dataset
from .tokenizer import LEVEL2ID


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class CBTDataset(Dataset):
    """Each item:
        input_ids   (block,)   char ids, right-padded with 0
        lm_targets  (block,)   input_ids shifted left by 1; pad/last = -100 (ignored)
        attn_pad    (block,)   1 for real tokens, 0 for pad
        level       ()         int level id
        contract    ()         int contract id
        label       ()         int 0/1 (incompatible/compatible)
    """
    def __init__(self, rows, tokenizer, contract_vocab, block_size):
        self.tok = tokenizer
        self.cv = contract_vocab
        self.block = block_size
        self.rows = rows
        self.n_trunc = 0
        for r in rows:
            if len(tokenizer.encode(r["text"])) + 1 > block_size:
                self.n_trunc += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = self.tok.encode(r["text"])[: self.block]
        n = len(ids)
        x = torch.zeros(self.block, dtype=torch.long)
        x[:n] = torch.tensor(ids, dtype=torch.long)
        # LM targets: next-char prediction
        y = torch.full((self.block,), -100, dtype=torch.long)
        if n >= 2:
            y[: n - 1] = torch.tensor(ids[1:], dtype=torch.long)
        pad = torch.zeros(self.block, dtype=torch.long)
        pad[:n] = 1
        return {
            "input_ids": x,
            "lm_targets": y,
            "attn_pad": pad,
            "level": torch.tensor(LEVEL2ID[r["level"]], dtype=torch.long),
            "contract": torch.tensor(self.cv.encode(r["contract"]), dtype=torch.long),
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }
