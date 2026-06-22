"""Char-level tokenizer + label vocabularies (level, contract)."""
import json

PAD = "\x00"  # reserved pad char
LEVELS = ["concept", "context", "task"]
LEVEL2ID = {l: i for i, l in enumerate(LEVELS)}


class CharTokenizer:
    def __init__(self, chars):
        # index 0 is always PAD
        self.itos = [PAD] + [c for c in chars if c != PAD]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = 0

    @classmethod
    def from_texts(cls, texts):
        chars = sorted({c for t in texts for c in t})
        return cls(chars)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != self.pad_id)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            itos = json.load(f)["itos"]
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = {c: i for i, c in enumerate(itos)}
        obj.pad_id = 0
        return obj


class ContractVocab:
    """Maps contract strings -> ids. Index 0 = <unk>."""
    def __init__(self, contracts):
        self.itos = ["<unk>"] + sorted(set(contracts))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def size(self):
        return len(self.itos)

    def encode(self, contract):
        return self.stoi.get(contract, 0)
