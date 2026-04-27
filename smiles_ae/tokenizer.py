"""Atom-wise SMILES tokenizer with persistent vocabulary."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

# Atom-wise SMILES regex (after Schwaller et al.).
# Matches multi-char tokens (Br, Cl, [...]) before single-char ones.
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]"
    r"|Br?|Cl?|N|O|S|P|F|I"
    r"|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$"
    r"|%[0-9]{2}|[0-9])"
)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class SmilesTokenizer:
    """Reversible token <-> id mapping with optional vocab extension."""

    def __init__(self, token_to_id: Optional[dict] = None):
        if token_to_id is None:
            token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.token_to_id: dict = dict(token_to_id)
        self.id_to_token: dict = {i: t for t, i in self.token_to_id.items()}

    # --- ids for special tokens ---
    @property
    def pad_id(self) -> int: return self.token_to_id[PAD_TOKEN]
    @property
    def bos_id(self) -> int: return self.token_to_id[BOS_TOKEN]
    @property
    def eos_id(self) -> int: return self.token_to_id[EOS_TOKEN]
    @property
    def unk_id(self) -> int: return self.token_to_id[UNK_TOKEN]
    @property
    def vocab_size(self) -> int: return len(self.token_to_id)

    # --- core ops ---
    @staticmethod
    def split(smiles: str) -> List[str]:
        return SMILES_REGEX.findall(smiles)

    def fit(self, smiles_iter: Iterable[str], min_freq: int = 1) -> "SmilesTokenizer":
        """Add new tokens found in `smiles_iter`. Existing ids are preserved."""
        counter: Counter = Counter()
        for s in smiles_iter:
            counter.update(self.split(s))
        next_id = len(self.token_to_id)
        # Deterministic order: by descending frequency, then alphabetic.
        for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq < min_freq or tok in self.token_to_id:
                continue
            self.token_to_id[tok] = next_id
            next_id += 1
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        return self

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in self.split(smiles)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(int(i), UNK_TOKEN)
            if tok == EOS_TOKEN:
                break
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        return cls(json.loads(Path(path).read_text()))
