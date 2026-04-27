"""Shared utilities: SMILES file reading and vocabulary extension."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import torch.nn as nn

from .model import SmilesAutoencoder
from .tokenizer import SmilesTokenizer


def read_smiles_file(path: str | Path) -> List[str]:
    """Read SMILES one per line. Tolerates CSV/TSV by taking the first column.

    Lines starting with '#' are treated as comments. A header row of "smiles"
    (case-insensitive) is skipped.
    """
    out: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # First column for CSV/TSV/whitespace-separated files.
            tok = line.split(",")[0].split("\t")[0].split()[0]
            if tok.lower() == "smiles":
                continue
            out.append(tok)
    return out


def read_labeled_smiles_csv(
    path: str | Path,
    smiles_col: str = "smiles",
    label_col: str = "y",
    split_col: Optional[str] = None,
    split: Optional[str] = None,
) -> Tuple[List[str], List[int]]:
    """Read SMILES + integer class labels from a CSV.

    If ``split_col`` and ``split`` are given, only rows where that column
    equals ``split`` are kept (e.g. ``split_col="split", split="train"``).
    Rows with empty SMILES or empty labels are skipped.
    """
    smiles: List[str] = []
    labels: List[int] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_col is not None and split is not None:
                if row.get(split_col) != split:
                    continue
            smi = (row.get(smiles_col) or "").strip()
            lbl = (row.get(label_col) or "").strip()
            if not smi or not lbl:
                continue
            smiles.append(smi)
            labels.append(int(float(lbl)))  # tolerate "1.0" / "0.0" formats
    return smiles, labels


def maybe_extend_vocab(
    model: SmilesAutoencoder,
    tokenizer: SmilesTokenizer,
    new_smiles,
) -> bool:
    """Add tokens from `new_smiles` to `tokenizer`. If new tokens were added,
    grow the model's embedding and output projection in place. The new rows
    are randomly initialised and ready for fine-tuning.

    Returns True if vocabulary was extended.
    """
    old_size = tokenizer.vocab_size
    tokenizer.fit(new_smiles)
    new_size = tokenizer.vocab_size
    if new_size == old_size:
        return False

    cfg = model.cfg
    cfg.vocab_size = new_size

    # Grow encoder embedding.
    old_e = model.encoder.embed
    new_e = nn.Embedding(new_size, cfg.embed_dim, padding_idx=cfg.pad_id)
    new_e.weight.data[:old_size] = old_e.weight.data
    model.encoder.embed = new_e

    # Grow decoder embedding.
    old_d = model.decoder.embed
    new_d = nn.Embedding(new_size, cfg.embed_dim, padding_idx=cfg.pad_id)
    new_d.weight.data[:old_size] = old_d.weight.data
    model.decoder.embed = new_d

    # Grow output projection.
    old_o = model.decoder.out
    new_o = nn.Linear(cfg.dec_hidden, new_size)
    new_o.weight.data[:old_size] = old_o.weight.data
    new_o.bias.data[:old_size] = old_o.bias.data
    model.decoder.out = new_o

    return True
