"""Dataset and DataLoader helpers."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from .tokenizer import SmilesTokenizer


class SmilesDataset(Dataset):
    """Tokenizes once on construction and filters by length."""

    def __init__(
        self,
        smiles_list: Sequence[str],
        tokenizer: SmilesTokenizer,
        max_len: int = 128,
        min_len: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data: List[List[int]] = []
        self.kept_smiles: List[str] = []
        for s in smiles_list:
            ids = tokenizer.encode(s, add_special=True)
            if min_len <= len(ids) <= max_len:
                self.data.append(ids)
                self.kept_smiles.append(s)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> List[int]:
        return self.data[i]


def collate(batch: Sequence[Sequence[int]], pad_id: int):
    """Pad to the longest sequence in the batch.

    Returns:
        src        (B, T)   full sequence including BOS and EOS — encoder input.
        src_len    (B,)     true length of each src sequence.
        tgt_in     (B, T-1) src shifted right by one (decoder input).
        tgt_out    (B, T-1) src shifted left by one (decoder target).
    """
    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
    T = int(lengths.max().item())
    src = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        src[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    tgt_in = src[:, :-1].clone()
    tgt_out = src[:, 1:].clone()
    return src, lengths, tgt_in, tgt_out


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate(b, pad_id),
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Labeled SMILES (for the joint molecular model)
# ---------------------------------------------------------------------------

class LabeledSmilesDataset(Dataset):
    """Like SmilesDataset but each item carries an integer class label."""

    def __init__(
        self,
        smiles_list: Sequence[str],
        labels: Sequence[int],
        tokenizer: SmilesTokenizer,
        max_len: int = 128,
        min_len: int = 3,
    ):
        if len(smiles_list) != len(labels):
            raise ValueError("smiles_list and labels must have the same length.")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data: List[List[int]] = []
        self.labels: List[int] = []
        self.kept_smiles: List[str] = []
        for s, y in zip(smiles_list, labels):
            ids = tokenizer.encode(s, add_special=True)
            if min_len <= len(ids) <= max_len:
                self.data.append(ids)
                self.labels.append(int(y))
                self.kept_smiles.append(s)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i], self.labels[i]


def collate_labeled(batch, pad_id: int):
    ids = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    src, lengths, tgt_in, tgt_out = collate(ids, pad_id)
    return src, lengths, tgt_in, tgt_out, torch.tensor(labels, dtype=torch.long)


def make_labeled_dataloader(
    dataset: Dataset,
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    if sampler is not None:
        shuffle = False  # mutually exclusive with sampler in DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda b: collate_labeled(b, pad_id),
        drop_last=drop_last,
    )


def class_balanced_sampler(
    labels: Iterable[int],
    n_classes: Optional[int] = None,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Weighted sampler implementing the paper's eq. (14): ``P_c = 1 - n_c/N``.

    Minority classes get a higher per-sample weight so that minibatches of
    the requested ``batch_size`` are roughly class-balanced. Sampling is
    with replacement, matching the reference implementation.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("class_balanced_sampler needs at least one label.")
    n_classes = n_classes if n_classes is not None else (max(labels) + 1)
    counts = [0] * n_classes
    for y in labels:
        counts[y] += 1
    N = len(labels)
    P_c = [1.0 - c / N for c in counts]
    weights = [P_c[y] for y in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=num_samples if num_samples is not None else N,
        replacement=True,
    )
