"""Programmatic API for computing reconstruction errors on new SMILES."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .data import collate
from .model import JointMolecularModel, SmilesAutoencoder
from .tokenizer import SmilesTokenizer


@torch.no_grad()
def compute_reconstruction_errors(
    model,
    tokenizer: SmilesTokenizer,
    smiles_list: Sequence[str],
    device: str = "cpu",
    batch_size: int = 128,
    max_len: int = 256,
    decode: bool = False,
) -> List[Dict]:
    """For each SMILES, return a dict with at minimum:

        smiles                 the input
        reconstruction_error   mean cross-entropy per token (NaN if skipped)
        unfamiliarity          log(reconstruction_error) — the U(x) of
                               van Tilborg et al. (2026) (NaN if skipped)
        n_tokens               number of non-pad target tokens
        reconstructed          greedy decode if decode=True else ''
        exact_match            reconstructed == smiles (only meaningful with decode=True)
        note                   reason for skipping (if any)

    If ``model`` is a :class:`JointMolecularModel`, each row also contains:

        predicted_class        argmax over class logits (int, NaN if skipped)
        prob_<i>                softmax probability for class i (one column per class)

    Sequences longer than ``max_len`` are returned with NaN error and a note.
    Output order matches input order.
    """
    is_jmm = isinstance(model, JointMolecularModel)
    n_classes = model.cls_cfg.n_classes if is_jmm else 0
    model.eval()
    pad_id = tokenizer.pad_id
    results: List[Dict] = [None] * len(smiles_list)  # type: ignore

    def _skipped(s: str, n_tokens: int, note: str) -> Dict:
        row = {
            "smiles": s,
            "reconstruction_error": float("nan"),
            "unfamiliarity": float("nan"),
            "n_tokens": n_tokens,
            "reconstructed": "",
            "exact_match": False,
            "note": note,
        }
        if is_jmm:
            row["predicted_class"] = float("nan")
            for c in range(n_classes):
                row[f"prob_{c}"] = float("nan")
        return row

    # Pass 1: tokenize, mark over-length sequences.
    encoded: List[tuple] = []  # list of (orig_idx, ids)
    for i, s in enumerate(smiles_list):
        ids = tokenizer.encode(s, add_special=True)
        if len(ids) > max_len:
            results[i] = _skipped(s, max(len(ids) - 2, 0), f"exceeds max_len={max_len}")
        elif len(ids) < 3:
            results[i] = _skipped(s, 0, "empty or untokenizable")
        else:
            encoded.append((i, ids))

    # Pass 2: batched forward through the model.
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        idxs = [c[0] for c in chunk]
        ids_list = [c[1] for c in chunk]
        src, src_len, tgt_in, tgt_out = collate(ids_list, pad_id)
        src = src.to(device); src_len = src_len.to(device)
        tgt_in = tgt_in.to(device); tgt_out = tgt_out.to(device)

        if is_jmm:
            logits, z, cls_logits = model(src, src_len, tgt_in)
            probs = F.softmax(cls_logits, dim=-1).cpu().tolist()
            preds = cls_logits.argmax(-1).cpu().tolist()
        else:
            logits, z = model(src, src_len, tgt_in)
            probs = None
            preds = None
        per_seq = model.per_sequence_loss(logits, tgt_out).cpu().tolist()

        decoded_strs = [""] * len(chunk)
        exact = [False] * len(chunk)
        if decode:
            gen = model.decoder.generate(z, max_len, tokenizer.bos_id, tokenizer.eos_id)
            gen = gen.cpu().tolist()
            for k in range(len(chunk)):
                decoded_strs[k] = tokenizer.decode(gen[k])
                exact[k] = decoded_strs[k] == smiles_list[idxs[k]]

        for k, idx in enumerate(idxs):
            err = per_seq[k]
            row = {
                "smiles": smiles_list[idx],
                "reconstruction_error": err if not math.isnan(err) else float("nan"),
                "unfamiliarity": math.log(err) if (not math.isnan(err) and err > 0) else float("nan"),
                "n_tokens": int((tgt_out[k] != pad_id).sum().item()),
                "reconstructed": decoded_strs[k],
                "exact_match": bool(exact[k]),
                "note": "",
            }
            if is_jmm:
                row["predicted_class"] = int(preds[k])
                for c in range(n_classes):
                    row[f"prob_{c}"] = float(probs[k][c])
            results[idx] = row

    return results  # type: ignore


@torch.no_grad()
def compute_ensemble_predictions(
    jmms,
    tokenizer: SmilesTokenizer,
    smiles_list,
    device: str = "cpu",
    batch_size: int = 128,
    max_len: int = 256,
    decode: bool = False,
) -> List[Dict]:
    """Average per-molecule predictions across an ensemble of JMMs.

    Implements the M-model anchored-ensemble inference of van Tilborg et al.
    (2026): the predicted probability is the mean over models, the predicted
    class is the argmax of that mean, and the prediction uncertainty is the
    Shannon entropy ``H(y|x) = -Σ p log p`` of the mean. Reconstruction
    error and unfamiliarity are averaged across models per molecule.

    ``jmms`` is a sequence of :class:`JointMolecularModel` instances (already
    moved to ``device``). The first model's ``cls_cfg`` defines ``n_classes``;
    all members must agree.
    """
    if not jmms:
        raise ValueError("compute_ensemble_predictions needs at least one model.")
    if not all(isinstance(m, JointMolecularModel) for m in jmms):
        raise TypeError("compute_ensemble_predictions only works on JMMs.")
    n_classes = jmms[0].cls_cfg.n_classes
    if any(m.cls_cfg.n_classes != n_classes for m in jmms):
        raise ValueError("All ensemble members must have the same n_classes.")

    # Per-model rows.
    per_model_rows = [
        compute_reconstruction_errors(
            m, tokenizer, smiles_list,
            device=device, batch_size=batch_size, max_len=max_len,
            decode=(decode and i == 0),  # only decode once
        )
        for i, m in enumerate(jmms)
    ]

    out: List[Dict] = []
    n = len(smiles_list)
    for i in range(n):
        ref = per_model_rows[0][i]
        # Skipped rows (NaN error) propagate as skipped in the ensemble.
        if math.isnan(ref["reconstruction_error"]):
            row = dict(ref)
            row["entropy"] = float("nan")
            out.append(row)
            continue

        recon_vals = [r[i]["reconstruction_error"] for r in per_model_rows]
        unfam_vals = [r[i]["unfamiliarity"] for r in per_model_rows]
        prob_stack = [
            [r[i][f"prob_{c}"] for c in range(n_classes)] for r in per_model_rows
        ]
        mean_recon = sum(recon_vals) / len(recon_vals)
        mean_unfam = sum(unfam_vals) / len(unfam_vals)
        mean_probs = [sum(p[c] for p in prob_stack) / len(prob_stack) for c in range(n_classes)]
        # H(y|x) = -Σ p log p, with 0 log 0 := 0.
        entropy = -sum(p * math.log(p) for p in mean_probs if p > 0.0)
        pred = max(range(n_classes), key=lambda c: mean_probs[c])
        row = {
            "smiles": ref["smiles"],
            "reconstruction_error": mean_recon,
            "unfamiliarity": mean_unfam,
            "n_tokens": ref["n_tokens"],
            "reconstructed": ref["reconstructed"],
            "exact_match": ref["exact_match"],
            "note": ref["note"],
            "predicted_class": pred,
            "entropy": entropy,
        }
        for c in range(n_classes):
            row[f"prob_{c}"] = mean_probs[c]
        out.append(row)
    return out
