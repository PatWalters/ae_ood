"""Training and evaluation loops."""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


def _move_batch(batch, device):
    src, src_len, tgt_in, tgt_out = batch
    return src.to(device), src_len.to(device), tgt_in.to(device), tgt_out.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
    teacher_forcing: bool = True,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        src, src_len, tgt_in, tgt_out = _move_batch(batch, device)
        logits, _ = model(src, src_len, tgt_in, teacher_forcing=teacher_forcing)
        loss = model.reconstruction_loss(logits, tgt_out)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bsz = src.size(0)
        total += loss.item() * bsz
        n += bsz
    return total / max(n, 1)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    teacher_forcing: bool = True,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        src, src_len, tgt_in, tgt_out = _move_batch(batch, device)
        logits, _ = model(src, src_len, tgt_in, teacher_forcing=teacher_forcing)
        loss = model.reconstruction_loss(logits, tgt_out)
        bsz = src.size(0)
        total += loss.item() * bsz
        n += bsz
    return total / max(n, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    save_path: Optional[str] = None,
    freeze_encoder: bool = False,
    log_every: int = 1,
    teacher_forcing: bool = True,
    quiet: bool = False,
) -> List[Dict]:
    """Train `model` and (if save_path is given) checkpoint the best val model."""
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

    best = float("inf")
    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, device, grad_clip,
            teacher_forcing=teacher_forcing,
        )
        val = (
            eval_loss(model, val_loader, device, teacher_forcing=teacher_forcing)
            if val_loader is not None
            else float("nan")
        )
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": val, "time_s": dt})
        if not quiet and epoch % log_every == 0:
            print(f"epoch {epoch:3d} | train {tr:.4f} | val {val:.4f} | {dt:5.1f}s")
        if save_path is not None and (val == val) and val < best:  # val==val filters NaN
            best = val
            model.save(save_path)
        elif save_path is not None and val_loader is None:
            # No validation: just keep latest.
            model.save(save_path)
    return history


# ---------------------------------------------------------------------------
# Joint molecular model training
# ---------------------------------------------------------------------------

def _move_joint_batch(batch, device):
    src, src_len, tgt_in, tgt_out, y = batch
    return (
        src.to(device),
        src_len.to(device),
        tgt_in.to(device),
        tgt_out.to(device),
        y.to(device),
    )


def joint_train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    gamma: float = 0.1,
    anchor_lambda: float = 0.0,
    grad_clip: float = 1.0,
    teacher_forcing: bool = True,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "cls": 0.0}
    n = 0
    for batch in loader:
        src, src_len, tgt_in, tgt_out, y = _move_joint_batch(batch, device)
        recon_logits, _, cls_logits = model(
            src, src_len, tgt_in, teacher_forcing=teacher_forcing
        )
        loss, l_recon, l_cls = model.joint_loss(
            recon_logits, tgt_out, cls_logits, y,
            gamma=gamma, anchor_lambda=anchor_lambda,
        )
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bsz = src.size(0)
        totals["loss"] += loss.item() * bsz
        totals["recon"] += l_recon.item() * bsz
        totals["cls"] += l_cls.item() * bsz
        n += bsz
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def joint_eval(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    gamma: float = 0.1,
    anchor_lambda: float = 0.0,
    teacher_forcing: bool = True,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "cls": 0.0}
    correct, n = 0, 0
    for batch in loader:
        src, src_len, tgt_in, tgt_out, y = _move_joint_batch(batch, device)
        recon_logits, _, cls_logits = model(
            src, src_len, tgt_in, teacher_forcing=teacher_forcing
        )
        loss, l_recon, l_cls = model.joint_loss(
            recon_logits, tgt_out, cls_logits, y,
            gamma=gamma, anchor_lambda=anchor_lambda,
        )
        bsz = src.size(0)
        totals["loss"] += loss.item() * bsz
        totals["recon"] += l_recon.item() * bsz
        totals["cls"] += l_cls.item() * bsz
        correct += int((cls_logits.argmax(-1) == y).sum().item())
        n += bsz
    out = {k: v / max(n, 1) for k, v in totals.items()}
    out["acc"] = correct / max(n, 1)
    return out


def _make_joint_optimizer(
    model: nn.Module,
    lr: float,
    decoder_lr: Optional[float],
    weight_decay: float,
) -> torch.optim.Optimizer:
    """If ``decoder_lr`` is given, the decoder uses its own (smaller) LR — the
    paper trains encoder/classifier at 3e-6 and the decoder at 3e-7 during
    joint fine-tuning.
    """
    if decoder_lr is None:
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    decoder_ids = {id(p) for p in decoder_params}
    other_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in decoder_ids
    ]
    return AdamW(
        [
            {"params": other_params, "lr": lr},
            {"params": decoder_params, "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )


def joint_fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int = 10,
    gamma: float = 0.1,
    anchor_lambda: float = 0.0,
    lr: float = 3e-6,
    decoder_lr: Optional[float] = None,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    save_path: Optional[str] = None,
    freeze_encoder: bool = False,
    freeze_decoder: bool = False,
    teacher_forcing: bool = True,
    log_every: int = 1,
) -> List[Dict]:
    """Jointly train ``model`` (a JointMolecularModel) and (if ``save_path``
    is given) checkpoint the best-validation model.
    """
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    if freeze_decoder:
        for p in model.decoder.parameters():
            p.requires_grad = False
    optimizer = _make_joint_optimizer(model, lr, decoder_lr, weight_decay)

    best = math.inf
    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = joint_train_one_epoch(
            model, train_loader, optimizer, device,
            gamma=gamma, anchor_lambda=anchor_lambda, grad_clip=grad_clip,
            teacher_forcing=teacher_forcing,
        )
        val = (
            joint_eval(
                model, val_loader, device,
                gamma=gamma, anchor_lambda=anchor_lambda,
                teacher_forcing=teacher_forcing,
            )
            if val_loader is not None
            else {"loss": float("nan"), "recon": float("nan"), "cls": float("nan"), "acc": float("nan")}
        )
        dt = time.time() - t0
        history.append({"epoch": epoch, "train": tr, "val": val, "time_s": dt})
        if epoch % log_every == 0:
            print(
                f"epoch {epoch:3d} | train loss {tr['loss']:.4f} "
                f"(recon {tr['recon']:.4f}, cls {tr['cls']:.4f}) | "
                f"val loss {val['loss']:.4f} (acc {val['acc']:.3f}) | {dt:5.1f}s"
            )
        track = val["loss"]
        if save_path is not None and not math.isnan(track) and track < best:
            best = track
            model.save(save_path)
        elif save_path is not None and val_loader is None:
            model.save(save_path)
    return history
