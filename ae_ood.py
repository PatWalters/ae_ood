#!/usr/bin/env python
"""ae_ood: Autoencoder out-of-distribution CLI.

Three subcommands consolidated into one entry point:

    python ae_ood.py pretrain  --smiles ... --out_dir ...
    python ae_ood.py finetune  --smiles ... --checkpoint_dir ... --out_dir ...
    python ae_ood.py evaluate  --smiles ... --checkpoint_dir ... --output_csv ...
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

import torch

from smiles_ae import (
    AutoencoderConfig,
    JointMolecularModel,
    LabeledSmilesDataset,
    SmilesAutoencoder,
    SmilesDataset,
    SmilesTokenizer,
    class_balanced_sampler,
    compute_ensemble_predictions,
    compute_reconstruction_errors,
    fit,
    joint_fit,
    load_checkpoint,
    make_dataloader,
    make_labeled_dataloader,
    maybe_extend_vocab,
    read_labeled_smiles_csv,
    read_smiles_file,
)


# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------

def add_pretrain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smiles", required=True, help="Path to SMILES file (one per line).")
    p.add_argument("--out_dir", required=True, help="Where to save tokenizer + checkpoints.")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=128)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--enc_hidden", type=int, default=256)
    p.add_argument("--dec_hidden", type=int, default=512)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--enc_layers", type=int, default=1)
    p.add_argument("--dec_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)

    # Encoder architecture. "gru" is the original bi-GRU; "cnn" is the
    # JMM-style 1D-CNN encoder of van Tilborg et al. (2026).
    p.add_argument("--encoder_type", choices=["gru", "cnn"], default="gru")
    p.add_argument("--cnn_layers", type=int, default=2,
                   help="Number of stacked Conv1d->ReLU->MaxPool1d->Dropout blocks (CNN encoder).")
    p.add_argument("--cnn_filters", type=int, default=256,
                   help="Number of convolutional filters / output channels (CNN encoder).")
    p.add_argument("--cnn_kernel_size", type=int, default=8,
                   help="Conv1d kernel size; max-pool kernel matches (CNN encoder).")
    p.add_argument("--cnn_max_len", type=int, default=128,
                   help="Fixed sequence length the CNN encoder pads/truncates inputs to.")

    # Decoder architecture. "gru" is the original GRU decoder; "lstm" is the
    # conditioned LSTM decoder of van Tilborg et al. (2026).
    p.add_argument("--decoder_type", choices=["gru", "lstm"], default="gru")
    p.add_argument("--no_teacher_forcing", action="store_true",
                   help="Train the decoder autoregressively from its own predictions "
                        "(matches van Tilborg et al. 2026). Slower; default uses "
                        "teacher forcing.")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda/cpu (auto-detected if omitted).")


def cmd_pretrain(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles = read_smiles_file(args.smiles)
    if not smiles:
        raise SystemExit(f"No SMILES read from {args.smiles}")
    random.shuffle(smiles)
    n_val = max(1, int(len(smiles) * args.val_frac))
    val_s, train_s = smiles[:n_val], smiles[n_val:]
    print(f"Loaded {len(smiles)} SMILES  |  train={len(train_s)}  val={len(val_s)}")

    tokenizer = SmilesTokenizer().fit(train_s)
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    cfg = AutoencoderConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        enc_hidden=args.enc_hidden,
        dec_hidden=args.dec_hidden,
        latent_dim=args.latent_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        encoder_type=args.encoder_type,
        cnn_layers=args.cnn_layers,
        cnn_filters=args.cnn_filters,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_max_len=args.cnn_max_len,
        decoder_type=args.decoder_type,
    )
    model = SmilesAutoencoder(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} on {device}")

    train_ds = SmilesDataset(train_s, tokenizer, args.max_len)
    val_ds = SmilesDataset(val_s, tokenizer, args.max_len)
    print(f"After length filter: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = make_dataloader(
        train_ds, args.batch_size, tokenizer.pad_id,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_ds, args.batch_size, tokenizer.pad_id,
        shuffle=False, num_workers=args.num_workers,
    )

    history = fit(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        save_path=str(out_dir / "best.pt"),
        teacher_forcing=not args.no_teacher_forcing,
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "args.json").write_text(
        json.dumps({k: v for k, v in vars(args).items() if k != "func"}, indent=2)
    )
    print(f"Done. Saved to {out_dir}")


# ---------------------------------------------------------------------------
# finetune
# ---------------------------------------------------------------------------

def add_finetune_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smiles", required=True)
    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory with best.pt and tokenizer.json from pretraining.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4, help="Lower LR is typical for fine-tuning.")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder weights and fine-tune only the decoder.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)


def cmd_finetune(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ck_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SmilesTokenizer.load(ck_dir / "tokenizer.json")
    model = SmilesAutoencoder.load(ck_dir / "best.pt")
    print(f"Loaded checkpoint from {ck_dir} (vocab {tokenizer.vocab_size})")

    smiles = read_smiles_file(args.smiles)
    if not smiles:
        raise SystemExit(f"No SMILES read from {args.smiles}")
    random.shuffle(smiles)
    n_val = max(1, int(len(smiles) * args.val_frac))
    val_s, train_s = smiles[:n_val], smiles[n_val:]
    print(f"Fine-tune set: train={len(train_s)}  val={len(val_s)}")

    extended = maybe_extend_vocab(model, tokenizer, train_s)
    if extended:
        print(f"Extended vocabulary to {tokenizer.vocab_size} tokens "
              f"(new rows are randomly initialised).")
    tokenizer.save(out_dir / "tokenizer.json")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = SmilesDataset(train_s, tokenizer, args.max_len)
    val_ds = SmilesDataset(val_s, tokenizer, args.max_len)
    print(f"After length filter: train={len(train_ds)}  val={len(val_ds)}")

    train_loader = make_dataloader(
        train_ds, args.batch_size, tokenizer.pad_id,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_ds, args.batch_size, tokenizer.pad_id,
        shuffle=False, num_workers=args.num_workers,
    )

    history = fit(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        freeze_encoder=args.freeze_encoder,
        save_path=str(out_dir / "best.pt"),
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "args.json").write_text(
        json.dumps({k: v for k, v in vars(args).items() if k != "func"}, indent=2)
    )
    print(f"Done. Saved to {out_dir}")


# ---------------------------------------------------------------------------
# joint_finetune (JMM: encoder + decoder + classifier on z, trained jointly)
# ---------------------------------------------------------------------------

def add_joint_finetune_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smiles", required=True,
                   help="Labeled CSV with smiles_col + label_col (and optional split_col).")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Directory with best.pt + tokenizer.json from a pretrained autoencoder.")
    p.add_argument("--out_dir", required=True)

    # Column names + optional split filtering.
    p.add_argument("--smiles_col", default="smiles")
    p.add_argument("--label_col", default="y")
    p.add_argument("--split_col", default=None,
                   help="If set, filter rows by split_col == train_split / val_split.")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default=None,
                   help="If set, use rows with split_col == val_split as validation. "
                        "Otherwise val_frac from train rows is held out.")
    p.add_argument("--val_frac", type=float, default=0.1)

    # Classifier head.
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--cls_hidden", type=int, nargs="+", default=[1024, 1024],
                   help="Hidden layer sizes of the classifier MLP.")
    p.add_argument("--cls_dropout", type=float, default=0.0)

    # Joint training schedule (defaults track the paper).
    p.add_argument("--gamma", type=float, default=0.1,
                   help="Weight on the classifier loss in L_recon + gamma * L_cls.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-6,
                   help="LR for encoder and classifier (paper: 3e-6).")
    p.add_argument("--decoder_lr", type=float, default=3e-7,
                   help="Separate LR for the decoder; pass 0 to use --lr for everything.")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--freeze_decoder", action="store_true")
    p.add_argument("--no_class_balance", action="store_true",
                   help="Skip the P_c = 1 - n_c/N class-balanced sampler.")
    p.add_argument("--no_teacher_forcing", action="store_true",
                   help="Train the decoder autoregressively from its own predictions.")

    # Anchored ensembling for approximate-Bayesian uncertainty (Pearce et al. 2020;
    # van Tilborg et al. 2026 eq. (10)).
    p.add_argument("--ensemble_size", type=int, default=1,
                   help="Train M JMMs with different seeds for anchored-ensemble "
                        "uncertainty. M=1 (default) saves best.pt; M>1 saves "
                        "best_0.pt..best_{M-1}.pt and a copy as best.pt.")
    p.add_argument("--anchor_lambda", type=float, default=0.0,
                   help="Strength of anchored-ensemble regularisation (paper: 3e-4). "
                        "Only meaningful with --ensemble_size > 1; otherwise the "
                        "anchor is the model's own random init and acts like L2 to "
                        "that init.")

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)


def _resolve_csv_column(path: str, requested: str) -> str:
    """Resolve a column name against a CSV header case-insensitively, so
    ``--smiles_col smiles`` matches a header named ``SMILES`` and vice
    versa. Returns the actual header string to pass to the CSV reader.
    """
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if requested in header:
        return requested
    for h in header:
        if h.lower() == requested.lower():
            return h
    raise SystemExit(
        f"Column {requested!r} not found in {path}; available: {header}"
    )


def cmd_joint_finetune(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ck_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SmilesTokenizer.load(ck_dir / "tokenizer.json")
    ae = SmilesAutoencoder.load(ck_dir / "best.pt")
    print(f"Loaded pretrained autoencoder from {ck_dir} (vocab {tokenizer.vocab_size})")

    smiles_col = _resolve_csv_column(args.smiles, args.smiles_col)

    # --- read labeled data, optionally split-filtered ---
    if args.val_split is not None:
        if args.split_col is None:
            raise SystemExit("--val_split requires --split_col.")
        train_smi, train_y = read_labeled_smiles_csv(
            args.smiles, smiles_col, args.label_col, args.split_col, args.train_split,
        )
        val_smi, val_y = read_labeled_smiles_csv(
            args.smiles, smiles_col, args.label_col, args.split_col, args.val_split,
        )
    else:
        smi, y = read_labeled_smiles_csv(
            args.smiles, smiles_col, args.label_col,
            args.split_col, args.train_split if args.split_col else None,
        )
        if not smi:
            raise SystemExit(f"No labeled rows read from {args.smiles}")
        idx = list(range(len(smi)))
        random.shuffle(idx)
        n_val = max(1, int(len(idx) * args.val_frac))
        val_idx = set(idx[:n_val])
        train_smi = [s for i, s in enumerate(smi) if i not in val_idx]
        train_y = [c for i, c in enumerate(y) if i not in val_idx]
        val_smi = [s for i, s in enumerate(smi) if i in val_idx]
        val_y = [c for i, c in enumerate(y) if i in val_idx]
    print(f"Labeled set: train={len(train_smi)}  val={len(val_smi)}")

    # --- grow vocabulary if the labeled set introduces new tokens ---
    extended = maybe_extend_vocab(ae, tokenizer, train_smi)
    if extended:
        print(f"Extended vocabulary to {tokenizer.vocab_size} tokens "
              f"(new rows are randomly initialised).")
    tokenizer.save(out_dir / "tokenizer.json")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LabeledSmilesDataset(train_smi, train_y, tokenizer, args.max_len)
    val_ds = LabeledSmilesDataset(val_smi, val_y, tokenizer, args.max_len)
    print(f"After length filter: train={len(train_ds)}  val={len(val_ds)}")

    train_loader_factory = lambda sampler: make_labeled_dataloader(
        train_ds, args.batch_size, tokenizer.pad_id,
        shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers,
    )
    val_loader = make_labeled_dataloader(
        val_ds, args.batch_size, tokenizer.pad_id,
        shuffle=False, num_workers=args.num_workers,
    ) if len(val_ds) > 0 else None

    decoder_lr = args.decoder_lr if args.decoder_lr and args.decoder_lr > 0 else None
    M = max(int(args.ensemble_size), 1)

    histories = []
    for m in range(M):
        seed_m = args.seed + m
        random.seed(seed_m)
        torch.manual_seed(seed_m)

        # Each ensemble member gets a fresh classifier head (with its own
        # random init = anchor) on top of the shared pretrained AE weights.
        jmm = JointMolecularModel.from_pretrained_ae(
            ae,
            n_classes=args.n_classes,
            hidden_dims=args.cls_hidden,
            dropout=args.cls_dropout,
        )
        if args.anchor_lambda > 0:
            jmm.initialize_anchors()
        jmm.to(device)
        if m == 0:
            n_params = sum(p.numel() for p in jmm.parameters())
            print(f"JMM parameters: {n_params:,} on {device} "
                  f"(gamma={args.gamma}, ensemble_size={M}, "
                  f"anchor_lambda={args.anchor_lambda})")

        sampler = None
        if not args.no_class_balance:
            sampler = class_balanced_sampler(train_ds.labels, n_classes=args.n_classes)
        train_loader = train_loader_factory(sampler)

        member_save = (
            str(out_dir / "best.pt") if M == 1 else str(out_dir / f"best_{m}.pt")
        )
        if M > 1:
            print(f"--- ensemble member {m+1}/{M} (seed={seed_m}) ---")
        history = joint_fit(
            jmm, train_loader, val_loader, device,
            epochs=args.epochs,
            gamma=args.gamma,
            anchor_lambda=args.anchor_lambda,
            lr=args.lr,
            decoder_lr=decoder_lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            save_path=member_save,
            freeze_encoder=args.freeze_encoder,
            freeze_decoder=args.freeze_decoder,
            teacher_forcing=not args.no_teacher_forcing,
        )
        histories.append({"member": m, "seed": seed_m, "history": history})

    # For ensembles, also drop a copy of member 0 as best.pt so single-model
    # tools that look for best.pt still work.
    if M > 1:
        first = out_dir / "best_0.pt"
        if first.exists():
            (out_dir / "best.pt").write_bytes(first.read_bytes())

    (out_dir / "history.json").write_text(json.dumps(histories, indent=2))
    (out_dir / "args.json").write_text(
        json.dumps({k: v for k, v in vars(args).items() if k != "func"}, indent=2)
    )
    print(f"Done. Saved {M} JMM checkpoint(s) to {out_dir}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def add_evaluate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smiles", required=True,
                   help="SMILES to evaluate. Plain text (one per line) or CSV with a smiles column.")
    p.add_argument("--checkpoint_dir", required=True,
                   help="Directory with best.pt and tokenizer.json.")
    p.add_argument("--output_csv", default="reconstruction_errors.csv")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--decode", action="store_true",
                   help="Also greedy-decode each input and check exact match.")
    p.add_argument("--device", default=None)


def cmd_evaluate(args: argparse.Namespace) -> None:
    ck_dir = Path(args.checkpoint_dir)

    tokenizer = SmilesTokenizer.load(ck_dir / "tokenizer.json")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Detect an ensemble: best_0.pt, best_1.pt, ... in the same directory.
    ensemble_paths = sorted(
        ck_dir.glob("best_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]) if p.stem.split("_", 1)[1].isdigit() else 0,
    )
    is_ensemble = len(ensemble_paths) > 1

    smiles = read_smiles_file(args.smiles)
    if not smiles:
        raise SystemExit(f"No SMILES read from {args.smiles}")

    if is_ensemble:
        jmms = []
        for p in ensemble_paths:
            m = load_checkpoint(p)
            if not isinstance(m, JointMolecularModel):
                raise SystemExit(
                    f"Ensemble member {p} is not a JMM checkpoint."
                )
            m.to(device).eval()
            jmms.append(m)
        n_classes = jmms[0].cls_cfg.n_classes
        print(f"Loaded JMM ensemble of {len(jmms)} from {ck_dir}")
        rows = compute_ensemble_predictions(
            jmms, tokenizer, smiles,
            device=device, batch_size=args.batch_size,
            max_len=args.max_len, decode=args.decode,
        )
        fieldnames = [
            "smiles", "reconstruction_error", "unfamiliarity", "n_tokens",
            "reconstructed", "exact_match", "note",
            "predicted_class", "entropy",
        ] + [f"prob_{c}" for c in range(n_classes)]
    else:
        model = load_checkpoint(ck_dir / "best.pt")
        model.to(device)
        is_jmm = isinstance(model, JointMolecularModel)
        print(f"Loaded {'JMM' if is_jmm else 'autoencoder'} from {ck_dir}")
        rows = compute_reconstruction_errors(
            model, tokenizer, smiles,
            device=device,
            batch_size=args.batch_size,
            max_len=args.max_len,
            decode=args.decode,
        )
        fieldnames = [
            "smiles", "reconstruction_error", "unfamiliarity", "n_tokens",
            "reconstructed", "exact_match", "note",
        ]
        if is_jmm:
            fieldnames.append("predicted_class")
            fieldnames += [f"prob_{c}" for c in range(model.cls_cfg.n_classes)]

    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    valid = [r["reconstruction_error"] for r in rows
             if r["reconstruction_error"] == r["reconstruction_error"]]
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    if valid:
        print(f"Mean reconstruction error: {mean(valid):.4f}")
        print(f"Min / Max: {min(valid):.4f} / {max(valid):.4f}")
    if args.decode:
        em = sum(1 for r in rows if r["exact_match"])
        print(f"Exact-match reconstructions: {em}/{len(rows)} = {em/max(len(rows),1):.2%}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ae_ood",
        description="Autoencoder out-of-distribution toolkit for SMILES.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("pretrain", help="Pretrain a SMILES autoencoder from scratch.")
    add_pretrain_args(p_pre)
    p_pre.set_defaults(func=cmd_pretrain)

    p_ft = sub.add_parser("finetune", help="Fine-tune a pretrained autoencoder on a new dataset.")
    add_finetune_args(p_ft)
    p_ft.set_defaults(func=cmd_finetune)

    p_jft = sub.add_parser(
        "joint_finetune",
        help="Joint fine-tune the JMM (autoencoder + classifier) on labeled SMILES.",
    )
    add_joint_finetune_args(p_jft)
    p_jft.set_defaults(func=cmd_joint_finetune)

    p_eval = sub.add_parser(
        "evaluate",
        help="Compute per-molecule reconstruction error / unfamiliarity (and class "
             "probabilities if the checkpoint is a JMM).",
    )
    add_evaluate_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
