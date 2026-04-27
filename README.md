# SMILES Autoencoder

A small, dependency-light PyTorch implementation of a sequence-to-sequence
autoencoder for SMILES strings. It supports four workflows:

1. **Pretrain** the model from scratch on a large, generic corpus.
2. **Fine-tune** the pretrained model on a smaller, domain-specific dataset.
3. **Joint fine-tune** the autoencoder together with a classifier head on
   labelled SMILES — the joint molecular model (JMM) of van Tilborg et al.
   (2026).
4. **Evaluate** new SMILES and report per-molecule reconstruction error
   (and class probabilities + unfamiliarity, when the checkpoint is a JMM).

## Architecture

```
SMILES tokens  ──►  embedding ──►  encoder ──►  linear ──►  z (latent vector)
                                                              │
                                                              ▼
                                                    init hidden state
                                                              │
SMILES tokens  ◄──  output ◄── GRU decoder (teacher forcing) ◄┘
```

- **Tokenizer**: atom-wise regex tokenizer that handles multi-char tokens
  (`Br`, `Cl`, `[C@@H]`, `%10`, …). Special tokens: `<pad>`, `<bos>`, `<eos>`,
  `<unk>`. Vocabulary is built from the training data and saved alongside
  the model.
- **Encoder** (selectable via `--encoder_type`):
  - `gru` (default): embedding → bidirectional GRU → linear projection to
    a fixed latent vector `z`.
  - `cnn`: the joint molecular model (JMM) encoder of van Tilborg et al.,
    *Nature Machine Intelligence* (2026). Embedding → stacked
    `Conv1d → ReLU → MaxPool1d → Dropout` blocks (all stride 1, no padding;
    max-pool kernel matches the conv kernel) → flatten → linear projection
    to `z`. Inputs are padded/truncated to a fixed length (`--cnn_max_len`)
    so the flattened feature has a fixed size. Hyperparameter ranges
    explored in the paper: 2–3 conv layers, 256–512 filters, kernel size
    6–8.
- **Decoder** (selectable via `--decoder_type`):
  - `gru` (default): linear projection from `z` initialises a
    unidirectional GRU that predicts the next token autoregressively.
  - `lstm`: the conditioned LSTM decoder of van Tilborg et al. (2026).
    `z` is projected to `h_0` of shape `(num_layers, B, dec_hidden)`; the
    cell state `c_0` is zero-initialised.

  Either decoder is trained with cross-entropy on next-token prediction
  (PAD ignored). Teacher forcing is on by default; pass
  `--no_teacher_forcing` to feed each step the model's own argmax
  prediction instead — the autoregressive training procedure used in the
  paper.
- **Classifier head** (used only by the JMM workflow): an MLP on top of
  `z` (configurable hidden sizes / dropout, default `[1024, 1024]`),
  ReLU activations, cross-entropy loss. The autoencoder and classifier
  share the same `z` and are trained jointly with
  `L_JMM = L_reconstruction + γ · L_classifier` (default `γ = 0.1`,
  matching the paper).
- **Reconstruction error** at evaluation time is the mean per-token
  cross-entropy on each input sequence — lower means the model can
  reproduce the molecule more faithfully. Optionally we also greedy-decode
  and report exact-match accuracy.
- **Unfamiliarity** `U(x) = log(reconstruction_error)` — the OOD/
  applicability-domain signal proposed in the paper. Higher means the
  model is less able to reproduce the molecule, i.e. it sits further from
  the learned data distribution. Reported in every evaluation run.

## Layout

```
smiles_ae/
    __init__.py        public API
    tokenizer.py       atom-wise regex tokenizer
    model.py           encoders (GRU + 1D-CNN), decoder, autoencoder,
                       Classifier, JointMolecularModel
    data.py            Dataset/collate (unlabeled + labeled), class-balanced sampler
    trainer.py         train_one_epoch, fit, joint_train_one_epoch, joint_fit
    inference.py       compute_reconstruction_errors (Python API)
    utils.py           file I/O, labeled CSV reader, vocab extension
ae_ood.py              unified CLI with `pretrain`, `finetune`, `joint_finetune`,
                       `evaluate` subcommands
requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

The only required dependency is PyTorch. (RDKit is not used here — feed the
SMILES already canonicalised if that matters for your application.)

## Data format

For the unsupervised workflows (`pretrain`, `finetune`, `evaluate`) a plain
text file with one SMILES per line is accepted. CSV/TSV with `smiles` as
the first column also works (a header row called `smiles` is skipped).
Lines starting with `#` are treated as comments.

For `joint_finetune` you need a CSV with **only two required columns**:
a SMILES column (either `smiles` or `SMILES` is auto-detected; override
with `--smiles_col`) and an integer-label column (default `y`,
configurable with `--label_col`). Any other columns in the file are
ignored. Minimal example:

```
smiles,y
Cc1ccc(C(=O)NC2CC(F)(F)C2)cc1-c1ccc2cc(NC(=O)C3CC3)ncc2c1,1
CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1,0
...
```

Every labelled row is used for training and `--val_frac` is held out at
random for validation.

## CLI

All workflows are exposed as subcommands of a single `ae_ood.py`
(autoencoder out-of-distribution) script:

```bash
python ae_ood.py {pretrain,finetune,joint_finetune,evaluate} ...
```

### Pretraining

Default (bi-GRU) encoder:

```bash
python ae_ood.py pretrain \
    --smiles data/zinc_train.smi \
    --out_dir runs/pretrain \
    --epochs 30 --batch_size 256 --lr 1e-3 \
    --embed_dim 128 --enc_hidden 256 --dec_hidden 512 --latent_dim 128
```

JMM-style 1D-CNN encoder + conditioned LSTM decoder (van Tilborg et al., 2026):

```bash
python ae_ood.py pretrain \
    --smiles data/zinc_train.smi \
    --out_dir runs/pretrain_cnn_lstm \
    --epochs 30 --batch_size 256 --lr 1e-3 \
    --encoder_type cnn \
    --cnn_layers 2 --cnn_filters 256 --cnn_kernel_size 8 --cnn_max_len 128 \
    --decoder_type lstm \
    --dec_layers 2 --dec_hidden 512 \
    --embed_dim 128 --latent_dim 128
    # add --no_teacher_forcing to match the paper's autoregressive training
```

This produces `runs/pretrain/best.pt`, `runs/pretrain/tokenizer.json`,
`runs/pretrain/history.json`, and `runs/pretrain/args.json`. The encoder
choice and its hyperparameters are stored in `best.pt` and reloaded
automatically by `finetune` / `evaluate`.

### Fine-tuning

```bash
python ae_ood.py finetune \
    --smiles data/my_actives.smi \
    --checkpoint_dir runs/pretrain \
    --out_dir runs/finetune \
    --epochs 10 --batch_size 64 --lr 2e-4
```

The fine-tuner loads the pretrained tokenizer and model. If the new dataset
contains tokens that were not in the pretraining vocabulary, the
embedding and output layers are grown automatically and the new rows are
randomly initialised — those weights are then learned during fine-tuning.

To freeze the encoder and only adapt the decoder:

```bash
python ae_ood.py finetune ... --freeze_encoder
```

### Joint fine-tuning (JMM)

Joint fine-tuning starts from a pretrained autoencoder, attaches a fresh
MLP classifier on top of `z`, and trains the whole stack with the joint
loss `L_JMM = L_reconstruction + γ · L_classifier`:

```bash
python ae_ood.py joint_finetune \
    --smiles labeled.csv \
    --checkpoint_dir runs/pretrain_cnn_lstm \
    --out_dir runs/jmm \
    --label_col y --n_classes 2 \
    --val_frac 0.1 \
    --cls_hidden 1024 1024 --cls_dropout 0.0 \
    --gamma 0.1 \
    --epochs 20 --batch_size 64 \
    --lr 3e-6 --decoder_lr 3e-7 \
    --ensemble_size 10 --anchor_lambda 3e-4
```

Notes:

- The SMILES column header is auto-detected: either `smiles` or `SMILES`
  works out of the box. Use `--smiles_col` to override.
- Every labelled row is used for training and `--val_frac` of it is held
  out at random for validation.
- Class imbalance is handled by a weighted sampler implementing the
  paper's `P_c = 1 − n_c / N` (eq. 14). Pass `--no_class_balance` to
  disable.
- `--lr` controls the learning rate of the encoder and classifier; the
  decoder gets `--decoder_lr` (defaults `3e-6` and `3e-7` per the paper).
  Pass `--decoder_lr 0` to use a single LR for all parameters.
- `--freeze_encoder` / `--freeze_decoder` are available if you want the
  classifier to specialise on a fixed `z`.
- The new vocabulary (if the labelled SMILES introduce tokens unseen in
  pretraining) is grown automatically — same mechanism as plain `finetune`.

#### Anchored ensembling (uncertainty)

`--ensemble_size M` trains *M* JMM members in a single CLI call, each
with its own random seed (`seed`, `seed+1`, …, `seed+M-1`) and its own
classifier-init anchor. The classifier loss is augmented with
`λ · Σ ‖θ − θ_anchor‖²` (eq. 10 of the paper, `λ` controlled by
`--anchor_lambda`, paper default `3e-4`). Outputs:

- `M = 1` (default): a single `best.pt`, identical to before.
- `M > 1`: `best_0.pt`, `best_1.pt`, …, `best_{M-1}.pt`, plus a copy of
  `best_0.pt` named `best.pt` so single-model tools that look for
  `best.pt` keep working.

`evaluate` auto-detects the ensemble from the presence of `best_*.pt`
files and averages the per-model class probabilities, derives a
prediction uncertainty `H(y|x) = -Σ p log p`, and averages the
reconstruction error / unfamiliarity across members.

The `--no_teacher_forcing` flag from `pretrain` is also available on
`joint_finetune` if you want the JMM's reconstruction branch trained
autoregressively.

### Computing reconstruction error on new molecules

```bash
python ae_ood.py evaluate \
    --smiles data/test.smi \
    --checkpoint_dir runs/finetune \
    --output_csv test_errors.csv \
    --decode
```

`evaluate` auto-detects whether the checkpoint is a plain autoencoder or a
JMM and emits the corresponding columns:

| column                | meaning                                                   |
|-----------------------|-----------------------------------------------------------|
| `smiles`              | original input                                            |
| `reconstruction_error`| mean cross-entropy per non-pad token (NaN if skipped)     |
| `unfamiliarity`       | `log(reconstruction_error)` — `U(x)` from van Tilborg 2026 |
| `n_tokens`            | number of target tokens contributing to the error         |
| `reconstructed`       | greedy-decoded SMILES if `--decode`, otherwise empty      |
| `exact_match`         | whether the decoded string equals the input               |
| `note`                | reason for skipping (e.g. exceeds `--max_len`)            |
| `predicted_class`     | argmax of the classifier logits (or argmax of the *mean* probabilities for an ensemble) — **JMM only** |
| `prob_<i>`            | softmax probability of class `i`, averaged across ensemble members — **JMM only** |
| `entropy`             | `H(y|x) = -Σ p log p` of the mean class probabilities — **ensemble only** |

The error is comparable across molecules of different lengths because it is
already a per-token average. If you would prefer the un-normalised total
negative log-likelihood, multiply the error by `n_tokens`.

## Python API

Everything the CLIs do is also exposed programmatically:

```python
import torch
from smiles_ae import (
    AutoencoderConfig, SmilesAutoencoder, SmilesTokenizer,
    SmilesDataset, make_dataloader, fit, maybe_extend_vocab,
    compute_reconstruction_errors,
)

# --- pretrain ---
tokenizer = SmilesTokenizer().fit(train_smiles)
cfg = AutoencoderConfig(vocab_size=tokenizer.vocab_size,
                        pad_id=tokenizer.pad_id,
                        bos_id=tokenizer.bos_id,
                        eos_id=tokenizer.eos_id)
model = SmilesAutoencoder(cfg)

train_ds = SmilesDataset(train_smiles, tokenizer, max_len=128)
train_loader = make_dataloader(train_ds, batch_size=128, pad_id=tokenizer.pad_id)
fit(model, train_loader, val_loader=None, device="cuda", epochs=20, lr=1e-3,
    save_path="pretrained.pt")
tokenizer.save("tokenizer.json")

# --- fine-tune ---
tokenizer = SmilesTokenizer.load("tokenizer.json")
model = SmilesAutoencoder.load("pretrained.pt")
maybe_extend_vocab(model, tokenizer, finetune_smiles)
ft_ds = SmilesDataset(finetune_smiles, tokenizer, max_len=128)
ft_loader = make_dataloader(ft_ds, batch_size=64, pad_id=tokenizer.pad_id)
fit(model, ft_loader, val_loader=None, device="cuda", epochs=10, lr=2e-4,
    save_path="finetuned.pt")

# --- evaluate ---
rows = compute_reconstruction_errors(
    model, tokenizer, new_smiles_list, device="cuda", decode=True,
)
for r in rows:
    print(r["smiles"], r["reconstruction_error"], r["exact_match"])
```

### Joint molecular model (programmatic)

```python
from smiles_ae import (
    JointMolecularModel, LabeledSmilesDataset, SmilesAutoencoder,
    SmilesTokenizer, class_balanced_sampler, compute_reconstruction_errors,
    joint_fit, load_checkpoint, make_labeled_dataloader,
    read_labeled_smiles_csv,
)

# Load a pretrained autoencoder and attach a fresh classifier head.
tokenizer = SmilesTokenizer.load("runs/pretrain_cnn/tokenizer.json")
ae = SmilesAutoencoder.load("runs/pretrain_cnn/best.pt")
jmm = JointMolecularModel.from_pretrained_ae(
    ae, n_classes=2, hidden_dims=[1024, 1024], dropout=0.0,
)

# Labeled data.
all_smi, all_y = read_labeled_smiles_csv(
    "labeled.csv", smiles_col="smiles", label_col="y",
)
n_val = max(1, len(all_smi) // 10)
val_smi, val_y = all_smi[:n_val], all_y[:n_val]
train_smi, train_y = all_smi[n_val:], all_y[n_val:]

train_ds = LabeledSmilesDataset(train_smi, train_y, tokenizer, max_len=128)
val_ds = LabeledSmilesDataset(val_smi, val_y, tokenizer, max_len=128)
sampler = class_balanced_sampler(train_ds.labels, n_classes=2)
train_loader = make_labeled_dataloader(
    train_ds, batch_size=64, pad_id=tokenizer.pad_id, sampler=sampler,
)
val_loader = make_labeled_dataloader(
    val_ds, batch_size=64, pad_id=tokenizer.pad_id, shuffle=False,
)

joint_fit(
    jmm, train_loader, val_loader, device="cuda",
    epochs=20, gamma=0.1, lr=3e-6, decoder_lr=3e-7,
    save_path="runs/jmm/best.pt",
)

# Evaluate (also works on plain autoencoders — same call).
jmm = load_checkpoint("runs/jmm/best.pt")
rows = compute_reconstruction_errors(jmm, tokenizer, val_smi, device="cuda")
for r in rows[:3]:
    print(r["smiles"], "U=%.3f" % r["unfamiliarity"],
          "pred=%d" % r["predicted_class"], "p1=%.3f" % r["prob_1"])
```

## Notes and tips

- **Validation split** is taken from the input file via `--val_frac`. The
  best-validation checkpoint is saved as `best.pt`.
- **Length filtering**: sequences whose total length (including BOS/EOS)
  exceeds `--max_len` are silently dropped during training. At evaluation
  they receive a NaN error and a `note` saying so.
- **Canonicalisation** is the user's responsibility. If you want to
  ignore stylistic differences ("CCO" vs "OCC") canonicalise both the
  training data and the evaluation inputs first (RDKit's
  `Chem.MolToSmiles(Chem.MolFromSmiles(s))` works well).
- **Out-of-distribution detection**: a high reconstruction error on a
  fine-tuned model is a useful signal that a molecule is unlike the
  fine-tuning set. Calibrate a threshold on a held-out in-domain set.
- **Determinism**: pass `--seed` to `ae_ood.py pretrain` / `ae_ood.py
  finetune` for reproducible splits and weight initialisation. CuDNN
  nondeterminism is not explicitly disabled.
- **CNN encoder length**: the 1D-CNN encoder needs a fixed input length
  because the flattened feature map feeds a single linear layer. Inputs
  shorter than `--cnn_max_len` are PAD-padded; longer inputs are
  truncated. Choose `--cnn_max_len` so it both covers your typical SMILES
  length and satisfies `cnn_max_len > 2 × cnn_layers × (cnn_kernel_size − 1)`
  (otherwise the feature length collapses and the encoder will error on
  construction).

## References

- van Tilborg, D., Rossen, L. & Grisoni, F. *Molecular deep learning at
  the edge of chemical space.* Nature Machine Intelligence (2026).
  doi:10.1038/s42256-026-01216-w. The architectural pieces of the JMM
  described in that paper are all wired up here:
  1. **1D-CNN encoder** (`--encoder_type cnn`).
  2. **Conditioned LSTM decoder** (`--decoder_type lstm`), with optional
     non-teacher-forcing autoregressive training (`--no_teacher_forcing`).
  3. **Classifier head + joint training** (`joint_finetune`) with the
     joint loss `L_recon + γ · L_cls`, the `P_c = 1 − n_c/N`
     class-balanced sampler, and separate learning rates for the decoder
     and the encoder/classifier.
  4. **Unfamiliarity** `U(x) = log L_reconstruction(x)`, reported by
     `evaluate` for both plain-AE and JMM checkpoints.
  5. **Anchored-ensemble uncertainty** (`--ensemble_size M
     --anchor_lambda 3e-4`) following Pearce et al. (2020). At
     evaluation time, ensembles are auto-detected and reported with mean
     class probabilities and `H(y|x)`.
