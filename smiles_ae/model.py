"""Seq2seq autoencoder for SMILES strings.

Two encoder architectures are supported:

* ``"gru"`` (default): token embedding -> bidirectional GRU -> linear
  projection to a latent vector z.
* ``"cnn"``: the joint molecular model (JMM) encoder of van Tilborg et al.,
  Nature Machine Intelligence (2026) — token embedding -> stacked 1D
  convolutional layers (Conv1d -> ReLU -> MaxPool1d -> Dropout, all with
  stride 1 and no padding) -> flatten -> linear projection to z. SMILES
  are padded/truncated to a fixed length so the flattened feature has a
  fixed size.

Decoder: z -> initial hidden state of a unidirectional GRU that predicts
the next token given the previous (teacher forcing at train time, greedy
at inference time).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoencoderConfig:
    vocab_size: int
    embed_dim: int = 128
    enc_hidden: int = 256
    dec_hidden: int = 512
    latent_dim: int = 128
    enc_layers: int = 1
    dec_layers: int = 1
    dropout: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Encoder choice. "gru" preserves the original bi-GRU encoder; "cnn"
    # selects the JMM-style 1D-CNN encoder of van Tilborg et al. (2026).
    encoder_type: str = "gru"

    # CNN encoder hyperparameters (ignored when encoder_type == "gru").
    cnn_layers: int = 2
    cnn_filters: int = 256
    cnn_kernel_size: int = 8
    cnn_max_len: int = 128

    # Decoder choice. "gru" preserves the original GRU decoder; "lstm" is
    # the conditioned LSTM decoder of van Tilborg et al. (2026).
    decoder_type: str = "gru"


class GRUEncoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_id)
        self.gru = nn.GRU(
            cfg.embed_dim,
            cfg.enc_hidden,
            cfg.enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.enc_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(cfg.enc_hidden * 2, cfg.latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h: (num_layers * 2, B, H)
        # Concatenate the final forward and backward hidden states of the top layer.
        h_last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2H)
        return self.to_latent(h_last)


class CNNEncoder(nn.Module):
    """1D-CNN encoder following the JMM of van Tilborg et al. (2026).

    Embedding -> N x {Conv1d -> ReLU -> MaxPool1d -> Dropout} -> flatten ->
    linear projection to z. Both Conv1d and MaxPool1d use stride 1 and no
    padding; max-pool kernel size matches the convolutional kernel size.
    Inputs are padded (or truncated) to ``cfg.cnn_max_len`` so the flattened
    feature map has a fixed size.
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_id)

        layers: list[nn.Module] = []
        in_channels = cfg.embed_dim
        T = cfg.cnn_max_len
        k = cfg.cnn_kernel_size
        for _ in range(cfg.cnn_layers):
            layers.append(nn.Conv1d(in_channels, cfg.cnn_filters, k, stride=1, padding=0))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(k, stride=1, padding=0))
            layers.append(nn.Dropout(cfg.dropout))
            in_channels = cfg.cnn_filters
            T -= (k - 1)  # conv with stride 1, no padding
            T -= (k - 1)  # max-pool with stride 1, no padding
        if T <= 0:
            raise ValueError(
                f"cnn_max_len={cfg.cnn_max_len} is too small for "
                f"{cfg.cnn_layers} layers of kernel size {cfg.cnn_kernel_size}: "
                f"feature length collapses to {T}."
            )
        self.conv = nn.Sequential(*layers)
        self._flatten_dim = cfg.cnn_filters * T
        self.to_latent = nn.Linear(self._flatten_dim, cfg.latent_dim)

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        T = self.cfg.cnn_max_len
        B, T_in = x.size()
        if T_in == T:
            return x
        if T_in < T:
            pad = x.new_full((B, T - T_in), self.cfg.pad_id)
            return torch.cat([x, pad], dim=1)
        return x[:, :T]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self._pad_or_truncate(x)
        emb = self.embed(x)               # (B, T, E)
        h = emb.transpose(1, 2)           # (B, E, T)
        h = self.conv(h)                  # (B, F, T')
        h = h.flatten(1)                  # (B, F * T')
        return self.to_latent(h)


def build_encoder(cfg: AutoencoderConfig) -> nn.Module:
    if cfg.encoder_type == "gru":
        return GRUEncoder(cfg)
    if cfg.encoder_type == "cnn":
        return CNNEncoder(cfg)
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type!r} (expected 'gru' or 'cnn').")


# Backwards-compatible alias for the original bi-GRU encoder.
Encoder = GRUEncoder


class _BaseDecoder(nn.Module):
    """Shared logic for the GRU and LSTM decoders.

    Subclasses define ``self.rnn_module`` (a GRU or LSTM) and the
    ``init_hidden`` method. ``rnn_module`` must accept ``(emb, hidden)`` and
    return ``(out, hidden_next)``.
    """

    rnn_module: nn.Module  # set by subclass

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_id)
        # Project z to an initial hidden state for every decoder layer.
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.dec_hidden * cfg.dec_layers)
        self.out = nn.Linear(cfg.dec_hidden, cfg.vocab_size)

    def _project_z(self, z: torch.Tensor) -> torch.Tensor:
        # (B, L*H) -> (B, L, H) -> (L, B, H)
        h = self.from_latent(z).view(z.size(0), self.cfg.dec_layers, self.cfg.dec_hidden)
        return torch.tanh(h).transpose(0, 1).contiguous()

    def init_hidden(self, z: torch.Tensor):
        raise NotImplementedError

    def forward(
        self,
        z: torch.Tensor,
        tgt_in: torch.Tensor,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        if teacher_forcing:
            h0 = self.init_hidden(z)
            emb = self.embed(tgt_in)
            out, _ = self.rnn_module(emb, h0)
            return self.out(out)
        # Autoregressive — match van Tilborg et al. (2026): feed the model's
        # own argmax prediction back as the next-step input. The first token
        # comes from tgt_in (BOS); subsequent tokens are predictions. The
        # cross-entropy loss is still computed against the teacher-forcing
        # tgt_out, but inputs at every step except the first are predicted
        # tokens whose argmax does not propagate gradients.
        h = self.init_hidden(z)
        B, T = tgt_in.size()
        tok = tgt_in[:, :1]
        all_logits = []
        for _ in range(T):
            emb = self.embed(tok)
            o, h = self.rnn_module(emb, h)
            logits = self.out(o)            # (B, 1, V)
            all_logits.append(logits)
            tok = logits.argmax(-1).detach()
        return torch.cat(all_logits, dim=1)

    @torch.no_grad()
    def generate(
        self, z: torch.Tensor, max_len: int, bos_id: int, eos_id: int
    ) -> torch.Tensor:
        """Greedy decode. Returns a (B, T) LongTensor; sequences end at EOS."""
        B = z.size(0)
        device = z.device
        h = self.init_hidden(z)
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        outs = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len):
            emb = self.embed(tok)
            o, h = self.rnn_module(emb, h)
            logits = self.out(o[:, -1])
            nxt = logits.argmax(-1)
            # Once a sequence has finished, keep emitting EOS so output is well-formed.
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            outs.append(nxt)
            finished = finished | (nxt == eos_id)
            tok = nxt.unsqueeze(1)
            if bool(finished.all()):
                break
        return torch.stack(outs, dim=1)


class GRUDecoder(_BaseDecoder):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__(cfg)
        # Attribute name kept as ``gru`` so legacy checkpoints continue to load.
        self.gru = nn.GRU(
            cfg.embed_dim,
            cfg.dec_hidden,
            cfg.dec_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.dec_layers > 1 else 0.0,
        )

    @property
    def rnn_module(self) -> nn.Module:  # type: ignore[override]
        return self.gru

    def init_hidden(self, z: torch.Tensor) -> torch.Tensor:
        return self._project_z(z)


class LSTMDecoder(_BaseDecoder):
    """Conditioned LSTM decoder of van Tilborg et al. (2026).

    The latent ``z`` is projected to an initial hidden state ``h_0`` of shape
    ``(num_layers, batch, dec_hidden)``; the cell state ``c_0`` is zero
    (matching the paper, which only specifies ``h_0`` initialisation).
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__(cfg)
        self.lstm = nn.LSTM(
            cfg.embed_dim,
            cfg.dec_hidden,
            cfg.dec_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.dec_layers > 1 else 0.0,
        )

    @property
    def rnn_module(self) -> nn.Module:  # type: ignore[override]
        return self.lstm

    def init_hidden(self, z: torch.Tensor):
        h = self._project_z(z)
        c = torch.zeros_like(h)
        return (h, c)


def build_decoder(cfg: AutoencoderConfig) -> nn.Module:
    if cfg.decoder_type == "gru":
        return GRUDecoder(cfg)
    if cfg.decoder_type == "lstm":
        return LSTMDecoder(cfg)
    raise ValueError(
        f"Unknown decoder_type: {cfg.decoder_type!r} (expected 'gru' or 'lstm')."
    )


# Backwards-compatible alias for the original GRU decoder.
Decoder = GRUDecoder


class SmilesAutoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(cfg)
        self.decoder = build_decoder(cfg)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
        teacher_forcing: bool = True,
    ):
        z = self.encoder(src, src_lengths)
        logits = self.decoder(z, tgt_in, teacher_forcing=teacher_forcing)
        return logits, z

    # --- losses ---
    def reconstruction_loss(
        self, logits: torch.Tensor, tgt_out: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.cfg.pad_id,
            reduction=reduction,
        )

    def per_sequence_loss(
        self, logits: torch.Tensor, tgt_out: torch.Tensor
    ) -> torch.Tensor:
        """Mean cross-entropy per non-pad token, returned per sequence in the batch."""
        B, T, V = logits.size()
        ce = F.cross_entropy(
            logits.reshape(-1, V),
            tgt_out.reshape(-1),
            ignore_index=self.cfg.pad_id,
            reduction="none",
        ).reshape(B, T)
        mask = (tgt_out != self.cfg.pad_id).float()
        return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)

    # --- persistence ---
    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "config": asdict(self.cfg)}, path)

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "SmilesAutoencoder":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = AutoencoderConfig(**ck["config"])
        model = cls(cfg)
        model.load_state_dict(ck["state_dict"])
        return model


# ---------------------------------------------------------------------------
# Joint molecular model (van Tilborg et al., Nat. Mach. Intell., 2026)
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    in_dim: int
    n_classes: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024])
    dropout: float = 0.0


class Classifier(nn.Module):
    """MLP classifier head on top of the latent z.

    The paper uses 2–3 fully connected layers with hidden size 1024 or 2048
    and ReLU activations, with class-imbalance handled at the sampler level.
    """

    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        prev = cfg.in_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, cfg.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class JointMolecularModel(nn.Module):
    """JMM: SMILES autoencoder + MLP classifier on z, trained jointly.

    Joint loss: ``L_JMM = L_reconstruction + gamma * L_classifier`` (paper
    uses ``gamma = 0.1``). The latent ``z`` is shared between reconstruction
    and property prediction so the encoder learns features that serve both.
    """

    def __init__(self, ae_cfg: AutoencoderConfig, cls_cfg: ClassifierConfig):
        super().__init__()
        if cls_cfg.in_dim != ae_cfg.latent_dim:
            raise ValueError(
                f"Classifier in_dim ({cls_cfg.in_dim}) must equal autoencoder "
                f"latent_dim ({ae_cfg.latent_dim})."
            )
        self.ae_cfg = ae_cfg
        self.cls_cfg = cls_cfg
        self.autoencoder = SmilesAutoencoder(ae_cfg)
        self.classifier = Classifier(cls_cfg)
        # Anchors for anchored-ensemble regularization (Pearce et al., 2020;
        # van Tilborg et al. 2026 eq. (10)). Set per ensemble member by
        # calling ``initialize_anchors()`` after construction.
        self._cls_anchors: list[torch.Tensor] | None = None

    @property
    def cfg(self) -> AutoencoderConfig:
        # Convenience: most callers reach into model.cfg for ae fields.
        return self.ae_cfg

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
        teacher_forcing: bool = True,
    ):
        recon_logits, z = self.autoencoder(
            src, src_lengths, tgt_in, teacher_forcing=teacher_forcing
        )
        cls_logits = self.classifier(z)
        return recon_logits, z, cls_logits

    def joint_loss(
        self,
        recon_logits: torch.Tensor,
        tgt_out: torch.Tensor,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
        gamma: float = 0.1,
        anchor_lambda: float = 0.0,
    ):
        L_recon = self.autoencoder.reconstruction_loss(recon_logits, tgt_out)
        L_cls = F.cross_entropy(cls_logits, labels)
        if anchor_lambda > 0.0 and self._cls_anchors is not None:
            L_cls = L_cls + anchor_lambda * self.anchor_loss()
        return L_recon + gamma * L_cls, L_recon, L_cls

    def initialize_anchors(self) -> None:
        """Snapshot current classifier weights as the anchor for anchored
        ensembling. Call this once after construction (and seeded init)
        before training each ensemble member.
        """
        self._cls_anchors = [
            p.detach().clone() for p in self.classifier.parameters()
        ]

    def anchor_loss(self) -> torch.Tensor:
        """Sum of squared deviations of classifier params from their anchors.

        Returns 0 if no anchors have been set (so callers can pass
        ``anchor_lambda > 0`` safely on un-anchored models — it's a no-op).
        """
        device = next(self.classifier.parameters()).device
        if self._cls_anchors is None:
            return torch.zeros((), device=device)
        terms = []
        for p, a in zip(self.classifier.parameters(), self._cls_anchors):
            terms.append(((p - a.to(device)) ** 2).sum())
        return torch.stack(terms).sum()

    # --- losses on the AE side, exposed for trainer compatibility ---
    def reconstruction_loss(self, *args, **kwargs):
        return self.autoencoder.reconstruction_loss(*args, **kwargs)

    def per_sequence_loss(self, *args, **kwargs):
        return self.autoencoder.per_sequence_loss(*args, **kwargs)

    @property
    def encoder(self) -> nn.Module:
        return self.autoencoder.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.autoencoder.decoder

    # --- persistence ---
    def save(self, path) -> None:
        anchors = (
            [a.detach().cpu() for a in self._cls_anchors]
            if self._cls_anchors is not None
            else None
        )
        torch.save(
            {
                "model_type": "jmm",
                "ae_config": asdict(self.ae_cfg),
                "cls_config": asdict(self.cls_cfg),
                "state_dict": self.state_dict(),
                "cls_anchors": anchors,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "JointMolecularModel":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        if ck.get("model_type") != "jmm":
            raise ValueError(
                f"Checkpoint at {path} is not a JointMolecularModel (model_type="
                f"{ck.get('model_type')!r}). Use SmilesAutoencoder.load instead."
            )
        ae_cfg = AutoencoderConfig(**ck["ae_config"])
        cls_cfg = ClassifierConfig(**ck["cls_config"])
        model = cls(ae_cfg, cls_cfg)
        model.load_state_dict(ck["state_dict"])
        if ck.get("cls_anchors") is not None:
            model._cls_anchors = [a.to(map_location) for a in ck["cls_anchors"]]
        return model

    @classmethod
    def from_pretrained_ae(
        cls,
        ae: "SmilesAutoencoder",
        n_classes: int = 2,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.0,
    ) -> "JointMolecularModel":
        """Wrap a pretrained autoencoder with a fresh classifier head."""
        cls_cfg = ClassifierConfig(
            in_dim=ae.cfg.latent_dim,
            n_classes=n_classes,
            hidden_dims=list(hidden_dims) if hidden_dims else [1024, 1024],
            dropout=dropout,
        )
        model = cls(ae.cfg, cls_cfg)
        model.autoencoder.load_state_dict(ae.state_dict())
        return model


def load_checkpoint(path, map_location: str = "cpu"):
    """Load either a SmilesAutoencoder or a JointMolecularModel based on
    the ``model_type`` field in the checkpoint.
    """
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if ck.get("model_type") == "jmm":
        return JointMolecularModel.load(path, map_location=map_location)
    return SmilesAutoencoder.load(path, map_location=map_location)
