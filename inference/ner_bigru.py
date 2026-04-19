"""
BiGRU NER inference wrapper.

Mirrors tan.hg.001116's `clinical-ner-bi-gru-genia-notebook.ipynb`:
Embedding → bidirectional GRU → Linear over NER_LABELS. Plain argmax decoding
(no CRF).

Status
------
Scaffolded stub. Replace `_GRUNER.__init__` with the exact architecture from
the teammate notebook once `bigru_ckpt.zip` is uploaded:
    best_model.pt     — state_dict
    vocab.json        — word → int
    label_list.json   — list[str], index = label id
    hparams.json      — {embed_dim, hidden_dim, num_layers, dropout, ...}
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE
from inference.entity_utils import (
    Entity,
    group_iob_tags,
    fill_entity_texts,
    whitespace_tokenize_with_offsets,
)


class _GRUNER(nn.Module):
    """
    Placeholder. Fill in from the teammate's actual `GRUNERModel` class.
    Keeping a minimal forward signature so tests can still import this file.
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)
        h, _ = self.gru(e)
        return self.classifier(h)


class BiGRUInference:
    """Word-level BiGRU NER."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        ckpt = self.model_dir / "best_model.pt"
        vocab_p = self.model_dir / "vocab.json"
        labels_p = self.model_dir / "label_list.json"
        hparams_p = self.model_dir / "hparams.json"
        missing = [p for p in (ckpt, vocab_p, labels_p) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "BiGRU checkpoint not ready yet. Missing:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\n\nAsk tan.hg.001116 to upload `bigru_ckpt.zip` "
                + "(best_model.pt + vocab.json + label_list.json + hparams.json) "
                + "to the NLP Drive folder, then extract into "
                + f"{self.model_dir}."
            )

        self.vocab = json.loads(vocab_p.read_text())
        self.label_list = json.loads(labels_p.read_text())
        hparams = json.loads(hparams_p.read_text()) if hparams_p.exists() else {}

        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

        self.model = _GRUNER(
            vocab_size=len(self.vocab),
            num_labels=len(self.label_list),
            embed_dim=hparams.get("emb_dim", hparams.get("embed_dim", 128)),
            hidden_dim=hparams.get("hidden_dim", 256),
            num_layers=hparams.get("num_layers", 1),
            dropout=hparams.get("dropout", 0.3),
            pad_idx=self.pad_id,
        )
        state = torch.load(ckpt, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        words, spans = whitespace_tokenize_with_offsets(text)
        if not words:
            return []

        ids = [self.vocab.get(w.lower(), self.unk_id) for w in words]
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        logits = self.model(x)[0]  # [seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)

        tags = [self.label_list[int(p)] for p in preds.tolist()]
        confidences = [float(c) for c in confs.tolist()]

        ents = group_iob_tags(words, tags, spans, confidences=confidences)
        fill_entity_texts(ents, text)
        return ents
