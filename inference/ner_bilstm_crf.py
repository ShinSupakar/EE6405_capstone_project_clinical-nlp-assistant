"""
BiLSTM-CRF NER inference wrapper.

Mirrors sweetprajna2003's `bilstm_crf_FINAL_(1).ipynb`: word-level model with
Embedding(vocab, 100) → BiLSTM(2 layers, hidden=128, bidirectional) → Linear
→ `torchcrf.CRF`. Inference uses the CRF's Viterbi `decode()`.

Checkpoint layout (under models/bilstm_crf/):
    best_model.pt     — state_dict of the full module
    word2idx.json     — token → int (lowercased keys, <PAD>=0, <UNK>=1)
    label2idx.json    — IOB label → int, matching NER_LABELS in config.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchcrf import CRF

from config import DEVICE
from inference.entity_utils import (
    Entity,
    group_iob_tags,
    fill_entity_texts,
    whitespace_tokenize_with_offsets,
)


class _BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _emissions(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x)
        h, _ = self.lstm(e)
        return self.fc(h)

    def decode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> list[list[int]]:
        emissions = self._emissions(x)
        if mask is None:
            mask = torch.ones(x.shape, dtype=torch.bool, device=x.device)
        return self.crf.decode(emissions, mask=mask)


class BiLSTMCRFInference:
    """Word-level BiLSTM-CRF NER. See module docstring for status."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        ckpt = self.model_dir / "best_model.pt"
        w2i = self.model_dir / "word2idx.json"
        l2i = self.model_dir / "label2idx.json"
        missing = [p for p in (ckpt, w2i, l2i) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "BiLSTM-CRF checkpoint not ready yet. Missing:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\n\nAsk sweetprajna2003 to upload `bilstm_crf_ckpt.zip` "
                + "(best_model.pt + word2idx.json + label2idx.json) to the "
                + "NLP Drive folder, then extract into "
                + f"{self.model_dir}."
            )

        self.word2idx = json.loads(w2i.read_text())
        self.label2idx = json.loads(l2i.read_text())
        self.idx2label = {int(v): k for k, v in self.label2idx.items()}
        self.unk_id = self.word2idx.get("<UNK>", 1)
        self.pad_id = self.word2idx.get("<PAD>", 0)

        self.model = _BiLSTMCRF(
            vocab_size=len(self.word2idx),
            num_labels=len(self.label2idx),
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

        ids = [self.word2idx.get(w.lower(), self.unk_id) for w in words]
        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        # Expect `self.model.decode(x) -> list[list[int]]` via CRF Viterbi.
        tag_ids = self.model.decode(x)[0]
        tags = [self.idx2label.get(int(t), "O") for t in tag_ids]
        confs = [1.0] * len(tags)  # CRF doesn't yield per-token posteriors cheaply

        ents = group_iob_tags(words, tags, spans, confidences=confs)
        fill_entity_texts(ents, text)
        return ents
