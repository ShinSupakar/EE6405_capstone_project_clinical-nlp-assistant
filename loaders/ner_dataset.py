"""
NER Dataset Loader — Clinical NLP Project
==========================================
Reads CoNLL-format files produced by the preprocessing notebook and returns
PyTorch-ready batches with subword-aligned labels for BERT-style models.

Usage (in your training notebook):
------------------------------------
    from loaders.ner_dataset import NERDataset, NER_LABEL2ID, NER_ID2LABEL

    train_ds = NERDataset(
        file_path="processed_data/ner/ner_train_merged.conll",
        tokenizer_name="dmis-lab/biobert-v1.1",
        max_length=512,
    )
    val_ds = NERDataset(
        file_path="processed_data/ner/bc5cdr_val.conll",
        tokenizer_name="dmis-lab/biobert-v1.1",
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    # Access label info
    print(train_ds.num_labels)      # number of distinct labels
    print(train_ds.label2id)        # {'O': 0, 'B-DRUG': 1, ...}

Each batch item is a dict with keys:
    input_ids       : LongTensor [seq_len]
    attention_mask  : LongTensor [seq_len]
    token_type_ids  : LongTensor [seq_len]
    labels          : LongTensor [seq_len]  (-100 = ignore in loss)

Subword alignment:
    BERT splits "methylprednisolone" → ["methyl", "##pred", "##nis", "##olone"]
    This loader assigns the word-level label to the FIRST subword and -100 to
    the rest, which is the standard approach for token classification with BERT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ── Default label schema (matches preprocessing_pipeline.ipynb output) ─────────
# Add or remove labels here if your project evolves.
NER_LABEL2ID: Dict[str, int] = {
    "O":            0,
    "B-DRUG":       1,
    "I-DRUG":       2,
    "B-DISEASE":    3,
    "I-DISEASE":    4,
    "B-PROTEIN":    5,
    "I-PROTEIN":    6,
    "B-DNA":        7,
    "I-DNA":        8,
    "B-RNA":        9,
    "I-RNA":       10,
    "B-CELL_LINE": 11,
    "I-CELL_LINE": 12,
    "B-CELL_TYPE": 13,
    "I-CELL_TYPE": 14,
}

NER_ID2LABEL: Dict[int, str] = {v: k for k, v in NER_LABEL2ID.items()}

IGNORE_INDEX = -100  # PyTorch's default value for ignored positions in CrossEntropyLoss


# ── File I/O ────────────────────────────────────────────────────────────────────

def read_conll(file_path: str | Path) -> List[Tuple[List[str], List[str]]]:
    """
    Read a CoNLL-format file.

    Expected format (tab-separated, blank line between sentences):
        Aspirin    B-DRUG
        reduces    O
        fever      O
        .          O
                                ← blank line
        Metformin  B-DRUG
        ...

    Returns:
        List of (tokens, labels) tuples, one per sentence.
    """
    sentences: List[Tuple[List[str], List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
            else:
                parts = line.split("\t")
                tokens.append(parts[0])
                labels.append(parts[1] if len(parts) > 1 else "O")

    if tokens:  # handle file with no trailing newline
        sentences.append((tokens, labels))

    return sentences


def load_label2id_from_file(json_path: str | Path) -> Dict[str, int]:
    """
    Load label2id from the label_list.json saved by the preprocessing notebook.
    Falls back to the default NER_LABEL2ID if the file is not found.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[NERDataset] label_list.json not found at {json_path}, using default label schema.")
        return NER_LABEL2ID

    with open(json_path) as f:
        label_list = json.load(f)

    # label_list.json contains sorted label names; assign IDs by position
    return {label: idx for idx, label in enumerate(label_list)}


# ── Dataset ─────────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """
    PyTorch Dataset for Named Entity Recognition.

    Tokenises with a HuggingFace tokenizer and aligns word-level IOB labels
    to subword tokens automatically.

    Args:
        file_path       : Path to a CoNLL-format file (output of preprocessing notebook).
        tokenizer_name  : HuggingFace model identifier. Recommended: "dmis-lab/biobert-v1.1"
                          or "allenai/scibert_scivocab_uncased".
        max_length      : Maximum number of tokens (including [CLS] and [SEP]).
                          Sequences longer than this are truncated.
        label2id        : Optional override for the label→integer mapping.
                          If None, uses NER_LABEL2ID above.
        label_json_path : Optional path to label_list.json produced by the notebook.
                          If provided, overrides label2id.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer_name: str = "dmis-lab/biobert-v1.1",
        max_length: int = 512,
        label2id: Optional[Dict[str, int]] = None,
        label_json_path: Optional[str | Path] = None,
    ) -> None:
        self.file_path   = Path(file_path)
        self.max_length  = max_length

        # Resolve label mapping
        if label_json_path is not None:
            self.label2id = load_label2id_from_file(label_json_path)
        else:
            self.label2id = label2id or NER_LABEL2ID

        self.id2label = {v: k for k, v in self.label2id.items()}

        print(f"[NERDataset] Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        print(f"[NERDataset] Reading: {self.file_path}")
        raw = read_conll(self.file_path)
        print(f"[NERDataset] Found {len(raw)} sentences in file.")

        self.examples: List[Dict[str, torch.Tensor]] = []
        n_truncated = 0
        n_unknown_labels = 0

        for tokens, labels in raw:
            encoded, was_truncated, had_unknown = self._encode(tokens, labels)
            if encoded is not None:
                self.examples.append(encoded)
                if was_truncated:
                    n_truncated += 1
                if had_unknown:
                    n_unknown_labels += 1

        print(
            f"[NERDataset] Encoded {len(self.examples)} examples "
            f"({n_truncated} truncated, {n_unknown_labels} had unknown labels → mapped to O)."
        )

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode(
        self,
        tokens: List[str],
        labels: List[str],
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], bool, bool]:
        """
        Tokenise a single sentence and align IOB labels to subword tokens.

        Returns:
            (encoded_dict, was_truncated, had_unknown_label)
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,   # tokens are already split
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        was_truncated    = len(tokens) > self.max_length - 2  # rough check
        had_unknown      = False
        word_ids         = encoding.word_ids()   # None for special tokens
        aligned_labels   = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # [CLS], [SEP], [PAD] → ignored by loss
                aligned_labels.append(IGNORE_INDEX)

            elif word_id != previous_word_id:
                # ── First subword of a new word ──
                raw_label = labels[word_id] if word_id < len(labels) else "O"
                label_id  = self.label2id.get(raw_label)

                if label_id is None:
                    # Unknown label (e.g. a GENIA tag the NER person doesn't need)
                    label_id    = self.label2id.get("O", 0)
                    had_unknown = True

                aligned_labels.append(label_id)

            else:
                # ── Continuation subword ──
                # Standard practice: ignore continuation subwords in loss (-100).
                # Some papers propagate I- tags; -100 is simpler and works well.
                aligned_labels.append(IGNORE_INDEX)

            previous_word_id = word_id

        return (
            {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding.get(
                    "token_type_ids",
                    torch.zeros_like(encoding["input_ids"])
                ).squeeze(0),
                "labels": torch.tensor(aligned_labels, dtype=torch.long),
            },
            was_truncated,
            had_unknown,
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    def get_label_list(self) -> List[str]:
        """Return labels in ID order (for passing to AutoModelForTokenClassification)."""
        return [self.id2label[i] for i in range(len(self.id2label))]
