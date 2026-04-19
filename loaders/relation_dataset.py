"""
Relation Extraction Dataset Loader — Clinical NLP Project
==========================================================
Reads the DrugBank CSV produced by the preprocessing notebook and returns
PyTorch-ready batches for a BERT-style relation extraction model.

Entity marking strategy:
    Drug and disease entities are wrapped in special tokens so the model
    can learn to focus on them:

        [E1] Metformin [/E1] is used to treat [E2] type 2 diabetes [/E2] .

    At inference time, the hidden states at [E1] and [E2] positions are
    concatenated and passed to a classification head — a standard approach
    from the "Matching the Blanks" and R-BERT papers.

Usage (in your training notebook):
------------------------------------
    from loaders.relation_dataset import RelationDataset, RELATION_LABEL2ID, RELATION_ID2LABEL

    train_ds = RelationDataset(
        file_path="processed_data/relations/drugbank_train.csv",
        tokenizer_name="dmis-lab/biobert-v1.1",
    )
    val_ds = RelationDataset(
        file_path="processed_data/relations/drugbank_val.csv",
        tokenizer_name="dmis-lab/biobert-v1.1",
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

Each batch item is a dict with keys:
    input_ids       : LongTensor [seq_len]
    attention_mask  : LongTensor [seq_len]
    e1_pos          : LongTensor scalar  — position of the [E1] token
    e2_pos          : LongTensor scalar  — position of the [E2] token
    labels          : LongTensor scalar  — relation class id

Recommended model head (R-BERT style):
    e1_hidden = output.hidden_states[-1][:, e1_pos, :]   # [batch, hidden]
    e2_hidden = output.hidden_states[-1][:, e2_pos, :]   # [batch, hidden]
    cls_hidden = output.hidden_states[-1][:, 0, :]       # [batch, hidden]
    logits = classifier(torch.cat([cls_hidden, e1_hidden, e2_hidden], dim=-1))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ── Relation label schema ────────────────────────────────────────────────────────
RELATION_LABEL2ID: Dict[str, int] = {
    "TREATS":         0,
    "CAUSES":         1,
    "INTERACTS_WITH": 2,
    "OTHER":          3,
}

RELATION_ID2LABEL: Dict[int, str] = {v: k for k, v in RELATION_LABEL2ID.items()}

# Special tokens used to mark entity spans
ENTITY_SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


# ── Dataset ──────────────────────────────────────────────────────────────────────

class RelationDataset(Dataset):
    """
    PyTorch Dataset for Drug–Disease Relation Extraction.

    Reads a CSV with at minimum these columns:
        sentence  : Text with [E1]..[/E1] and [E2]..[/E2] markers already inserted.
        relation  : One of TREATS, CAUSES, INTERACTS_WITH, OTHER.
        drug      : Drug entity name (stored for reference, not used in encoding).
        disease   : Disease entity name (stored for reference, not used in encoding).

    All four of these columns are produced by the preprocessing notebook.

    Args:
        file_path       : Path to a DrugBank CSV file.
        tokenizer_name  : HuggingFace model identifier.
        max_length      : Maximum token length (sequences are truncated to this).
        label2id        : Optional label override. Defaults to RELATION_LABEL2ID.
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer_name: str = "dmis-lab/biobert-v1.1",
        max_length: int = 256,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.file_path  = Path(file_path)
        self.max_length = max_length
        self.label2id   = label2id or RELATION_LABEL2ID
        self.id2label   = {v: k for k, v in self.label2id.items()}

        # ── Load tokenizer and add entity marker tokens ───────────────────────
        print(f"[RelationDataset] Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ENTITY_SPECIAL_TOKENS}
        )
        if added:
            print(f"[RelationDataset] Added {added} special tokens: {ENTITY_SPECIAL_TOKENS}")
            print(
                "  IMPORTANT: After loading your model, call:\n"
                "    model.resize_token_embeddings(len(tokenizer))\n"
                "  so the embedding matrix matches the expanded vocabulary."
            )

        # Cache token IDs for [E1] and [E2] — used to find entity positions
        self._e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self._e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")

        # ── Load CSV ─────────────────────────────────────────────────────────
        print(f"[RelationDataset] Reading: {self.file_path}")
        df = pd.read_csv(self.file_path)
        print(f"[RelationDataset] {len(df)} rows loaded.")
        print(f"[RelationDataset] Label distribution:\n{df['relation'].value_counts().to_string()}")

        # ── Encode ────────────────────────────────────────────────────────────
        self.examples: List[Dict] = []
        n_missing_markers = 0

        for _, row in df.iterrows():
            encoded, marker_ok = self._encode(row)
            self.examples.append(encoded)
            if not marker_ok:
                n_missing_markers += 1

        if n_missing_markers > 0:
            print(
                f"[RelationDataset] WARNING: {n_missing_markers} examples had missing "
                "[E1]/[E2] markers after truncation. Their e1_pos/e2_pos will be 0."
            )

        # Store raw df for inspection (e.g. error analysis)
        self.df = df

    # ── Encoding ────────────────────────────────────────────────────────────────

    def _encode(self, row: pd.Series):
        """
        Tokenise a single sentence and find the positions of [E1] and [E2].

        Returns:
            (encoded_dict, marker_ok)
            marker_ok is False if [E1] or [E2] was truncated out of the sequence.
        """
        sentence = str(row["sentence"])
        label    = str(row["relation"])

        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)  # [seq_len]

        # Find positions of [E1] and [E2] marker tokens
        e1_positions = (input_ids == self._e1_id).nonzero(as_tuple=True)[0]
        e2_positions = (input_ids == self._e2_id).nonzero(as_tuple=True)[0]

        marker_ok = len(e1_positions) > 0 and len(e2_positions) > 0

        # If truncated out, default to position 0 (CLS) — model will still train,
        # just with degraded signal for this example.
        e1_pos = int(e1_positions[0]) if len(e1_positions) > 0 else 0
        e2_pos = int(e2_positions[0]) if len(e2_positions) > 0 else 0

        label_id = self.label2id.get(label, self.label2id.get("OTHER", 3))

        return (
            {
                "input_ids":      input_ids,
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "e1_pos":         torch.tensor(e1_pos,   dtype=torch.long),
                "e2_pos":         torch.tensor(e2_pos,   dtype=torch.long),
                "labels":         torch.tensor(label_id, dtype=torch.long),
            },
            marker_ok,
        )

    # ── Dataset interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

    # ── Convenience properties ───────────────────────────────────────────────────

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    @property
    def vocab_size(self) -> int:
        """Use this to resize your model's embedding layer after loading."""
        return len(self.tokenizer)

    def get_label_list(self) -> List[str]:
        return [self.id2label[i] for i in range(len(self.id2label))]
