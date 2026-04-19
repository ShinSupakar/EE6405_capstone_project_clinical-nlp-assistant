"""
Unlabeled Clinical Text Dataset Loader — Clinical NLP Project
=============================================================
Loads the MTSamples sentence file produced by the preprocessing notebook.

This dataset is used by two teammates:
  1. ASR demo teammate — needs raw sentences as a realistic clinical text corpus.
  2. (Optional) Language model pre-training teammate — can fine-tune BioBERT
     further on clinical text before NER/RE training.

Usage — raw sentences (ASR teammate):
--------------------------------------
    from loaders.unlabeled_dataset import UnlabeledDataset

    dataset = UnlabeledDataset("processed_data/unlabeled/mtsamples_sentences.txt")

    # Iterate over raw strings
    for sentence in dataset:
        print(sentence)   # "The patient presented with chest pain..."

    # Or filter by medical specialty (uses the specialty-specific files)
    cardio_ds = UnlabeledDataset(
        "processed_data/unlabeled/mtsamples_cardiovascular_pulmonary.txt"
    )

Usage — tokenized (language model fine-tuning):
------------------------------------------------
    dataset = UnlabeledDataset(
        file_path="processed_data/unlabeled/mtsamples_sentences.txt",
        tokenizer_name="dmis-lab/biobert-v1.1",
        max_length=256,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Each batch item has keys: input_ids, attention_mask
    for batch in loader:
        outputs = model(**batch)

Usage — MLM (masked language modelling):
-----------------------------------------
    Use HuggingFace's DataCollatorForLanguageModeling to apply masking on-the-fly:

    from transformers import DataCollatorForLanguageModeling
    from loaders.unlabeled_dataset import UnlabeledDataset

    dataset   = UnlabeledDataset("...mtsamples_sentences.txt", tokenizer_name="dmis-lab/biobert-v1.1")
    collator  = DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm_probability=0.15)
    loader    = DataLoader(dataset, batch_size=16, collate_fn=collator)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class UnlabeledDataset(Dataset):
    """
    Dataset for unlabeled clinical text (MTSamples transcriptions).

    When tokenizer_name is None, __getitem__ returns raw strings (for the ASR
    teammate, or for any pipeline that needs plain text).

    When tokenizer_name is provided, __getitem__ returns a dict with
    input_ids and attention_mask tensors ready for a BERT model.

    Args:
        file_path       : Path to a plain text file with one sentence per line.
                          Produced by the preprocessing notebook.
        tokenizer_name  : HuggingFace model identifier. Pass None (default) to
                          get raw strings instead of tensor dicts.
        max_length      : Maximum token length (used only when tokenizing).
        deduplicate     : Remove duplicate sentences (default True).
        min_word_count  : Skip sentences shorter than this many words (default 3).
    """

    def __init__(
        self,
        file_path: str | Path,
        tokenizer_name: Optional[str] = None,
        max_length: int = 256,
        deduplicate: bool = True,
        min_word_count: int = 3,
    ) -> None:
        self.file_path      = Path(file_path)
        self.max_length     = max_length
        self.tokenizer      = None
        self._tokenized     = tokenizer_name is not None

        # ── Load sentences ────────────────────────────────────────────────────
        print(f"[UnlabeledDataset] Reading: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # Filter very short sentences
        sentences = [s for s in raw_lines if len(s.split()) >= min_word_count]

        # Deduplicate while preserving order
        if deduplicate:
            seen: set = set()
            unique: List[str] = []
            for s in sentences:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            sentences = unique

        self.sentences = sentences
        print(
            f"[UnlabeledDataset] {len(sentences)} sentences loaded "
            f"(filtered from {len(raw_lines)} raw lines)."
        )

        # ── Optionally tokenize ───────────────────────────────────────────────
        if tokenizer_name is not None:
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError("Install transformers: pip install transformers")

            print(f"[UnlabeledDataset] Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print(
                f"[UnlabeledDataset] Tokenizing {len(sentences)} sentences "
                f"(max_length={max_length})…"
            )
            self._encodings = self.tokenizer(
                sentences,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            print("[UnlabeledDataset] Tokenization complete.")
        else:
            self._encodings = None
            print("[UnlabeledDataset] No tokenizer provided — returning raw strings.")

    # ── Dataset interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Union[str, Dict[str, torch.Tensor]]:
        """
        Returns:
            str  — if no tokenizer was provided (raw sentence).
            dict — if tokenizer was provided:
                   { input_ids: Tensor, attention_mask: Tensor }
        """
        if self._encodings is None:
            return self.sentences[idx]

        return {
            "input_ids":      self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
        }

    # ── Convenience methods ──────────────────────────────────────────────────────

    def get_raw_sentences(self) -> List[str]:
        """Return all sentences as a plain Python list of strings."""
        return self.sentences

    def get_sample(self, n: int = 5) -> List[str]:
        """Return a random sample of n sentences for quick inspection."""
        import random
        return random.sample(self.sentences, min(n, len(self.sentences)))

    def stats(self) -> Dict[str, float]:
        """Return basic statistics about the corpus."""
        lengths = [len(s.split()) for s in self.sentences]
        return {
            "num_sentences":  len(self.sentences),
            "avg_words":      sum(lengths) / max(len(lengths), 1),
            "min_words":      min(lengths),
            "max_words":      max(lengths),
            "total_words":    sum(lengths),
        }
