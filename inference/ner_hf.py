"""
Generic HuggingFace NER wrapper.

Used for every token-classification model saved with `trainer.save_model(...)`:
- BERT (bert-base-cased)
- BioBERT (dmis-lab/biobert-base-cased-v1.1)
- BART (facebook/bart-base, via teammate's BartForTokenClassificationFull)

The BART checkpoint has a custom head so its model class must be imported
externally and passed in via `model_cls`. For BERT / BioBERT the default
`AutoModelForTokenClassification` does the right thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from config import DEVICE, MAX_LENGTH_NER
from inference.entity_utils import (
    Entity,
    group_iob_tags,
    fill_entity_texts,
    whitespace_tokenize_with_offsets,
)


class HFNerInference:
    """
    Wraps a HuggingFace token classifier for entity extraction.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing `pytorch_model.bin` (or safetensors), `config.json`,
        and tokenizer files — i.e. the folder written by `trainer.save_model`.
    model_cls : callable, optional
        Factory for the model. Defaults to
        `AutoModelForTokenClassification.from_pretrained`. Pass a custom one
        for BART which uses `BartForTokenClassificationFull`.
    """

    def __init__(
        self,
        model_dir: str | Path,
        model_cls: Callable | None = None,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Expected HF NER checkpoint at {self.model_dir}. "
                f"Did you extract the teammate-uploaded zip into this path?"
            )

        # Check that required files are actually present — transformers
        # throws a cryptic `stat: path should be ... not NoneType` when the
        # folder exists but is empty (common while teammates are still
        # uploading). Surface a clean message instead.
        required = ["config.json"]
        weights_options = ["model.safetensors", "pytorch_model.bin"]
        missing_required = [f for f in required if not (self.model_dir / f).exists()]
        has_weights = any((self.model_dir / w).exists() for w in weights_options)
        if missing_required or not has_weights:
            bits = []
            if missing_required:
                bits.append("missing: " + ", ".join(missing_required))
            if not has_weights:
                bits.append(
                    "no weights file (need one of: "
                    + ", ".join(weights_options) + ")"
                )
            raise FileNotFoundError(
                f"HF NER checkpoint at {self.model_dir} is incomplete "
                f"({'; '.join(bits)}). Extract the full teammate-uploaded "
                f"checkpoint folder into this path."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        factory = model_cls or AutoModelForTokenClassification.from_pretrained
        self.model = factory(str(self.model_dir))
        self.model.to(DEVICE).eval()

        cfg = getattr(self.model, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        if id2label is None:
            raise ValueError(
                f"Checkpoint at {self.model_dir} has no id2label in config.json. "
                f"Re-save the model with `trainer.save_model` after setting "
                f"model.config.id2label."
            )
        # id2label keys from HF can be str or int depending on JSON load; normalize.
        self.id2label = {int(k): v for k, v in id2label.items()}

    @torch.no_grad()
    def predict(self, text: str) -> list[Entity]:
        """Run NER on a single string; return a list of Entity spans."""
        if not text.strip():
            return []

        # 1) word-level tokenization so we can track char offsets into the
        #    ORIGINAL text. We hand the words to the HF tokenizer with
        #    is_split_into_words=True so the tokenizer builds word_ids for us.
        words, word_spans = whitespace_tokenize_with_offsets(text)
        if not words:
            return []

        # 2) subword tokenize
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH_NER,
            padding=True,
            return_tensors="pt",
        )
        enc_device = {k: v.to(DEVICE) for k, v in enc.items()}

        # 3) forward
        outputs = self.model(**enc_device)
        logits = outputs.logits[0]  # [seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(-1)
        confs, _ = probs.max(dim=-1)

        # 4) collapse subwords → word level: for each word_id, take the first
        #    subword's prediction (standard HF alignment).
        word_ids = enc.word_ids(batch_index=0)
        n_words = len(words)
        word_tags: list[str] = ["O"] * n_words
        word_confs: list[float] = [0.0] * n_words
        seen_word = [False] * n_words
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid >= n_words:  # truncated
                break
            if seen_word[wid]:
                continue
            label_id = int(preds[pos].item())
            word_tags[wid] = self.id2label.get(label_id, "O")
            word_confs[wid] = float(confs[pos].item())
            seen_word[wid] = True

        # 5) IOB → spans
        ents = group_iob_tags(words, word_tags, word_spans, confidences=word_confs)
        fill_entity_texts(ents, text)
        return ents
