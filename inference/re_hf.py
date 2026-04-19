"""
HuggingFace relation-extraction wrapper.

Used for both BERT-RE (`bert-base-cased`) and BioBERT-RE
(`dmis-lab/biobert-base-cased-v1.1`) — both are sequence-classification heads
fine-tuned with entity markers `[DRUG] … [/DRUG]` and `[DISEASE] … [/DISEASE]`
spliced into the input sentence.

Because `trainer.save_model(...)` persists the resized embedding matrix AND
the extra special tokens, a plain round-trip through `from_pretrained` works
without having to re-`add_special_tokens` at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DEVICE, MAX_LENGTH_RE, RE_CONFIDENCE_THRESHOLD, RE_MARKERS
from inference.entity_utils import (
    Entity,
    RECandidate,
    build_re_candidates,
)


class HFReInference:
    """
    Wraps a HuggingFace sequence classifier for drug/disease relation extraction.

    Parameters
    ----------
    model_dir : str | Path
        Directory written by `trainer.save_model`. Must contain the resized
        tokenizer (i.e. `[DRUG]`, `[/DRUG]`, `[DISEASE]`, `[/DISEASE]` already
        in vocab) and the matching model weights.
    confidence_threshold : float, optional
        Predictions below this drop from the output. Defaults to
        `RE_CONFIDENCE_THRESHOLD` in config.
    """

    def __init__(
        self,
        model_dir: str | Path,
        confidence_threshold: float | None = None,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Expected HF RE checkpoint at {self.model_dir}. "
                f"Did you extract the teammate-uploaded RE zip into this path?"
            )

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
                f"HF RE checkpoint at {self.model_dir} is incomplete "
                f"({'; '.join(bits)}). Extract the full teammate-uploaded "
                f"checkpoint folder into this path."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model.to(DEVICE).eval()

        cfg = getattr(self.model, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg is not None else None
        if id2label is None:
            raise ValueError(
                f"RE checkpoint at {self.model_dir} has no id2label in "
                f"config.json. Re-save the model with `trainer.save_model` "
                f"after setting model.config.id2label = {{0: 'side_effect', "
                f"1: 'treats'}} (or whatever the actual mapping is)."
            )
        self.id2label = {int(k): v for k, v in id2label.items()}

        self.threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else RE_CONFIDENCE_THRESHOLD
        )

    @torch.no_grad()
    def predict(
        self,
        text: str,
        entities: Sequence[Entity],
    ) -> list[dict]:
        """
        Extract relations between every (drug, disease) pair that co-occurs in
        a sentence.

        Returns
        -------
        list of dict with keys:
            e1 : Entity (always the DRUG)
            e2 : Entity (always the DISEASE)
            relation : str — one of `id2label.values()` (usually `treats` /
                `side_effect`)
            confidence : float in [0, 1] (softmax max)
            sentence : str — the sentence substring, without markers (handy
                for the UI)

        Pairs whose confidence is below `self.threshold` are dropped so the UI
        doesn't have to filter them out again.
        """
        if not entities:
            return []

        candidates: list[RECandidate] = build_re_candidates(
            text, entities, markers=RE_MARKERS
        )
        if not candidates:
            return []

        marked_sentences = [c.sentence for c in candidates]
        enc = self.tokenizer(
            marked_sentences,
            truncation=True,
            max_length=MAX_LENGTH_RE,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        outputs = self.model(**enc)
        logits = outputs.logits  # [batch, num_labels]
        probs = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)

        results: list[dict] = []
        for cand, pred_id, conf in zip(candidates, preds.tolist(), confs.tolist()):
            conf = float(conf)
            if conf < self.threshold:
                continue
            relation = self.id2label.get(int(pred_id), str(pred_id))
            results.append(
                {
                    "e1": cand.drug,
                    "e2": cand.disease,
                    "relation": relation,
                    "confidence": conf,
                    "sentence": cand.sentence_plain,
                }
            )
        return results
