"""
BART token-classification NER wrapper.

shinjinisupakar's `bart-full-encoder-decoder-optimized.ipynb` trains a custom
class `BartForTokenClassificationFull` that HuggingFace does not ship. The
architecture is:

    BartModel (encoder + decoder, 6 layers each, d_model=768)
      → decoder last_hidden_state  [B, seq, 768]
      → Linear(768, num_labels)    [B, seq, 5]

Decoder inputs are the same token ids as encoder inputs (no right-shift), so
position `i` of the output aligns with input token `i` for clean NER tagging.

Expected checkpoint layout (HF Trainer save_pretrained):
    models/bart_ner/
      config.json           — id2label / label2id / BartConfig
      model.safetensors     — state_dict (model.encoder.*, model.decoder.*, classifier.*)
      tokenizer.json
      tokenizer_config.json
      vocab.json
      merges.txt
      special_tokens_map.json
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, BartConfig, BartModel

from config import DEVICE, MAX_LENGTH_NER
from inference.entity_utils import (
    Entity,
    group_iob_tags,
    fill_entity_texts,
    whitespace_tokenize_with_offsets,
)


class _BartForTokenClassificationFull(nn.Module):
    """Re-implementation of the teammate's custom class for inference-only use."""

    def __init__(self, config: BartConfig, num_labels: int):
        super().__init__()
        self.model = BartModel(config)
        self.classifier = nn.Linear(config.d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        return self.classifier(outputs.last_hidden_state)


class BartNerInference:
    """BART NER with the custom BartForTokenClassificationFull head."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        cfg_p = self.model_dir / "config.json"
        weights = self.model_dir / "model.safetensors"
        tok_files = self.model_dir / "tokenizer.json"
        missing = [p for p in (cfg_p, weights, tok_files) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "BART NER checkpoint not ready yet. Missing:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\n\nExpected HF Trainer layout (config.json + "
                + "model.safetensors + tokenizer files) from "
                + "results_bart_full_fast/checkpoint-*."
            )

        config = BartConfig.from_pretrained(str(self.model_dir))
        self.id2label = {int(k): v for k, v in config.id2label.items()}
        num_labels = len(self.id2label)

        self.model = _BartForTokenClassificationFull(config, num_labels=num_labels)
        state = load_file(str(weights))
        # BART ties encoder/decoder embed_tokens to `model.shared.weight`, so
        # those keys are expected-missing and safe to skip.
        result = self.model.load_state_dict(state, strict=False)
        unexpected = [k for k in result.unexpected_keys]
        if unexpected:
            raise RuntimeError(f"Unexpected state_dict keys: {unexpected[:5]}")
        self.model.to(DEVICE).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    @torch.no_grad()
    def predict(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        words, word_spans = whitespace_tokenize_with_offsets(text)
        if not words:
            return []

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH_NER,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)

        logits = self.model(input_ids, attn_mask)[0]  # [seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(-1)
        confs, _ = probs.max(dim=-1)

        word_ids = enc.word_ids(batch_index=0)
        n = len(words)
        tags = ["O"] * n
        conf_per_word = [0.0] * n
        seen = [False] * n
        for pos, wid in enumerate(word_ids):
            if wid is None or wid >= n or seen[wid]:
                continue
            tags[wid] = self.id2label.get(int(preds[pos].item()), "O")
            conf_per_word[wid] = float(confs[pos].item())
            seen[wid] = True

        ents = group_iob_tags(words, tags, word_spans, confidences=conf_per_word)
        fill_entity_texts(ents, text)
        return ents
