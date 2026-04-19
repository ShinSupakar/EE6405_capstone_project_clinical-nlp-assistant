"""
spaCy NER wrapper.

sweetprajna2003's `notebookd7ab3aec43.ipynb` produces a saved spaCy model
directory. At load time we call `spacy.load(path)` and at predict time we
iterate `doc.ents`, which already gives char offsets. No IOB grouping needed.

spaCy does not expose a token-level confidence out of the box. We fall back
to `1.0` for each entity — the UI can note this. (A more honest signal would
be `get_pipe('ner').predict` beam scores, but that depends on the model type
and is overkill for this demo.)

Status
------
Scaffolded — pending upload of `spacy_ner_ckpt.zip` (extracted to the dir
passed as `model_dir`).
"""

from __future__ import annotations

from pathlib import Path

from inference.entity_utils import Entity, remap_label


class SpacyNerInference:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise FileNotFoundError(
                f"spaCy NER model not found at {self.model_dir}. "
                f"Ask sweetprajna2003 to upload `spacy_ner_ckpt.zip` (the full "
                f"directory from `models/spacy/bc5cdr/` — including meta.json, "
                f"vocab/, ner/, config.cfg) to the NLP Drive folder, then "
                f"extract into {self.model_dir}."
            )

        # Import spacy lazily so missing-package errors surface clearly.
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "spaCy is not installed. Run `pip install spacy` "
                "(and `python -m spacy download en_core_web_sm` if the "
                "teammate model depends on a shared vocab)."
            ) from e

        self.nlp = spacy.load(str(self.model_dir))

    def predict(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        doc = self.nlp(text)
        ents: list[Entity] = []
        for ent in doc.ents:
            ents.append(
                Entity(
                    text=ent.text,
                    label=remap_label(ent.label_),
                    confidence=1.0,  # spaCy doesn't expose this cheaply
                    char_start=ent.start_char,
                    char_end=ent.end_char,
                )
            )
        return ents
