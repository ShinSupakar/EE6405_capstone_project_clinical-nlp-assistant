"""
Pure-logic helpers shared by every NER / RE wrapper.

Nothing in this module touches a model; it's safe to unit-test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

from config import LABEL_DISPLAY_MAP, LABEL_CSS_CLASS


# ----------------------------------------------------------------------------
# Canonical entity dict shape used across the whole UI.
# Every NER wrapper MUST produce entities that serialize to this shape so that
# downstream code (Tab 1 DataFrame, Tab 2 RE pair generation, Tab 3 SOAP
# template) can consume them interchangeably.
# ----------------------------------------------------------------------------

@dataclass
class Entity:
    text: str            # the literal substring from the input
    label: str           # display label: DRUG / DISEASE (already remapped)
    confidence: float    # in [0, 1]
    char_start: int      # inclusive offset into the ORIGINAL input string
    char_end: int        # exclusive offset
    sentence_idx: int = 0  # populated later by SOAP generator if needed

    def to_dict(self) -> dict:
        return asdict(self)

    def to_table_row(self) -> dict:
        """Row shape for the Streamlit DataFrame on Tab 1."""
        return {
            "Entity": self.text,
            "Label": self.label,
            "Confidence Score": f"{self.confidence * 100:.1f}%",
        }


# ----------------------------------------------------------------------------
# Display remapping
# ----------------------------------------------------------------------------

def remap_label(raw_label: str) -> str:
    """
    Convert a model's raw label (`Chemical`, `Disease`, possibly with B-/I-
    prefix stripped) into the UI-facing label (`DRUG`, `DISEASE`).

    Unknown labels pass through unchanged so downstream code can notice them.
    """
    return LABEL_DISPLAY_MAP.get(raw_label, raw_label.upper())


def css_class_for(display_label: str) -> str:
    """CSS class suffix (`.entity-<suffix>`) for a display label."""
    return LABEL_CSS_CLASS.get(display_label, "drug")  # safe default


# ----------------------------------------------------------------------------
# IOB → span grouping
# ----------------------------------------------------------------------------

def group_iob_tags(
    words: Sequence[str],
    tags: Sequence[str],
    word_char_spans: Sequence[tuple[int, int]],
    confidences: Sequence[float] | None = None,
) -> list[Entity]:
    """
    Walk parallel sequences of (word, IOB tag, char span) and emit Entity
    spans. Confidence per entity is the mean of its token confidences.

    Expected tag format: "O", "B-<Type>", "I-<Type>". Anything else is treated
    as O.
    """
    if not (len(words) == len(tags) == len(word_char_spans)):
        raise ValueError(
            f"length mismatch: words={len(words)} tags={len(tags)} "
            f"spans={len(word_char_spans)}"
        )
    if confidences is not None and len(confidences) != len(words):
        raise ValueError(
            f"confidences length {len(confidences)} != words {len(words)}"
        )

    entities: list[Entity] = []
    cur_type: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    cur_confs: list[float] = []

    def flush():
        nonlocal cur_type, cur_start, cur_end, cur_confs
        if cur_type is not None and cur_start is not None and cur_end is not None:
            conf = sum(cur_confs) / len(cur_confs) if cur_confs else 1.0
            # Note: we slice from the original text, not from the word list,
            # so the UI can rely on char offsets for highlighting.
            entities.append(
                Entity(
                    text="",  # filled in by caller from original string
                    label=remap_label(cur_type),
                    confidence=float(conf),
                    char_start=cur_start,
                    char_end=cur_end,
                )
            )
        cur_type = None
        cur_start = None
        cur_end = None
        cur_confs = []

    for i, tag in enumerate(tags):
        start, end = word_char_spans[i]
        conf_i = confidences[i] if confidences is not None else 1.0

        if tag == "O" or not tag:
            flush()
            continue

        prefix, _, ent_type = tag.partition("-")
        if not ent_type:
            flush()
            continue

        if prefix == "B" or cur_type != ent_type:
            flush()
            cur_type = ent_type
            cur_start = start
            cur_end = end
            cur_confs = [conf_i]
        else:  # I-<same type> continues
            cur_end = end
            cur_confs.append(conf_i)

    flush()
    return entities


def fill_entity_texts(entities: list[Entity], source_text: str) -> list[Entity]:
    """Replace each entity's empty `text` field with the substring from source."""
    for e in entities:
        e.text = source_text[e.char_start : e.char_end]
    return entities


# ----------------------------------------------------------------------------
# Word-level tokenization that preserves char offsets (for the word-level
# models: BiLSTM-CRF, BiGRU).
# ----------------------------------------------------------------------------

def whitespace_tokenize_with_offsets(
    text: str,
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Simple whitespace tokenizer that records each token's (start, end) char
    offset. Matches how the BiLSTM-CRF / BiGRU notebooks tokenize: split on
    whitespace, lowercase downstream.
    """
    words: list[str] = []
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not text[i].isspace():
            i += 1
        words.append(text[start:i])
        spans.append((start, i))
    return words, spans


# ----------------------------------------------------------------------------
# Highlighted HTML rendering (used by Tab 1's BioBERT visualization)
# ----------------------------------------------------------------------------

def render_highlighted_html(text: str, entities: Iterable[Entity]) -> str:
    """
    Build an HTML string with `<span class="entity-<css>">…</span>` wrapped
    around every entity span. Uses char offsets (not string replace) so
    overlapping / repeated substrings work correctly.
    """
    ents = sorted(entities, key=lambda e: (e.char_start, e.char_end))

    # Drop any overlaps, keeping the first. This is rare in practice but
    # protects the renderer from crashing.
    cleaned: list[Entity] = []
    last_end = -1
    for e in ents:
        if e.char_start >= last_end:
            cleaned.append(e)
            last_end = e.char_end

    parts: list[str] = []
    cursor = 0
    for e in cleaned:
        if e.char_start > cursor:
            parts.append(_html_escape(text[cursor : e.char_start]))
        css = css_class_for(e.label)
        parts.append(
            f"<span class='entity-{css}' title='{e.label} · {e.confidence:.2f}'>"
            f"<b>{_html_escape(text[e.char_start : e.char_end])}</b></span>"
        )
        cursor = e.char_end
    if cursor < len(text):
        parts.append(_html_escape(text[cursor:]))
    return "".join(parts)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ----------------------------------------------------------------------------
# RE pair generation — given NER entities, build (drug, disease) candidates
# ----------------------------------------------------------------------------

@dataclass
class RECandidate:
    drug: Entity
    disease: Entity
    sentence: str            # the text segment containing both, with markers
    sentence_plain: str      # same segment without markers (for display)


def find_sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Very lightweight sentence splitter: split on `.`, `?`, `!` followed by
    whitespace. Returns list of (start, end) char spans. Good enough for
    clinical notes; the SOAP generator will optionally use spaCy instead.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] in ".?!":
            # advance past the punctuation and any trailing whitespace
            end = i + 1
            while end < n and text[end].isspace():
                end += 1
            spans.append((start, end))
            start = end
            i = end
        else:
            i += 1
    if start < n:
        spans.append((start, n))
    return spans


def build_re_candidates(
    text: str,
    entities: Sequence[Entity],
    markers: dict[str, str] | None = None,
) -> list[RECandidate]:
    """
    Build all (drug, disease) pairs that co-occur within a single sentence,
    wrapping each with the entity markers the RE tokenizer was trained with.

    `markers` defaults to config.RE_MARKERS via the caller (import here kept
    off the hot path).
    """
    from config import RE_MARKERS

    markers = markers or RE_MARKERS
    drugs = [e for e in entities if e.label == "DRUG"]
    diseases = [e for e in entities if e.label == "DISEASE"]
    sentence_spans = find_sentence_spans(text)

    def sentence_of(e: Entity) -> tuple[int, int] | None:
        for s, ee in sentence_spans:
            if s <= e.char_start < ee:
                return s, ee
        return None

    candidates: list[RECandidate] = []
    for d in drugs:
        ds = sentence_of(d)
        if ds is None:
            continue
        for di in diseases:
            dis = sentence_of(di)
            if dis != ds:
                continue
            s_start, s_end = ds
            sent = text[s_start:s_end]
            # insert markers — process rightmost span first so indices stay
            # valid during mutation
            e1_local = (d.char_start - s_start, d.char_end - s_start)
            e2_local = (di.char_start - s_start, di.char_end - s_start)
            if e1_local[0] < e2_local[0]:
                left_lbl, right_lbl = "drug", "disease"
                left, right = e1_local, e2_local
            else:
                left_lbl, right_lbl = "disease", "drug"
                left, right = e2_local, e1_local
            # splice markers
            marked = (
                sent[: left[0]]
                + markers[f"{left_lbl}_open"]
                + " "
                + sent[left[0] : left[1]]
                + " "
                + markers[f"{left_lbl}_close"]
                + sent[left[1] : right[0]]
                + markers[f"{right_lbl}_open"]
                + " "
                + sent[right[0] : right[1]]
                + " "
                + markers[f"{right_lbl}_close"]
                + sent[right[1] :]
            )
            candidates.append(
                RECandidate(
                    drug=d,
                    disease=di,
                    sentence=marked,
                    sentence_plain=sent.strip(),
                )
            )
    return candidates


# ----------------------------------------------------------------------------
# Dedup helpers
# ----------------------------------------------------------------------------

def dedupe_entities(entities: Sequence[Entity]) -> list[Entity]:
    """Collapse exact-offset duplicates, keeping the highest-confidence one."""
    best: dict[tuple[int, int, str], Entity] = {}
    for e in entities:
        k = (e.char_start, e.char_end, e.label)
        if k not in best or e.confidence > best[k].confidence:
            best[k] = e
    return sorted(best.values(), key=lambda e: e.char_start)
