"""
Rule-based SOAP note generator.

Consumes the BioBERT entities (best NER) + BioBERT-RE relations (best RE) and
emits a structured S/O/A/P markdown block for Tab 3.

There is no ML here — it's deterministic template filling. This is intentional:
SOAP generation was not part of the trained pipeline, and a predictable
template demos well while staying honest about what the models did and didn't
produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from inference.entity_utils import Entity, find_sentence_spans


# Temporal cue words that indicate historical / known / chronic conditions.
# If one of these precedes a DISEASE in the same sentence, we route that
# disease to the Objective section (PMH) rather than Subjective.
HISTORY_CUES = (
    "history of",
    "known",
    "diagnosed",
    "past medical history",
    "chronic",
    "pmh",
    "previously",
)


_TRAILING_PUNCT = ",.;:"


def _clean_text(s: str) -> str:
    """Strip whitespace and trailing sentence-punctuation from an entity span.

    NER spans often pick up a comma or period ("Metformin."); without cleaning
    we end up deduping "Metformin" against "Metformin." as if they were
    different drugs.
    """
    return s.strip().rstrip(_TRAILING_PUNCT).strip()


def _dedupe_entities(entities: Sequence[Entity]) -> list[Entity]:
    """Collapse entities whose cleaned text + label match, keeping the first."""
    seen: set[tuple[str, str]] = set()
    out: list[Entity] = []
    for e in entities:
        k = (_clean_text(e.text).lower(), e.label)
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
    return out


@dataclass
class SoapBuckets:
    """Intermediate structure — which entities/relations land where."""

    subjective_diseases: list[Entity]
    objective_history: list[Entity]
    objective_medications: list[Entity]
    assessment_pairs: list[tuple[Entity, Entity, str, float]]  # (drug, disease, relation, conf)
    assessment_unpaired_diseases: list[Entity]
    plan_side_effects: list[tuple[Entity, Entity, float]]
    plan_new_drugs: list[Entity]
    plan_untreated_diseases: list[Entity]


def _sentence_index_map(text: str) -> dict[int, int]:
    """Map each char offset to the 0-indexed sentence it belongs to."""
    spans = find_sentence_spans(text)
    offsets: dict[int, int] = {}
    for idx, (s, e) in enumerate(spans):
        for c in range(s, e):
            offsets[c] = idx
    return offsets


def _sentence_contains_history_cue(sent: str) -> bool:
    low = sent.lower()
    return any(cue in low for cue in HISTORY_CUES)


def _bucket(
    transcript: str,
    entities: Sequence[Entity],
    relations: Sequence[dict],
) -> SoapBuckets:
    """
    Sort entities and relations into S / O / A / P buckets via simple rules.

    relations: list of dicts shaped like
        {"e1": Entity, "e2": Entity, "relation": "treats"|"side_effect", "confidence": float}
    """
    sentences = find_sentence_spans(transcript)
    sent_texts = [transcript[s:e] for s, e in sentences]

    def sent_idx_of(e: Entity) -> int:
        for i, (s, end) in enumerate(sentences):
            if s <= e.char_start < end:
                return i
        return 0

    # Collapse near-duplicate spans (e.g. "Metformin" vs "Metformin.") so they
    # don't all show up as separate items in the SOAP note.
    entities = _dedupe_entities(entities)

    # Split entities by type
    diseases = [e for e in entities if e.label == "DISEASE"]
    drugs = [e for e in entities if e.label == "DRUG"]

    # Diseases mentioned in a history-cue sentence → Objective (PMH)
    objective_history: list[Entity] = []
    subjective_diseases: list[Entity] = []
    for d in diseases:
        idx = sent_idx_of(d)
        if _sentence_contains_history_cue(sent_texts[idx]):
            objective_history.append(d)
        else:
            subjective_diseases.append(d)

    # Drugs that appear on the left side of a `treats` relation → Objective
    # (current meds). Everything else → candidate new-script plan items.
    treats_pairs = [r for r in relations if r["relation"] == "treats"]
    side_effect_pairs = [r for r in relations if r["relation"] == "side_effect"]

    # Identity for cross-referencing — uses cleaned text so that "Metformin"
    # in an entity and "Metformin." in a relation still match.
    def drug_key(e: Entity) -> tuple[str, str]:
        return (_clean_text(e.text).lower(), e.label)

    treated_drug_keys = {drug_key(r["e1"]) for r in treats_pairs}
    treated_disease_keys = {drug_key(r["e2"]) for r in treats_pairs}

    # If no `treats` relations fired at all, every detected drug is by default
    # "currently taken" — better than leaving Current Medications empty when
    # the transcript clearly lists medications. This is the common case when
    # the RE model is biased toward side_effect and never emits treats.
    if treats_pairs:
        objective_medications = [
            d for d in drugs if drug_key(d) in treated_drug_keys
        ]
        plan_new_drugs = [
            d for d in drugs if drug_key(d) not in treated_drug_keys
        ]
    else:
        objective_medications = list(drugs)
        plan_new_drugs = []

    assessment_pairs = [
        (r["e1"], r["e2"], r["relation"], r["confidence"]) for r in treats_pairs
    ]
    assessment_unpaired_diseases = [
        d for d in diseases if drug_key(d) not in treated_disease_keys
    ]

    plan_side_effects = [
        (r["e1"], r["e2"], r["confidence"]) for r in side_effect_pairs
    ]
    plan_untreated_diseases = [
        d for d in diseases if drug_key(d) not in treated_disease_keys
    ]

    return SoapBuckets(
        subjective_diseases=subjective_diseases,
        objective_history=objective_history,
        objective_medications=objective_medications,
        assessment_pairs=assessment_pairs,
        assessment_unpaired_diseases=assessment_unpaired_diseases,
        plan_side_effects=plan_side_effects,
        plan_new_drugs=plan_new_drugs,
        plan_untreated_diseases=plan_untreated_diseases,
    )


def _fmt_entities(entities: Iterable[Entity]) -> str:
    items = [f"**{_clean_text(e.text)}**" for e in entities]
    return ", ".join(items) if items else "_(none detected)_"


def generate_soap_note(
    transcript: str,
    entities: Sequence[Entity],
    relations: Sequence[dict],
) -> str:
    """
    Build a SOAP-style markdown note from NER entities + RE relations.

    Parameters
    ----------
    transcript : str
        The original clinical text / transcript shown on Tab 1.
    entities : list of Entity
        Output from the best NER model (BioBERT) — each with
        text/label/confidence/char offsets, already display-remapped to
        DRUG / DISEASE.
    relations : list of dict
        Output from the best RE model (BioBERT-RE), shape:
        `{"e1": Entity, "e2": Entity, "relation": str, "confidence": float}`.
        Empty list is allowed — template falls back gracefully.

    Returns
    -------
    str : Markdown with four `### ` sections (S / O / A / P).
    """
    b = _bucket(transcript, entities, relations)

    lines: list[str] = []

    # --- S ---------------------------------------------------------------
    lines.append("### Subjective (S)")
    if b.subjective_diseases:
        lines.append(
            "Patient presents with: " + _fmt_entities(b.subjective_diseases) + "."
        )
    else:
        lines.append("_Presenting complaints not explicitly detected in transcript._")
    lines.append("")

    # --- O ---------------------------------------------------------------
    lines.append("### Objective (O)")
    lines.append("- **Past Medical History:** " + _fmt_entities(b.objective_history))
    lines.append("- **Current Medications:** " + _fmt_entities(b.objective_medications))
    lines.append("")

    # --- A ---------------------------------------------------------------
    lines.append("### Assessment (A)")
    if b.assessment_pairs:
        for i, (drug, disease, rel, conf) in enumerate(b.assessment_pairs, 1):
            lines.append(
                f"{i}. **{_clean_text(disease.text)}** — managed with "
                f"**{_clean_text(drug.text)}** (_{rel}_, conf {conf:.2f})"
            )
    if b.assessment_unpaired_diseases:
        start = len(b.assessment_pairs) + 1
        for i, d in enumerate(b.assessment_unpaired_diseases, start):
            lines.append(
                f"{i}. **{_clean_text(d.text)}** — no treatment relation detected."
            )
    if not b.assessment_pairs and not b.assessment_unpaired_diseases:
        lines.append("_No diseases detected._")
    lines.append("")

    # --- P ---------------------------------------------------------------
    lines.append("### Plan (P)")
    plan_items: list[str] = []
    if b.plan_side_effects:
        seen_pairs: set[tuple[str, str]] = set()
        for drug, disease, conf in b.plan_side_effects:
            key = (_clean_text(drug.text).lower(), _clean_text(disease.text).lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            plan_items.append(
                f"Monitor **{_clean_text(drug.text)}** for potential "
                f"**{_clean_text(disease.text)}** "
                f"(_side effect_, conf {conf:.2f})."
            )
    if b.plan_new_drugs:
        plan_items.append(
            "Consider / continue: " + _fmt_entities(b.plan_new_drugs) + "."
        )
    if b.plan_untreated_diseases:
        plan_items.append(
            "Diseases without detected active treatment: "
            + _fmt_entities(b.plan_untreated_diseases)
            + " — workup / evaluate."
        )
    if not plan_items:
        plan_items.append("_No plan items synthesized from relations._")

    for i, item in enumerate(plan_items, 1):
        lines.append(f"{i}. {item}")

    return "\n".join(lines)
