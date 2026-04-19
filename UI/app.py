"""
Clinical NLP Assistant — Streamlit front-end.

Three tabs:
  1. NER across all six trained models (2 rows × 3 cols).
  2. Relation extraction from BERT-RE and BioBERT-RE, pulling entities from
     BioBERT-NER via session state.
  3. Template-based SOAP note generation from BioBERT outputs + export button.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and `inference/` importable when this file is run as the
# Streamlit entry point (`streamlit run UI/app.py`). The parent dir is the
# repo root.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from config import MODEL_METADATA, RE_CONFIDENCE_THRESHOLD
from inference import (
    Entity,
    NER_REGISTRY,
    RE_REGISTRY,
    dedupe_entities,
    generate_soap_note,
    render_highlighted_html,
    safe_predict_ner,
    safe_predict_re,
)


# ---------------------------------------------------------------------------
# Page config & CSS (carried over from the original mock UI)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical NLP Assistant",
    layout="wide",
    page_icon="🏥",
)

st.markdown(
    """
<style>
    .entity-disease { background-color: #ffcccc; color: #212529; padding: 2px 4px; border-radius: 4px; border: 1px solid #ff9999; }
    .entity-drug { background-color: #ccffcc; color: #212529; padding: 2px 4px; border-radius: 4px; border: 1px solid #99ff99; }
    .entity-symptom { background-color: #ccccff; color: #212529; padding: 2px 4px; border-radius: 4px; border: 1px solid #9999ff; }
    .entity-procedure { background-color: #ffffcc; color: #212529; padding: 2px 4px; border-radius: 4px; border: 1px solid #ffff99; }
    .soap-box { background-color: #f8f9fa; color: #212529; padding: 20px; border-radius: 10px; border-left: 5px solid #0056b3; }
    .soap-box h1, .soap-box h2, .soap-box h3, .soap-box h4 { color: #0056b3; }
    .err-box  { background-color: #fff3cd; color: #212529; padding: 10px; border-radius: 6px; border-left: 4px solid #ffc107; font-size: 13px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🏥 Clinical NLP Assistant Dashboard")
st.markdown(
    "### End-to-End Pipeline: Clinical Note → Entity Recognition → Relation Extraction → SOAP Note"
)
st.write("---")


# ---------------------------------------------------------------------------
# Sample transcripts — presets for the Tab 1 input. Each is engineered to
# exercise a different slice of the pipeline so the demo doesn't live or die
# by one input.
# ---------------------------------------------------------------------------

SAMPLE_PRESETS = {
    "Default case": (
        "Patient is a 65-year-old male presenting with severe chest pain and "
        "shortness of breath. He has a history of hypertension and Type 2 "
        "diabetes. Currently taking Metformin and Lisinopril. The patient "
        "reported experiencing nausea this morning. Plan is to order an ECG "
        "and start Aspirin immediately."
    ),
    "SOAP-heavy case": (
        "55-year-old female with a known history of asthma and type 2 diabetes. "
        "She takes Metformin daily for diabetes and Lisinopril for hypertension. "
        "Reports persistent dry cough since starting Lisinopril last month. "
        "Currently experiencing mild nausea which she attributes to Metformin. "
        "Examination reveals elevated blood pressure. Plan to start Amlodipine "
        "and monitor for side effects."
    ),
    "RE stress test": (
        "Patient takes Metformin daily for type 2 diabetes. "
        "Lisinopril was prescribed to manage hypertension. "
        "Developed a persistent dry cough after starting Lisinopril. "
        "Reports nausea since initiating Metformin. "
        "Aspirin 81 mg given for chest pain."
    ),
}

# Back-compat alias — older code paths still reference SAMPLE_TEXT.
SAMPLE_TEXT = SAMPLE_PRESETS["Default case"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

MODELS_WITHOUT_REAL_CONFIDENCE = {"ner_bilstm_crf", "ner_spacy"}


def entities_to_df(entities: list[Entity], show_confidence: bool = True) -> pd.DataFrame:
    rows = [e.to_table_row() for e in entities]
    df = pd.DataFrame(rows, columns=["Entity", "Label", "Confidence Score"])
    if not show_confidence and not df.empty:
        df["Confidence Score"] = "—"
    return df


def relations_to_df(relations: list[dict]) -> pd.DataFrame:
    rows = []
    for r in relations:
        rows.append(
            {
                "Entity 1 (DRUG)": r["e1"].text,
                "Relation ➔": r["relation"],
                "Entity 2 (DISEASE)": r["e2"].text,
                "Confidence": f"{r['confidence'] * 100:.1f}%",
            }
        )
    return pd.DataFrame(
        rows, columns=["Entity 1 (DRUG)", "Relation ➔", "Entity 2 (DISEASE)", "Confidence"]
    )


def summary_row(display_name: str, reg_key: str, entities: list[Entity]) -> dict:
    meta_key = reg_key.replace("ner_", "")
    meta = MODEL_METADATA.get(meta_key, {})
    avg_conf = (
        sum(e.confidence for e in entities) / len(entities) if entities else 0.0
    )
    if reg_key in MODELS_WITHOUT_REAL_CONFIDENCE:
        avg_display = "—"
    elif entities:
        avg_display = f"{avg_conf * 100:.1f}%"
    else:
        avg_display = "—"
    return {
        "Model": display_name,
        "Entities": len(entities),
        "Avg Confidence": avg_display,
        "Target F1": meta.get("target_f1", "—"),
        "Notes": meta.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 1. Entity Recognition (NER Models)",
        "🔗 2. Relation Extraction",
        "📝 3. SOAP Note Generation",
    ]
)


# ==========================================================================
# TAB 1 — NER comparison across all six models
# ==========================================================================

with tab1:
    st.header("Medical Entity Recognition — Six-Model Comparison")
    st.markdown(
        "Same input text is fed into all six trained NER models. Outputs are "
        "shown side-by-side; the combined visualization below uses BioBERT "
        "(our best performer) offsets."
    )

    # Initialise the text_area's state key once so preset buttons can
    # overwrite it cleanly via session_state (Streamlit disallows passing
    # `value=` alongside `key=` once the widget has been rendered).
    if "ui_input_text" not in st.session_state:
        st.session_state["ui_input_text"] = st.session_state.get(
            "input_text", SAMPLE_TEXT
        )

    # Preset chips row — one click loads a curated transcript into the box.
    st.markdown("**Load a sample transcript:**")
    preset_cols = st.columns(len(SAMPLE_PRESETS))
    for (label, text), col in zip(SAMPLE_PRESETS.items(), preset_cols):
        with col:
            if st.button(f"📋 {label}", key=f"preset_{label}", use_container_width=True):
                st.session_state["ui_input_text"] = text
                st.rerun()

    col_input, col_info = st.columns([2, 1])
    with col_input:
        input_text = st.text_area(
            "Patient Transcript / Clinical Note Input",
            height=180,
            key="ui_input_text",
        )
    with col_info:
        st.info(
            "**Dataset context:**\nAll six models were trained on BC5CDR "
            "(5-label IOB: O, B-Chemical, I-Chemical, B-Disease, I-Disease). "
            "Labels are remapped to **DRUG / DISEASE** for the UI."
        )
        run_ner = st.button("Run NER Inference 🚀", use_container_width=True)

    if run_ner:
        st.session_state["input_text"] = input_text

        # Predict with every model in the registry; errors surface inline.
        results: dict[str, list[Entity]] = {}
        errors: dict[str, str] = {}

        with st.spinner("Running all six NER models…"):
            for display_name, loader, key in NER_REGISTRY:
                ents, err = safe_predict_ner(loader, input_text)
                if err:
                    errors[key] = err
                ents = dedupe_entities(ents)
                st.session_state[key] = ents
                results[key] = ents

        st.divider()

        # Row 1: BiLSTM-CRF, BiGRU, spaCy
        # Row 2: BERT, BioBERT, BART
        row1 = NER_REGISTRY[:3]
        row2 = NER_REGISTRY[3:]

        for row in (row1, row2):
            cols = st.columns(3)
            for col, (name, _loader, key) in zip(cols, row):
                with col:
                    st.subheader(name)
                    meta_key = key.replace("ner_", "")
                    meta = MODEL_METADATA.get(meta_key, {})
                    st.caption(
                        f"Target F1: {meta.get('target_f1', '—')} · "
                        f"{meta.get('notes', '')}"
                    )
                    if key in errors:
                        st.markdown(
                            f"<div class='err-box'>⚠️ {errors[key]}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        show_conf = key not in MODELS_WITHOUT_REAL_CONFIDENCE
                        df = entities_to_df(results[key], show_confidence=show_conf)
                        if df.empty:
                            st.caption("_No entities detected._")
                        else:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            if not show_conf:
                                st.caption(
                                    "_Confidence shown as \"—\": this model's "
                                    "decoder (CRF Viterbi for BiLSTM-CRF, "
                                    "rule-based span matching for spaCy) does "
                                    "not expose per-entity probabilities._"
                                )

        st.divider()

        # Highlighted visualization based on BioBERT offsets
        st.markdown("### BioBERT Highlighted Visualization")
        st.markdown(
            "Legend: "
            "<span class='entity-drug'>DRUG</span> &nbsp;|&nbsp; "
            "<span class='entity-disease'>DISEASE</span>",
            unsafe_allow_html=True,
        )
        biobert_ents = results.get("ner_biobert", [])
        if biobert_ents:
            html = render_highlighted_html(input_text, biobert_ents)
            st.markdown(
                f"<div style='line-height: 1.8; font-size: 16px; "
                f"border: 1px solid #ccc; padding: 15px; border-radius: 5px;'>"
                f"{html}</div>",
                unsafe_allow_html=True,
            )
        elif "ner_biobert" in errors:
            st.warning("BioBERT not loaded — highlight preview unavailable.")
        else:
            st.caption("_BioBERT produced no entities for this input._")

        # Summary table
        st.markdown("### Per-Model Summary")
        summary_rows = [
            summary_row(name, key, results.get(key, []))
            for name, _loader, key in NER_REGISTRY
        ]
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================================
# TAB 2 — Relation Extraction (BERT-RE vs BioBERT-RE)
# ==========================================================================

with tab2:
    st.header("Relation Extraction — BERT-RE vs BioBERT-RE")
    st.markdown(
        "Each (DRUG, DISEASE) pair within the same sentence is passed through "
        "both BERT-RE and BioBERT-RE after wrapping with "
        "`[DRUG] … [/DRUG]` / `[DISEASE] … [/DISEASE]` markers. Only binary "
        "labels are predicted: **treats** / **side_effect** "
        f"(confidence threshold: {RE_CONFIDENCE_THRESHOLD:.2f})."
    )

    biobert_ents = st.session_state.get("ner_biobert")
    input_text = st.session_state.get("input_text")

    if not input_text or biobert_ents is None:
        st.warning(
            "⚠️ Run Tab 1 (NER) first — RE needs BioBERT entities from the "
            "previous step as input."
        )
    else:
        entity_preview = ", ".join(f"{e.text} ({e.label})" for e in biobert_ents) or "—"
        st.text_area(
            "Entities carried over from BioBERT (Tab 1)",
            value=entity_preview,
            disabled=True,
            height=80,
        )

        if st.button("Extract Relationships 🔗", key="run_re"):
            from inference.entity_utils import build_re_candidates

            re_results: dict[str, list[dict]] = {}
            re_errors: dict[str, str] = {}

            with st.spinner("Running BERT-RE and BioBERT-RE…"):
                for display_name, loader, key in RE_REGISTRY:
                    rels, err = safe_predict_re(loader, input_text, biobert_ents)
                    if err:
                        re_errors[key] = err
                    st.session_state[key] = rels
                    re_results[key] = rels

            n_candidates = len(build_re_candidates(input_text, biobert_ents))

            st.divider()
            if n_candidates == 0:
                st.warning(
                    "⚠️ No (DRUG, DISEASE) pairs co-occur within a single "
                    "sentence of this note, so there's nothing for RE to "
                    "classify. Try a note where a drug and a disease appear "
                    "in the same sentence (e.g. _\"Aspirin was prescribed for "
                    "chest pain.\"_)."
                )
            c1, c2 = st.columns(2)
            for col, (name, _loader, key) in zip((c1, c2), RE_REGISTRY):
                with col:
                    st.subheader(name)
                    if key in re_errors:
                        st.markdown(
                            f"<div class='err-box'>⚠️ {re_errors[key]}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        df = relations_to_df(re_results[key])
                        if df.empty:
                            if n_candidates == 0:
                                st.caption(
                                    f"_No candidate pairs to score "
                                    f"(see note above)._"
                                )
                            else:
                                st.caption(
                                    f"_{n_candidates} pair(s) evaluated; "
                                    f"none passed the "
                                    f"{RE_CONFIDENCE_THRESHOLD:.2f} "
                                    f"confidence threshold._"
                                )
                        else:
                            st.dataframe(
                                df, use_container_width=True, hide_index=True
                            )


# ==========================================================================
# TAB 3 — SOAP Note Generation
# ==========================================================================

with tab3:
    st.header("Automated SOAP Note Generation")
    st.markdown(
        "Template-based synthesis using **BioBERT** (best NER) entities and "
        "**BioBERT-RE** (best RE) relations from the previous tabs. The "
        "generator is rule-based — no LLM — and deterministic."
    )

    input_text = st.session_state.get("input_text")
    biobert_ents = st.session_state.get("ner_biobert")
    biobert_rels = st.session_state.get("re_biobert")

    missing = []
    if not input_text or biobert_ents is None:
        missing.append("Tab 1 (NER)")
    if biobert_rels is None:
        missing.append("Tab 2 (RE)")

    if missing:
        st.warning(
            "⚠️ Run " + " and ".join(missing) + " first to populate session state."
        )
    else:
        if st.button("Generate Structured SOAP Note 📝"):
            with st.spinner("Synthesizing S / O / A / P sections…"):
                soap_md = generate_soap_note(
                    transcript=input_text,
                    entities=biobert_ents,
                    relations=biobert_rels,
                )
            st.session_state["soap_md"] = soap_md

        soap_md = st.session_state.get("soap_md")
        if soap_md:
            st.divider()
            st.markdown(
                f"<div class='soap-box'>\n\n{soap_md}\n\n</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Formatting derived from BioBERT NER + BioBERT-RE outputs. "
                "SOAP template is rule-based (see inference/soap_generator.py)."
            )
            st.download_button(
                label="💾 Export Note to Markdown",
                data=soap_md,
                file_name="soap_note.md",
                mime="text/markdown",
            )
