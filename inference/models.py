"""
Cached model loaders + top-level predict entry points.

Every loader is wrapped with `@st.cache_resource` so Streamlit keeps the model
in memory across reruns — without this, every UI rerun would re-load BioBERT
from disk (~10–30 s on CPU, unusable).

If Streamlit is not available (e.g. running a smoke test from the REPL), the
loaders fall back to a plain no-op decorator so imports don't break.
"""

from __future__ import annotations

from typing import Callable

# ---- cache_resource fallback when Streamlit is absent ---------------------
try:
    import streamlit as st
    _cache_resource = st.cache_resource
except Exception:  # pragma: no cover
    def _cache_resource(fn: Callable) -> Callable:
        return fn


from config import (
    BILSTM_CRF_DIR,
    BIGRU_DIR,
    BERT_NER_DIR,
    BIOBERT_NER_DIR,
    BART_NER_DIR,
    SPACY_NER_DIR,
    RE_BERT_DIR,
    RE_BIOBERT_DIR,
)


# ---------------------------------------------------------------------------
# NER loaders
# ---------------------------------------------------------------------------

@_cache_resource
def load_bilstm_crf():
    from inference.ner_bilstm_crf import BiLSTMCRFInference
    return BiLSTMCRFInference(BILSTM_CRF_DIR)


@_cache_resource
def load_bigru():
    from inference.ner_bigru import BiGRUInference
    return BiGRUInference(BIGRU_DIR)


@_cache_resource
def load_bert_ner():
    from inference.ner_hf import HFNerInference
    return HFNerInference(BERT_NER_DIR)


@_cache_resource
def load_biobert_ner():
    from inference.ner_hf import HFNerInference
    return HFNerInference(BIOBERT_NER_DIR)


@_cache_resource
def load_bart_ner():
    from inference.ner_bart import BartNerInference
    return BartNerInference(BART_NER_DIR)


@_cache_resource
def load_spacy_ner():
    from inference.ner_spacy import SpacyNerInference
    return SpacyNerInference(SPACY_NER_DIR)


# ---------------------------------------------------------------------------
# RE loaders
# ---------------------------------------------------------------------------

@_cache_resource
def load_bert_re():
    from inference.re_hf import HFReInference
    return HFReInference(RE_BERT_DIR)


@_cache_resource
def load_biobert_re():
    from inference.re_hf import HFReInference
    return HFReInference(RE_BIOBERT_DIR)


# ---------------------------------------------------------------------------
# Registry — used by the UI to iterate all 6 NER models uniformly
# ---------------------------------------------------------------------------

# display_name, loader, session_state key
NER_REGISTRY: list[tuple[str, Callable, str]] = [
    ("BiLSTM-CRF", load_bilstm_crf, "ner_bilstm_crf"),
    ("BiGRU",      load_bigru,      "ner_bigru"),
    ("spaCy",      load_spacy_ner,  "ner_spacy"),
    ("BERT",       load_bert_ner,   "ner_bert"),
    ("BioBERT",    load_biobert_ner,"ner_biobert"),
    ("BART",       load_bart_ner,   "ner_bart"),
]

RE_REGISTRY: list[tuple[str, Callable, str]] = [
    ("BERT-RE",    load_bert_re,    "re_bert"),
    ("BioBERT-RE", load_biobert_re, "re_biobert"),
]


# ---------------------------------------------------------------------------
# Top-level predict helpers the UI can call. They swallow FileNotFoundError
# so one missing checkpoint doesn't brick the whole tab — the UI shows a
# yellow "not loaded yet" state instead.
# ---------------------------------------------------------------------------

def safe_predict_ner(loader: Callable, text: str):
    """
    Returns (entities, error_str). `entities` is [] on error; `error_str` is
    None on success.
    """
    try:
        model = loader()
    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:  # catch-all so the UI never crashes during demo
        return [], f"Load failed: {type(e).__name__}: {e}"

    try:
        return model.predict(text), None
    except Exception as e:
        return [], f"Predict failed: {type(e).__name__}: {e}"


def safe_predict_re(loader: Callable, text: str, entities):
    try:
        model = loader()
    except FileNotFoundError as e:
        return [], str(e)
    except Exception as e:
        return [], f"Load failed: {type(e).__name__}: {e}"

    try:
        return model.predict(text, entities), None
    except Exception as e:
        return [], f"Predict failed: {type(e).__name__}: {e}"
