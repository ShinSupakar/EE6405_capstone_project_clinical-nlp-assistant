"""
Inference package — wraps every trained NER / RE checkpoint behind a
consistent `predict(...)` interface.

Public API:
    Entity              — dataclass every NER wrapper returns
    render_highlighted_html
    build_re_candidates
    generate_soap_note
    NER_REGISTRY / RE_REGISTRY  — iterate over all models in the UI
    safe_predict_ner / safe_predict_re  — UI-side wrappers that swallow errors
"""

from inference.entity_utils import (
    Entity,
    RECandidate,
    remap_label,
    css_class_for,
    render_highlighted_html,
    whitespace_tokenize_with_offsets,
    find_sentence_spans,
    build_re_candidates,
    dedupe_entities,
    group_iob_tags,
    fill_entity_texts,
)
from inference.soap_generator import generate_soap_note
from inference.models import (
    NER_REGISTRY,
    RE_REGISTRY,
    safe_predict_ner,
    safe_predict_re,
    load_bilstm_crf,
    load_bigru,
    load_bert_ner,
    load_biobert_ner,
    load_bart_ner,
    load_spacy_ner,
    load_bert_re,
    load_biobert_re,
)

__all__ = [
    "Entity",
    "RECandidate",
    "remap_label",
    "css_class_for",
    "render_highlighted_html",
    "whitespace_tokenize_with_offsets",
    "find_sentence_spans",
    "build_re_candidates",
    "dedupe_entities",
    "group_iob_tags",
    "fill_entity_texts",
    "generate_soap_note",
    "NER_REGISTRY",
    "RE_REGISTRY",
    "safe_predict_ner",
    "safe_predict_re",
    "load_bilstm_crf",
    "load_bigru",
    "load_bert_ner",
    "load_biobert_ner",
    "load_bart_ner",
    "load_spacy_ner",
    "load_bert_re",
    "load_biobert_re",
]
