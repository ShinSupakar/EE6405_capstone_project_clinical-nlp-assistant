"""
Central configuration for the Clinical NLP capstone project.

All paths, label schemas, display maps, and runtime constants live here so the
UI and inference wrappers stay in lockstep with how models were trained.
"""

from pathlib import Path
import torch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

# Repo root = this file's directory
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

# Per-model checkpoint directories. Each must match what is extracted from the
# teammate-uploaded zips (see README / plan doc for exact layouts).
BILSTM_CRF_DIR = MODELS_DIR / "bilstm_crf"   # best_model.pt + word2idx.json + label2idx.json
BIGRU_DIR      = MODELS_DIR / "bigru"        # best_model.pt + vocab.json + label_list.json + hparams.json
BERT_NER_DIR   = MODELS_DIR / "bert_ner"     # HF format (saved via trainer.save_model)
BIOBERT_NER_DIR = MODELS_DIR / "biobert_ner" # HF format
BART_NER_DIR   = MODELS_DIR / "bart_ner"     # HF format (config.json + model.safetensors + tokenizer files); custom BartForTokenClassificationFull head
SPACY_NER_DIR  = MODELS_DIR / "spacy_ner"    # spaCy directory
RE_BERT_DIR    = MODELS_DIR / "re_bert"      # HF format with [DRUG] / [/DRUG] / [DISEASE] / [/DISEASE]
RE_BIOBERT_DIR = MODELS_DIR / "re_biobert"   # HF format

# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Prefer CUDA, then Apple Metal, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# ----------------------------------------------------------------------------
# Label schemas — exactly as the models were trained
# ----------------------------------------------------------------------------

# All 6 NER models share this 5-label BC5CDR schema.
# NOTE: this does NOT match loaders/ner_dataset.py (15 labels). That loader is
# for training only; do not use it at inference time.
NER_LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
NER_LABEL2ID = {lbl: i for i, lbl in enumerate(NER_LABELS)}
NER_ID2LABEL = {i: lbl for lbl, i in NER_LABEL2ID.items()}

# Relation extraction is BINARY.
# NOTE: this does NOT match loaders/relation_dataset.py (4 labels). Same caveat.
RE_LABELS = ["side_effect", "treats"]
RE_LABEL2ID = {lbl: i for i, lbl in enumerate(RE_LABELS)}
RE_ID2LABEL = {i: lbl for lbl, i in RE_LABEL2ID.items()}

# Relation-extraction entity markers the tokenizer was trained with.
RE_MARKERS = {
    "drug_open": "[DRUG]",
    "drug_close": "[/DRUG]",
    "disease_open": "[DISEASE]",
    "disease_close": "[/DISEASE]",
}

# ----------------------------------------------------------------------------
# Display remapping — the models output "Chemical" / "Disease" but the UI
# colour palette / copy uses DRUG / DISEASE. This is a *display-only* remap.
# ----------------------------------------------------------------------------

LABEL_DISPLAY_MAP = {
    "Chemical": "DRUG",
    "Disease": "DISEASE",
    # spaCy may emit uppercase or its own label names:
    "CHEMICAL": "DRUG",
    "DISEASE": "DISEASE",
    "DRUG": "DRUG",
}

# CSS class suffix (must match the .entity-<suffix> rules in UI/app.py)
LABEL_CSS_CLASS = {
    "DRUG": "drug",
    "DISEASE": "disease",
    # kept for backwards-compat with the original mock data colours, even
    # though the trained models will never emit these:
    "SYMPTOM": "symptom",
    "PROCEDURE": "procedure",
}

# ----------------------------------------------------------------------------
# Tokenization / runtime limits
# ----------------------------------------------------------------------------

MAX_LENGTH_NER = 512   # transformer NER
MAX_LENGTH_RE = 256    # relation extraction (shorter sentences)

# Confidence threshold below which RE pairs are dropped from the UI.
RE_CONFIDENCE_THRESHOLD = 0.50

# ----------------------------------------------------------------------------
# Model metadata for the summary table on Tab 1
# ----------------------------------------------------------------------------

# Reported val-set metrics from the team's notebooks (update when they share
# final numbers). Used only for display in the summary table; not load-bearing.
MODEL_METADATA = {
    "bilstm_crf": {"name": "BiLSTM-CRF",  "target_f1": ">=0.85",          "notes": "baseline"},
    "bigru":      {"name": "BiGRU",        "target_f1": "~0.73 (BC5CDR)",  "notes": "alt baseline"},
    "bert":       {"name": "BERT",         "target_f1": "0.833 (BC5CDR)",  "notes": "bert-base-cased"},
    "biobert":    {"name": "BioBERT",      "target_f1": "0.876 (BC5CDR)",  "notes": "best — dmis-lab/biobert-v1.1"},
    "bart":       {"name": "BART",         "target_f1": "pending team report", "notes": "facebook/bart-base + custom TokenClassificationFull head"},
    "spacy":      {"name": "spaCy",        "target_f1": "pending team report", "notes": "rule+stat hybrid"},
}


# ----------------------------------------------------------------------------
# Model roster — flat list of every model wired up end-to-end, including RE.
# Used by the sidebar comparison view.
# ----------------------------------------------------------------------------

MODEL_ROSTER = [
    # NER (6)
    {"name": "BiLSTM-CRF", "task": "NER", "target_f1": ">=0.85",             "notes": "baseline, word-level + CRF"},
    {"name": "BiGRU",      "task": "NER", "target_f1": "~0.73 (BC5CDR)",     "notes": "alt baseline"},
    {"name": "BERT",       "task": "NER", "target_f1": "0.833 (BC5CDR)",     "notes": "bert-base-cased"},
    {"name": "BioBERT",    "task": "NER", "target_f1": "0.876 (BC5CDR)",     "notes": "★ best NER — drives RE + SOAP"},
    {"name": "BART",       "task": "NER", "target_f1": "pending team report","notes": "encoder-decoder + token-cls head"},
    {"name": "spaCy",      "task": "NER", "target_f1": "pending team report","notes": "rule+stat hybrid"},
    # RE (2)
    {"name": "BERT-RE",    "task": "RE",  "target_f1": "pending team report","notes": "bert-base + [DRUG]/[DISEASE] markers"},
    {"name": "BioBERT-RE", "task": "RE",  "target_f1": "pending team report","notes": "★ best RE — drives SOAP; note domain-shift caveat"},
]
