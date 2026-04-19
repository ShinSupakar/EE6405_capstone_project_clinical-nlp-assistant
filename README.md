# Clinical NLP Assistant with Entity Recognition & Relation Extraction

End-to-end clinical NLP pipeline: unstructured clinical text → Named Entity Recognition (NER) → Relation Extraction (RE) → structured SOAP note.

Developed as a group project (**Table D24**) covering six NER models, two RE models, and a rule-based SOAP note generator, all demo-able through a Streamlit web app.

---

## Team

| Member | Contribution |
|---|---|
| Gaur Prajna | spaCy and BiLSTM-CRF NER models |
| Tan Hong Guan | BiGRU NER model |
| Supakar Shinjini | BERT, BioBERT, BART NER models |
| Borra Kailash Sai Kumar | Relation Extraction pipeline (BERT-RE, BioBERT-RE) |
| Gupta Chaitanya | Dataset preprocessing; Streamlit web app + end-to-end integration |

---

## Pipeline

```
Clinical Note  →  NER (DRUG / DISEASE)  →  RE (treats / side_effect)  →  SOAP Note (S / O / A / P)
```

Best models from each stage — **BioBERT** (NER, F1 ≈ 0.876) and **BioBERT-RE** — feed the rule-based SOAP note generator in the demo.

---

## Repo structure

```
clinical-nlp-assistant/
├── README.md
├── requirements.txt          # pinned versions the demo was validated against
├── config.py                 # paths and runtime config
├── re_eval_set.py            # RE evaluation set builder
├── preprocessing/
│   └── preprocessing_pipeline.ipynb    # BC5CDR / GENIA / DrugBank / MTSamples
├── notebooks/                # one per model — training + evaluation
├── inference/                # inference wrappers + SOAP generator
├── loaders/                  # PyTorch datasets for NER / RE
├── UI/
│   └── app.py                # Streamlit demo (3 tabs: NER → RE → SOAP)
├── data/
│   └── correct_treats.jsonl  # small sample labels
└── models/                   # NOT in repo — see "Downloading model weights" below
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/<org-or-user>/clinical-nlp-assistant.git
cd clinical-nlp-assistant
```

### 2. Install dependencies

Python **3.9** is required (the venv used for the demo was Python 3.9; `spacy==3.7.5` and `thinc` wheels are the reason — see `requirements.txt` for details).

```bash
python3.9 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download model weights

Trained model weights (~4 GB total) are hosted on Google Drive:

**[Google Drive — trained model weights](https://drive.google.com/drive/folders/139EqsK7C2v9DQCrhh1Qbvo6uFU6g1-E7)**

Download the `models/` folder from Drive and place it at the repo root so the structure becomes:

```
clinical-nlp-assistant/
└── models/
    ├── bart_ner/
    ├── bert_ner/
    ├── bigru/
    ├── bilstm_crf/
    ├── biobert_ner/
    ├── re_bert/
    ├── re_biobert/
    └── spacy_ner/
```

### 4. Run the demo

```bash
streamlit run UI/app.py
```

The dashboard opens at `http://localhost:8501` with three tabs:

1. **Entity Recognition** — side-by-side output from all six NER models on the same transcript.
2. **Relation Extraction** — BERT-RE vs BioBERT-RE on the BioBERT entities.
3. **SOAP Note Generation** — structured clinical note from the best NER + RE outputs.

---

## Datasets

| Dataset | Purpose |
|---|---|
| **BC5CDR** | Biomedical NER — chemical + disease labels (5-tag BIO) |
| **GENIA** | NER augmentation; tokens labeled `O` since non-chemical/disease |
| **DrugBank** | Drug standardization + RE pair generation |
| **MTSamples** | Raw SOAP-style clinical notes used as demo input |

All preprocessing — cleaning, BIO conversion, GENIA trigger alignment, DrugBank deduplication — is in `preprocessing/preprocessing_pipeline.ipynb`.

---

## How each model was trained

See `notebooks/` for each model's training + evaluation notebook. Each notebook is self-contained and references the processed datasets produced by the preprocessing notebook.

---

## Demo

A short walkthrough of the three-tab pipeline is in the project presentation. The demo runs end-to-end on the "Default case" or "SOAP-heavy case" sample transcripts shipped in the app.

---

## Contributing (team workflow)

1. Clone the repo and create a branch:
   ```bash
   git checkout -b <your-name>/<what-youre-adding>
   ```
2. Add your training notebook to `notebooks/` (one notebook per model).
3. Clear large cell outputs before committing (`Kernel → Restart & Clear Output` in Jupyter) — keeps notebooks diff-friendly.
4. Commit and push:
   ```bash
   git add notebooks/<your-notebook>.ipynb
   git commit -m "add <model> training notebook"
   git push origin <your-name>/<what-youre-adding>
   ```
5. Open a pull request on GitHub and merge into `main`.

---

## License

MIT — see `LICENSE` (optional; pick whatever your team prefers or omit for a private repo).
