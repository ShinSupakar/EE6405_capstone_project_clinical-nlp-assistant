"""
Hand-labeled in-distribution eval set for RE (BC5CDR schema, binary + optional no_relation).

Purpose: measure how RE performs on SOAP-note-style sentences that resemble the
demo app's inputs, NOT on LLM-augmented conversational training data.

Each row: the full sentence, the two entity spans (drug, disease), and the
expected relation label. Entities are case-sensitive text matches; char offsets
can be computed from str.find() since no entity appears twice in the same sentence.

Usage (teammate side):
    for row in EVAL_SET:
        pred, conf = re_model.predict(row["sentence"], row["drug"], row["disease"])
        ...
    Then: classification_report(y_true, y_pred, digits=3)
"""

EVAL_SET = [
    # --- treats (5) -------------------------------------------------------
    {"sentence": "Patient takes Metformin daily for type 2 diabetes.",
     "drug": "Metformin", "disease": "type 2 diabetes", "label": "treats"},
    {"sentence": "Lisinopril was prescribed to manage hypertension.",
     "drug": "Lisinopril", "disease": "hypertension", "label": "treats"},
    {"sentence": "Started on Atorvastatin for hyperlipidemia.",
     "drug": "Atorvastatin", "disease": "hyperlipidemia", "label": "treats"},
    {"sentence": "Aspirin 81 mg given for chest pain.",
     "drug": "Aspirin", "disease": "chest pain", "label": "treats"},
    {"sentence": "Amoxicillin administered to treat pneumonia.",
     "drug": "Amoxicillin", "disease": "pneumonia", "label": "treats"},

    # --- side_effect (5) --------------------------------------------------
    {"sentence": "Developed a persistent dry cough after starting Lisinopril.",
     "drug": "Lisinopril", "disease": "dry cough", "label": "side_effect"},
    {"sentence": "Reports nausea since initiating Metformin.",
     "drug": "Metformin", "disease": "nausea", "label": "side_effect"},
    {"sentence": "Experienced myopathy attributed to Atorvastatin.",
     "drug": "Atorvastatin", "disease": "myopathy", "label": "side_effect"},
    {"sentence": "Patient complains of GI bleeding on Aspirin therapy.",
     "drug": "Aspirin", "disease": "GI bleeding", "label": "side_effect"},
    {"sentence": "Rash developed following Penicillin administration.",
     "drug": "Penicillin", "disease": "Rash", "label": "side_effect"},

    # --- no_relation distractors (5) --------------------------------------
    # These co-occur in the same sentence but are not clinically linked.
    # If the model is binary {treats, side_effect} with no `no_relation` head,
    # its prediction here is "whatever it's biased toward". These cases are the
    # cleanest evidence of over-prediction. Add a no_relation class to fix.
    {"sentence": "History of diabetes; prescribed Amoxicillin for a sinus infection.",
     "drug": "Amoxicillin", "disease": "diabetes", "label": "no_relation"},
    {"sentence": "Admitted with hypertension, currently on Omeprazole for reflux.",
     "drug": "Omeprazole", "disease": "hypertension", "label": "no_relation"},
    {"sentence": "History of asthma; started Metformin today for newly diagnosed diabetes.",
     "drug": "Metformin", "disease": "asthma", "label": "no_relation"},
    {"sentence": "Reports chronic back pain; takes Atorvastatin daily for cholesterol.",
     "drug": "Atorvastatin", "disease": "back pain", "label": "no_relation"},
    {"sentence": "On Warfarin long-term; presented today with a sprained ankle.",
     "drug": "Warfarin", "disease": "sprained ankle", "label": "no_relation"},
]
