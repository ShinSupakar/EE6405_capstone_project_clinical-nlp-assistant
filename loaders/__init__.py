"""
loaders — Clinical NLP Project
================================
Reusable PyTorch dataset classes for all three task types.

Quick import:
    from loaders import NERDataset, RelationDataset, UnlabeledDataset
    from loaders import NER_LABEL2ID, NER_ID2LABEL
    from loaders import RELATION_LABEL2ID, RELATION_ID2LABEL

See each module's docstring for full usage examples.
"""

from .ner_dataset      import NERDataset,      NER_LABEL2ID,      NER_ID2LABEL
from .relation_dataset import RelationDataset, RELATION_LABEL2ID, RELATION_ID2LABEL
from .unlabeled_dataset import UnlabeledDataset

__all__ = [
    "NERDataset",
    "NER_LABEL2ID",
    "NER_ID2LABEL",
    "RelationDataset",
    "RELATION_LABEL2ID",
    "RELATION_ID2LABEL",
    "UnlabeledDataset",
]
