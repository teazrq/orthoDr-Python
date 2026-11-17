"""Bundled example data exported without numerical conversion from R."""

from importlib import resources
import json

import numpy as np


def load_skcm():
    """Load orthoDr's aligned TCGA melanoma clinical and gene-expression data.

    Returns a dictionary with ``clinical`` (469 by 4), ``genes`` (469 by 20),
    ``clinical_columns``, ``gene_names``, and ``subject_ids``. Eight missing
    ages remain NaN. Rows are retained in their original R order. If a future
    source has different row identifiers, ``gene_subject_ids`` is also given
    rather than silently reordering or joining observations. ``provenance``
    records the pinned R source, source-file checksums, and numerical checks.

    Each call returns independently owned arrays. NumPy is the only data
    dependency; neither pandas nor an R installation is needed.
    """
    data = resources.files("orthodr").joinpath("data")
    metadata = json.loads(data.joinpath("skcm.json").read_text(encoding="utf-8"))
    with data.joinpath("skcm_clinical.npy").open("rb") as stream:
        clinical = np.load(stream, allow_pickle=False)
    with data.joinpath("skcm_genes.npy").open("rb") as stream:
        genes = np.load(stream, allow_pickle=False)
    result = dict(clinical=clinical, genes=genes,
                  clinical_columns=np.array(metadata["clinical_columns"], dtype=str),
                  gene_names=np.array(metadata["gene_names"], dtype=str),
                  subject_ids=np.array(metadata["subject_ids"], dtype=str),
                  provenance=metadata["provenance"])
    if metadata["gene_subject_ids"] != metadata["subject_ids"]:
        result["gene_subject_ids"] = np.array(metadata["gene_subject_ids"], dtype=str)
    return result
