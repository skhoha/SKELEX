from __future__ import annotations

"""

Utility for loading BTXRD dataset using metadata as the primary source.

Usage
-----
>>> data = load_btxrd_dataset(
...     image_dir="/path/to/images",
...     annotation_dir="/path/to/annotations",
...     metadata_path="/path/to/metadata.xlsx",
... )
>>> len(data)
12456

The returned list is ready for conversion to a Hugging Face `Dataset` or any
custom dataloader.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


# -----------------------------------------------------------------------
# Helper functions to convert metadata flags → class & mask indices
# -----------------------------------------------------------------------
def make_class_list(meta: Dict[str, int | float | str]) -> List[int]:
    """Return a list of class indices based on metadata flags."""

    classes: List[int] = []

    # Append location
    location = meta.get("Location")
    if location is not None:
        classes.append(location)

    if meta.get("Tumor type") == 1:
        classes.append(0)

    # Tumor classification
    if meta.get("malignant(+lch+low grade chondrosarcoma(axial))") == 1:
        classes.append(1)
    elif meta.get("intermediate(gct+chondroblastoma+osteoblastoma+ACT)") == 1:
        classes.append(2)
    elif meta.get("benign") == 1:
        classes.append(3)
    elif meta.get("No_bonetu") == 1:
        classes.append(4)
    else :
        raise ValueError(f"Unknown tumor class in metadata: {meta}")
    
    # Fracture classification
    if meta.get("neoplastic pathologic fracture") == 1:
        classes.append(34)
    elif meta.get("non-pathologic fracture") == 1:
        classes.append(35)
    elif meta.get("No_fracture") == 1:
        classes.append(36)
    else :
        raise ValueError(f"Unknown fracture class in metadata: {meta}")
    
    # Implant
    if meta.get("Implant") == 1:
        classes.append(37)

    return classes


def make_mask_list(meta: Dict[str, int | float | str]) -> List[int]:
    """Return a list of mask indices based on metadata flags."""
    masks: List[int] = []
    # Every label is unmasked
    return masks


# -----------------------
# Main loader
# -----------------------


def load_snuhbt_dataset(
    image_dir: os.PathLike,
    metadata_path: os.PathLike,
) -> List[Dict]:
    """Create a list of dataset entries driven by metadata.

    Parameters
    ----------
    image_dir      : Directory with radiograph image files.
    metadata_path  : Path to the csv sheet containing per-image metadata.

    Returns
    -------
    A list of dictionaries, one per image, with keys:

        image_id, image, bbox, label, classes, masks

    Images lacking an annotation file will contain empty bbox and label
    lists but are still included so no negative cases are dropped.
    """

    # Load metadata and index by image ID for O(1) look‑ups
    metadata_df = pd.read_csv(metadata_path)
    metadata_df.set_index("file_name", inplace=True)

    data_entries: List[Dict] = []

    for file_name, meta_row in metadata_df.iterrows():
        meta = meta_row.to_dict()

        image_path = Path(image_dir) / file_name
        if not image_path.exists():
            # Skip if the image itself is missing – nothing to train/evaluate on
            continue
        
        bboxes, box_labels = [], []  # keep negatives!

        labels = make_class_list(meta)
        masks = make_mask_list(meta)

        if 0 in labels:
            bboxes.append([0.5, 0.5, 1.0, 1.0])
            box_labels.append("normal")

        data_entries.append(
            {
                "image_id": file_name,
                "image": str(image_path),
                "bbox": bboxes,
                "box_label": box_labels,
                "labels": labels,
                "masks": masks,
            }
        )

    return data_entries