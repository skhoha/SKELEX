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
from typing import Dict, List, Any, Mapping

import pandas as pd


def _to_bool(v: Any) -> bool:
    if v is None: 
        return False
    if isinstance(v, (int, float)):
        return v != 0
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


# ------------------------------------------------------------------
# Helper functions to convert metadata flags → class & mask indices
# ------------------------------------------------------------------
def make_class_list(meta: Dict[str, int | float | str]) -> List[int]:
    """Return a list of class indices based on metadata flags."""

    classes: List[int] = []

    tumor = _to_bool(meta.get("tumor"))
    benign = _to_bool(meta.get("benign"))
    malignant = _to_bool(meta.get("malignant"))

    if tumor:
        classes.append(0)
        if benign and malignant:
            raise ValueError("Sample cannot be both benign and malignant.")
        if benign:
            classes.append(3)
        if malignant:
            classes.append(1)
    else :
        classes.append(4)

    # every sample gets the No fracture label 36
    classes.append(36)

    return classes


def make_mask_list(meta: Dict[str, int | float | str]) -> List[int]:
    """Return a list of mask indices based on metadata flags."""

    masks: List[int] = []

    tumor = _to_bool(meta.get("tumor"))
    if tumor:
        # 5‒37 inclusive
        masks.extend(range(5, 38))
    else:
        masks.append(0)  # abnormality label is masked, since some of the x-ray looks abnormal
        masks.extend(range(5, 34))  # 5‒33 inclusive

    return masks


# -----------------------
# Annotation parser
# -----------------------

def _parse_annotation(json_path: os.PathLike) -> tuple[list[list[float]], list[str]]:
    """Extract normalized YOLO-style bboxes + box_labels from a LabelMe JSON file."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bboxes: list[list[float]] = []
    box_labels: list[str] = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue

        (x1, y1), (x2, y2) = shape["points"]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        w_img = data["imageWidth"]
        h_img = data["imageHeight"]

        cx = (x_min + x_max) / 2 / w_img
        cy = (y_min + y_max) / 2 / h_img
        w = (x_max - x_min) / w_img
        h = (y_max - y_min) / h_img

        bboxes.append([cx, cy, w, h])
        box_labels.append(shape["label"])

    return bboxes, box_labels


# ------------------------
# Main loader
# ------------------------


def load_btxrd_dataset(
    image_dir: os.PathLike,
    annotation_dir: os.PathLike,
    metadata_path: os.PathLike,
) -> List[Dict]:
    """Create a list of dataset entries driven by metadata.

    Parameters
    ----------
    image_dir      : Directory with radiograph image files.
    annotation_dir : Directory with LabelMe JSON files.
    metadata_path  : Path to the Excel sheet containing per-image metadata.

    Returns
    -------
    A list of dictionaries, one per image, with keys::

        image_id, image, bbox, box_label, labels, masks

    Images lacking an annotation file will contain empty ``bbox`` and ``label``
    lists but are still included so no negative cases are dropped.
    """

    # Load metadata and index by image ID for O(1) look‑ups
    metadata_df = pd.read_excel(metadata_path)
    metadata_df.set_index("image_id", inplace=True)

    data_entries: List[Dict] = []

    for image_id, meta_row in metadata_df.iterrows():
        meta = meta_row.to_dict()

        image_path = Path(image_dir) / image_id
        if not image_path.exists():
            # Skip if the image itself is missing – nothing to train/evaluate on
            continue

        # Expect annotation file to share the stem with the image, .json suffix
        json_path = Path(annotation_dir) / (Path(image_id).stem + ".json")

        if json_path.exists():
            bboxes, box_labels = _parse_annotation(json_path)
        else:
            bboxes, box_labels = [[0.5, 0.5, 1.0, 1.0]], ["normal"]  # keep negatives!

        data_entries.append(
            {
                "image_id": image_id,
                "image": str(image_path),
                "bbox": bboxes,
                "box_label": box_labels,
                "labels": make_class_list(meta),
                "masks": make_mask_list(meta),
            }
        )
        
    return data_entries