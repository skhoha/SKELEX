from __future__ import annotations

"""

Utility for loading FRACATLAS dataset using metadata as the primary source.

Usage
-----
>>> data = load_fracatlas_dataset(
...     image_dir="/path/to/images",
...     annotation_dir="/path/to/bounding_box_YOLO_annotations",
...     metadata_path="/path/to/metadata.csv",
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


def load_fracatlas_dataset(
    image_dir: os.PathLike, 
    annotation_dir: os.PathLike,
    metadata_path: os.PathLike,
) -> List[Dict]:
    """Create a list of dataset entries driven by metadata.

    Parameters
    ----------
    image_dir      : Directory with radiograph image files. fractured_dir = image_dir/Fractured, no_fracture_dir = image_dir/no_fracture_dir
    annotation_dir : Directory with YOLO txt files
    metadata_path  : Path to the csv file containing per-image metadata.

    Returns
    -------
    A list of dictionaries, one per image, with keys
        image_id, image, bbox, box_label, labels, masks

    Images lacking an annotation file will contain empty bbox and label
    lists are still included so no negative cases are dropped.
    """

    import os
    import pandas as pd
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    fractured_dir = os.path.join(image_dir, "Fractured")
    no_fracture_dir = os.path.join(image_dir, "no_fracture")
    meta_df = pd.read_csv(metadata_path)
    
    def find_image_path(image_id):
        for folder in [fractured_dir, no_fracture_dir]:
            path = os.path.join(folder, image_id)
            if os.path.exists(path):
                return path
        return None

    data_entries = []

    for idx, row in meta_df.iterrows():
        image_id = row["image_id"]
        image_path = find_image_path(image_id)
        image_basename = os.path.splitext(image_id)[0]
        label_path = os.path.join(annotation_dir, f"{image_basename}.txt")

        if image_path and os.path.exists(label_path):
            boxes, classes, box_classes, masks = [], [], [], []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append([cx, cy, w, h])
                    box_classes.append(["fractured"])
            if not boxes:
                boxes.append([0.5, 0.5, 1.0, 1.0])
                box_classes.append("normal")

            h = int(row["hardware"])
            f = int(row["fractured"])

            if h == 1 and f == 1:
                classes.extend([37, 35, 0])
            elif h == 0 and f == 1:
                classes.extend([35, 0])
            elif h == 1 and f == 0:
                classes.extend([36, 37, 0])
            elif h == 0 and f == 0:
                classes.append(36)
            else:
                print(row["image_id"], h, f, "error")
            classes.append(4)

            if f == 1 or h == 1 :
                # 5‒33 inclusive
                masks.extend(range(5, 34))
            else :
                masks.append(0)  # abnormality label is masked, since some of the x-ray looks abnormal
                masks.extend(range(5, 34))  # 5‒33 inclusive

            data_entries.append({
                "image_id": image_id,
                "image": str(image_path),
                "bbox": boxes,
                "box_label": box_classes,
                "labels": classes,
                "masks": masks,
            })
    return data_entries
