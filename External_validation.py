"""
External validation of Region-guided bone tumor classification on BTXRD center 2 and 3.

Pipeline:
1. Load BTXRD metadata (Excel) and images.
2. For each image:
   - Detect bone regions with a YOLO detector.
   - Crop each region + whole image.
   - Run a 38-label region-guided classifier (multi-head).
   - Aggregate per-region probabilities into per-image summary features.
3. Save enriched CSV and plot normalized confusion matrices
   for tumor presence and benign/malignant/normal subtype.

"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import PIL
import PIL.Image
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as T
from datasets import Dataset, Features, Image, Value
from matplotlib import pyplot as plt
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix
from transformers import AutoConfig, ViTForImageClassification
from ultralytics import YOLO

# --------------------------------------------------------
# Global config
# --------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = (224, 224)

def build_valid_transforms(normalize: bool = True) -> T.Compose:
    """Build validation-time image transforms."""
    if normalize:
        return T.Compose(
            [
                T.Resize(IMG_SIZE),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(IMG_SIZE),
                T.ToTensor(),
            ]
        )
    
# --------------------------------------------------------
# Confusion matrix analysis
# --------------------------------------------------------

def analyze_and_plot_confusion(
    df: pd.DataFrame,
    output_dir: str | Path,
    threshold_bonetumor: float = 0.5,
    threshold_malignant: float = 0.5,
) -> None:
    """
    Analyze predictions and plot confusion matrices.
    
    Args:
        df: DataFrame with ground-truth columns:
            - 'tumor' (0/1)
            - 'benign' (0/1)
            - 'malignant' (0/1)
            and model-derived probability or aggregated score columns:
            - 'max_class_1_2_3_sum'
            - 'max_class_1'
        output_dir: Directory where figures and CSVs will be saved.
        threshold_bonetumor: Threshold for tumor presence.
        threshold_malignant: Threshold for malignant vs benign.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

       
    # Add prediction columns
    df = df.copy()
    df['predicted tumor'] = (df['max_class_1_2_3_sum'] > threshold_bonetumor).astype(int)

    # Malignant/benign subclassification only if tumor is present
    df['predicted malignant'] = df.apply(
        lambda row: 0 if row['predicted tumor'] == 0 else int(row["max_class_1"] >= threshold_malignant),
        axis=1
    )
    df['predicted benign'] = df.apply(
        lambda row: 0 if row['predicted tumor'] == 0 else int(row["max_class_1"] < threshold_malignant),
        axis=1
    )

    # Confusion matrices
    # Binary tumor detection
    binary_cm = confusion_matrix(df['tumor'], df['predicted tumor'])

    # 3-way: normal / benign / malignant
    true_labels = df.apply(
        lambda row: 'benign' if row['benign'] == 1 else ('malignant' if row['malignant'] == 1 else 'normal'), axis=1
    )
    pred_labels = df.apply(
        lambda row: 'benign' if row['predicted benign'] == 1 else ('malignant' if row['predicted malignant'] == 1 else 'normal'), axis=1
    )
    conf_matrix_3way = pd.crosstab(true_labels, pred_labels)

    # Row-normalized matrices (per-class accuracy profiles)
    def _row_normalize(cm: np.ndarray) -> np.ndarray:
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

    binary_cm_ratio = _row_normalize(binary_cm)
    cm3_ratio = _row_normalize(conf_matrix_3way.values)


    # Plotting function
    def plot_confusion_matrix(ax, cm, labels, title):
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 12}, ax=ax, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=14, weight='bold', pad=12)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_aspect('equal')

    # Create side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_confusion_matrix(
        axes[0],
        binary_cm_ratio,
        labels=['Normal', 'Tumor'],
        title='Tumor Detection',
    )

    plot_confusion_matrix(
        axes[1],
        cm3_ratio,
        labels=conf_matrix_3way.columns,
        title='Benign / Malignant / Normal',
    )

    plt.tight_layout()

    # Save figure to PDF
    pdf_path = output_dir / "confusion_matrices_ratio.pdf"
    svg_path = output_dir / "confusion_matrices_ratio.svg"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrices (ratios) saved to: {pdf_path} and {svg_path}")

    # Save raw confusion matrices to CSV
    binary_cm_df = pd.DataFrame(binary_cm, index=['True Normal', 'True Tumor'], columns=['Pred Normal', 'Pred Tumor'])
    binary_cm_csv_path = output_dir / "binary_confusion_matrix.csv"
    binary_cm_df.to_csv(binary_cm_csv_path)

    three_way_cm_csv_path = output_dir / "three_way_confusion_matrix.csv"
    conf_matrix_3way.to_csv(three_way_cm_csv_path)

    print(f"Raw confusion matrices saved to: {binary_cm_csv_path} and {three_way_cm_csv_path}")



# --------------------------------------------------------
# BTXRD loader
# --------------------------------------------------------

def load_BTXRD_dataset(excel_path: str, image_dir: str
) -> Dataset:
        
    """
    Load BTXRD metadata and construct a Hugging Face Dataset.

    Expected columns in the Excel file include:
    - image_id
    - tumor, benign, malignant
    - subtype and anatomical labels as 0/1 indicators

    Args:
        excel_path: Path to BTXRD Excel file.
        image_dir: Directory containing the images referenced by 'image_id'.

    Returns:
        A Hugging Face Dataset with PIL images and typed columns.
    """

    # Load the metadata xlsx
    df = pd.read_excel(excel_path)

    # Convert the 'image_id' column to full file paths
    df['image'] = df['image_id'].apply(lambda x: os.path.join(image_dir, x))
    df = df.drop(columns=['image_id'])
    
    # Define dataset features 
    features = Features({
        "image": Image(),  # Load images as PIL objects
        "center": Value("int32"),
        "age": Value("int32"),
        "gender": Value("string"),
        "hand": Value("int32"),
        "ulna": Value("int32"),
        "radius": Value("int32"),
        "humerus": Value("int32"),
        "foot": Value("int32"),
        "tibia": Value("int32"),
        "fibula": Value("int32"),
        "femur": Value("int32"),
        "hip bone": Value("int32"),
        "ankle-joint": Value("int32"),
        "knee-joint": Value("int32"),
        "hip-joint": Value("int32"),
        "wrist-joint": Value("int32"),
        "elbow-joint": Value("int32"),
        "shoulder-joint": Value("int32"),
        "tumor": Value("int32"),
        "benign": Value("int32"),
        "malignant": Value("int32"),
        "osteochondroma": Value("int32"),
        "multiple osteochondromas": Value("int32"),
        "simple bone cyst": Value("int32"),
        "giant cell tumor": Value("int32"),
        "osteofibroma": Value("int32"),
        "synovial osteochondroma": Value("int32"),
        "other bt": Value("int32"),
        "osteosarcoma": Value("int32"),
        "other mt": Value("int32"),
        "upper limb": Value("int32"),
        "lower limb": Value("int32"),
        "pelvis": Value("int32"),
        "frontal": Value("int32"),
        "lateral": Value("int32"),
        "oblique": Value("int32"),
    })

    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast(features)
    return dataset

# --------------------------------------------------------
# Detection + classification helper functions
# --------------------------------------------------------

def load_yolo_model(model_path):
    """Load a YOLO model (Ultralytics) for region detection."""
    return YOLO(model_path)

def load_classifier_model(checkpoint_name: str = "google/vit-large-patch16-224", checkpoint_path: str = "./models/skelex", num_labels: int = 38, device: torch.device | str = "cpu"
) -> ViTForImageClassification:
    """
    Load a ViT-based region-guided classifier from a safetensors checkpoint.

    Args:
        checkpoint_name: SKELEX.
        checkpoint_path: Path to .safetensors file with fine-tuned weights.
        num_labels: Number of output labels (38 in this work).
        device: Torch device.
    Returns:
        A ViTForImageClassification model on the specified device.
    """

     # Model
    config = AutoConfig.from_pretrained(checkpoint_name, num_labels=num_labels)
    model = ViTForImageClassification(config)
    model.load_state_dict(load_file(checkpoint_path, device="cpu"))
    model.to(device)
    model.eval()
    return model

def predict_classes_as_tensor(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits of shape [B, 38] into probability tensor [B, 38]
    for the multi-head structure:

      - Binary:      idx 0, 37 (sigmoid per-label)
      - 4-way:       idx 1..4  (softmax over 4)
      - 29-way:      idx 5..33 (softmax over 29)
      - 3-way:       idx 34..36 (softmax over 3)
    """

    batch_size, total_labels = logits.shape
    predictions = torch.zeros((batch_size, total_labels), dtype=torch.float, device=logits.device)  # Initialize prediction tensor
    
    # Binary Labels: Index 0 and 37
    binary_logits = logits[:, [0, 37]]
    binary_predictions = torch.sigmoid(binary_logits)
    predictions[:, 0] = binary_predictions[:, 0]
    predictions[:, 37] = binary_predictions[:, 1]

    # 4-Way Classification: (1, 2, 3, 4)
    four_way_logits = logits[:, [1, 2, 3, 4]]
    four_way_predictions = torch.softmax(four_way_logits, dim=1)  
    for i, idx in enumerate([1, 2, 3, 4]):
        predictions[:, idx] = four_way_predictions[:, i]

    # 29-Way Classification: (5 to 33)
    twenty_nine_way_logits = logits[:, 5:34]
    twenty_nine_way_predictions = torch.softmax(twenty_nine_way_logits, dim=1)
    for i in range(29):  
        predictions[:, 5 + i] = twenty_nine_way_predictions[:, i]

    # 3-Way Classification: (34, 35, 36)
    three_way_logits = logits[:, [34, 35, 36]]
    three_way_predictions = torch.softmax(three_way_logits, dim=1)
    for i, idx in enumerate([34, 35, 36]):  
        predictions[:, idx] = three_way_predictions[:, i]

    return predictions


def process_dataset_sample(
    sample: Dict[str, Any],
    yolo_model: YOLO,
    classifier_model: ViTForImageClassification,
    tfms: T.Compose,
    device: torch.device | str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Run region detection + region-guided classification for a single BTXRD sample.

    Args:
        sample: A row from the HuggingFace Dataset, including 'image' (PIL.Image).
        yolo_model: anatomical region detector.
        classifier_model: Region-guided classifier.
        tfms: Torchvision transforms applied to each cropped region.
        device: Torch device used for classification.

    Returns:
        A list of dicts with keys:
         - 'bounding_box': [x1, y1, x2, y2]
         - 'detected_object': YOLO class name or "whole image"
         - 'classification': np.ndarray of shape (38,)
        or None if no bounding boxes are detected.
    """
    # Convert PIL image to OpenCV format (YOLO works with numpy arrays)
    pil_img = sample["image"]
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    image = np.array(pil_img)  # Convert PIL to numpy
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    
    # Step 1: Detect bounding boxes using YOLO
    results = yolo_model.predict(source=image_bgr, save=False, save_txt=False, device = device)
    if len(results[0].boxes.data) == 0:
        print("No bounding boxes detected.")
        return None

    det = results[0]
    boxes_data = det.boxes.data.cpu().numpy()  # [N, 6] (x1,y1,x2,y2,conf,class)
    class_ids = boxes_data[:, -1].astype(int)
    boxes = boxes_data[:, :4]
    
    class_names = [det.names[cid] for cid in class_ids]
    
    # Step 2: Crop regions based on bounding boxes
    cropped_regions = []
    final_boxes: List[List[int]] = []
    final_labels: List[str] = []

    h, w = image_bgr.shape[:2]
    for box, label in zip(boxes, class_names):
        if np.any(np.isnan(box)):
            continue
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2, :]
        cropped_regions.append(crop)
        final_boxes.append([x1, y1, x2, y2])
        final_labels.append(label)

    cropped_regions.append(image)
    final_boxes.append([0, 0, w, h])
    final_labels.append("whole image")


    # Step 3: Classify each cropped region using the classifier model
    outputs: List[Dict[str, Any]] = []
    for region, box, lbl in zip(cropped_regions, final_boxes, final_labels):
        region_pil = PIL.Image.fromarray(region)
        inputs = tfms(region_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier_model(inputs).logits
        probs = predict_classes_as_tensor(logits).cpu().numpy().flatten()

        outputs.append(
            {
                "bounding_box": box,
                "detected_object": lbl,
                "classification": probs,
            }
        )

    return outputs

def compute_max_values(outputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    excluded_labels = {"right", "left", "abnormal bone lesion"}

    max_vals = {
        "class_0": {"val": -float("inf"), "box": None, "label": None},
        "class_37": {"val": -float("inf"), "box": None, "label": None},
        "class_1": {"val": -float("inf"), "box": None, "label": None},
        "class_2": {"val": -float("inf"), "box": None, "label": None},
        "class_3": {"val": -float("inf"), "box": None, "label": None},
        "class_1_2_3_sum": {"val": -float("inf"), "box": None, "label": None},
        "class_34": {"val": -float("inf"), "box": None, "label": None},
        "class_35": {"val": -float("inf"), "box": None, "label": None},
        "class_36": {"val": -float("inf"), "box": None, "label": None},
       }
    for item in outputs:
        label = item["detected_object"].strip().lower()
        if label in excluded_labels:
            continue
    
        cls = item["classification"]
        box = item["bounding_box"]
        det_label = item["detected_object"]
    
        if cls[0] > max_vals["class_0"]["val"]:
            max_vals["class_0"] = {"val": cls[0], "box": box, "label": det_label}
        
        if cls[37] > max_vals["class_37"]["val"]:
            max_vals["class_37"] = {"val": cls[37], "box": box, "label": det_label}
        
        if cls[1] > max_vals["class_1"]["val"]:
            max_vals["class_1"] = {"val": cls[1], "box": box, "label": det_label}
        
        if cls[2] > max_vals["class_2"]["val"]:
            max_vals["class_2"] = {"val": cls[2], "box": box, "label": det_label}
        
        if cls[3] > max_vals["class_3"]["val"]:
            max_vals["class_3"] = {"val": cls[3], "box": box, "label": det_label}

        if (cls[1] + cls[2] + cls[3]) > max_vals["class_1_2_3_sum"]["val"]:
            max_vals["class_1_2_3_sum"] = {
                "val": cls[1] + cls[2] + cls[3],
                "box": box,
                "label": det_label,
            }  

        if cls[34] > max_vals["class_34"]["val"]:
            max_vals["class_34"] = {"val": cls[34], "box": box, "label": det_label}
        
        if cls[35] > max_vals["class_35"]["val"]:
            max_vals["class_35"] = {"val": cls[35], "box": box, "label": det_label}

        if cls[36] > max_vals["class_36"]["val"]:
            max_vals["class_36"] = {"val": cls[36], "box": box, "label": det_label}

    return max_vals
    

def Test_BTXRD(
        excel_path: str,
        image_dir: str,
        yolo_ckpt: str,
        classifier_name: str,
        classifier_ckpt: str,
        output_dir: str,
        num_labels: int = 38,
        normalize_tfms: bool = True,
        threshold_bonetumor: float = 0.5,
        threshold_malignant: float = 0.5,
        device: str = "cpu",
) -> None:
    device_t = torch.device(device)

    # Models and transforms
    
    yolo_model = load_yolo_model(yolo_ckpt)
    classifier_model = load_classifier_model(
        checkpoint_name=classifier_name,
        checkpoint_path=classifier_ckpt,
        num_labels=num_labels,
        device=device_t,
    )
    tfms = build_valid_transforms(normalize=normalize_tfms)

    dataset = load_BTXRD_dataset(excel_path=excel_path, image_dir=image_dir)

    updated_rows: List[Dict[str, Any]] = []

    # Per-image processing
    for i, row in enumerate(dataset):
        outputs = process_dataset_sample(
            row,
            yolo_model=yolo_model,
            classifier_model=classifier_model,
            tfms=tfms,
            device=device_t,
        )
        if outputs is None or len(outputs) == 0:
            enriched_row = dict(row)
            for key in [
                "class_0",
                "class_37",
                "class_1",
                "class_2",
                "class_3",
                "class_1_2_3_sum",
                "class_34",
                "class_35",
                "class_36",
            ]:
                enriched_row[f"max_{key}"] = 0.0
                enriched_row[f"max_{key}_box"] = None
                enriched_row[f"max_{key}_label"] = None
            updated_rows.append(enriched_row)
            continue

        max_vals = compute_max_values(outputs)
        enriched_row = dict(row)
        for key, info in max_vals.items():
            enriched_row[f"max_{key}"] = float(info["val"])
            enriched_row[f"max_{key}_box"] = info["box"]
            enriched_row[f"max_{key}_label"] = info["label"]
        updated_rows.append(enriched_row)


    # 4. Save enriched CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(updated_rows)
    csv_path = output_dir / "btxrd_region_guided_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved enriched CSV to: {csv_path}")

    # 5. Confusion matrices
    analyze_and_plot_confusion(
        df,
        output_dir=output_dir,
        threshold_bonetumor=threshold_bonetumor,
        threshold_malignant=threshold_malignant,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Region-guided bone tumor evaluation on BTXRD."
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to BTXRD Excel file with metadata and labels.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing BTXRD images (referenced by image_id).",
    )
    parser.add_argument(
        "--yolo_ckpt",
        type=str,
        required=True,
        help="Path to YOLO detector checkpoint (.pt).",
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default="google/vit-large-patch16-224",
        help="Hugging Face model ID for the classification backbone.",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        required=True,
        help="Path to classifier weights (.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save enriched CSV and confusion matrices.",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=38,
        help="Number of output labels of the classifier.",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable ImageNet normalization in transforms.",
    )
    parser.add_argument(
        "--threshold_bonetumor",
        type=float,
        default=0.5,
        help="Threshold for tumor presence decision.",
    )
    parser.add_argument(
        "--threshold_malignant",
        type=float,
        default=0.5,
        help="Threshold for malignant vs benign among tumor cases.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    return parser.parse_args()






def main():
    args = parse_args()
    Test_BTXRD(
        excel_path=args.excel_path,
        image_dir=args.image_dir,
        yolo_ckpt=args.yolo_ckpt,
        classifier_name=args.classifier_name,
        classifier_ckpt=args.classifier_ckpt,
        output_dir=args.output_dir,
        num_labels=args.num_labels,
        normalize_tfms=not args.no_normalize,
        threshold_bonetumor=args.threshold_bonetumor,
        threshold_malignant=args.threshold_malignant,
        device=args.device,
    )
   

if __name__ == "__main__":
    main()