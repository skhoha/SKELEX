"""
Region-guided multi-head classifier training script.

This script reproduces the multi-headed bone tumor classifier training used in the SKELEX, 
using a combined dataset of SNUH-bonetu, BTXRD, FracAtlas.

"""


import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    auc,
    confusion_matrix,
)
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import ViTForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup

# Import from your package
from skelex.data.btxrd_loader import load_btxrd_dataset
from skelex.data.fracatlas_loader import load_fracatlas_dataset
from skelex.data.snuh_bonetu_loader import load_snuhbt_dataset
from skelex.data.dataset import CombinedDataset
from skelex.utils.transform_utils import (
    RandomIoRCrop,
    ToTensor,
    Normalize,
    Compose,
    Resize,
    RandomRotation,
    ColorJitter,
    ToMultiHot,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train region-guided multi-headed bone tumor classifier."
    )

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI args override config values.",
    )

    # Dataset paths
    parser.add_argument("--btxrd_image_dir", type=str, required=True)
    parser.add_argument("--btxrd_annotation_dir", type=str, required=True)
    parser.add_argument("--btxrd_metadata", type=str, required=True)

    parser.add_argument("--fracatlas_image_dir", type=str, required=True)
    parser.add_argument("--fracatlas_annotation_dir", type=str, required=True)
    parser.add_argument("--fracatlas_metadata", type=str, required=True)

    parser.add_argument("--snuh_bonetu_image_dir", type=str, required=True)
    parser.add_argument("--snuh_bonetu_metadata", type=str, required=True)


    # Class definition file
    parser.add_argument(
        "--class_file",
        type=str,
        required=True,
        help="Text file with lines of the form 'index: class_name'.",
    )

    # Model / training hyperparameters
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="facebook/vit-mae-large",
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)

    # Splits / CV
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.10,
        help="Fraction of data reserved as held-out test set.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/region_guided",
        help="Root directory for splits, logs, metrics and checkpoints.",
    )

    # Dataloader workers
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # If config is provided, load it and use it as defaults
    if args.config is not None:
        config_path = Path(args.config)
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # For each key in config, if the corresponding arg is still None or default,
        # fill it from the config. CLI always wins.
        for k, v in cfg.items():
            if not hasattr(args, k):
                continue
            current = getattr(args, k)
            # If current is None OR equal to the parser default, override with config
            # (simple heuristic; good enough here)
            if current is None or (isinstance(current, (int, float, str)) and current == parser.get_default(k)):
                setattr(args, k, v)

    # Final sanity checks for required fields when using config
    required_keys = [
        "btxrd_image_dir",
        "btxrd_annotation_dir",
        "btxrd_metadata",
        "fracatlas_image_dir",
        "fracatlas_annotation_dir",
        "fracatlas_metadata",
        "snuh_bonetu_image_dir",
        "snuh_bonetu_metadata",
        "class_file",
    ]
    missing = [k for k in required_keys if getattr(args, k) is None]
    if missing:
        raise ValueError(
            f"Missing required arguments (provide via --config or CLI): {missing}"
        )

    return args

# --------------------------------------------------------
# Helpers / utilities
# --------------------------------------------------------

def load_class_names(class_file: str):
    """
    Load class names from a text file with lines of the form 'id: class_name'.
    """
    mapping: Dict[int, str] = {}
    with open(class_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(": ", 1)
            key = key.strip()
            idx = int(key)
            value = value.strip()

            mapping[idx] = value

    num_classes = max(mapping.keys()) + 1
    missing = [i for i in range(num_classes) if i not in mapping]
    if missing:
        raise ValueError(f"{class_file} missing class ids: {missing}")

    return [mapping[i] for i in range(num_classes)]


def make_transforms(num_classes: int, img_size: int):
    img_size_tuple = (img_size, img_size)
    train_tsfm = Compose([
        RandomIoRCrop(scale_range=(0.2, 1.0), min_ior=0.6),
        RandomRotation(10),
        Resize(img_size_tuple),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToMultiHot(num_classes),
    ])
    valid_tsfm = Compose([
        Resize(img_size_tuple),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToMultiHot(num_classes),
    ])
    return train_tsfm, valid_tsfm

def subset_entries(entries, indices_tensor):
    idx_list = indices_tensor.tolist() if isinstance(indices_tensor, torch.Tensor) else list(indices_tensor)
    return [entries[i] for i in idx_list]


def _to_jsonable(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collate_fn(batch):
    return {
        "image_id": [b["image_id"] for b in batch],
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]).float(),
        "masks": torch.stack([x["masks"] for x in batch]).float(),
    }


def param_count(model):
    params = [(p.numel(), p.requires_grad) for p in model.parameters()]
    trainable = sum(n for n, is_trainable in params if is_trainable)
    total = sum(count for count, _ in params)
    frac = (trainable / total) * 100
    return total, trainable, frac

def logits_to_probs_multhead(logits):
    probs = torch.zeros_like(logits).float()
    probs[:, [0, 37]] = torch.sigmoid(logits[:, [0, 37]])
    probs[:, 1:5]     = torch.softmax(logits[:, 1:5], dim=1)
    probs[:, 5:34]    = torch.softmax(logits[:, 5:34], dim=1)
    probs[:, 34:37]   = torch.softmax(logits[:, 34:37], dim=1)
    return probs

# --------------------------------------------------------
# Loss and metrics
# --------------------------------------------------------

class MultiLabelLoss(nn.Module):
    """
    Custom multi-task loss:
      - Binary: classes [0, 37]
      - 4-way:  classes [1, 2, 3, 4]
      - 29-way: classes [5..33]
      - 3-way:  classes [34, 35, 36]
    Uses masks to exclude unknown labels.
    Logically multi-head, but implemented as a single shared output tensor partitioned into task-specific channel groups.
    """
    def __init__(self, weights=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        self.weights = weights or {
            "binary": 1.0,
            "4way": 1.0,
            "29way": 1.0,
            "3way": 1.0,
        }

    def forward(self, predictions, targets, masks):
        """
        predictions, targets, masks : [B, C] where C=38
        """
        loss_total = 0.0

        # Binary (abnormal + implant) -> indices [0, 37]
        binary_predictions = torch.cat([predictions[:, [0]], predictions[:, [37]]], dim=1)
        binary_targets = torch.cat([targets[:, [0]], targets[:, [37]]], dim=1)
        binary_mask = torch.cat([masks[:, [0]], masks[:, [37]]], dim=1)

        binary_loss = self.bce_loss(binary_predictions, binary_targets)
        binary_loss = binary_loss * (1 - binary_mask)  # mask out 1s
        binary_loss = binary_loss.sum() / ((1 - binary_mask).sum() + 1e-6)
        loss_total += self.weights["binary"] * binary_loss

        # 4-way classification (1, 2, 3, 4)
        idx_4way = (1 - masks[:, [1, 2, 3, 4]]).sum(dim=1) == 4
        if idx_4way.any():
            four_way_predictions = predictions[idx_4way][:, [1, 2, 3, 4]]
            four_way_targets = torch.argmax(targets[idx_4way][:, [1, 2, 3, 4]], dim=1)
            four_way_loss = self.ce_loss(four_way_predictions, four_way_targets).mean()
            loss_total += self.weights["4way"] * four_way_loss

        # 29-way classification (5..33)
        idx_29way = (1 - masks[:, 5:34]).sum(dim=1) == 29
        if idx_29way.any():
            twenty_nine_way_predictions = predictions[idx_29way][:, 5:34]
            twenty_nine_way_targets = torch.argmax(targets[idx_29way][:, 5:34], dim=1)
            twenty_nine_way_loss = self.ce_loss(twenty_nine_way_predictions, twenty_nine_way_targets).mean()
            loss_total += self.weights["29way"] * twenty_nine_way_loss

        # 3-way classification (34, 35, 36)
        idx_3way = (1 - masks[:, [34, 35, 36]]).sum(dim=1) == 3
        if idx_3way.any():
            three_way_predictions = predictions[idx_3way][:, [34, 35, 36]]
            three_way_targets = torch.argmax(targets[idx_3way][:, [34, 35, 36]], dim=1)
            three_way_loss = self.ce_loss(three_way_predictions, three_way_targets).mean()
            loss_total += self.weights["3way"] * three_way_loss

        return loss_total


def _flatten_unmasked(y_true, y_prob, y_mask):
    """Return flattened y_true, y_prob where mask == 0 (known)."""
    keep = (y_mask == 0)
    return y_true[keep], y_prob[keep]


def _group_confusion_mtx_mc(y_true, y_prob, y_mask, idxs0):
    """
    Multiclass group confusion matrix using one-hot y_true and argmax y_prob.
    idxs0: iterable of class IDs (e.g., [1,2,3,4])

    Returns (cm, used_count) where cm shape is [K,K] with K=len(idxs0).
    """
    idxs0 = np.asarray(idxs0, dtype=int)  # kept same as original
    # require: unmasked for all group columns
    unmasked = (y_mask[:, idxs0] == 0).all(axis=1)
    # require: exactly one positive in y_true within the group
    one_hot = (y_true[:, idxs0].sum(axis=1) == 1)

    keep = unmasked & one_hot
    if not np.any(keep):
        K = len(idxs0)
        return np.zeros((K, K), dtype=int), 0

    yt_group = y_true[keep][:, idxs0]
    yp_group = y_prob[keep][:, idxs0]

    y_true_idx = yt_group.argmax(axis=1)
    y_pred_idx = yp_group.argmax(axis=1)

    K = len(idxs0)
    labels = np.arange(K)
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels)
    return cm, keep.sum()


def _group_confusion_mtx_binary(y_true, y_prob, y_mask, idx0, threshold=0.5):
    """
    Binary confusion matrix for a single class ID.
    Returns (cm, used_count) where cm shape is [2,2] (rows=true [0,1], cols=pred [0,1]).
    """
    idx0 = int(idx0)
    unmasked = (y_mask[:, idx0] == 0)
    if not np.any(unmasked):
        return np.zeros((2, 2), dtype=int), 0

    yt = y_true[unmasked, idx0].astype(np.uint8)
    yp = (y_prob[unmasked, idx0] >= threshold).astype(np.uint8)

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return cm, unmasked.sum()


def compute_masked_metrics(y_true, y_prob, y_mask, threshold=0.5):
    """
    y_true, y_prob, y_mask: numpy arrays [N, C]
      mask==1 means 'unknown' -> exclude from metrics

    Returns:
      dict with per-label and aggregate metrics
      grouped-task confusion matrices using class IDs:
        - '4_way'  (classes 1-4)
        - '29_way' (classes 5-33)
        - '3_way'  (classes 34-36)
        - 'binary' (class 37, threshold 0.5)
    """
    C = y_true.shape[1]
    per_label = {
        "auroc": [np.nan] * C,
        "auprc": [np.nan] * C,
        "f1": [np.nan] * C,
        "prec": [np.nan] * C,
        "recall": [np.nan] * C,
        "bacc": [np.nan] * C,
        "support_pos": [0] * C,
        "support_total": [0] * C,
    }

    # per-label
    for c in range(C):
        m = (y_mask[:, c] == 0)
        yt = y_true[m, c]
        yp = y_prob[m, c]
        if len(yt) == 0:
            continue
        pos = int(yt.sum())
        neg = len(yt) - pos
        per_label["support_pos"][c] = pos
        per_label["support_total"][c] = len(yt)

        # AUROC
        if pos > 0 and neg > 0:
            try:
                per_label["auroc"][c] = roc_auc_score(yt, yp)
            except Exception:
                per_label["auroc"][c] = np.nan
        # AUPRC
        if pos > 0:
            try:
                per_label["auprc"][c] = average_precision_score(yt, yp)
            except Exception:
                per_label["auprc"][c] = np.nan

        # thresholded metrics
        yhat = (yp >= threshold).astype(np.uint8)
        if pos > 0 and neg > 0:
            per_label["f1"][c] = f1_score(yt, yhat, zero_division=0)
            per_label["prec"][c] = precision_score(yt, yhat, zero_division=0)
            per_label["recall"][c] = recall_score(yt, yhat, zero_division=0)
            per_label["bacc"][c] = balanced_accuracy_score(yt, yhat)

    # micro (flatten all unmasked)
    y_true_flat, y_prob_flat = _flatten_unmasked(y_true, y_prob, y_mask)
    if (y_true_flat.sum() > 0) and (len(y_true_flat) - y_true_flat.sum() > 0):
        micro_auroc = roc_auc_score(y_true_flat, y_prob_flat)
    else:
        micro_auroc = np.nan
    micro_auprc = average_precision_score(y_true_flat, y_prob_flat) if y_true_flat.sum() > 0 else np.nan

    # macro
    macro_auroc = np.nanmean(per_label["auroc"])
    macro_auprc = np.nanmean(per_label["auprc"])
    macro_f1 = np.nanmean(per_label["f1"])

    # F1/precision/recall/bacc micro over flattened (thresholded)
    y_pred_flat = (y_prob_flat >= threshold).astype(np.uint8)
    if (y_true_flat.sum() > 0) and (len(y_true_flat) - y_true_flat.sum() > 0):
        micro_f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        micro_prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        micro_recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        micro_bacc = balanced_accuracy_score(y_true_flat, y_pred_flat)
    else:
        micro_f1 = micro_prec = micro_recall = micro_bacc = np.nan

    # Grouped confusion matrices
    grp_4 = list(range(1, 5))     # 1-4
    grp_29 = list(range(5, 34))   # 5-33
    grp_3 = list(range(34, 37))   # 34-36
    idx_bin = 37                  # 37

    cm_4, n4 = _group_confusion_mtx_mc(y_true, y_prob, y_mask, grp_4)
    cm_29, n29 = _group_confusion_mtx_mc(y_true, y_prob, y_mask, grp_29)
    cm_3, n3 = _group_confusion_mtx_mc(y_true, y_prob, y_mask, grp_3)
    cm_bin, nb = _group_confusion_mtx_binary(y_true, y_prob, y_mask, idx_bin, threshold=threshold)

    return {
        "per_label": per_label,
        "micro": {
            "auroc": micro_auroc,
            "auprc": micro_auprc,
            "f1": micro_f1,
            "precision": micro_prec,
            "recall": micro_recall,
            "bacc": micro_bacc,
        },
        "macro": {
            "auroc": macro_auroc,
            "auprc": macro_auprc,
            "f1": macro_f1,
        },
        "confusion_matrices": {
            "4_way": {"matrix": cm_4, "labels": grp_4, "num_samples": n4},
            "29_way": {"matrix": cm_29, "labels": grp_29, "num_samples": n29},
            "3_way": {"matrix": cm_3, "labels": grp_3, "num_samples": n3},
            "binary": {"matrix": cm_bin, "labels": [0, 1], "num_samples": nb},
        },
    }


def save_roc_curve(fpr, tpr, roc_auc, epoch, label_name, phase, out_root: Path):
    """
    Save the ROC curve plot for a specific label.
    """
    directory = out_root / phase / f"epoch_{epoch}"
    directory.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {label_name} (Epoch {epoch})")
    plt.legend(loc="lower right")

    plt.savefig(directory / f"roc_curve_{label_name}.png", dpi=300)
    plt.close()



def save_all_roc_curves_masked(y_true, y_prob, y_mask, epoch, phase, class_names, out_root: Path):
    """
    Save ROC curves for all labels with at least one positive and one negative
    unmasked sample.
    """
    C = y_true.shape[1]
    for i in range(C):
        m = (y_mask[:, i] == 0)
        yt = y_true[m, i]
        yp = y_prob[m, i]
        if len(yt) == 0 or yt.sum() == 0 or (len(yt) - yt.sum()) == 0:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        save_roc_curve(fpr, tpr, roc_auc, epoch, class_names[i], phase, out_root)


# --------------------------------------------------------
# Training loop
# --------------------------------------------------------

def train(
    model_name: str,
    pretrained_path: str,
    train_dataset,
    val_dataset,
    test_dataset,
    class_names,
    run_root: Path,
    batch_size: int = 16,
    epochs: int = 1,
    lr: float = 2e-4,
    num_workers: int = 4,
):
    """
    Main training loop with Accelerate support.
    """
    accelerator = Accelerator()
    safe_model_name = model_name.replace("/", "_")
    run_root.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    valid_dl = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    model = ViTForImageClassification.from_pretrained(
        pretrained_path,
        num_labels=len(class_names),
    ).to(accelerator.device)

    total, trainable, frac = param_count(model)
    accelerator.print(f"{total = :,} | {trainable = :,} | {frac:.2f}% trainable")

    # Loss, optimizer, scheduler
    loss_fn = MultiLabelLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    num_training_steps = epochs * len(train_dl)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, scheduler, train_dl, valid_dl, test_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, valid_dl, test_dl
    )

    roc_root = run_root / "roc_curves"
    metrics_root = roc_root  # per-label metrics CSVs
    ckpt_root = run_root / "checkpoints"

    # -------------- EPOCH LOOP --------------
    for epoch in range(1, epochs + 1):
        # -------------- TRAIN --------------
        model.train()
        running_loss = 0.0
        all_probs = []
        all_labels = []
        all_masks = []

        for batch in train_dl:
            out = model(batch["pixel_values"])
            logits = out.logits

            loss = loss_fn(logits, batch["labels"], batch["masks"])
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            probs = logits_to_probs_multhead(logits)
            probs, labels, masks = accelerator.gather_for_metrics(
                (probs.detach(), batch["labels"], batch["masks"])
            )
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(masks.cpu())

        train_loss = running_loss / len(train_dl)
        y_prob = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_mask = torch.cat(all_masks).numpy()

        train_metrics = compute_masked_metrics(y_true, y_prob, y_mask, threshold=0.5)

        accelerator.print(
            f"\nEpoch {epoch}"
            f"\nTrain loss: {train_loss:.4f} | "
            f"AUROC micro/macro: {train_metrics['micro']['auroc']:.3f} / {train_metrics['macro']['auroc']:.3f} | "
            f"AUPRC micro/macro: {train_metrics['micro']['auprc']:.3f} / {train_metrics['macro']['auprc']:.3f}"
        )
        if accelerator.is_main_process:
            save_all_roc_curves_masked(
                y_true,
                y_prob,
                y_mask,
                epoch,
                "train",
                class_names,
                roc_root,
            )

        # -------------- VALID --------------
        model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []
        all_masks = []

        for batch in valid_dl:
            with torch.no_grad():
                out = model(batch["pixel_values"])
                logits = out.logits

            loss = loss_fn(logits, batch["labels"], batch["masks"])
            running_loss += loss.item()

            probs = logits_to_probs_multhead(logits)
            probs, labels, masks = accelerator.gather_for_metrics(
                (probs.detach(), batch["labels"], batch["masks"])
            )
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(masks.cpu())

        valid_loss = running_loss / len(valid_dl)
        y_prob = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_labels).numpy()
        y_mask = torch.cat(all_masks).numpy()

        valid_metrics = compute_masked_metrics(y_true, y_prob, y_mask, threshold=0.5)
        accelerator.print(
            f"Valid loss: {valid_loss:.4f} | "
            f"AUROC micro/macro: {valid_metrics['micro']['auroc']:.3f} / {valid_metrics['macro']['auroc']:.3f} | "
            f"AUPRC micro/macro: {valid_metrics['micro']['auprc']:.3f} / {valid_metrics['macro']['auprc']:.3f}"
        )

        if accelerator.is_main_process:
            # ROC curves
            save_all_roc_curves_masked(
                y_true,
                y_prob,
                y_mask,
                epoch,
                "valid",
                class_names,
                roc_root,
            )

            # per-label metrics CSV
            rows = []
            for i, cname in enumerate(class_names):
                rows.append(
                    {
                        "label": cname,
                        "auroc": valid_metrics["per_label"]["auroc"][i],
                        "auprc": valid_metrics["per_label"]["auprc"][i],
                        "f1": valid_metrics["per_label"]["f1"][i],
                        "precision": valid_metrics["per_label"]["prec"][i],
                        "recall": valid_metrics["per_label"]["recall"][i],
                        "bacc": valid_metrics["per_label"]["bacc"][i],
                        "support_pos": valid_metrics["per_label"]["support_pos"][i],
                        "support_total": valid_metrics["per_label"]["support_total"][i],
                    }
                )
            rows.append({"label": "micro_avg_auroc", "auroc": valid_metrics["micro"]["auroc"]})
            rows.append({"label": "micro_avg_auprc", "auprc": valid_metrics["micro"]["auprc"]})
            rows.append({"label": "macro_avg_auroc", "auroc": valid_metrics["macro"]["auroc"]})
            rows.append({"label": "macro_avg_auprc", "auprc": valid_metrics["macro"]["auprc"]})

            metrics_root.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_csv(metrics_root / "per_label_metrics_valid.csv", index=False)

            # checkpoint
            ckpt_dir = ckpt_root / f"{safe_model_name}_epoch{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_model(model, ckpt_dir, max_shard_size="2GB")

        # -------------- TEST --------------
        model.eval()
        all_probs, all_labels, all_masks, all_ids = [], [], [], []

        for batch in test_dl:
            with torch.no_grad():
                out = model(batch["pixel_values"])
                logits = out.logits

            probs = logits_to_probs_multhead(logits)
            probs, labels, masks = accelerator.gather_for_metrics(
                (probs.detach(), batch["labels"], batch["masks"])
            )

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(masks.cpu())

            g_ids = gather_object(list(batch["image_id"]))
            all_ids.extend(g_ids)

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_mask = torch.cat(all_masks).numpy()

    num_classes = y_prob.shape[1]
    prob_df = pd.DataFrame(y_prob, columns=[f"prob_class{i}" for i in range(num_classes)])
    true_df = pd.DataFrame(y_true, columns=[f"true_class{i}" for i in range(num_classes)])
    mask_df = pd.DataFrame(y_mask, columns=[f"mask_class{i}" for i in range(num_classes)])
    id_df = pd.DataFrame({"id": all_ids})

    test_root = roc_root / "test"
    test_root.mkdir(parents=True, exist_ok=True)

    df_all = pd.concat([id_df, true_df, mask_df, prob_df], axis=1)
    df_all.to_csv(test_root / "predictions.csv", index=False)

    test_metrics = compute_masked_metrics(y_true, y_prob, y_mask, threshold=0.5)

    accelerator.print(
        f"\nTEST AUROC micro/macro: {test_metrics['micro']['auroc']:.3f} / {test_metrics['macro']['auroc']:.3f}\n"
        f"TEST AUPRC micro/macro: {test_metrics['micro']['auprc']:.3f} / {test_metrics['macro']['auprc']:.3f}\n"
        f"TEST F1 micro/macro   : {test_metrics['micro']['f1']:.3f} / {test_metrics['macro']['f1']:.3f}"
    )

    if accelerator.is_main_process:
        # ROC curves for test
        save_all_roc_curves_masked(
            y_true,
            y_prob,
            y_mask,
            epoch="final",
            phase="test",
            class_names=class_names,
            out_root=roc_root,
        )

        # per-label test metrics
        rows = []
        for i, cname in enumerate(class_names):
            rows.append(
                {
                    "label": cname,
                    "auroc": test_metrics["per_label"]["auroc"][i],
                    "auprc": test_metrics["per_label"]["auprc"][i],
                    "f1": test_metrics["per_label"]["f1"][i],
                    "precision": test_metrics["per_label"]["prec"][i],
                    "recall": test_metrics["per_label"]["recall"][i],
                    "bacc": test_metrics["per_label"]["bacc"][i],
                    "support_pos": test_metrics["per_label"]["support_pos"][i],
                    "support_total": test_metrics["per_label"]["support_total"][i],
                }
            )
        rows.append({"label": "micro_avg_auroc", "auroc": test_metrics["micro"]["auroc"]})
        rows.append({"label": "micro_avg_auprc", "auprc": test_metrics["micro"]["auprc"]})
        rows.append({"label": "macro_avg_auroc", "auroc": test_metrics["macro"]["auroc"]})
        rows.append({"label": "macro_avg_auprc", "auprc": test_metrics["macro"]["auprc"]})

        df = pd.DataFrame(rows)
        df.to_csv(test_root / "per_label_metrics_test.csv", index=False)

        # confusion matrices
        conf_root = test_root / "confusion_matrices"
        conf_root.mkdir(parents=True, exist_ok=True)

        cms = test_metrics["confusion_matrices"]

        def _names_from_ids(ids, class_names_):
            return [class_names_[i] for i in ids]

        group_label_names = {
            "4_way": _names_from_ids(cms["4_way"]["labels"], class_names),
            "29_way": _names_from_ids(cms["29_way"]["labels"], class_names),
            "3_way": _names_from_ids(cms["3_way"]["labels"], class_names),
            "binary": ["neg", f"pos:{class_names[37]}"],
        }

        def _save_cm(name, matrix, rowcol_names, num_samples):
            df_cm = pd.DataFrame(matrix, index=rowcol_names, columns=rowcol_names)
            df_cm.to_csv(conf_root / f"{name}.csv", index=True)

            np.save(conf_root / f"{name}.npy", matrix)

            meta = {
                "group": name,
                "labels": rowcol_names,
                "num_samples_used": int(num_samples),
                "shape": list(matrix.shape),
            }
            with open(conf_root / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            fig, ax = plt.subplots(
                figsize=(
                    max(4, len(rowcol_names) * 0.4),
                    max(3.5, len(rowcol_names) * 0.4),
                )
            )
            im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("count", rotation=90, va="center")

            ax.set_xticks(np.arange(len(rowcol_names)))
            ax.set_yticks(np.arange(len(rowcol_names)))
            ax.set_xticklabels(rowcol_names, rotation=90)
            ax.set_yticklabels(rowcol_names)

            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{name} confusion matrix (N={num_samples})")

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=7)

            plt.tight_layout()
            plt.savefig(conf_root / f"{name}.png", dpi=300)
            plt.close(fig)

        _save_cm("4_way", cms["4_way"]["matrix"], group_label_names["4_way"], cms["4_way"]["num_samples"])
        _save_cm("29_way", cms["29_way"]["matrix"], group_label_names["29_way"], cms["29_way"]["num_samples"])
        _save_cm("3_way", cms["3_way"]["matrix"], group_label_names["3_way"], cms["3_way"]["num_samples"])
        _save_cm("binary", cms["binary"]["matrix"], group_label_names["binary"], cms["binary"]["num_samples"])

# --------------------------------------------------------
# Main: build splits, run CV, call train()
# --------------------------------------------------------

def main():
    args = parse_args()

    # Output root for this run (seed + timestamp)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"seed{args.split_seed}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load classes and transforms
    class_names = load_class_names(args.class_file)
    assert len(class_names) == args.num_classes, (
        f"num_classes={args.num_classes} but class_file has {len(class_names)} entries."
    )
    train_tsfm, valid_tsfm = make_transforms(args.num_classes, args.img_size)

    # Load datasets
    btxrd_entries = load_btxrd_dataset(
        image_dir=args.btxrd_image_dir,
        annotation_dir=args.btxrd_annotation_dir,
        metadata_path=args.btxrd_metadata,
    )

    fracatlas_entries = load_fracatlas_dataset(
        image_dir=args.fracatlas_image_dir,
        annotation_dir=args.fracatlas_annotation_dir,
        metadata_path=args.fracatlas_metadata,
    )

    snuh_bonetu_entries = load_snuhbt_dataset(
        image_dir=args.snuh_bonetu_image_dir,
        metadata_path=args.snuh_bonetu_metadata,
    )


    # Merge all entries
    all_entries = (
        btxrd_entries
        + fracatlas_entries
        + snuh_bonetu_entries
    )

    full_dataset = CombinedDataset(entries=all_entries, transform=None)
    N = len(full_dataset)

    # Create global permutation
    idxs = torch.randperm(N, generator=torch.Generator().manual_seed(args.split_seed))

    # Fixed test split
    n_test = max(1, int(round(args.test_fraction * N)))
    test_idx = idxs[:n_test]
    remain_idx = idxs[n_test:]  # CV pool

    test_dataset = CombinedDataset(
        subset_entries(all_entries, test_idx),
        transform=valid_tsfm,
    )

    # Save top-level summary
    summary = {
        "seed": args.split_seed,
        "created_at": stamp,
        "N_total": N,
        "n_test": int(test_idx.numel()),
        "n_cv_pool": int(remain_idx.numel()),
        "cv_folds": args.n_splits,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Fixed test snapshot
    test_recs = [
        {"global_index": int(i), **_to_jsonable(all_entries[int(i)])}
        for i in test_idx.tolist()
    ]
    _write_jsonl(out_root / "test.jsonl", test_recs)

    # K-fold CV over remaining indices
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.split_seed)
    remain_idx_np = remain_idx.numpy()

    for fold_id, (tr_rel, va_rel) in enumerate(kfold.split(remain_idx_np), start=1):
        tr_abs = remain_idx_np[tr_rel]
        va_abs = remain_idx_np[va_rel]

        fold_dir = out_root / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save splits indices
        splits_meta = {
            "fold": fold_id,
            "train_idx": [int(i) for i in tr_abs.tolist()],
            "val_idx": [int(i) for i in va_abs.tolist()],
            "test_idx": [int(i) for i in test_idx.tolist()],
        }
        (fold_dir / "splits.json").write_text(json.dumps(splits_meta, indent=2), encoding="utf-8")

        # Save rich entries
        train_recs = [
            {"global_index": int(i), **_to_jsonable(all_entries[int(i)])}
            for i in tr_abs.tolist()
        ]
        val_recs = [
            {"global_index": int(i), **_to_jsonable(all_entries[int(i)])}
            for i in va_abs.tolist()
        ]
        _write_jsonl(fold_dir / "train.jsonl", train_recs)
        _write_jsonl(fold_dir / "val.jsonl", val_recs)

        # Build datasets
        train_dataset = CombinedDataset(
            subset_entries(all_entries, tr_abs),
            transform=train_tsfm,
        )
        val_dataset = CombinedDataset(
            subset_entries(all_entries, va_abs),
            transform=valid_tsfm,
        )

        model_name = f"facebook/vit-mae-large_fold{fold_id}"
        run_root = fold_dir / "run"

        train(
            model_name=model_name,
            pretrained_path=args.pretrained_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            class_names=class_names,
            run_root=run_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()


