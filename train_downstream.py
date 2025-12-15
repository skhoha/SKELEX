"""
Musculoskeletal X-ray tasks evaluation with multiple backbones

* Compares several backbones on the exact same splits.
* Keeps a fixed stratified train/valid/test split per random-seed. 
* No patient leakage to test split if patient_id is present in the dataset.
* Within each fold, stores the checkpoint that achieves the best
  validation metric (AUROC for classification, MAE for regression)
  and evaluates it once on the held-out test set.
* Produces three artefacts per backbone:
     *_folds.csv  - per-fold metrics on val & test
     *_summary.json - mean ± SD across folds (val & test)
     ckpts/...pt - the model state dict with the overall best val score

Supported datasets (via --data):
  pesplanus, boneage, mura, fracatlas, kneeoa,
  pediatricfx, pediatricfx_implant, pediatricfx_ao,
  fracatlas_implant, btxrd, btxrd_mb, btxrd_subtype
  
Usage example
-------------

python train_downstream.py \
    --data fracatlas \
    --backbones skelex vit-i21k resnet-101 \
    --cv 5 \
    --epochs 50 \
    --seed 2025 \
    --test_split 0.1 \
    --bs 64 \
    --lr 5e-5 \
    --outdir ./results
"""

import argparse
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.utils import save_image

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize

from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    ViTForImageClassification,
)

# --------------------------------------------------------
# Global config
# --------------------------------------------------------

IMNET_MEAN, IMNET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMNET_MEAN_T = torch.tensor(IMNET_MEAN).view(1, 3, 1, 1)  # [1,C,1,1]
_IMNET_STD_T  = torch.tensor(IMNET_STD).view(1, 3, 1, 1)

# --------------------------------------------------------
# Utilities
# --------------------------------------------------------

def denorm(x: torch.Tensor) -> torch.Tensor:
    """
    Inverts torchvision Normalize(IMNET_MEAN, IMNET_STD).
    Accepts [C,H,W] or [B,C,H,W]. Returns float in [0,1].
    """

    if x.dim() == 3:                       # [C,H,W] -> [1,C,H,W]
        x = x.unsqueeze(0)
        squeeze_back = True
    elif x.dim() == 4:                     # [B,C,H,W]
        squeeze_back = False
    else:
        raise ValueError(f"Expected 3D/4D tensor, got {x.shape}")

    mean = _IMNET_MEAN_T.to(x.device, dtype=x.dtype)
    std  = _IMNET_STD_T.to(x.device, dtype=x.dtype)
    y = (x * std) + mean
    y = y.clamp_(0, 1)
    return y.squeeze(0) if squeeze_back else y


def save_train_preview(tr_ds, out_dir: Path, n:int=5):
    """
    Saves first n augmented training images for logging the training (after current train transforms),
    de-normalized, as PNGs labeled in filename.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(tr_ds))):
        x, y = tr_ds[i][0], tr_ds[i][1]              # [C,H,W], normalized
        x_den = denorm(x)                            # back to [0,1]
        lbl = int(y.item()) if hasattr(y, "item") else int(y)
        save_image(x_den.cpu(), str(out_dir / f"{i:02d}_label-{lbl}.png"))


def probs_to_columns(problem: str, y_prob, num_classes: int):
    """
    Returns a dict of probability columns depending on task type.
    For binary: {'prob': ...}; for multiclass: {'prob_0':..., 'prob_1':...}
    """
    if problem == "regression" or y_prob is None:
        return {}
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        return {"prob": y_prob}
    cols = {}
    for c in range(num_classes):
        cols[f"prob_{c}"] = y_prob[:, c]
    return cols


def load_rgb_8bit(p: Path) -> Image.Image:
    """
    Open an image and return as 8-bit RGB after per-image min-max normalization.
    Works for 8-bit, 16-bit, and float grayscale or RGB inputs.
    """
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)  # fix orientation

    # Convert to numpy (keep original precision)
    arr = np.array(img)
    img.close()

    # If RGB(A), drop alpha
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Convert all channels to float for normalization
    arr = arr.astype(np.float32)

    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if hi <= lo or not np.isfinite([lo, hi]).all():
        arr_norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_norm = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    # If grayscale, promote to RGB
    if arr_norm.ndim == 2:
        return Image.fromarray(arr_norm).convert("RGB")
    else:
        return Image.fromarray(arr_norm)


def _filter_meta(d: Dict, cols: Optional[List[str]]):
    if cols is None: 
        return dict(d)
    return {k: d[k] for k in cols if k in d}
    
    
def collate_with_meta(batch):
    # Custom collate: keep paths and meta as simple Python lists (no tensor conversion).
    norm = []
    for item in batch:
        if len(item) == 2:
            x, y = item; p=None; m=None
        elif len(item) == 3:
            x, y, p = item; m=None
        else:
            x, y, p, m = item
        norm.append((x, y, p, m))

    xs, ys, paths, metas = zip(*norm)
    xs = default_collate(xs)
    ys = default_collate(ys)
    # keep as plain python lists (no tensor conversion / dict merging)
    paths = list(paths)
    metas = list(metas)
    return xs, ys, paths, metas


# --------------------------------------------------------
# Stratification helpers
# --------------------------------------------------------


def get_labels_for_stratification(ds, ds_name: str) -> np.ndarray:
    """
    Return integer labels for stratified splitting based on the dataset name.

    Parameters
    ----------
    ds : Dataset
        dataset instance (must expose either `.samples` or `.meta` DataFrame).
    ds_name : str
        Canonical (case-insensitive) dataset name. Supported aliases:
          - pesplanusds, kneeoads
          - pediatricwristfxds, pediatricwristfx_implant_ds, pediatricwristfx_ao_ds
          - murads
          - btxrdds, btxrd_subtype_ds, btxrd_mb_ds
          - boneageds
          - fracatlas_implant_ds, fracatlasds

    Returns np.ndarray of dtype int
    """
    key = (ds_name or "").strip().lower()

    if key in {"pesplanusds", "kneeoads"}:
        # Expect ds.samples as list of (path, label)
        return np.asarray([lbl for _, lbl in ds.samples], dtype=int)

    elif key in {"pediatricwristfxds"}:
        if not hasattr(ds, "meta") or "fracture_visible" not in ds.meta:
            raise ValueError("Expected 'fracture_visible' in ds.meta for pediatric wrist dataset.")
        return ds.meta["fracture_visible"].astype(int).to_numpy()
    
    elif key in {"pediatricwristfx_implant_ds"}:
        if not hasattr(ds, "meta") or "metal" not in ds.meta:
            raise ValueError("Expected 'metal' in ds.meta for pediatric wrist dataset.")
        return ds.meta["metal"].astype(int).to_numpy()

    elif key in {"pediatricwristfx_ao_ds"}:
        if not hasattr(ds, "meta") or "ao_class_enc" not in ds.meta:
            raise ValueError("Expected 'ao_class_enc' in ds.meta for pediatric wrist dataset.")
        return ds.meta["ao_class_enc"].astype(int).to_numpy()

    elif key in {"fracatlasds"}:
        if not hasattr(ds, "meta") or "fractured" not in ds.meta:
            raise ValueError("Expected 'fractured' in ds.meta for fracatlas fracture presence.")
        return ds.meta["fractured"].astype(int).to_numpy() 
    
    elif key in {"fracatlas_implant_ds"}:
        if not hasattr(ds, "meta") or "hardware" not in ds.meta:
            raise ValueError("Expected 'hardware' in ds.meta for fracatlas hardware presence.")
        return ds.meta["hardware"].astype(int).to_numpy() 

    elif key in {"murads"}:
        if not hasattr(ds, "meta") or "abnormal" not in ds.meta:
            raise ValueError("Expected 'abnormal' in ds.meta for MURA.")
        return ds.meta["abnormal"].astype(int).to_numpy()

    elif key in {"btxrdds"}:
        # Binary tumor presence
        if not hasattr(ds, "meta") or "tumor" not in ds.meta:
            raise ValueError("Expected 'tumor' in ds.meta for BTXRD tumor presence.")
        return ds.meta["tumor"].astype(int).to_numpy()

    elif key in {"btxrd_mb_ds"}:
        # malignant or benign
        if not hasattr(ds, "meta") or "malignant" not in ds.meta:
            raise ValueError("Expected 'malignant' in ds.meta for BTXRD malignant, benign classification.")
        return ds.meta["malignant"].astype(int).to_numpy()

    elif key in {"btxrd_subtype_ds"}:
        # Multiclass subtype dataset:
        # Btxrd_subtype_DS stores integer labels in '__label__' (0..8)
        if not hasattr(ds, "meta"):
            raise ValueError("Expected ds.meta for BTXRD subtype.")
        label_col = "__label__" if "__label__" in ds.meta.columns else (
            "label" if "label" in ds.meta.columns else None
        )
        if label_col is None:
            raise ValueError("Expected '__label__' (or 'label') in ds.meta for BTXRD subtype.")
        return ds.meta[label_col].astype(int).to_numpy()
    
    else:
        raise ValueError(
            f"Could not derive labels for dataset name '{ds_name}'. "
            "Provide a supported name"
        )

def split_train_test_no_patient_leak(ds, y_labels, test_size, seed):
    """
    If the dataset has a 'patient_id' column (pediatricfx, MURA), do a patient-level stratified split for the TEST set, 
    then map back to sample indices.
    Otherwise, fall back to StratifiedShuffleSplit or ShuffleSplit.
    Test sample count may deviate slightly from the desired fraction because patients have variable numbers of images.
    """
    # pediatricfx or MURA cases
    if hasattr(ds, "meta") and isinstance(ds.meta, pd.DataFrame) and "patient_id" in ds.meta.columns:
        df = ds.meta.reset_index(drop=True).copy()
        df["__y__"] = np.asarray(y_labels) if y_labels is not None else 0
        
        uniques = pd.unique(df["__y__"].dropna())
        is_binary = (len(uniques) <= 2) and set(uniques).issubset({0, 1})
        
        if is_binary:
            # Patient-level label: "any positive makes positive" (simple & robust)  
            pat = df.groupby("patient_id")["__y__"].max().reset_index()
        else :
            pat = df.groupby("patient_id")["__y__"].first().reset_index()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        pat_tr_idx, pat_te_idx = next(sss.split(np.zeros(len(pat)), pat["__y__"]))

        te_patients = set(pat.loc[pat_te_idx, "patient_id"].astype(str))
        # Map back to sample indices
        te_mask = df["patient_id"].astype(str).isin(te_patients).to_numpy()
        te_idx = np.where(te_mask)[0]
        tr_idx = np.where(~te_mask)[0]
        return tr_idx, te_idx

    # Else fallback to StratifiedShuffleSplit or ShuffleSplit
    if y_labels is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(sss.split(np.zeros(len(ds)), y_labels))
    else:
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(ss.split(np.zeros(len(ds))))
    return tr_idx, te_idx

# --------------------------------------------------------
# 1. Transforms
# --------------------------------------------------------

def _img_tfm(train: bool):
    aug = [
        transforms.RandomRotation(10),
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ] if train else [transforms.Resize(IMG_SIZE)]

    return transforms.Compose([
        *aug,
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])

# --------------------------------------------------------
# 2. Dataset definitions
# --------------------------------------------------------

class PesplanusDS(Dataset):
    def __init__(self, root, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        paths, labels = [], []
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(("png", "jpg", "jpeg")):
                    labels.append(1 if "pesplanus-1" in r.lower() else 0)
                    paths.append(Path(r) / f)
        self.samples = [
            (p, l)
            for i, (p, l) in enumerate(zip(paths, labels))
            if idxs is None or i in idxs
        ]
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "PesplanusDS"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, l = self.samples[i]
        img = load_rgb_8bit(p)
        x = self.tfm(img)
        y = torch.tensor(l, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = None
            if self.return_meta:
                meta = _filter_meta({
                    "filename": p.name,
                    "stem": p.stem,
                    "parent": p.parent.name,
                }, self.meta_cols)
            return (x, y, str(p), meta)
        return (x, y)


class PediatricwristfxDS(Dataset):
    def __init__(self, csv_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_csv(csv_path)
        if "fracture_visible" in df.columns:
            df["fracture_visible"] = pd.to_numeric(df["fracture_visible"], errors="coerce").fillna(0).astype(int)
        else:
            df["fracture_visible"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "PediatricwristfxDS"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / f"{row.filestem}.png"
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.fracture_visible, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y

class Pediatricwristfx_implant_DS(Dataset):
    def __init__(self, csv_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_csv(csv_path)
        if "metal" in df.columns:
            df["metal"] = pd.to_numeric(df["metal"], errors="coerce").fillna(0).astype(int)
        else:
            df["metal"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "Pediatricwristfx_implant_DS"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / f"{row.filestem}.png"
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.metal, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y

class Pediatricwristfx_AO_DS(Dataset):
    """
    Dataset returning (image, label) where label is an int in {0, ..., n_classes-1}
    based on the CSV column 'ao_classification'. Expects columns: 'filestem', 'ao_classification'.
    """
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        idxs: Optional[List[int]] = None,
        train: bool = True,
        return_path: bool = False,
        return_meta: bool = False,
        meta_cols: Optional[List[str]] = None,
        label_col: str = "ao_classification",
        image_ext: str = ".png",
    ):
        df = pd.read_csv(csv_path)

        # basic validation
        required_cols = {"filestem", label_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")

        
        # build a stable label mapping: sort unique values as strings
        df[label_col]=df[label_col].str.rstrip()
        unique_labels = sorted(df[label_col].astype(str).unique())
        class_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        # subset rows if idxs provided
        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        df = df.reset_index(drop=True)

        # store encoded labels
        df["ao_class_enc"] = df[label_col].astype(str).map(class_to_idx).astype(int)

        # save attributes
        self.meta = df
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "Pediatricwristfx_AO_DS"
        self.label_col = label_col
        self.classes_: List[str] = unique_labels
        self.class_to_idx_: Dict[str, int] = class_to_idx
        self.image_ext = image_ext

    def __len__(self):
        return len(self.meta)

    @property
    def num_classes(self) -> int:
        return len(self.classes_)

    def decode_label(self, y_int: int) -> str:
        return self.classes_[int(y_int)]

    def __getitem__(self, i: int):
        row = self.meta.iloc[i]
        path = self.dir / f"{row.filestem}{self.image_ext}"
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.ao_class_enc, dtype=torch.long)  # integer class label

        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta

        return x, y

class BtxrdDS(Dataset):
    """
    Inputs image directory and xlsx file. xlsx file should have image_id where image name and extension is noted. xlsx file also should have "tumor" column
    Returns (image, label) where label is an int in {0, 1}.
    """
    def __init__(self, excel_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_excel(excel_path)
        if "tumor" in df.columns:
            df["tumor"] = pd.to_numeric(df["tumor"], errors="coerce").fillna(0).astype(int)
        else:
            df["tumor"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "BtxrdDS"
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / row.image_id
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.tumor, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y


class Btxrd_mb_DS(Dataset):
    """
    Inputs image directory and xlsx file. xlsx file should have image_id where image name and extension is noted. xlsx file also should have "malignant" column
    Returns (image, label) where label is an int in {0, 1}.
    """
    def __init__(self, excel_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_excel(excel_path)
        if "malignant" in df.columns:
            df["malignant"] = pd.to_numeric(df["malignant"], errors="coerce").fillna(0).astype(int)
        else:
            df["malignant"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "Btxrd_mb_DS"
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / row.image_id
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.malignant, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y
    

class Btxrd_subtype_DS(Dataset):
    """
    Expects an Excel file with:
      - 'image_id' column: image file name (with extension)
      - Nine binary columns (0/1) named exactly as below:

        0: 'osteochondroma'
        1: 'multiple osteochondromas'
        2: 'simple bone cyst'
        3: 'giant cell tumor'
        4: 'osteofibroma'
        5: 'synovial osteochondroma'
        6: 'other bt'
        7: 'osteosarcoma'
        8: 'other mt'

    Returns:
      (image_tensor, label_int) by default
      Optionally: (image_tensor, label_int, str(path)) or also meta dict if requested
    """
    CLASS_NAMES = [
        "osteochondroma",
        "multiple osteochondromas",
        "simple bone cyst",
        "giant cell tumor",
        "osteofibroma",
        "synovial osteochondroma",
        "other bt",
        "osteosarcoma",
        "other mt",
    ]
    NAME_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        excel_path: str,
        img_dir: str,
        idxs: Optional[List[int]] = None,
        train: bool = True,
        return_path: bool = False,
        return_meta: bool = False,
        meta_cols: Optional[List[str]] = None,
        strict_one_hot: bool = True,
    ):
        """
        Args:
            excel_path: path to the .xlsx with columns described above
            img_dir: directory containing the images
            idxs: optional subset of row indices to keep
            train: passed to your transform factory
            return_path: if True, include the image file path in the return tuple
            return_meta: if True, include selected metadata in the return tuple
            meta_cols: which metadata columns to keep when return_meta=True
            strict_one_hot: if True, require exactly one of the nine class columns == 1 for each row
                            If False, pick argmax after coercion to numeric (ties resolved by first max)
        """
        df = pd.read_excel(excel_path)

        # Validate required columns
        missing = [c for c in ["image_id"] + self.CLASS_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in Excel: {missing}")
        
        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        # Coerce the class columns to numeric 0/1
        class_df = df[self.CLASS_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0)

        if strict_one_hot:
            # Require exactly one positive per row
            sums = class_df.sum(axis=1).astype(int)
            bad_rows = np.where(sums != 1)[0]
            if len(bad_rows) > 0:
                # Give a short preview of the first few offending rows for easier debugging
                preview = df.iloc[bad_rows[:5]][["image_id"] + self.CLASS_NAMES]
                raise ValueError(
                    f"Each row must have exactly one of the nine class columns == 1, "
                    f"but {len(bad_rows)} rows violate this. Example(s):\n{preview.to_string(index=False)}"
                )
            labels = class_df.idxmax(axis=1).map(self.NAME_TO_IDX).astype(int).values
        else:
            # Fall back: choose argmax (ties resolved by first occurrence)
            labels = class_df.values.argmax(axis=1).astype(int)

        df = df.copy()
        df["__label__"] = labels

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "Btxrd_subtype_DS"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / row.image_id
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(int(row["__label__"]), dtype=torch.long)

        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y

class KneeoaDS(Dataset):
    """
    Root directory structure:
        root/
          0/  img0.png, img1.jpg, ...
          1/  ...
          2/
          3/
          4/

    Returns (image, label) where label is an int in {0..4}.
    """
    def __init__(self, root, idxs=None, train=True, grades=(0,1,2,3,4),
                 extensions=(".png",".jpg",".jpeg",".bmp",".tif",".tiff"),
                 return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        self.root = Path(root)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.grades = [int(g) for g in grades]
        exts = {e.lower() for e in extensions}

        samples = []
        for g in self.grades:
            d = self.root / str(g)
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    samples.append((p, g))

        if not samples:
            raise RuntimeError(
                f"No images found under {self.root} for grades {self.grades} "
                f"with extensions {sorted(exts)}"
            )
        
        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            samples = [samples[i] for i in idxs]

        self.samples = samples
        self.class_counts = Counter([g for _, g in samples])
        self.class_to_idx = {str(g): g for g in self.grades}
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "KneeoaDS"
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = load_rgb_8bit(path)
        img = self.tfm(img)
        target = torch.tensor(label, dtype=torch.long)
        if self.return_path or self.return_meta:
            meta = None
            if self.return_meta:
                meta = _filter_meta({
                    "filename": path.name,
                    "stem": path.stem,
                    "parent": path.parent.name,
                    "grade_dir": str(label),
                }, self.meta_cols)
            return img, target, str(path), meta
        return img, target

class BoneAgeDS(Dataset):
    def __init__(self, csv_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_csv(csv_path)
        self.meta = df.iloc[idxs].reset_index(drop=True) if idxs is not None else df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "BoneAgeDS"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / f"{row.id}.png"
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.boneage, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y


class MURADS(Dataset):
    def __init__(self, csv_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_csv(csv_path)
        if "abnormal" in df.columns:
            df["abnormal"] = pd.to_numeric(df["abnormal"], errors="coerce").fillna(0).astype(int)
        else:
            df["abnormal"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "MURADS"
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / row.filestem
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.abnormal, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y


class FracAtlasDS(Dataset):
    """
    Inputs image directory and csv file. csv file should have image_id where image name and extension is noted. csv file also should have "fractured" column
    Returns (image, label) where label is an int in {0, 1}.
    """
    def __init__(self, csv_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_csv(csv_path)
        if "fractured" in df.columns:
            df["fractured"] = pd.to_numeric(df["fractured"], errors="coerce").fillna(0).astype(int)
        else:
            df["fractured"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "FracAtlasDS"
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / row.image_id
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.fractured, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y

class FracAtlas_implant_DS(Dataset):
    """
    Inputs image directory and csv file. csv file should have image_id where image name and extension is noted. csv file also should have "hardware" column
    Returns (image, label) where label is an int in {0, 1}.
    """
    def __init__(self, csv_path, img_dir, idxs=None, train=True, return_path: bool=False, return_meta: bool=False, meta_cols: Optional[List[str]]=None):
        df = pd.read_csv(csv_path)
        if "hardware" in df.columns:
            df["hardware"] = pd.to_numeric(df["hardware"], errors="coerce").fillna(0).astype(int)
        else:
            df["hardware"] = 0

        if idxs is not None:
            if isinstance(idxs, torch.Tensor):
                idxs = idxs.tolist()
            df = df.iloc[idxs]

        self.meta = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.tfm = _img_tfm(train)
        self.return_path = return_path
        self.return_meta = return_meta
        self.meta_cols = meta_cols
        self.name = "FracAtlas_implant_DS"
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        path = self.dir / row.image_id
        img = load_rgb_8bit(path)
        x = self.tfm(img)
        y = torch.tensor(row.hardware, dtype=torch.float32)
        if self.return_path or self.return_meta:
            meta = _filter_meta(row.to_dict(), self.meta_cols) if self.return_meta else None
            return x, y, str(path), meta
        return x, y

# --------------------------------------------------------
# 3.  Model / training utility helpers
# --------------------------------------------------------

def build_model(backbone_ckpt: str, num_labels: int, problem: str):
    cfg = AutoConfig.from_pretrained(backbone_ckpt)

    if getattr(cfg, "model_type", None) == "vit_mae":
        model = ViTForImageClassification.from_pretrained(
            backbone_ckpt,
            num_labels=num_labels,
            problem_type=problem,
        )
        return model.to(DEVICE)
    else : 
        model = AutoModelForImageClassification.from_pretrained(
            backbone_ckpt,
            num_labels=num_labels,
            problem_type=problem,
            ignore_mismatched_sizes=True,  # head resize
        )
        return model.to(DEVICE)


def build_param_groups_with_llrd(model, base_lr: float, weight_decay: float, layer_decay: float):
    """
    Creates optimizer param groups with layer-wise LR decay for ViT models.
    If not a ViT or layer_decay==1.0, returns a single default group.
    """
    # If not ViT or no decay requested → single group
    if layer_decay >= 0.999 or not hasattr(model, "vit") or not hasattr(model.vit, "encoder"):
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "lr": base_lr, "weight_decay": weight_decay}]

    # Collect layers
    encoder = model.vit.encoder
    n_layers = len(getattr(encoder, "layer", []))
    # Map param name -> depth (0 = embeddings, 1..n_layers = blocks, n_layers+1 = head)
    def depth_from_name(name: str) -> int:
        if "vit.embeddings" in name: 
            return 0
        for i in range(n_layers):
            if f"vit.encoder.layer.{i}." in name:
                return i + 1
        # classifier/head or pooler: treat as top
        return n_layers + 1

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = depth_from_name(name)
        lr = base_lr * (layer_decay ** (n_layers + 1 - d))
        # No wd for biases/LayerNorms
        if name.endswith(".bias") or "LayerNorm.weight" in name or "layernorm" in name.lower():
            wd = 0.0
        else:
            wd = weight_decay
        key = (lr, wd)
        groups.setdefault(key, []).append(p)

    param_groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]
    return param_groups


def build_cosine_warmup_scheduler(optimizer, warmup_epochs: int, max_epochs: int):
    """
    Epoch-wise cosine schedule with linear warmup over `warmup_epochs`.
    """
    def lr_lambda(epoch: int):
        # epoch starts at 0 in PyTorch schedulers
        e = epoch + 1
        if e <= warmup_epochs:
            return e / max(1, warmup_epochs)
        # Cosine from 1.0 down to 0.0 over remaining epochs
        progress = (e - warmup_epochs) / max(1, (max_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, loss_fn, optim=None):
    training = optim is not None
    model.train() if training else model.eval()
    ctx = torch.enable_grad() if training else torch.no_grad()

    y_true_list, y_pred_list, y_prob_list, running_loss = [], [], [], 0.0
    path_list = []
    meta_list = []

    with ctx:
        for batch in loader:
            # Possible batch shapes: (x,y), (x,y,paths), (x,y,paths,metas)
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x, y = batch[0], batch[1]
                paths = batch[2]
                if isinstance(paths, (list, tuple, np.ndarray)):
                    path_list.extend(list(paths))
                else:
                    path_list.append(paths)
                if len(batch) >= 4:
                    metas = batch[3]
                    if isinstance(metas, (list, tuple, np.ndarray)):
                        meta_list.extend(list(metas))
                    else:
                        meta_list.append(metas)
            else:
                x, y = batch

            x, y = x.to(DEVICE), y.to(DEVICE)

            if training:
                optim.zero_grad()

            logits = model(x).logits

            if isinstance(loss_fn, nn.MSELoss):
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                loss = loss_fn(logits, y)
                y_pred_batch = logits.detach().cpu().numpy()
                y_true_batch = y.detach().cpu().numpy()
                y_pred_list.append(y_pred_batch)
                y_true_list.append(y_true_batch)
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                # Ensure integer class indices in range
                assert y.dtype == torch.long, f"CE expects Long targets, got {y.dtype}"
                C = logits.shape[1]
                bad = (y < 0) | (y >= C) | torch.isnan(y)
                if bad.any():
                    bad_idx = torch.nonzero(bad, as_tuple=False).squeeze(-1).tolist()
                    print(f"[BAD LABEL] C={C}  targets[min,max]=({int(y.min())},{int(y.max())})  idxs={bad_idx}")
                    try:
                        # if paths were provided (e.g., test loader)
                        print("paths for bad:", [paths[i] for i in bad_idx])
                    except Exception:
                        pass
                    raise ValueError("Found out-of-range target(s) for CrossEntropyLoss.")
                loss = loss_fn(logits, y)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
                pred = logits.argmax(dim=1).detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                y_prob_list.append(prob)
                y_pred_list.append(pred)
                y_true_list.append(y_np)
            else:
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                loss = loss_fn(logits, y)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                pred = (prob > 0.5).astype(int)
                y_np = y.detach().cpu().numpy().astype(int)
                y_prob_list.append(prob)
                y_pred_list.append(pred)
                y_true_list.append(y_np)

            if training:
                loss.backward()
                optim.step()
            running_loss += loss.item() * x.size(0)

    n = len(loader.dataset)
    mean_loss = running_loss / n

    if isinstance(loss_fn, nn.MSELoss):
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_prob = None
    else:
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_prob = np.concatenate(y_prob_list, axis=0)

    paths_out = path_list if len(path_list) == len(y_true) else None
    metas_out = meta_list if len(meta_list) == len(y_true) else None
    return mean_loss, y_true, y_pred, y_prob, paths_out, metas_out


# --- Metrics helpers ---

def compute_metrics(problem: str, y_true, y_pred, y_prob=None):
    if problem == "regression":
            return {
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
            }
    
    if y_prob is None:
        # defensive guard
        return {
            "BACC": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "AUROC": float("nan"),
        }
    y_prob = np.asarray(y_prob)
    
    if problem in ("single_label_classification", "multi_label_classification"):
        # classification
        if y_prob.ndim == 1:
            # binary
            return {
                "BACC": balanced_accuracy_score(y_true, y_pred),
                "F1": f1_score(y_true, y_pred),
                "AUROC": roc_auc_score(y_true, y_prob),
            }
        # multiclass
        num_classes = y_prob.shape[1]

        try:
            macro_auc = roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr",
                labels=np.arange(num_classes)
            )
        except ValueError:
            # fall back to “only classes with both pos & neg”
            Y = label_binarize(y_true, classes=np.arange(num_classes))
            aucs = []
            for c in range(num_classes):
                pos = Y[:, c].sum()
                neg = (Y[:, c] == 0).sum()
                if pos > 0 and neg > 0:
                    aucs.append(roc_auc_score(Y[:, c], y_prob[:, c]))
            macro_auc = float(np.mean(aucs)) if aucs else float("nan")

        return {
            "BACC": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "AUROC": macro_auc,
        }
    else:
        raise ValueError(f"Unknown problem type: {problem}")
    
# --------------------------------------------------------
# 4.  Cross-validation loop
# --------------------------------------------------------

def cv_loop(
    args,
    DS_cls,
    ds_kwargs: dict,
    problem: str,
    train_val_indices: np.ndarray,
    test_indices: np.ndarray,
    backbones: List[str],
):
    """Run CV for **all** backbones on identical splits.

    Returns
    -------
    results : Dict[str, List[dict]]
        key = backbone alias, value = list with per-fold metric dicts
    """

    # --- initial DS & split ---
    full_ds = DS_cls(**ds_kwargs, idxs=train_val_indices, train=True)
    labels = get_labels_for_stratification(full_ds, full_ds.name) if problem in ("single_label_classification", "multi_label_classification") else None
    
    num_classes_tv = None
    if labels is not None:
        num_classes_tv = int(np.max(labels)) + 1

    splitter = (
        StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        if problem in ("single_label_classification", "multi_label_classification")
        else KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    )

    results = {bk: [] for bk in backbones}
    
    # --- iterate folds ---
    for fold_id, (tr_rel, vl_rel) in enumerate(splitter.split(np.zeros(len(full_ds)), labels), 1):
        print(f"\n─── Fold {fold_id}/{args.cv} ─────────────────────────────────────")

        # train set inside CV (train_val_idx) 
        # Create loaders once and reuse for each backbone
        tr_abs = train_val_indices[tr_rel]
        vl_abs = train_val_indices[vl_rel]

        tr_ds = DS_cls(**ds_kwargs, idxs=tr_abs, train=True,  return_path=False)
        vl_ds = DS_cls(**ds_kwargs, idxs=vl_abs, train=False, return_path=False)
        te_ds = DS_cls(**ds_kwargs, idxs=test_indices, train=False,
                    return_path=True, return_meta=True, meta_cols=getattr(args, "meta_cols", None))

        test_split = str(int(len(te_ds) / (len(tr_ds) + len(vl_ds) + len(te_ds)) * 100))
        tr_ld = DataLoader(tr_ds, args.bs, shuffle=True,  num_workers=4, pin_memory=True)
        vl_ld = DataLoader(vl_ds, args.bs, shuffle=False, num_workers=4, pin_memory=True)
        te_ld = DataLoader(te_ds, args.bs, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_with_meta)

        # --- evaluate each backbone ---
        for bk in backbones:
            alias = bk
            checkpoint_dir = Path(args.outdir) / "ckpts" / DS_cls.__name__ / alias
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / f"fold{fold_id}.pt"

            # --- Save first 5 augmented train images (de-normalized) ---
            preview_dir = Path(args.outdir) / "previews" / DS_cls.__name__ / alias / f"fold{fold_id}"
            save_train_preview(tr_ds, preview_dir, n=5)

            sample_y = next(iter(tr_ld))[1]
            if problem == "single_label_classification":
                if num_classes_tv > 2:            # multiclass
                    num_labels = num_classes_tv
                    criterion = nn.CrossEntropyLoss()
                else:                              # binary
                    num_labels = 1
                    criterion = nn.BCEWithLogitsLoss()
                    
            elif problem == "regression":
                num_labels = 1
                criterion = nn.MSELoss()
            else:  # multilabel case for future
                num_labels = num_classes_tv
                criterion = nn.BCEWithLogitsLoss()

            model = build_model(args.backbone_map[bk], num_labels, problem)

            if alias == "resnet-101":
                base_lr = 5e-4
                base_wd = 1e-4
                ld = 1.0
            else:
                base_lr = args.lr
                base_wd = args.weight_decay
                ld = args.layer_decay
            # Param groups (LLRD for ViT; single group otherwise)
            param_groups = build_param_groups_with_llrd(
                model, base_lr=base_lr, weight_decay=base_wd, layer_decay=ld
            )

            optim = torch.optim.AdamW(
                param_groups,
                lr=base_lr,           # per-backbone base_lr
                weight_decay=0.0,     # wd handled per-group
                betas=(args.beta1, args.beta2)
            )

            # Cosine LR with warmup (epoch-wise)
            scheduler = build_cosine_warmup_scheduler(
                optim, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs
            )

            best_val_metric = float("inf") if problem == "regression" else -float("inf")
            best_state = None

            for ep in range(1, args.epochs + 1):
                tr_loss, *_ = run_epoch(model, tr_ld, criterion, optim)
                vl_loss, y_true, y_pred, y_prob, _, _ = run_epoch(model, vl_ld, criterion)
                val_metrics = compute_metrics(problem, y_true, y_pred, y_prob)
                if problem == "regression":
                    val_metric = val_metrics["MAE"]
                    improved = val_metric < best_val_metric
                else :
                    val_metric = val_metrics["AUROC"]
                    improved = val_metric > best_val_metric

                if improved:
                    best_val_metric = val_metric
                    best_state = model.state_dict()

                if args.verbose:
                    is_cls = problem in ("single_label_classification", "multi_label_classification")
                    msg = f"[{alias}] Ep{ep:02d} trL{tr_loss:.3f} vlL{vl_loss:.3f} "
                    msg += f"vlAUC{val_metric:.3f}" if is_cls else f"vlMAE{val_metric:.2f}"
                    print(msg)

                scheduler.step()

            # save ckpt for this fold/backbone
            if best_state is None:
                best_state = model.state_dict()
            torch.save(best_state, ckpt_path)
            model.load_state_dict(best_state)

            # --- final test evaluation + per-sample CSV ---
            _, y_t, y_p, y_pb, paths, metas = run_epoch(model, te_ld, criterion)
            fold_metrics = compute_metrics(problem, y_t, y_p, y_pb)
            results[alias].append(fold_metrics)

            pred_dir = Path(args.outdir) / "predictions" / DS_cls.__name__ / test_split / alias
            pred_dir.mkdir(parents=True, exist_ok=True)

            # Build base frame with predictions
            if problem == "regression":
                base_df = pd.DataFrame({
                    "path": paths if paths is not None else [None]*len(y_t),
                    "y_true": y_t,
                    "y_pred": y_p,
                })
            else:
                num_classes = 1 if (y_pb is None or (np.asarray(y_pb).ndim==1)) else np.asarray(y_pb).shape[1]
                prob_cols = probs_to_columns(problem, y_pb, num_classes)
                base = {
                    "path": paths if paths is not None else [None]*len(y_t),
                    "y_true": y_t.astype(int),
                    "y_pred": y_p.astype(int),
                }
                base.update(prob_cols)
                base_df = pd.DataFrame(base)

            # Metadata frame
            if metas is not None:
                # metas is a list of dicts (may contain None)
                metas_clean = [m if isinstance(m, dict) else {} for m in metas]
                meta_df = pd.DataFrame(metas_clean)
                # Avoid column name collisions 
                for c in list(meta_df.columns):
                    if c in base_df.columns and c != "path":
                        meta_df.rename(columns={c: f"meta_{c}"}, inplace=True)
                out_df = pd.concat([meta_df, base_df], axis=1)
            else:
                out_df = base_df

            out_df.to_csv(pred_dir / f"fold{fold_id}.csv", index=False)

            del model, optim
            torch.cuda.empty_cache()


    # --- aggregate ---
    agg = {
        bk: {
            k: (np.mean([d[k] for d in folds]), np.std([d[k] for d in folds]))
            for k in folds[0]
        }
        for bk, folds in results.items()
    }

    return results, agg

# --------------------------------------------------------
# 5.  IO helpers
# --------------------------------------------------------

def dump_results(outdir: str, tag: str, all_folds: Dict[str, List[dict]], agg: Dict[str, Dict]):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{tag}_{ts}"

    # one CSV per backbone
    for bk, folds in all_folds.items():
        pd.DataFrame(folds).to_csv(Path(outdir, f"{base}_{bk}_folds.csv"), index_label="fold")

    # summary JSON (all backbones)
    with open(Path(outdir, f"{base}_summary.json"), "w") as f:
        json.dump({bk: {
            k: {"mean": float(v[0]), "std": float(v[1])} for k, v in met.items()
        } for bk, met in agg.items()}, f, indent=2)

    print(f"✓ saved results to {Path(outdir).resolve()}")

# --------------------------------------------------------
# 6.  CLI parser
# --------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=["pesplanus", "boneage", "mura", "fracatlas", "kneeoa", "pediatricfx", "btxrd", "btxrd_mb", "btxrd_subtype", "pediatricfx_implant", "pediatricfx_ao", "fracatlas_implant"])
    p.add_argument("--backbones", nargs="+", required=True, help="List of backbone aliases to compare (e.g. skelex vit-i21k)")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--test_split", type=float, default=0.1, help="Fraction of data held out for final testing")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--outdir", default="F:/runs")
    p.add_argument("--verbose", default=True, action="store_true")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--layer_decay", type=float, default=0.75,
                help="Layer-wise LR decay factor (<=1.0). 1.0 disables LLRD.")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--meta_cols", nargs="*", default=None,
               help="Optional list of metadata columns to include in test CSV (CSV-backed datasets). If unset, include all.")
    return p.parse_args()

# --------------------------------------------------------
# 7.  Utility: set seeds
# --------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------------------
# 8.  Main
# --------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)

    # Map backbone alias (HuggingFace checkpoint path)
    args.backbone_map = {
        "skelex": "./models/skelex/",
        "vit-i21k": "google/vit-large-patch16-224-in21k",
        "resnet-101" : "./models/resnet_101/",
    }

    missing = [bk for bk in args.backbones if bk not in args.backbone_map]
    if missing:
        raise ValueError(f"Unknown backbones: {missing}. Update backbone_map in the script.")

    # --- dataset-specific keyword factories ---
    if args.data == "pesplanus":
        DS_cls = PesplanusDS
        ds_kwargs = {"root": "/data/Pesplanus Dataset"}
        problem = "single_label_classification"
    elif args.data == "boneage":
        DS_cls = BoneAgeDS
        ds_kwargs = {
            "csv_path": "/data/RSNA Bone age/boneage-training-dataset.csv",
            "img_dir": "/data/RSNA Bone age/boneage-training-dataset",
        }
        problem = "regression"
    elif args.data == "mura":
        DS_cls = MURADS
        ds_kwargs = {
            "csv_path": "/data/MURA-v1.1/whole_image_metadata.csv",
            "img_dir": "/data/MURA-v1.1/",
        }
        problem = "single_label_classification"
    elif args.data == "fracatlas":
        DS_cls = FracAtlasDS
        ds_kwargs = {
            "csv_path": "./data/FracAtlas/dataset.csv",
            "img_dir": "./data/FracAtlas/images/all_images",
        }
        problem = "single_label_classification"

    elif args.data == "fracatlas_implant":
        DS_cls = FracAtlas_implant_DS
        ds_kwargs = {
            "csv_path": "/data/FracAtlas/dataset.csv",
            "img_dir": "/data/FracAtlas/images/all_images",
        }
        problem = "single_label_classification"

    elif args.data == "kneeoa":
        DS_cls = KneeoaDS
        ds_kwargs = {"root": "/data/kneeKL224_integrated"} 
        problem = "single_label_classification"

    elif args.data == "pediatricfx":
        DS_cls = PediatricwristfxDS
        ds_kwargs = {
            "csv_path": "/data/pediatric_wrist_trauma/dataset_uncertain_filtered.csv",
            "img_dir": "/data/pediatric_wrist_trauma/images",
        }
        problem = "single_label_classification"

    elif args.data == "pediatricfx_implant":
        DS_cls = Pediatricwristfx_implant_DS
        ds_kwargs = {
            "csv_path": "/data/pediatric_wrist_trauma/dataset.csv",
            "img_dir": "/data/pediatric_wrist_trauma/images",

        }
        problem = "single_label_classification"
    elif args.data == "pediatricfx_ao":
        DS_cls = Pediatricwristfx_AO_DS
        ds_kwargs = {
            "csv_path": "/data/pediatric_wrist_trauma/dataset_uncertain_filtered_v2_ao_top_20.csv",
            "img_dir": "/data/pediatric_wrist_trauma/images",
        }
        problem = "single_label_classification"
    elif args.data == "btxrd":
        DS_cls = BtxrdDS
        ds_kwargs = {
            "excel_path": "/data/BTXRD/dataset.xlsx",
            "img_dir": "/data/BTXRD/images",
        }
        problem = "single_label_classification"

    elif args.data == "btxrd_mb":
        DS_cls = Btxrd_mb_DS
        ds_kwargs = {
            "excel_path": "/data/BTXRD/dataset_tumoronly.xlsx",
            "img_dir": "/data/BTXRD/images",
        }
        problem = "single_label_classification"

    elif args.data == "btxrd_subtype":
        DS_cls = Btxrd_subtype_DS
        ds_kwargs = {
            "excel_path": "/data/BTXRD/dataset_tumoronly.xlsx",
            "img_dir": "/data/BTXRD/images",
        }
        problem = "single_label_classification"

    else:
        raise RuntimeError()

    # -----------------------------------------------------------------
    # split once for test set (same indices for all backbones)
    tmp_ds = DS_cls(**ds_kwargs, train=True)
    y_labels = get_labels_for_stratification(tmp_ds, tmp_ds.name) if problem in ("single_label_classification", "multi_label_classification") else None
    
    train_val_idx, test_idx = split_train_test_no_patient_leak(
        ds=tmp_ds,
        y_labels=y_labels,
        test_size=args.test_split,
        seed=args.seed
    )

    # -----------------------------------------------------------------
    folds, summary = cv_loop(
        args,
        DS_cls,
        ds_kwargs,
        problem,
        train_val_idx,
        test_idx,
        args.backbones,
    )
    tag = f"{args.data}_{'-'.join(args.backbones)}_test{int(args.test_split * 100)}"
    dump_results(
        args.outdir,
        tag,
        folds,
        summary,
    )

if __name__ == "__main__":
    main()
