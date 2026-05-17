# SKELEX: musculoSKELEtal X-ray foundation model

SKELEX is a foundation model for musculoskeletal
radiographs, pretrained with a masked-autoencoding (MAE) objective on a
large in-house collection of clinical X-rays.

This repository contains the code released alongside the paper:

- a Jupyter walkthrough showing reconstruction and anomaly-map inference
  end-to-end (`notebooks/SKELEX_walkthrough.ipynb`) or [`colab_link`](https://colab.research.google.com/drive/1HxwGWF1vbcmyFKgGfDHOubyoMuPJ4HK4?usp=sharing).
- pretrained weight loading and unsupervised reconstruction / anomaly
  heatmap visualization (`unsupervised_heatmap.py`),
- a **standardized downstream evaluation** comparing SKELEX with
  ViT, ViT-MAE, ResNet, BiomedCLIP and RadioDINO on twelve musculoskeletal benchmarks
  (`train_downstream.py`),
- **region-guided multi-head fine-tuning** for bone-tumor & implant
  classification (`multihead_classifier_train.py`),

---

## 1. Environment setup

```bash
conda create -y -n skelex_env python=3.10
conda activate skelex_env
pip install torch torchvision
pip install -r requirements.txt
```

The code has been tested with Python 3.10, PyTorch 2.1+ and CUDA 11.8.

---

## 2. Pretrained weights

The SKELEX checkpoint is released on the HuggingFace Hub:

- HuggingFace: [`skhoha/SKELEX`](https://huggingface.co/skhoha/SKELEX)

The SKELEX checkpoints are also available via google drive:

- ['checkpoints'](https://drive.google.com/drive/folders/19vA7SF-ek0Rkumz9EEivRM6vwi00z16x?usp=sharing)

Load the model directly:

```python
from transformers import AutoImageProcessor, ViTMAEForPreTraining

processor = AutoImageProcessor.from_pretrained("skhoha/SKELEX")
model = ViTMAEForPreTraining.from_pretrained("skhoha/SKELEX")
```

For supervised fine-tuning, load the same encoder into a classification head
(the decoder weights are unused):

```python
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("skhoha/SKELEX")
```

A complete walkthrough (loading, reconstruction, anomaly heatmaps, and a
Gradio demo) is provided in [`colab`](https://colab.research.google.com/drive/1HxwGWF1vbcmyFKgGfDHOubyoMuPJ4HK4?usp=sharing)..

---

## 3. Unsupervised reconstruction & anomaly heatmaps

`unsupervised_heatmap.py` runs SKELEX in a multi-mask fashion and
overlays the average reconstruction error on each input radiograph.

```bash
python unsupervised_heatmap.py \
  --model-path skhoha/SKELEX \
  --in-dirs /path/to/images \
  --out /path/to/output_dir \
  --passes 10
```

---

## 4. Downstream evaluation (`train_downstream.py`)

This script trains and evaluates multiple backbones on the same stratified
splits across twelve musculoskeletal benchmarks. Per fold and per backbone,
it stores the best-val checkpoint, per-sample predictions, per-fold metrics
and a JSON summary (mean ± SD across folds).

Supported `--data` values:

```
pesplanus  boneage  mura
fracatlas  fracatlas_implant
kneeoa
pediatricfx  pediatricfx_implant  pediatricfx_ao
btxrd  btxrd_mb  btxrd_subtype
```

### 4.1 Configure backbones and dataset paths

Open `train_downstream.py` and edit the two pieces below.

Backbone alias → checkpoint:

```python
args.backbone_map = {
    "skelex":             "skhoha/SKELEX",
    "vit-i21k":           "google/vit-large-patch16-224-in21k",
    "resnet-101":         "microsoft/resnet-101",
    "vit-mae":            "./models/vit_mae_large/",
    "biomedclip":         "./models/biomedclip/",
    "radio-dino-snarcy":  "hf_hub:Snarcy/RadioDino-b16",
}
```

Then set dataset-specific paths, for example:

```python
elif args.data == "fracatlas":
    DS_cls = FracAtlasDS
    ds_kwargs = {
        "csv_path": "/path/to/FracAtlas/dataset.csv",
        "img_dir":  "/path/to/FracAtlas/images/all_images",
    }
    problem = "single_label_classification"
```
Do the same for other datasets (boneage, mura, kneeoa, pediatricfx, btxrd, …) to match directory structure.


### 4.2 Run

```bash
python train_downstream.py \
  --data {DATASET_NAME} \
  --backbones {BACKBONE_1} {BACKBONE_2} ... \
  --cv 5 \
  --epochs 50 \
  --bs 64 \
  --lr 5e-5 \
  --test_split 0.1 \
  --outdir ./runs \
  --num_workers 4
```

---

## 5. Region-guided multi-head fine-tuning

`multihead_classifier_train.py` trains a single ViT classifier with multiple
task-specific output heads (binary abnormality, 4-way bone tumor classification,
29-way anatomical region, 3-way fracture type, plus an implant head).

Provide dataset paths and the class file either via CLI flags or a YAML
config (`--config /path/to/config.yaml`). Required arguments:

```bash
python multihead_classifier_train.py \
  --btxrd_image_dir       /path/to/BTXRD/images \
  --btxrd_annotation_dir  /path/to/BTXRD/annotations \
  --btxrd_metadata        /path/to/BTXRD/metadata.xlsx \
  --fracatlas_image_dir       /path/to/FracAtlas/images \
  --fracatlas_annotation_dir  /path/to/FracAtlas/YOLO_annotations \
  --fracatlas_metadata        /path/to/FracAtlas/metadata.csv \
  --snuh_bonetu_image_dir /path/to/SNUH_BoneTu/images \
  --snuh_bonetu_metadata  /path/to/SNUH_BoneTu/metadata.csv \
  --class_file            configs/classes.txt \
  --pretrained_path       skhoha/SKELEX \
  --num_classes 38 \
  --batch_size 64 \
  --epochs 30 \
  --lr 5e-5 \
  --n_splits 5 \
  --test_fraction 0.1 \
  --output_dir ./outputs/skelex_result \
  --num_workers 4
```

The 38 class indices are listed in [`configs/classes.txt`](configs/classes.txt).

---


## Acknowledgements

This project was built on top of excellent open-source work including
[ViT](https://github.com/google-research/vision_transformer),
[MAE](https://github.com/facebookresearch/mae),
[timm](https://github.com/huggingface/pytorch-image-models), and
[HuggingFace Transformers](https://github.com/huggingface/transformers). We thank the
authors and developers for their contributions.

## License
- CC-BY-NC-ND-4.0



