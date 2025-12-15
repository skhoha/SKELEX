# SKELEX: MusculoSKELEtal X-ray foundation model
- A Foundation model of musculoskeletal X-ray using masked autoencoding strategy.


### Setup environment
- Create and activate a python 3 conda environment:
```bash
  conda create -y -n "skelex_env" python=3.10
  pip install torch torchvision
  pip install -r requirements.txt
```

### Pre-trained Bone-MAE models
- You can download the pretrained Bone-MAE model by google drive
- ["Google drive access"](https://drive.google.com/drive/folders/19vA7SF-ek0Rkumz9EEivRM6vwi00z16x?usp=sharing)
- An example of loading the pretrained model can be found at: zero_shot_heatmap.py



### Zeroshot SKELEX Reconstruction & Heatmap Visualization
example run
```bash
python zero_shot_heatmap.py \
  --model-path ./models/skelex
  --in-dirs /path/to/images \
  --out /path/to/output_dir \
  --passes 10
```


### Downstream evaluation (`train_downstream.py`)
This script evaluates multiple backbones (e.g., SKELEX, ViT-L, ResNet-101) on several musculoskeletal X-ray benchmarks using identical data splits.

Supported datasets:
- `pesplanus`
- `boneage`
- `mura`
- `fracatlas`
- `fracatlas_implant`
- `kneeoa`
- `pediatricfx`
- `pediatricfx_implant`
- `pediatricfx_ao`
- `btxrd`
- `btxrd_mb`
- `btxrd_subtype`

#### 1. Configure dataset paths and backbones

Open `scripts/train_downstream.py` and edit:

```python
args.backbone_map = {
    "skelex": "/path/to/skelex",
    "vit-i21k": "google/vit-large-patch16-224-in21k",
    "resnet-101": "/path/to/resnet101_checkpoint/ or microsoft/resnet-101",
}
```

Then set dataset-specific paths, for example:

```python
elif args.data == "fracatlas":
    DS_cls = FracAtlasDS
    ds_kwargs = {
        "csv_path": "/path/to/FracAtlas/dataset.csv",
        "img_dir": "/path/to/FracAtlas/images/all_images",
    }
    problem = "single_label_classification"
```
Do the same for other datasets (boneage, mura, kneeoa, pediatricfx, btxrd, …) to match directory structure.

#### 2. Basic usage

```bash
python scripts/train_downstream.py \
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

### Region-guided multi-headed bone tumor classifier
- This script trains the region-guided multi-headed bone tumor classifier used in SKELEX, combining multiple musculoskeletal radiograph datasets into a single multi-task learning framework.
- Backbone: ViT-large based SKELEX
- Training data:
  - BTXRD (bone tumor radiographs)
  - FracAtlas (fracture and implant annotations)
  - SNUH-bonetu (institutional bone tumor dataset)

#### Training command
```bash
python train_region_guided.py \
  --btxrd_image_dir ./data/BTXRD/images \
  --btxrd_annotation_dir ./data/BTXRD/Annotations \
  --btxrd_metadata ./data/BTXRD/dataset_c1.xlsx \
  --fracatlas_image_dir ./data/FracAtlas/images \
  --fracatlas_annotation_dir ./data/FracAtlas/Annotations/YOLO \
  --fracatlas_metadata ./data/FracAtlas/dataset.csv \
  --snuh_bonetu_image_dir ./data/SNUH-bonetu/images \
  --snuh_bonetu_metadata ./data/SNUH-bonetu/metadata.csv \
  --class_file ./configs/classes.txt \
  --pretrained_path ./models/skelex \
  --num_classes 38 \
  --batch_size 64 \
  --epochs 30 \
  --lr 5e-5 \
  --n_splits 5 \
  --test_fraction 0.1 \
  --output_dir ./outputs/skelex_result \
  --num_workers 4
```

## Acknowledgements
This project builds upon a number of outstanding open-source libraries and research codebases, including but not limited to Vision Transformer (ViT) implementations, the timm library, Hugging Face Transformers and Datasets, and masked autoencoder (MAE) frameworks. We sincerely thank the authors and developers of these repositories for making their work publicly available and for their invaluable contributions to the research community.

## License
- CC-BY-NC-ND-4.0



