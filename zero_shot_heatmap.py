"""
MAE reconstruction & hotspot visualization

"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

from transformers import ViTMAEForPreTraining
from transformers import AutoImageProcessor as _ImageProcessorFactory

# -------------------------
# Configuration dataclasses
# -------------------------


@dataclass
class HotspotConfig:
    n_passes: int = 10
    mask_ratio: float = 0.75
    use_l2: bool = True
    alpha: float = 0.55
    plot_norm: bool = True            # normalize heatmap for plotting

@dataclass
class IOConfig:
    out_root: Path = Path("output")
    image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    output_format: str = "png"  # png, jpeg, pdf, ...

@dataclass
class ModelConfig:
    model_path: Optional[Path] = None
    device: str = "auto"  # "cuda", "cpu", "auto"
    imagenet_normalize: bool = True  # respect model processor normalization

# --------------
# Util functions
# --------------

def setup_logging():
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

def pick_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_images(dirs: Sequence[Path], exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for base in dirs:
        if not base.exists():
            logging.warning(f"[skip] missing: {base}")
            continue
        files.extend([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return files

# ------------------------
# Image & preprocessing
# ------------------------

def _to_uint8_minmax(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def load_rgb_8bit(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        arr = np.array(img)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr8 = _to_uint8_minmax(arr)
    if arr8.ndim == 2:
        return Image.fromarray(arr8).convert("RGB")
    return Image.fromarray(arr8)

def apply_clahe_rgb(pil_img: Image.Image, clip_limit=2.0, tile_grid=(8, 8)) -> Image.Image:
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    gray = clahe.apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    rgb2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb2)

def preprocess_image(path: Path) -> Image.Image:
    img = load_rgb_8bit(path)
    return img

# ------------------------
# Model & visualization
# ------------------------

def load_model_and_processor(cfg: ModelConfig):
    if cfg.model_path is None:
        raise ValueError("model_path must be provided.")
    model = ViTMAEForPreTraining.from_pretrained(str(cfg.model_path))
    processor = _ImageProcessorFactory.from_pretrained(str(cfg.model_path))
    if not cfg.imagenet_normalize:
        if hasattr(processor, "do_normalize"):
            processor.do_normalize = False
        elif hasattr(processor, "image_std"):
            processor.image_std = [1.0, 1.0, 1.0]
            processor.image_mean = [0.0, 0.0, 0.0]
    device = pick_device(cfg.device)
    model = model.to(device).eval()
    return model, processor, device

def _nhwc(tchw: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nchw->nhwc", tchw)

def denorm_imagenet(nhwc: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.as_tensor(mean, device=nhwc.device).view(1, 1, 1, -1)
    s = torch.as_tensor(std, device=nhwc.device).view(1, 1, 1, -1)
    x = nhwc * s + m
    return x.clamp(0, 1)

def show_image(ax: plt.Axes, image_nhwc: torch.Tensor, title: str,
               mean: Sequence[float], std: Sequence[float],
               imagenet_normalize: bool = True) -> None:
    x = image_nhwc
    if imagenet_normalize:
        x = denorm_imagenet(x, mean, std)
    npimg = (x.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)[0]
    if npimg.ndim == 3 and npimg.shape[-1] == 3:
        ax.imshow(npimg, norm=colors.NoNorm())
    else:
        ax.imshow(npimg.squeeze(), cmap="gray", norm=colors.NoNorm())
    ax.set_title(title, fontsize=14)
    ax.axis("off")

@torch.inference_mode()
def mae_forward(model: ViTMAEForPreTraining,
                processor,
                pil_img: Image.Image,
                device: torch.device):
    batch = processor(images=pil_img, return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device)
    kwargs = {}
    outputs = model(pixel_values, **kwargs)

    rec = model.unpatchify(outputs.logits)               # [N, 3, H, W]
    rec = _nhwc(rec).detach()                            # [N, H, W, 3]

    mask = outputs.mask.detach()                         # [N, L]
    mask = mask.unsqueeze(-1).repeat(1, 1, model.config.patch_size**2 * 3)
    mask = model.unpatchify(mask)                        # [N, 3, H, W]
    mask = _nhwc(mask)                                   # [N, H, W, 3]

    x = _nhwc(pixel_values)                              # [N, H, W, 3]
    return x, rec, mask

def save_reconstruction_panel(out_path: Path,
                              x: torch.Tensor, rec: torch.Tensor, mask: torch.Tensor,
                              processor, imagenet_normalize: bool):
    fig, axs = plt.subplots(1, 4, figsize=(24, 24))
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + rec * mask

    show_image(axs[0], x, "original", mean, std, imagenet_normalize)
    show_image(axs[1], im_masked, "masked", mean, std, imagenet_normalize)
    show_image(axs[2], rec, "reconstruction", mean, std, imagenet_normalize)
    show_image(axs[3], im_paste, "reconstruction + visible", mean, std, imagenet_normalize)

    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ---------- Hotspot + Quality metrics ----------

@torch.inference_mode()
def compute_hotspot_map(
    model: ViTMAEForPreTraining,
    processor,
    pil_img: Image.Image,
    device: torch.device,
    n_passes: int = 10,
    mask_ratio: float = 0.75,
    use_l2: bool = True,
    imagenet_normalize: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Multi-pass SKELEX error map (masked-only regions averaged over passes).
    Returns (err_map_normed[H,W], mean_err_float, base_img_uint8[H,W,3]).
    """
    x0, _, _ = mae_forward(model, processor, pil_img, device)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    x_img = denorm_imagenet(x0, mean, std) if imagenet_normalize else x0
    base = (x_img[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    H, W = x0.shape[1], x0.shape[2]
    err_accum = torch.zeros((1, H, W), device=device)
    mask_hits = torch.zeros((1, H, W), device=device)

    for _ in range(int(n_passes)):
        x, rec, mask = mae_forward(model, processor, pil_img, device) 
        x_dn = denorm_imagenet(x, mean, std) if imagenet_normalize else x
        rec_dn = denorm_imagenet(rec, mean, std) if imagenet_normalize else rec

        delta = (rec_dn - x_dn) * mask
        per_px = (delta ** 2).mean(dim=-1, keepdim=False) if use_l2 else delta.abs().mean(dim=-1, keepdim=False)

        err_accum += per_px
        mask_hits += mask[..., 0]

    err_map = torch.zeros_like(err_accum)
    hit = mask_hits > 0
    err_map[hit] = err_accum[hit] / mask_hits[hit]

    em = err_map[0]
    em_cpu = em.detach().cpu().numpy()
    mean_err = float(np.nanmean(em_cpu[np.isfinite(em_cpu)]))

    return em_cpu, mean_err, base


def save_hotspot_overlay(out_path: Path,
                         err_map: np.ndarray,
                         base_rgb: np.ndarray,
                         alpha: float = 0.55,
                         plot_norm: bool = True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(base_rgb, norm=colors.NoNorm())
    if plot_norm:
        im = ax.imshow(err_map, cmap="hot", alpha=alpha, vmin=0.0, vmax=1.0) #jet
    else:
        im = ax.imshow(err_map, cmap="hot", alpha=alpha)
   

    # add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Error value")

    ax.axis("off")

    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0)
    
    plt.close(fig)

# ------------------------
# Main pipeline
# ------------------------

def process_images(
    image_paths: Sequence[Path],
    model_cfg: ModelConfig,
    hs_cfg: HotspotConfig,
    io_cfg: IOConfig,
) -> None:
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k): return x  # fallback

    model, processor, device = load_model_and_processor(model_cfg)

    rows: List[dict] = []
    for p in tqdm(image_paths, desc="Visualizing"):
        try:
            pil = preprocess_image(p)

            rel = Path(*p.parts[-2:])
            out_dir = io_cfg.out_root / rel.parent
            ensure_dir(out_dir)

            stem = p.stem
            panel_path   = out_dir / f"{stem}.{io_cfg.output_format}"
            hotspot_path = out_dir / f"{stem}_hotspot.{io_cfg.output_format}"

            # Reconstruction panel (single pass just for visuals)
            x, rec, mask = mae_forward(model, processor, pil, device)
            save_reconstruction_panel(panel_path, x, rec, mask, processor, model_cfg.imagenet_normalize)
            logging.info(f"Saved panel → {panel_path}")

            # Hotspot
            emap, mean_err, base = compute_hotspot_map(
                model, processor, pil, device,
                n_passes=hs_cfg.n_passes,
                mask_ratio=hs_cfg.mask_ratio,
                use_l2=hs_cfg.use_l2,
                imagenet_normalize=model_cfg.imagenet_normalize,
            )
            save_hotspot_overlay(hotspot_path, emap, base, alpha=hs_cfg.alpha, plot_norm=hs_cfg.plot_norm)
            logging.info(f"Saved hotspot → {hotspot_path}")

            record = {"path": str(p), "mean_error": mean_err}


            rows.append(record)

        except Exception as e:
            logging.exception(f"[error] {p}: {e}")

    if rows:
        summary_csv = io_cfg.out_root / "err_mean_summary.csv"
        df = pd.DataFrame(rows)
        df.to_csv(summary_csv, index=False)
        logging.info(f"Per-image summary saved → {summary_csv}")

# ------------------------
# CLI
# ------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SKELEX reconstruction, hotspot")
    ap.add_argument("--model-path", type=Path, default="/models/skelex", required=True, help="SKELEX model checkpoint dir")
    ap.add_argument("--in-dirs", type=Path, nargs="+", required=True, help="One or more input directories")
    ap.add_argument("--out", type=Path, default="/outputs/zero_shot", required=True, help="Root output directory")
    ap.add_argument("--format", type=str, default="png", help="Output format: png|jpeg|pdf")
    ap.add_argument("--passes", type=int, default=10, help="Number of passes for hotspot/MSSIM")
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--l2", default=True, action="store_true", help="Hotspot: Use L2 (else L1)")
    ap.add_argument("--alpha", type=float, default=0.55, help="Hotspot overlay alpha")
    ap.add_argument("--plot-norm", default=False, action="store_true")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--imagenet-norm", default=True, action="store_true", help="ImageNet normalization in processor")

    return ap

def main(argv=None):
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    model_cfg = ModelConfig(
        model_path=args.model_path,
        device=args.device,
        imagenet_normalize=args.imagenet_norm,
    )
    hs_cfg = HotspotConfig(
        n_passes=args.passes,
        mask_ratio=args.mask_ratio,
        use_l2=args.l2,
        alpha=args.alpha,
        plot_norm=args.plot_norm,
    )

    io_cfg = IOConfig(
        out_root=args.out,
        output_format=args.format,
    )

    imgs = list_images(args.in_dirs, io_cfg.image_exts)
    if not imgs:
        logging.warning("No images found. Check --in-dirs and extensions.")
        return
    process_images(imgs, model_cfg, hs_cfg, io_cfg)

if __name__ == "__main__":
    main()