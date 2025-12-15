import os
import uuid
import torchvision.transforms.functional as F
import torch
import random
from torchvision.ops import box_iou

from torchvision.transforms import Resize as TVResize

def box_ior(boxes: torch.Tensor, crop: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    IoR for each reference box in `boxes` against a single `crop`.
    IoR = area(intersection) / area(reference box)
    boxes: [N, 4] (xmin, ymin, xmax, ymax)
    crop : [4]    (xmin, ymin, xmax, ymax)
    returns: [N]
    """
    # Intersection
    ixmin = torch.maximum(boxes[:, 0], crop[0])
    iymin = torch.maximum(boxes[:, 1], crop[1])
    ixmax = torch.minimum(boxes[:, 2], crop[2])
    iymax = torch.minimum(boxes[:, 3], crop[3])

    iw = (ixmax - ixmin).clamp(min=0)
    ih = (iymax - iymin).clamp(min=0)
    inter = iw * ih

    # Reference (box) areas
    bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    ref_area = bw * bh

    return inter / (ref_area + eps)

class RandomIoRCrop:
    def __init__(self, scale_range=(0.5, 1.0), min_ior=0.5, max_attempts=10):
        self.scale_range = scale_range
        self.min_ior = min_ior
        self.max_attempts = max_attempts

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        boxes = target["boxes"]
        box_labels = target["box_labels"]

        if len(boxes) == 0:
            return image, target

        boxes = boxes.clone()

        for _ in range(self.max_attempts):
            scale = random.uniform(*self.scale_range)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)

            # skip degenerate crops
            if new_w <= 0 or new_h <= 0 or (orig_w == new_w and orig_h == new_h):
                continue

            left = random.randint(0, orig_w - new_w)
            top = random.randint(0, orig_h - new_h)
            crop = torch.tensor([left, top, left + new_w, top + new_h], dtype=torch.float32)

            # IoR: how much of each reference box is inside the crop
            iors = box_ior(boxes, crop)

            if (iors > self.min_ior).any():
                # apply crop to image
                image = F.crop(image, top, left, new_h, new_w)

                new_boxes, new_labels = [], []
                for box, label, ior in zip(boxes, box_labels, iors):
                    if ior > self.min_ior:
                        xmin, ymin, xmax, ymax = box
                        new_xmin = max(xmin - left, 0)
                        new_ymin = max(ymin - top, 0)
                        new_xmax = min(xmax - left, new_w)
                        new_ymax = min(ymax - top, new_h)

                        if new_xmax > new_xmin and new_ymax > new_ymin:
                            new_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
                            new_labels.append(label)

                if len(new_boxes) > 0:
                    target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
                    target["box_labels"] = new_labels
                    return image, target

        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)  # Converts PIL to CxHxW tensor in [0,1]
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
        return image, target

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees  # max angle in degrees

    def __call__(self, image, target):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle)
        return image, target
    
class ColorJitter:
    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, target):
        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = F.adjust_brightness(image, factor)

        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = F.adjust_contrast(image, factor)

        return image, target
    

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size  # tuple (H, W)

    def __call__(self, image, target):
        original_size = image.size  # (W, H)
        image = F.resize(image, self.size)  # PIL resize to HxW

        # Resize bounding boxes accordingly
        orig_w, orig_h = original_size
        new_w, new_h = self.size

        if "boxes" in target:
            resized_boxes = []
            for box in target["boxes"]:
                x1, y1, x2, y2 = box
                x1 = x1 * new_w / orig_w
                y1 = y1 * new_h / orig_h
                x2 = x2 * new_w / orig_w
                y2 = y2 * new_h / orig_h
                resized_boxes.append([x1, y1, x2, y2])
            target["boxes"] = torch.tensor(resized_boxes, dtype=torch.float32)

        return image, target


class ToMultiHot:
    def __init__(self, num_labels, labels_key='labels', mask_key='masks'):
        self.num_labels = num_labels
        self.labels_key = labels_key
        self.mask_key = mask_key

    def __call__(self, image, target):
        # indices of present labels
        label_idx = torch.as_tensor(target.get(self.labels_key, []), dtype=torch.long)
        # indices to mark as "masked/unknown"
        mask_idx  = torch.as_tensor(target.get(self.mask_key, []), dtype=torch.long)

        labels = torch.zeros(self.num_labels, dtype=torch.float32)
        masks  = torch.zeros(self.num_labels, dtype=torch.float32)

        if label_idx.numel() > 0:
            labels.index_fill_(0, label_idx, 1.0)

        if mask_idx.numel() > 0:
            masks.index_fill_(0, mask_idx, 1.0)

        target['labels'] = labels
        target['masks']  = masks
        return image, target