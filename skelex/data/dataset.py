import torch
from torch.utils.data import Dataset
from PIL import Image
import random

class CombinedDataset(Dataset):
    def __init__(self, entries, transform=None):
        """
        entries: list of dicts, each with keys at least:
          - 'image'      : path
          - 'bbox'       : list[[cx, cy, w, h], ...]  (YOLO-style, relative to 1.0)
          - 'box_labels' : list[int]  (detection labels for each box)
          - 'labels'    : list[int]  (indices for 38-d multi-label task)
          - 'masks'      : list[int]  (indices to mask out)
        transform: callable(image, target) -> (image, target)
                   MUST eventually populate target['labels'] and target['masks'] as 38-d tensors.
        """
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = entry["image"]
        image = Image.open(img_path).convert("RGB")

        # Convert [cx, cy, w, h] --> [xmin, ymin, xmax, ymax] in pixels
        W, H = image.size
        boxes = []
        for cx, cy, w, h in entry["bbox"]:
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            boxes.append([x1, y1, x2, y2])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "box_labels": entry["box_label"],  # raw list, or convert to tensor if needed
            "image_id": entry["image_id"],
            "labels": entry["labels"],
            "masks": entry["masks"]
        }

        if self.transform:
            image, target = self.transform(image, target)

        return {
            'pixel_values': image,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "box_labels": target["box_labels"],  # raw list, or convert to tensor if needed
            "image_id": target["image_id"],
            "labels": target["labels"],
            "masks": target["masks"]
        }