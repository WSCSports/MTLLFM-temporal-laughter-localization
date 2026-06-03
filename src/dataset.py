"""
Dataset reader for the released temporal laughter annotations.

Each annotation file (annotations/temporal_ur_funny.json,
annotations/temporal_smile.json) is a list of records:

    {
      "name": "<clip identifier>",
      "split": "train" | "dev" | "test",
      "laughter_events": [
        {"start": <sec>, "end": <sec>, "intensity": ...,
         "source": ..., "speaker": ...},
        ...
      ]
    }

We release annotations only. The dataset pairs each record with locally extracted
features (see extract_features.py); records without cached features are skipped so
the pipeline runs on whatever subset of the media you have obtained.
"""
import json
import os
from typing import List, Optional

import torch
from torch.utils.data import Dataset

AUDIO_T = 250   # HuBERT frames per 5 s segment
VISION_T = 50   # MAE frames per 5 s segment (10 fps)
AUDIO_DIM = 1024
VISION_DIM = 768


def _fix_temporal(t: torch.Tensor, target_T: int) -> torch.Tensor:
    if t.dim() == 1:
        t = t.unsqueeze(0)
    T = t.shape[0]
    if T > target_T:
        return t[:target_T]
    if T < target_T:
        pad = t[-1:].repeat(target_T - T, 1)
        return torch.cat([t, pad], dim=0)
    return t


class LaughterDataset(Dataset):
    def __init__(
        self,
        annotations_path: str,
        features_dir: str,
        split: Optional[str] = None,
    ):
        with open(annotations_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.items: List[dict] = []
        self.missing = 0
        for rec in records:
            if split is not None and rec.get("split") != split:
                continue
            stem = os.path.splitext(rec["name"])[0]
            feat_path = os.path.join(features_dir, stem + ".pt")
            if not os.path.exists(feat_path):
                self.missing += 1
                continue
            self.items.append({
                "name": rec["name"],
                "feat_path": feat_path,
                "events": rec.get("laughter_events", []),
                "label": 1 if rec.get("laughter_events") else 0,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        blob = torch.load(item["feat_path"], map_location="cpu")
        audio = _fix_temporal(blob["audio"].float(), AUDIO_T)
        vision = _fix_temporal(blob["vision"].float(), VISION_T)
        return {
            "name": item["name"],
            "audio": audio,
            "vision": vision,
            "label": item["label"],
            "events": item["events"],
        }
