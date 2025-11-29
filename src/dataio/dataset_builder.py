"""Dataset utilities for multi-view 3D QA and ARKit instruction tuning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    path_glob: str
    num_views: int
    image_size: int
    task: str


class MultiViewJsonDataset(Dataset):
    """Lazy JSON loader that reads multi-view samples."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.files = sorted(Path().glob(config.path_glob))
        self.index: List[Dict] = []
        for file in self.files:
            # Handle both .json and .jsonl formats
            if file.suffix == '.jsonl':
                # JSONL: one JSON object per line
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.index.append(json.loads(line))
            else:
                # Regular JSON: single object or array
                records = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(records, dict):
                    records = records.get("data") or records.get("samples") or []
                if not isinstance(records, list):
                    raise ValueError(f"Expected a JSON array in {file}, got {type(records)}")
                self.index.extend(records)
        if not self.index:
            raise FileNotFoundError(f"No samples found for pattern {config.path_glob}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_image(self, rel_path: str) -> Image.Image:
        """Resolve and load an image path with clear errors if missing."""
        candidates = []
        rel_path_obj = Path(rel_path)
        if rel_path_obj.is_absolute():
            candidates.append(rel_path_obj)
        else:
            candidates.append(rel_path_obj)
            candidates.append(Path("data/raw") / rel_path_obj)
        for path in candidates:
            if path.exists():
                return Image.open(path).convert("RGB")
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Image not found for sample: tried {tried}")

    def __getitem__(self, idx: int) -> Dict:
        sample = self.index[idx]
        images = sample["images"][: self.config.num_views]
        pil_images = [self._load_image(img) for img in images]
        return {
            "images": pil_images,
            "geom_token": sample.get("geom_token"),
            "question": sample.get("question") or sample.get("instruction"),
            "answer": sample.get("answer") or sample.get("action_json"),
            "task": sample.get("task", self.config.task),
        }


class MultiSourceDataset(Dataset):
    """Interleave multiple datasets roughly according to mix ratios."""

    def __init__(self, datasets: Dict[str, MultiViewJsonDataset], mix_ratio: Dict[str, float]) -> None:
        self.datasets = datasets
        self.mix_ratio = mix_ratio
        self.order = self._build_schedule()
        self.dataset_lengths = {k: len(v) for k, v in datasets.items()}
        self.total_length = sum(self.dataset_lengths.values())
        self.random = random.Random(0)

    def _build_schedule(self) -> List[str]:
        total = sum(self.mix_ratio.values())
        schedule = []
        for name, weight in self.mix_ratio.items():
            count = max(1, int(round(weight / total * 100)))
            schedule.extend([name] * count)
        return schedule

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict:
        ds_name = self.order[idx % len(self.order)]
        dataset = self.datasets[ds_name]
        sample_idx = self.random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]
