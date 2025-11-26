"""Collate utilities for stacking multi-view samples before feeding VGGT."""

from __future__ import annotations

from typing import Dict, List

import torch
from torchvision import transforms


def build_default_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # VGGT will handle its own normalization internally
        ]
    )


class MultiViewCollator:
    def __init__(self, image_size: int, tokenizer, max_length: int, num_vis_tokens: int = 128, geom_tokens: int = 8) -> None:
        self.transform = build_default_transform(image_size)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Reserve space for visual tokens
        self.num_vis_tokens = num_vis_tokens
        self.geom_tokens = geom_tokens
        self.min_text_length = num_vis_tokens + geom_tokens + 64  # Visual + geom + minimum text

    def __call__(self, batch: List[Dict]) -> Dict:
        pixel_batches = []
        texts = []
        answers = []
        geom = []
        for sample in batch:
            tensor_views = [self.transform(img) for img in sample["images"]]
            stack = torch.stack(tensor_views, dim=0)
            pixel_batches.append(stack)
            question = sample["question"]
            answer = sample["answer"]
            # Put image token AFTER question to avoid overwriting answer labels
            prompt = f"{question}\n<image>\n"
            texts.append(prompt)
            answers.append(answer)
            geom.append(sample.get("geom_token"))
        pixel_tensor = torch.stack(pixel_batches, dim=0)
        # Convert to bfloat16 if needed for mixed precision training
        # This will be handled by the model's autocast context
        pad_id = self.tokenizer.pad_token_id
        input_ids_list = []
        label_ids_list = []
        max_len = 0
        for prompt, answer in zip(texts, answers):
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + answer_ids)[: self.max_length]
            labels_seq = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
            max_len = max(max_len, len(ids))
            input_ids_list.append(ids)
            label_ids_list.append(labels_seq)
        
        # Ensure minimum length to accommodate visual tokens
        max_len = max(max_len, self.min_text_length)
        
        for ids, labels_seq in zip(input_ids_list, label_ids_list):
            pad_amount = max_len - len(ids)
            if pad_amount > 0:
                ids += [pad_id] * pad_amount
                labels_seq += [-100] * pad_amount
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()
        labels = torch.tensor(label_ids_list, dtype=torch.long)
        
        geom_batch = None
        if all(g is not None for g in geom):
            geom_batch = {}
            keys = geom[0].keys()
            for key in keys:
                geom_batch[key] = torch.tensor([g[key] for g in geom], dtype=torch.float32)
        return {
            "pixel_values": pixel_tensor,
            "geom_token": geom_batch,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
