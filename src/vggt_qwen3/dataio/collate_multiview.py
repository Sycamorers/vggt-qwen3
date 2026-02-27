"""Collate utilities for stacking multi-view samples before feeding VGGT."""

from __future__ import annotations

import json
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
        # Minimum sequence length target; we still enforce a hard reserved span
        # right after <image> at the token level.
        self.min_text_length = num_vis_tokens + geom_tokens + 64  # Visual + geom + minimum text
        if self.max_length <= self.num_vis_tokens + self.geom_tokens + 16:
            raise ValueError(
                f"max_length={self.max_length} is too small for "
                f"num_vis_tokens={self.num_vis_tokens}, geom_tokens={self.geom_tokens}"
            )

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
            answer_obj = sample["answer"]
            # Ensure answers are strings (serialize dicts, lists, etc.)
            if not isinstance(answer_obj, str):
                answer = json.dumps(answer_obj, ensure_ascii=False)
            else:
                answer = answer_obj
            # Put image token AFTER question to avoid overwriting answer labels
            prompt = f"{question}\n<image>\n"
            texts.append(prompt)
            answers.append(answer)
            geom.append(sample.get("geom_token"))
        pixel_tensor = torch.stack(pixel_batches, dim=0)
        # Convert to bfloat16 if needed for mixed precision training
        # This will be handled by the model's autocast context
        pad_id = self.tokenizer.pad_token_id
        image_id = self.tokenizer.convert_tokens_to_ids("<image>")
        if image_id is None or (self.tokenizer.unk_token_id is not None and image_id == self.tokenizer.unk_token_id):
            raise ValueError("MultiViewCollator requires a valid '<image>' token in the tokenizer vocabulary.")

        reserved_span = self.num_vis_tokens + self.geom_tokens
        input_ids_list = []
        label_ids_list = []
        max_len = 0
        for prompt, answer in zip(texts, answers):
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            # Ensure exactly one <image> token and insert a reserved padding span
            # immediately after it so the injection window never touches answer tokens.
            image_positions = [idx for idx, tid in enumerate(prompt_ids) if tid == image_id]
            if not image_positions:
                raise ValueError(f"Prompt is missing <image> token: {prompt!r}")
            if len(image_positions) > 1:
                raise ValueError(f"Prompt has multiple <image> tokens; expected exactly one: {prompt!r}")
            img_pos = image_positions[0]

            if reserved_span > 0:
                reserved_tokens = [pad_id] * reserved_span
                prompt_ids = (
                    prompt_ids[: img_pos + 1]
                    + reserved_tokens
                    + prompt_ids[img_pos + 1 :]
                )

            answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + answer_ids)[: self.max_length]
            # Labels are -100 on the entire prompt (including <image> and reserved span),
            # and equal to answer token IDs on the answer region.
            labels_seq = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
            max_len = max(max_len, len(ids))
            input_ids_list.append(ids)
            label_ids_list.append(labels_seq)

        # Ensure minimum length to accommodate visual tokens + some slack
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
        if any(g is not None for g in geom):
            geom_batch = {}
            template = next((g for g in geom if g is not None), None)
            assert template is not None
            for key, template_val in template.items():
                stacked = []
                template_tensor = torch.tensor(template_val, dtype=torch.float32)
                for g in geom:
                    if g is None:
                        stacked.append(torch.zeros_like(template_tensor))
                    else:
                        stacked.append(torch.tensor(g[key], dtype=torch.float32))
                geom_batch[key] = torch.stack(stacked, dim=0)
            geom_batch["mask"] = torch.tensor([g is not None for g in geom], dtype=torch.bool)
        return {
            "pixel_values": pixel_tensor,
            "geom_token": geom_batch,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
