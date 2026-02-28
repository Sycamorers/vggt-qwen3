from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch


class DummyTextModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden_size})()
        self.emb = torch.nn.Embedding(16, hidden_size)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        # Return a single "answer" token per example.
        bsz = inputs_embeds.size(0)
        # Token id 1 repeated once per sample.
        return torch.ones(bsz, 1, dtype=torch.long)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = DummyTextModel()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # Simple linear projection of flattened image tensor.
        b, v, c, h, w = images.shape
        flat = images.view(b, v, c * h * w)
        # Produce one "visual token" per batch.
        return flat.mean(dim=1, keepdim=True)


def _make_dummy_sample() -> Dict:
    import PIL.Image

    img = PIL.Image.new("RGB", (8, 8), color=(128, 128, 128))
    return {
        "images": [img],
        "geom_token": None,
        "question": "What color is the object?",
        "answer": "gray",
        "task": "scanqa",
        "scene_id": "scene_dummy",
        "question_id": "dummy-0",
    }


def test_run_inference_with_dummy_model(tmp_path: Path):
    from vggt_qwen3.inference.qa_inference import build_tokenizer, run_inference

    # Use a small built‑in tokenizer to avoid network access.
    class TinyTok:
        def __init__(self):
            self.vocab = {"<image>": 1, "What": 2, "color": 3, "is": 4, "the": 5, "object?": 6}
            self.pad_token = None
            self.eos_token = "</s>"
            self.pad_token_id = 0
            self.eos_token_id = 0

        def get_vocab(self):
            return self.vocab

        def add_tokens(self, tokens):
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab) + 1

        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, 0)

        def __call__(self, text, return_tensors=None):
            # Very approximate: just return a few ids.
            ids = [2, 3, 4, 5, 6, 1]  # "What color is the object? <image>"
            attn = [1] * len(ids)
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            }

        def decode(self, ids, skip_special_tokens=True):
            return "gray"

        def convert_ids_to_tokens(self, ids):
            return ["tok"] * len(ids)

    tokenizer = TinyTok()
    model = DummyModel()
    device = torch.device("cpu")

    samples = [_make_dummy_sample()]
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        image_size=8,
        max_new_tokens=4,
        output_path=None,
        dataset_name="scanqa",
        debug=False,
        events_path=None,
        short_answer_only=True,
        debug_one=None,
        generation_kwargs={"max_new_tokens": 4, "do_sample": False, "num_beams": 1},
        verbose_samples=0,
    )

    assert len(results) == 1
    rec = results[0]
    assert rec["question_id"] == "dummy-0"
    assert isinstance(rec["prediction"], str)
    assert len(rec["prediction"]) > 0

