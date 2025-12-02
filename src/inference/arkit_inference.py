"""ARKit / RoomPlan inference entrypoint.

This script loads a trained VGGT-Qwen3 checkpoint and runs inference on
ARKit synthetic data records (JSON produced by `synth_roomplan_instructions.py`).
It mirrors the training data path so you can point it at the same
`data/processed/arkit_synth/*.json` files and run the first N scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import load_sharded_checkpoint

from src.dataio.dataset_builder import DatasetConfig, MultiViewJsonDataset
from src.dataio.collate_multiview import build_default_transform
from src.models.projector_perceiver import PerceiverConfig
from src.models.vggt_qwen3_vlm import VisionLanguageConfig


def load_yaml(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(config_path: str, device: torch.device) -> torch.nn.Module:
    """Rebuild VGGT-Qwen3 model for inference using a stage config YAML.

    This re-implements the minimal pieces of `build_model` from train_sft but
    keeps the model in eval mode and exposes generate().
    """
    cfg = load_yaml(config_path)
    model_cfg = cfg["model"]

    projector_cfg = load_yaml(model_cfg["projector"])
    vlm_cfg = VisionLanguageConfig(
        text_model_name=model_cfg["name_or_path"],
        vision_ckpt_dir=model_cfg["vision_backbone"],
        num_vis_tokens=model_cfg["num_vis_tokens"],
        geom_tokens=model_cfg.get("geom_tokens", 0),
        projector_cfg=PerceiverConfig(**projector_cfg),
        freeze_vision=True,
        dtype=model_cfg.get("dtype", "bfloat16"),
    )

    # Rebuild the text model and projector in a lightweight wrapper so we can
    # call `.generate`. We intentionally do not rebuild the training wrapper
    # which expected loss labels.
    from src.models.vggt_qwen3_vlm import VGGTQwen3VLM

    base = VGGTQwen3VLM(vlm_cfg).to(device)
    base.eval()
    return base, cfg


def load_checkpoint_if_available(model: torch.nn.Module, ckpt_dir: Optional[str]) -> None:
    """Optionally load a checkpoint directory (Stage 2/3 weights).

    This is aware of the DeepSpeed ZeRO-3 layout used in training:
    if a merged fp32 folder created by `zero_to_fp32.py` is present, we
    prefer that; otherwise we fall back to any *.bin / *.safetensors.

    Recommended layout (more professional naming):
      ckpts/stage2_3d/
        pytorch_model/              # ZeRO shards from training
        pytorch_model_fp32/         # directory created by zero_to_fp32.py
          pytorch_model-00001-of-00005.bin
          ...
    """
    if not ckpt_dir:
        return
    path = Path(ckpt_dir)
    if not path.exists():
        print(f"⚠️  Checkpoint directory {path} does not exist; running with base weights.")
        return

    # Prefer merged fp32 folder if user ran zero_to_fp32.py (Stage 1/2 training).
    merged_root = path / "pytorch_model_fp32"
    legacy_merged_root = path / "pytorch_model_fp32.bin"

    # If a sharded HF-style checkpoint exists, load it fully (not just shard 0).
    for candidate in (merged_root, legacy_merged_root):
        index_file = candidate / "pytorch_model.bin.index.json"
        if candidate.exists() and index_file.exists():
            print(f"🔄 Loading sharded checkpoint from {candidate} (using index)")
            load_sharded_checkpoint(model, candidate)
            return

    weight_files = []
    if merged_root.exists():
        weight_files = [merged_root] if merged_root.is_file() else sorted(merged_root.glob("*.bin"))
    elif legacy_merged_root.exists() and legacy_merged_root.is_dir():
        weight_files = sorted(legacy_merged_root.glob("*.bin"))
    else:
        weight_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))

    if not weight_files:
        print(f"⚠️  No model weights found in {path}; using base HF weights.")
        return

    print(f"🔄 Loading checkpoint weights from {weight_files[0]}")
    state = torch.load(weight_files[0], map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"   Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def load_arkit_samples(
    glob_pattern: str,
    max_scenes: int,
    num_views: int,
    image_size: int,
) -> List[Dict]:
    """Load up to `max_scenes` ARKit synthetic samples from JSON/JSONL files.

    This uses the same DatasetConfig fields as training, so preprocessing
    (multi-view selection, image paths) is consistent with Stage 3.
    """
    cfg = DatasetConfig(
        path_glob=glob_pattern,
        num_views=num_views,
        image_size=image_size,
        task="arkit_synth",
    )
    dataset = MultiViewJsonDataset(cfg)
    max_scenes = min(max_scenes, len(dataset))
    return [dataset[i] for i in range(max_scenes)]


def build_tokenizer(model_name: str, tokenizer_path: Optional[str] = None):
    path = tokenizer_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"])
    tokenizer.padding_side = "left"
    return tokenizer


def _insert_vision_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    inputs_embeds: torch.Tensor,
    vis_tokens: torch.Tensor,
    image_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace the <image> token with the full visual token span, expanding the
    sequence and attention mask accordingly.
    """
    positions = (input_ids == image_token_id).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return inputs_embeds, attention_mask

    # Assume one <image> token per sample (matching training prompts).
    b = positions[0, 0]
    pos = positions[0, 1]
    vis_len = vis_tokens.shape[1]

    prefix = inputs_embeds[:, :pos, :]
    suffix = inputs_embeds[:, pos + 1 :, :]
    new_inputs = torch.cat([prefix, vis_tokens, suffix], dim=1)

    attn_prefix = attention_mask[:, :pos]
    attn_suffix = attention_mask[:, pos + 1 :]
    vis_attn = torch.ones(
        (attention_mask.size(0), vis_len),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    new_mask = torch.cat([attn_prefix, vis_attn, attn_suffix], dim=1)
    return new_inputs, new_mask


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    tokenizer,
    samples: List[Dict],
    device: torch.device,
    image_size: int,
    max_new_tokens: int = 256,
    output_path: Optional[Path] = None,
    compute_metrics: bool = True,
) -> Tuple[List[Dict], Optional[Dict[str, float]]]:
    """Run autoregressive generation for each ARKit sample.

    Returns:
        results: list of per-sample dicts including prediction and reference.
        metrics: optional summary dict (e.g., rough exact-match accuracy).
    """
    results: List[Dict] = []
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate existing file
        output_path.write_text("", encoding="utf-8")

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    total_with_ref = 0
    total_exact_match = 0

    for idx, sample in enumerate(samples):
        images = sample["images"]
        question = sample.get("question") or sample.get("instruction") or ""
        reference = sample.get("answer") or sample.get("action_json")
        prompt = f"{question}\n<image>\n"

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Encode images via VGGT and inject tokens at <image> position
        if hasattr(model, "encode_images"):
            transform = build_default_transform(image_size)
            views = torch.stack([transform(img) for img in images], dim=0).unsqueeze(0)
            vis_tokens = model.encode_images(views.to(device))
            text_dtype = model.text_model.get_input_embeddings().weight.dtype
            inputs_embeds = model.text_model.get_input_embeddings()(input_ids).to(text_dtype)
            inputs_embeds, attention_mask = _insert_vision_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                vis_tokens=vis_tokens.to(text_dtype),
                image_token_id=image_token_id,
            )
            generated = model.text_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        else:
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        record = {
            "index": idx,
            "scene_id": sample.get("scene_id"),
            "question": question,
            "prediction": text,
            "reference": reference,
        }
        results.append(record)

        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Simple exact string match metric if reference available
        if compute_metrics and reference is not None:
            total_with_ref += 1
            if isinstance(reference, (dict, list)):
                ref_str = json.dumps(reference, sort_keys=True)
            else:
                ref_str = str(reference)
            if ref_str.strip() == text.strip():
                total_exact_match += 1

        print(f"[{idx}] {question}")
        print(f" → {text}")
        if reference is not None:
            print(f"   (reference) {reference}")
        print("-" * 80)

    metrics: Optional[Dict[str, float]] = None
    if compute_metrics and total_with_ref > 0:
        metrics = {
            "num_samples": len(samples),
            "num_with_reference": total_with_ref,
            "exact_match": total_exact_match / float(total_with_ref),
        }
        print(
            f"\nSummary over {total_with_ref} samples with reference:"
            f" exact_match = {metrics['exact_match']:.3f}"
        )

    return results, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARKit / RoomPlan inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2_arkit.yaml",
        help="Stage config used for model reconstruction.",
    )
    parser.add_argument(
        "--arkit_glob",
        type=str,
        default="data/processed/arkit_synth/*.json",
        help="Glob pattern for ARKit synthetic JSON files.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional path to a trained checkpoint directory (e.g. ckpts/stage2_arkit/step_10000).",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=10,
        help="Number of scenes to run (first N).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="ckpts/arkit_infer/arkit_predictions.jsonl",
        help="Where to save predictions + references (JSONL).",
    )
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="Disable simple exact-match metric computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda, cuda:0, cpu). Defaults to cuda if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_cfg = load_yaml(args.config)
    model_name = stage_cfg["model"]["name_or_path"]
    tokenizer_path = stage_cfg["model"].get("tokenizer_path")

    tokenizer = build_tokenizer(model_name, tokenizer_path)
    model, full_cfg = build_model_from_config(args.config, device=device)
    # Ensure embeddings cover any tokenizer expansion (e.g., added <image> token).
    model.text_model.resize_token_embeddings(len(tokenizer))
    load_checkpoint_if_available(model, args.checkpoint_dir)

    data_cfg = full_cfg["data"]
    num_views = data_cfg.get("num_views", 10)
    image_size = data_cfg.get("image_size", 448)

    samples = load_arkit_samples(
        args.arkit_glob,
        args.num_scenes,
        num_views=num_views,
        image_size=image_size,
    )

    output_path = Path(args.output_jsonl) if args.output_jsonl else None

    run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        image_size=image_size,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
        compute_metrics=not args.no_metrics,
    )


if __name__ == "__main__":
    main()
