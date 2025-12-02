"""Generic QA inference for ScanQA/SQA3D using VGGT-Qwen3.

Saves predictions alongside references for quick quality checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import random
from transformers import AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict as hf_load_state_dict

from src.dataio.dataset_builder import DatasetConfig, MultiViewJsonDataset
from src.dataio.collate_multiview import build_default_transform
from src.models.projector_perceiver import PerceiverConfig
from src.models.vggt_qwen3_vlm import VisionLanguageConfig


def load_yaml(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_config(config_path: str, device: torch.device):
    cfg = load_yaml(config_path)
    model_cfg = cfg["model"]
    proj_cfg = load_yaml(model_cfg["projector"])
    vlm_cfg = VisionLanguageConfig(
        text_model_name=model_cfg["name_or_path"],
        vision_ckpt_dir=model_cfg["vision_backbone"],
        num_vis_tokens=model_cfg["num_vis_tokens"],
        geom_tokens=model_cfg.get("geom_tokens", 0),
        projector_cfg=PerceiverConfig(**proj_cfg),
        freeze_vision=True,
        dtype=model_cfg.get("dtype", "bfloat16"),
    )
    from src.models.vggt_qwen3_vlm import VGGTQwen3VLM

    model = VGGTQwen3VLM(vlm_cfg).to(device)
    model.eval()
    return model, cfg


def load_checkpoint_if_available(model: torch.nn.Module, ckpt_dir: Optional[str]) -> None:
    if not ckpt_dir:
        return
    path = Path(ckpt_dir)
    if not path.exists():
        print(f"⚠️  Checkpoint directory {path} does not exist; running with base weights.")
        return

    orig_device = next(model.parameters()).device
    model.to("cpu")

    merged_root = path / "pytorch_model_fp32"
    legacy_merged_root = path / "pytorch_model_fp32.bin"
    for candidate in (merged_root, legacy_merged_root):
        index_file = candidate / "pytorch_model.bin.index.json"
        if candidate.exists() and index_file.exists():
            shard_files = []
            try:
                with index_file.open("r", encoding="utf-8") as f:
                    idx = json.load(f)
                shard_files = sorted({v["filename"] for v in idx.get("weight_map", {}).values()})
            except Exception:
                pass
            print(f"🔄 Loading sharded checkpoint from {candidate} (index found, {len(shard_files)} shard files)")
            try:
                load_sharded_checkpoint(model, candidate, strict=False)
                print("   ✔️ Loaded sharded checkpoint via load_sharded_checkpoint (strict=False)")
            except Exception as e:
                print(f"⚠️  load_sharded_checkpoint failed ({e}); falling back to flat files if any")
                flat_bins = sorted(candidate.glob("*.bin"))
                if flat_bins:
                    print(f"   Trying first flat bin: {flat_bins[0]}")
                    state = torch.load(flat_bins[0], map_location="cpu")
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    print(f"   Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            model.to(orig_device)
            return

    weight_files: List[Path] = []
    if merged_root.exists():
        weight_files = [merged_root] if merged_root.is_file() else sorted(merged_root.glob("*.bin"))
    elif legacy_merged_root.exists() and legacy_merged_root.is_dir():
        weight_files = sorted(legacy_merged_root.glob("*.bin"))
    else:
        weight_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
    if not weight_files:
        print(f"⚠️  No model weights found in {path}; using base HF weights.")
        model.to(orig_device)
        return

    print(f"🔄 Loading checkpoint weights from {weight_files[0]} (no index; {len(weight_files)} candidate files)")
    state = torch.load(weight_files[0], map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"   Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.to(orig_device)


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
    positions = (input_ids == image_token_id).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return inputs_embeds, attention_mask
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
    max_new_tokens: int = 64,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    results: List[Dict] = []
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    transform = build_default_transform(image_size)

    for idx, sample in enumerate(samples):
        images = sample["images"]
        question = sample.get("question") or sample.get("instruction") or ""
        reference = sample.get("answer")
        prompt = f"{question}\n<image>\n"

        print(f"\n{'='*80}")
        print(f"[Sample {idx}] Processing question: {question}")
        print(f"Number of images: {len(images)}")
        print(f"Prompt: {repr(prompt)}")

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Image token ID: {image_token_id}")
        print(f"'<image>' in prompt: {'<image>' in prompt}")

        vis = torch.stack([transform(img) for img in images], dim=0).unsqueeze(0).to(device)
        vis_tokens = model.encode_images(vis)
        print(f"Visual tensor shape: {vis.shape}")
        print(f"Vision tokens shape: {vis_tokens.shape}")

        text_dtype = model.text_model.get_input_embeddings().weight.dtype
        inputs_embeds = model.text_model.get_input_embeddings()(input_ids).to(text_dtype)
        print(f"Initial inputs_embeds shape: {inputs_embeds.shape}")
        
        inputs_embeds, attention_mask = _insert_vision_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            vis_tokens=vis_tokens.to(text_dtype),
            image_token_id=image_token_id,
        )
        print(f"After vision insertion - inputs_embeds shape: {inputs_embeds.shape}")
        print(f"After vision insertion - attention_mask shape: {attention_mask.shape}")

        prompt_len = inputs_embeds.shape[1]
        print(f"Prompt length: {prompt_len}, generating max {max_new_tokens} tokens...")
        print(f"EOS token ID: {tokenizer.eos_token_id}, PAD token ID: {tokenizer.pad_token_id}")
        
        generated = model.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )
        print(f"Generated shape: {generated.shape}")
        print(f"Full generated IDs: {generated[0].tolist()}")
        
        # Decode the full sequence - the tokenizer will handle removing the prompt text
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        # Remove the original question from the decoded text since it gets echoed
        if text.startswith(question):
            text = text[len(question):].strip()
        # Also try to remove common prompt artifacts
        text = text.replace("<image>", "").strip()
        
        # Extract concise answer - take first sentence or phrase before repetition
        # For QA tasks, answers should be short like "brown", "left", "desk", etc.
        if "." in text:
            # Take first sentence
            text = text.split(".")[0].strip()
        
        # If still verbose (e.g., "The table next to you is brown"), 
        # try to extract the key answer (last word/phrase)
        if len(text.split()) > 5:
            # Common pattern: "The X is Y" -> extract Y
            if " is " in text.lower():
                parts = text.lower().split(" is ")
                if len(parts) >= 2:
                    # Get everything after the last "is"
                    text = parts[-1].strip()
        
        print(f"Decoded text: {repr(text)}")

        record = {
            "index": idx,
            "task": sample.get("task"),
            "scene_id": sample.get("scene_id"),
            "question": question,
            "prediction": text,
            "reference": reference,
        }
        results.append(record)
        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[{idx}] {question}")
        print(f" → {text}")
        if reference is not None:
            print(f"   (reference) {reference}")
        print("-" * 80)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScanQA/SQA3D QA inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_3d.yaml",
        help="Stage config used for model reconstruction.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="data/processed/scanqa/*.jsonl",
        help="Glob pattern for QA JSONL files.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a trained checkpoint directory (e.g. ckpts/stage2_3d).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to run (unique scenes, random order).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="ckpts/qa_infer/qa_predictions.jsonl",
        help="Where to save predictions + references (JSONL).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda, cuda:0, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_cfg = load_yaml(args.config)
    model_name = stage_cfg["model"]["name_or_path"]
    tokenizer_path = stage_cfg["model"].get("tokenizer_path")

    tokenizer = build_tokenizer(model_name, tokenizer_path)
    model, full_cfg = build_model_from_config(args.config, device=device)
    model.text_model.resize_token_embeddings(len(tokenizer))
    load_checkpoint_if_available(model, args.checkpoint_dir)

    data_cfg = full_cfg["data"]
    num_views = data_cfg.get("num_views", 1)
    image_size = data_cfg.get("image_size", 448)

    ds_cfg = DatasetConfig(
        path_glob=args.glob,
        num_views=num_views,
        image_size=image_size,
        task="qa",
    )
    dataset = MultiViewJsonDataset(ds_cfg)
    # Randomly pick up to num_samples unique scenes to avoid re-using the full training set.
    rng = random.Random(args.seed)
    all_indices = list(range(len(dataset)))
    rng.shuffle(all_indices)
    seen = set()
    picked: List[int] = []
    for idx in all_indices:
        # Access raw metadata instead of loading images
        scene_id = dataset.index[idx].get("scene_id")
        if scene_id in seen:
            continue
        seen.add(scene_id)
        picked.append(idx)
        if len(picked) >= args.num_samples:
            break
    # Now load the actual samples with images
    samples = [dataset[i] for i in picked]

    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    run_inference(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        device=device,
        image_size=image_size,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
