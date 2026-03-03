"""Generic QA inference for ScanQA/SQA3D using VGGT-Qwen3.

Saves predictions alongside references for quick quality checks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import random
from transformers import AutoTokenizer
try:
    # Older Transformers versions expose load_sharded_checkpoint here.
    from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict as hf_load_state_dict
except Exception:  # pragma: no cover - fallback for newer versions
    from transformers.modeling_utils import load_state_dict as hf_load_state_dict  # type: ignore
    load_sharded_checkpoint = None  # type: ignore[misc,assignment]

from vggt_qwen3.dataio.dataset_builder import DatasetConfig, MultiViewJsonDataset
from vggt_qwen3.dataio.collate_multiview import build_default_transform
from vggt_qwen3.models.projector_perceiver import PerceiverConfig
from vggt_qwen3.models.vggt_qwen3_vlm import VisionLanguageConfig


DATASET_DEFAULTS: Dict[str, Dict[str, str]] = {
    "scanqa": {
        "glob": "data/processed/scanqa/test_split.jsonl",
        "output": "outputs/qa/scanqa/scanqa_predictions_test.jsonl",
    },
    "sqa3d": {
        "glob": "data/processed/sqa3d/test_split.jsonl",
        "output": "outputs/qa/sqa3d/sqa3d_predictions_test.jsonl",
    },
}


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
    from vggt_qwen3.models.vggt_qwen3_vlm import VGGTQwen3VLM

    model = VGGTQwen3VLM(vlm_cfg).to(device)
    model.eval()
    return model, cfg


def load_checkpoint_if_available(
    model: torch.nn.Module,
    ckpt_dir: Optional[str],
    allow_base_fallback: bool = False,
) -> Dict[str, object]:
    """
    Load a fine-tuned checkpoint into `model` if available.

    Returns a small summary dict:
      {
        "requested_path": Optional[str],
        "resolved_path": Optional[str],
        "used_base_weights": bool,
        "missing_keys": int,
        "unexpected_keys": int,
      }
    """
    summary: Dict[str, object] = {
        "requested_path": ckpt_dir,
        "resolved_path": None,
        "used_base_weights": False,
        "missing_keys": 0,
        "unexpected_keys": 0,
    }

    if not ckpt_dir:
        # No explicit checkpoint requested: run with base weights.
        summary["used_base_weights"] = True
        return summary

    path = Path(ckpt_dir)
    summary["resolved_path"] = str(path)
    if not path.exists():
        # Explicit path but directory is missing: do not silently fall back.
        if not allow_base_fallback:
            # Suggest nearby checkpoints (if any) under ckpts/.
            candidates_root = Path("ckpts")
            existing: List[str] = []
            if candidates_root.exists():
                for p in sorted(candidates_root.glob("**/*")):
                    if not p.is_dir():
                        continue
                    if (p / "pytorch_model_fp32").exists() or (p / "zero_to_fp32.py").exists():
                        existing.append(str(p))
            msg_lines = [
                f"Checkpoint directory '{path}' does not exist and --allow_base_fallback is not set.",
                "Refusing to silently fall back to base HF weights.",
                "",
                "Actionable suggestions:",
                f"  - Double-check the path passed to --checkpoint_dir (current: {ckpt_dir!r}).",
                "  - If you just finished training, point --checkpoint_dir to the training output root,",
                "    e.g. 'ckpts/stage1_3d' or 'ckpts/stage1_3d_debug'.",
                "  - If you want to run with base weights only, re-run with --allow_base_fallback.",
            ]
            if existing:
                msg_lines.append("")
                msg_lines.append("Known checkpoint roots under 'ckpts/':")
                for c in existing[:10]:
                    msg_lines.append(f"  - {c}")
                if len(existing) > 10:
                    msg_lines.append(f"  (and {len(existing) - 10} more ...)")
            raise FileNotFoundError("\n".join(msg_lines))

        print(
            f"⚠️  Checkpoint directory {path} does not exist; "
            "running with base HF weights (--allow_base_fallback)."
        )
        summary["used_base_weights"] = True
        return summary

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
                weight_map = idx.get("weight_map", {}) or {}
                # HF-style index: param_name -> shard_filename
                shard_files = sorted(set(weight_map.values()))
            except Exception:
                # Fallback: just grab all bin files in the folder.
                shard_files = sorted(str(p.name) for p in candidate.glob("*.bin"))

            print(
                f"🔄 Loading sharded checkpoint from {candidate} "
                f"(index found, {len(shard_files)} shard files)"
            )

            if load_sharded_checkpoint is not None:
                try:
                    load_sharded_checkpoint(model, candidate, strict=False)
                    print(
                        "   ✔️ Loaded sharded checkpoint via load_sharded_checkpoint (strict=False)"
                    )
                    # We don't have per-key stats here; report as unknown.
                    summary["missing_keys"] = -1
                    summary["unexpected_keys"] = -1
                except Exception as e:
                    print(f"⚠️  load_sharded_checkpoint failed ({e}); falling back to manual shard loading")
                    last_missing: List[str] = []
                    last_unexpected: List[str] = []
                    for shard_name in shard_files:
                        shard_path = candidate / shard_name
                        print(f"   → Loading shard {shard_path}")
                        state = torch.load(shard_path, map_location="cpu")
                        missing, unexpected = model.load_state_dict(state, strict=False)
                        last_missing, last_unexpected = missing, unexpected
                        print(
                            f"     Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                        )
                    summary["missing_keys"] = len(last_missing)
                    summary["unexpected_keys"] = len(last_unexpected)
            else:
                # For newer Transformers where load_sharded_checkpoint is absent,
                # manually iterate over the shards.
                print("   ℹ️ transformers.load_sharded_checkpoint not available; loading shards one-by-one.")
                last_missing = []
                last_unexpected = []
                for shard_name in shard_files:
                    shard_path = candidate / shard_name
                    print(f"   → Loading shard {shard_path}")
                    state = torch.load(shard_path, map_location="cpu")
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    last_missing, last_unexpected = missing, unexpected
                    print(
                        f"     Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                    )
                summary["missing_keys"] = len(last_missing)
                summary["unexpected_keys"] = len(last_unexpected)

            model.to(orig_device)
            summary["used_base_weights"] = False
            return summary

    # DeepSpeed ZeRO checkpoints (no HF fp32 export yet).
    if (path / "zero_to_fp32.py").exists() or (path / "pytorch_model").exists():
        try:
            from deepspeed.utils.zero_to_fp32 import (  # type: ignore[import]
                get_fp32_state_dict_from_zero_checkpoint,
            )
        except Exception as exc:  # pragma: no cover - depends on deepspeed install
            model.to(orig_device)
            raise RuntimeError(
                f"Checkpoint at '{path}' appears to be a DeepSpeed ZeRO checkpoint "
                "but `deepspeed` is not available.\n\n"
                "Install deepspeed in your environment, or run the offline converter:\n"
                f"  python {path}/zero_to_fp32.py {path} {path}/pytorch_model_fp32 --safe_serialization\n"
                "and then point --checkpoint_dir to the generated fp32 folder."
            ) from exc

        print(f"🔄 Converting DeepSpeed ZeRO checkpoint at {path} to fp32 state_dict")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(path))  # already on CPU
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"   Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        model.to(orig_device)
        summary["missing_keys"] = len(missing)
        summary["unexpected_keys"] = len(unexpected)
        summary["used_base_weights"] = False
        return summary

    weight_files: List[Path] = []
    if merged_root.exists():
        weight_files = [merged_root] if merged_root.is_file() else sorted(merged_root.glob("*.bin"))
    elif legacy_merged_root.exists() and legacy_merged_root.is_dir():
        weight_files = sorted(legacy_merged_root.glob("*.bin"))
    else:
        weight_files = list(path.glob("*.bin")) + list(path.glob("*.safetensors"))
    if not weight_files:
        model.to(orig_device)
        if not allow_base_fallback:
            raise RuntimeError(
                f"No model weights found in '{path}'. Expected either a merged fp32 "
                "checkpoint (pytorch_model_fp32/...) or a DeepSpeed ZeRO checkpoint. "
                "Refusing to silently fall back to base weights. If you really want "
                "to ignore the checkpoint, rerun with --allow_base_fallback."
            )
        print(
            f"⚠️  No model weights found in {path}; "
            "using base HF weights (--allow_base_fallback)."
        )
        summary["used_base_weights"] = True
        return summary

    print(f"🔄 Loading checkpoint weights from {weight_files[0]} (no index; {len(weight_files)} candidate files)")
    state = torch.load(weight_files[0], map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"   Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.to(orig_device)
    summary["missing_keys"] = len(missing)
    summary["unexpected_keys"] = len(unexpected)
    summary["used_base_weights"] = False
    return summary


def build_tokenizer(model_name: str, tokenizer_path: Optional[str] = None):
    path = tokenizer_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"])
    tokenizer.padding_side = "left"
    return tokenizer


def log_model_and_checkpoint_summary(
    model: torch.nn.Module,
    model_name: str,
    ckpt_info: Dict[str, object],
) -> None:
    """Print a one-line summary of the base model and checkpoint coverage."""

    def _fmt_params(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.2f}K"
        return str(n)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    used_base = bool(ckpt_info.get("used_base_weights", False))
    requested = ckpt_info.get("requested_path")
    resolved = ckpt_info.get("resolved_path")

    if used_base and (requested or resolved):
        ckpt_desc = f"base-only (requested={resolved or requested})"
    elif used_base:
        ckpt_desc = "base-only"
    else:
        ckpt_desc = str(resolved or requested or "unknown")

    missing = int(ckpt_info.get("missing_keys", 0) or 0)
    unexpected = int(ckpt_info.get("unexpected_keys", 0) or 0)
    missing_str = "n/a" if missing < 0 else str(missing)
    unexpected_str = "n/a" if unexpected < 0 else str(unexpected)

    print(
        "[model] "
        f"base={model_name} | "
        f"checkpoint={ckpt_desc} | "
        f"missing_keys={missing_str} unexpected_keys={unexpected_str} | "
        f"trainable_params={_fmt_params(trainable_params)}/{_fmt_params(total_params)}"
    )


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


def _postprocess_short_answer(text: str) -> str:
    """Heuristic post-processing for short-answer mode."""
    import re

    if not text:
        return text

    line = text.strip().splitlines()[0]
    # Strip simple surrounding quotes.
    line = line.strip().strip("\"'")

    # Remove verbose prefixes like "The answer is:" / "Answer:" etc.
    line = re.sub(
        r"^\s*(the\s+answer\s+is|the\s+correct\s+answer\s+is|answer|it\s+is|it's)[:\s-]*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    return line.strip()


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    tokenizer,
    samples: List[Dict],
    device: torch.device,
    image_size: int,
    max_new_tokens: int = 32,
    output_path: Optional[Path] = None,
    dataset_name: str = "qa",
    debug: bool = False,
    events_path: Optional[Path] = None,
    short_answer_only: bool = False,
    debug_one: Optional[int] = None,
    generation_kwargs: Optional[Dict[str, object]] = None,
    verbose_samples: int = 0,
) -> List[Dict]:
    logger = logging.getLogger("vggt_qwen3.qa_inference")

    results: List[Dict] = []
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")

    stats: Dict[str, int] = {
        "total_samples_seen": len(samples),
        "total_batches": len(samples),  # one sample per forward
        "total_generations_attempted": 0,
        "total_predictions_written": 0,
        "total_non_empty_predictions": 0,
    }

    def log_event(event: str, payload: Dict) -> None:
        if events_path is None:
            return
        record = {"event": event, "dataset": dataset_name, **payload}
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    transform = build_default_transform(image_size)

    # Default decoding policy: deterministic QA, short answers.
    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.1,
        }
    # For logging we want an explicit int.
    effective_max_new = int(generation_kwargs.get("max_new_tokens", max_new_tokens))

    logger.info(
        "Starting inference: dataset=%s, num_samples=%d, max_new_tokens=%d",
        dataset_name,
        len(samples),
        effective_max_new,
    )
    log_event(
        "start",
        {
            "num_samples": len(samples),
            "image_size": image_size,
            "max_new_tokens": max_new_tokens,
        },
    )

    for idx, sample in enumerate(samples):
        stats["total_generations_attempted"] += 1

        do_debug = debug or (debug_one is not None and idx == debug_one)

        images = sample["images"]
        question = sample.get("question") or sample.get("instruction") or ""
        reference = sample.get("answer")
        task = sample.get("task") or dataset_name
        situation = sample.get("situation")

        # Dataset-specific prompt templates.
        if task == "sqa3d" and situation:
            user_content = (
                f"Situation: {situation}\n"
                f"Question: {question}\n"
                "<image>\n"
                "Answer with a short, concrete phrase grounded in the image."
            )
        else:
            user_content = (
                f"{question}\n<image>\n"
                "Answer with a short, concrete phrase grounded in the image."
            )

        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if do_debug:
            print(f"\n{'='*80}")
            print(f"[Sample {idx}] Dataset: {task}  Scene: {sample.get('scene_id')}")
            print(f"Question: {question}")
            if situation:
                print(f"Situation: {situation}")
            print(f"Number of images: {len(images)}")
            print("Prompt (chat template) preview:")
            print(f"  first 300 chars: {repr(prompt[:300])}")
            if len(prompt) > 300:
                print(f"  last 300 chars:  {repr(prompt[-300:])}")

        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if do_debug:
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Image token ID: {image_token_id}")
            print(f"'<image>' in prompt: {'<image>' in prompt}")
            num_image_tokens = int((input_ids == image_token_id).sum().item())
            print(f"Count of '<image>' tokens in input_ids: {num_image_tokens}")
            decoded_prompt = tokenizer.decode(
                input_ids[0], skip_special_tokens=False
            )
            print(f"Decoded prompt (no skip_special): {repr(decoded_prompt)}")

        vis = torch.stack([transform(img) for img in images], dim=0).unsqueeze(0).to(
            device
        )
        vis_tokens = model.encode_images(vis)

        if do_debug:
            print(f"Visual tensor shape: {vis.shape}")
            print(f"Vision tokens shape: {vis_tokens.shape}")
            # Basic stats on conditioning tensors.
            for name, tensor in (("vis", vis), ("vis_tokens", vis_tokens)):
                t = tensor.detach().float()
                mean = t.mean().item()
                std = t.std().item()
                min_val = t.min().item()
                max_val = t.max().item()
                frac_zeros = float((t == 0).float().mean().item())
                print(
                    f"{name}: shape={tuple(tensor.shape)}, "
                    f"mean={mean:.4e}, std={std:.4e}, "
                    f"min={min_val:.4e}, max={max_val:.4e}, "
                    f"zero_fraction={frac_zeros:.4f}"
                )

        # Conditioning must not silently vanish.
        if vis_tokens.numel() == 0:
            raise RuntimeError(
                f"Sample {idx} produced empty vision tokens; "
                "VGGT conditioning appears to be missing."
            )
        if float(vis_tokens.detach().abs().sum().item()) == 0.0:
            raise RuntimeError(
                f"Sample {idx} produced all-zero vision tokens; "
                "VGGT conditioning appears to have collapsed."
            )

        text_dtype = model.text_model.get_input_embeddings().weight.dtype
        inputs_embeds = model.text_model.get_input_embeddings()(input_ids).to(
            text_dtype
        )

        if do_debug:
            print(f"Initial inputs_embeds shape: {inputs_embeds.shape}")

        inputs_embeds, attention_mask = _insert_vision_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            vis_tokens=vis_tokens.to(text_dtype),
            image_token_id=image_token_id,
        )

        if do_debug:
            print(
                f"After vision insertion - inputs_embeds shape: {inputs_embeds.shape}"
            )
            print(
                f"After vision insertion - attention_mask shape: {attention_mask.shape}"
            )

        prompt_len = inputs_embeds.shape[1]
        if do_debug:
            print(
                f"Prompt length: {prompt_len}, generating max {effective_max_new} tokens..."
            )
            print(
                f"EOS token ID: {tokenizer.eos_token_id}, PAD token ID: {tokenizer.pad_token_id}"
            )

        generated = model.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

        if do_debug:
            print(f"Generated shape: {generated.shape}")
            full_ids = generated[0].tolist()
            print(f"Full generated IDs: {full_ids}")
        else:
            full_ids = generated[0].tolist()

        # When using `inputs_embeds`, HF generate() may return only the newly
        # generated tokens (prompt not included). Decode the generated tokens
        # directly, falling back to a more permissive decode if needed.
        if generated.shape[1] > prompt_len:
            new_tokens = generated[0, prompt_len:]
        else:
            new_tokens = generated[0]

        decoded_primary = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        decoded_fallback = None
        if not decoded_primary:
            decoded_fallback = tokenizer.decode(
                new_tokens, skip_special_tokens=False
            ).strip()
            text = decoded_fallback or decoded_primary
        else:
            text = decoded_primary

        if do_debug:
            print(f"New token IDs: {new_tokens.tolist()}")
            tokens_str = tokenizer.convert_ids_to_tokens(new_tokens.tolist())
            print(f"New tokens: {tokens_str}")
            print(f"Decoded text (skip_special=True): {repr(decoded_primary)}")
            if decoded_fallback is not None:
                print(
                    f"Decoded text (skip_special=False): {repr(decoded_fallback)}"
                )

        if short_answer_only:
            text = _postprocess_short_answer(text)

        stats["total_predictions_written"] += 1
        if text.strip():
            stats["total_non_empty_predictions"] += 1

        record = {
            "index": idx,
            "task": task,
            "scene_id": sample.get("scene_id"),
            "question_id": sample.get("question_id"),
            "question": question,
            "prediction": text,
            "reference": reference,
        }
        results.append(record)

        if output_path is not None:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if verbose_samples > 0 and idx < verbose_samples:
            print(f"[{idx}] [{task}] {question}")
            print(f" → {text}")
            if reference is not None:
                print(f"   (reference) {reference}")
            print("-" * 80)

        log_event(
            "sample",
            {
                "index": idx,
                "scene_id": sample.get("scene_id"),
                "task": task,
                "has_reference": reference is not None,
                "prediction_empty": not bool(text.strip()),
            },
        )

    logger.info(
        "Finished inference: dataset=%s, samples=%d, generations=%d, non_empty_predictions=%d",
        dataset_name,
        stats["total_samples_seen"],
        stats["total_generations_attempted"],
        stats["total_non_empty_predictions"],
    )
    log_event("summary", stats)

    if stats["total_predictions_written"] == 0 or stats["total_non_empty_predictions"] == 0:
        msg = (
            "Inference completed but produced no non-empty predictions.\n"
            f"  Dataset: {dataset_name}\n"
            f"  Samples seen: {stats['total_samples_seen']}\n"
            f"  Generations attempted: {stats['total_generations_attempted']}\n"
            f"  Predictions written: {stats['total_predictions_written']}\n"
            f"  Non-empty predictions: {stats['total_non_empty_predictions']}\n\n"
            "Likely causes include:\n"
            "  - Data glob matched no files or produced empty JSONL records.\n"
            "  - Tokenizer.decode removed all content (special-tokens-only output).\n"
            "  - Model generated only padding/EOS tokens.\n"
            "  - Output path is unwritable (check filesystem permissions).\n\n"
            "Re-run with --debug and inspect per-sample prompts and token IDs "
            "to pinpoint the issue."
        )
        logger.error(msg.replace("\n", " "))
        raise RuntimeError(msg)

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
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset to run: scanqa, sqa3d, or scanqa+sqa3d (combined). "
            "If unset, --glob/--output_jsonl control inputs/outputs directly."
        ),
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern for QA JSONL files (overrides dataset default when provided).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a trained checkpoint directory (e.g. ckpts/stage1_3d).",
    )
    parser.add_argument(
        "--allow_base_fallback",
        action="store_true",
        help=(
            "If set, allows running with base HF weights when --checkpoint_dir "
            "is missing or contains no loadable weights. By default this is "
            "disabled to avoid silently ignoring a bad checkpoint path."
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help=(
            "Optional cap on the number of records to run. "
            "0 or a negative value means 'use all records'. "
            "When used together with --unique_scenes, the cap "
            "applies to the deduplicated scene set."
        ),
    )
    parser.add_argument(
        "--unique_scenes",
        action="store_true",
        help=(
            "If set, evaluate at most one QA example per scene_id. "
            "By default, every record in the split is evaluated."
        ),
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
        default=None,
        help=(
            "Where to save predictions + references (JSONL). "
            "Defaults to dataset-specific paths under outputs/qa/ when --dataset is set."
        ),
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (CPU, tiny subset, verbose per-sample logging).",
    )
    parser.add_argument(
        "--debug_max_samples",
        type=int,
        default=8,
        help="Maximum number of samples to run in --debug mode.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling-based decoding instead of deterministic greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (only used when --do_sample is set).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (only used when --do_sample is set).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling (only used when --do_sample is set).",
    )
    parser.add_argument(
        "--short_answer_only",
        action="store_true",
        help=(
            "Post-process model outputs to enforce concise short answers "
            "(first line, no leading 'The answer is', etc.)."
        ),
    )
    parser.add_argument(
        "--debug_one",
        type=int,
        default=None,
        help=(
            "If set, print a detailed multimodal debug dump for the given "
            "sample index (0-based) in addition to any --debug output."
        ),
    )
    parser.add_argument(
        "--verbose_samples",
        type=int,
        default=0,
        help=(
            "Print at most this many sample predictions to stdout "
            "(0 disables per-sample printing)."
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        help="Logging level: debug, info, warning, error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    log_level = level_map.get(str(args.log_level).lower(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.debug:
        device = torch.device("cpu")
    else:
        device = (
            torch.device(args.device)
            if args.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    stage_cfg = load_yaml(args.config)
    model_name = stage_cfg["model"]["name_or_path"]
    tokenizer_path = stage_cfg["model"].get("tokenizer_path")

    tokenizer = build_tokenizer(model_name, tokenizer_path)
    model, full_cfg = build_model_from_config(args.config, device=device)
    model.text_model.resize_token_embeddings(len(tokenizer))
    ckpt_info = load_checkpoint_if_available(
        model,
        args.checkpoint_dir,
        allow_base_fallback=args.allow_base_fallback,
    )
    log_model_and_checkpoint_summary(model, model_name, ckpt_info)

    data_cfg = full_cfg["data"]
    num_views = data_cfg.get("num_views", 1)
    image_size = data_cfg.get("image_size", 448)

    # Resolve which dataset(s) to run.
    datasets_to_run: List[Tuple[str, str, str]] = []
    if args.dataset is None:
        glob_pattern = args.glob or DATASET_DEFAULTS["scanqa"]["glob"]
        output_jsonl = args.output_jsonl or "outputs/qa/qa_predictions.jsonl"
        datasets_to_run.append(("qa", glob_pattern, output_jsonl))
    else:
        ds = args.dataset.lower()
        if ds in ("scanqa", "sqa3d"):
            defaults = DATASET_DEFAULTS.get(ds)
            if defaults is None:
                raise ValueError(f"No defaults registered for dataset '{ds}'.")
            glob_pattern = args.glob or defaults["glob"]
            output_jsonl = args.output_jsonl or defaults["output"]
            datasets_to_run.append((ds, glob_pattern, output_jsonl))
        elif ds in ("scanqa+sqa3d", "scanqa_sqa3d", "both", "combined"):
            if args.glob or args.output_jsonl:
                print(
                    "⚠️  Ignoring custom --glob/--output_jsonl in combined mode; "
                    "using dataset defaults for both ScanQA and SQA3D."
                )
            for ds_name in ("scanqa", "sqa3d"):
                defaults = DATASET_DEFAULTS[ds_name]
                datasets_to_run.append((ds_name, defaults["glob"], defaults["output"]))
        else:
            raise ValueError(
                f"Unsupported --dataset value '{args.dataset}'. "
                "Use 'scanqa', 'sqa3d', or 'scanqa+sqa3d'."
            )

    # Build generation kwargs, making sure only supported flags are passed to `generate`.
    gen_kwargs: Dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "num_beams": 1,
        "repetition_penalty": 1.1,
    }
    if args.do_sample:
        if args.temperature is not None:
            gen_kwargs["temperature"] = float(args.temperature)
        if args.top_p is not None:
            gen_kwargs["top_p"] = float(args.top_p)
        if args.top_k is not None:
            gen_kwargs["top_k"] = int(args.top_k)
    else:
        # Deterministic decoding; ignore sampling hyperparameters but warn once.
        if any(v is not None for v in (args.temperature, args.top_p, args.top_k)):
            print(
                "⚠️  Sampling hyperparameters (temperature/top_p/top_k) were provided "
                "but --do_sample is not set; ignoring them and using deterministic decoding."
            )

    for dataset_name, glob_pattern, output_jsonl in datasets_to_run:
        print("\n" + "=" * 80)
        print(f"Running Stage-1 QA inference for dataset: {dataset_name}")
        print(f"  Data glob:   {glob_pattern}")
        print(f"  Output JSONL:{output_jsonl}")
        print("=" * 80)

        ds_cfg = DatasetConfig(
            path_glob=glob_pattern,
            num_views=num_views,
            image_size=image_size,
            task=dataset_name,
        )
        dataset = MultiViewJsonDataset(ds_cfg)

        if len(dataset) == 0:
            raise RuntimeError(
                f"Dataset '{dataset_name}' appears empty for glob '{glob_pattern}'. "
                "Check that test_split.jsonl exists and contains records."
            )

        total_records = len(dataset)
        all_indices = list(range(total_records))

        # ------------------------------------------------------------------
        # Build the list of indices to evaluate.
        # By default we iterate ALL records in deterministic order.
        # Optional flags:
        #   --unique_scenes  → collapse to one record per scene_id.
        #   --num_samples>0  → random subset (with fixed seed) of indices.
        #   --debug          → further cap to --debug_max_samples.
        # ------------------------------------------------------------------
        sampling_notes: List[str] = []

        # Optional unique-scene mode.
        if args.unique_scenes:
            seen_scenes = set()
            unique_indices: List[int] = []
            for idx in all_indices:
                scene_id = dataset.index[idx].get("scene_id")
                key = scene_id if scene_id is not None else f"__idx_{idx}"
                if key in seen_scenes:
                    continue
                seen_scenes.add(key)
                unique_indices.append(idx)
            base_indices = unique_indices
            sampling_notes.append(
                f"unique_scenes=True (reduced from {total_records} to {len(base_indices)} records)"
            )
        else:
            base_indices = all_indices

        # Optional random subsampling.
        limit = args.num_samples if args.num_samples and args.num_samples > 0 else None
        if limit is not None and len(base_indices) > limit:
            rng = random.Random(args.seed)
            shuffled = base_indices.copy()
            rng.shuffle(shuffled)
            selected_indices = shuffled[:limit]
            sampling_notes.append(
                f"num_samples={limit} (random subset with seed={args.seed})"
            )
        else:
            selected_indices = base_indices

        # Optional debug cap (always applied last).
        if args.debug and args.debug_max_samples is not None and args.debug_max_samples > 0:
            if len(selected_indices) > args.debug_max_samples:
                selected_indices = selected_indices[: args.debug_max_samples]
                sampling_notes.append(f"debug_max_samples={args.debug_max_samples}")

        num_selected = len(selected_indices)

        print(f"  Total records in split: {total_records}")
        if sampling_notes:
            print("  Active sampling/filtering:")
            for note in sampling_notes:
                print(f"    - {note}")
        else:
            print("  Active sampling/filtering: none (full split).")
        print(f"  Records selected for inference: {num_selected}")

        # Materialize samples (this is where images are loaded).
        samples = [dataset[i] for i in selected_indices]

        output_path = Path(output_jsonl) if output_jsonl else None
        events_path = (
            output_path.with_suffix(output_path.suffix + ".events.jsonl")
            if output_path is not None
            else None
        )

        results = run_inference(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            device=device,
            image_size=image_size,
            max_new_tokens=args.max_new_tokens,
            output_path=output_path,
            dataset_name=dataset_name,
            debug=args.debug,
            events_path=events_path,
            short_answer_only=args.short_answer_only,
            debug_one=args.debug_one,
            generation_kwargs=gen_kwargs,
            verbose_samples=args.verbose_samples,
        )

        num_predictions = len(results)
        if num_predictions != num_selected:
            raise RuntimeError(
                f"Sanity check failed for dataset '{dataset_name}': "
                f"selected {num_selected} records but wrote {num_predictions} predictions."
            )

        summary_path = str(output_jsonl) if output_jsonl else "<stdout only>"
        print(
            f"✅ Dataset '{dataset_name}': loaded {total_records} records, "
            f"ran inference on {num_selected} records, "
            f"wrote {num_predictions} predictions to {summary_path}."
        )


if __name__ == "__main__":
    main()
