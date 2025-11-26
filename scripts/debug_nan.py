#!/usr/bin/env python3
"""Debug script to find source of NaN loss."""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataio.dataset_builder import MultiViewJsonlDataset
from src.dataio.collate_multiview import MultiViewCollator
from src.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig
from src.models.projector_perceiver import PerceiverConfig
from transformers import AutoTokenizer


def check_tensor(name, tensor, step=""):
    """Check if tensor contains NaN or Inf."""
    if tensor is None:
        print(f"{step}{name}: None")
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item() if not has_nan else float('nan')
    max_val = tensor.max().item() if not has_inf else float('inf')
    mean_val = tensor.float().mean().item() if not (has_nan or has_inf) else float('nan')
    
    status = "✓" if not (has_nan or has_inf) else "✗"
    print(f"{step}{status} {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
          f"min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, "
          f"nan={has_nan}, inf={has_inf}")


def main():
    print("=" * 80)
    print("NaN Debug Script")
    print("=" * 80)
    
    # Load one batch
    print("\n1. Loading data...")
    dataset = MultiViewJsonlDataset(
        jsonl_path="data/processed/scanqa/train.jsonl",
        num_views=6,
        image_size=378,
    )
    print(f"Dataset size: {len(dataset)}")
    
    collator = MultiViewCollator(
        image_size=378,
        tokenizer=None,  # Will set after loading model
        max_length=512,
    )
    
    # Load model
    print("\n2. Loading model...")
    config = VisionLanguageConfig(
        text_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        vision_ckpt_dir="third_party/vggt",
        num_vis_tokens=64,
        geom_tokens=0,
        projector_cfg=PerceiverConfig(
            depth=6,
            num_latents=64,
            dim_head=128,
            num_heads=16,
            ff_mult=4,
        ),
        freeze_vision=True,
        dtype="bfloat16",
    )
    
    model = VGGTQwen3VLM(config)
    collator.tokenizer = model.tokenizer
    model = model.to("cuda").to(torch.bfloat16)
    model.eval()  # Start in eval mode
    
    print(f"Model dtype: {model.model_dtype}")
    print(f"Text model dtype: {model.text_model.dtype}")
    
    # Get one sample
    print("\n3. Processing one sample...")
    sample = dataset[0]
    batch = collator([sample])
    
    print("\n4. Checking batch tensors...")
    check_tensor("pixel_values", batch["pixel_values"])
    check_tensor("input_ids", batch["input_ids"])
    check_tensor("attention_mask", batch["attention_mask"])
    check_tensor("labels", batch["labels"])
    if batch.get("geom_token"):
        for k, v in batch["geom_token"].items():
            check_tensor(f"geom_token[{k}]", v)
    
    # Move to GPU
    images = batch["pixel_values"].to("cuda")
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    labels = batch["labels"].to("cuda")
    geom_token = batch.get("geom_token")
    if geom_token:
        geom_token = {k: v.to("cuda") for k, v in geom_token.items()}
    
    print("\n5. Checking GPU tensors...")
    check_tensor("images (GPU)", images)
    
    # Step through model
    print("\n6. Encoding images...")
    with torch.no_grad():
        B, V = images.shape[:2]
        images_flat = images.view(B * V, *images.shape[2:])
        check_tensor("images_flat", images_flat, "  ")
        
        images_converted = images_flat.to(dtype=model.model_dtype)
        check_tensor("images_converted", images_converted, "  ")
        
        # Vision encoder
        print("  Running vision encoder...")
        agg = model.vision_model.aggregator(images_converted)
        check_tensor("vision_agg", agg, "  ")
        
        # Projector
        agg_reshaped = agg.view(B, V, -1, agg.shape[-1])[:, :, :model.num_vis_tokens, :]
        agg_reshaped = agg_reshaped.reshape(B, -1, agg_reshaped.shape[-1])
        check_tensor("agg_reshaped", agg_reshaped, "  ")
        
    print("  Running projector (WITH gradients)...")
    model.train()  # Enable gradients for projector
    proj = model.projector(agg_reshaped)
    check_tensor("projector_output", proj, "  ")
    
    vis_tokens = proj.to(model.text_model.dtype)
    check_tensor("vis_tokens", vis_tokens, "  ")
    
    # Get text embeddings
    print("\n7. Getting text embeddings...")
    with torch.no_grad():
        inputs_embeds = model.text_model.get_input_embeddings()(input_ids)
        check_tensor("inputs_embeds", inputs_embeds, "  ")
        
        # Replace image tokens
        image_token_id = model.tokenizer.convert_tokens_to_ids("<image>")
        image_positions = (input_ids == image_token_id).nonzero(as_tuple=False)
        print(f"  Found {len(image_positions)} image token positions")
        
        for batch_idx, pos in image_positions:
            pos = pos.item()
            span = vis_tokens[batch_idx]
            end = pos + span.size(0)
            inputs_embeds[batch_idx, pos:end, :] = span
            attention_mask[batch_idx, pos:end] = 1
            labels[batch_idx, pos:end] = -100
        
        check_tensor("inputs_embeds (after replace)", inputs_embeds, "  ")
    
    # Forward pass
    print("\n8. Running forward pass...")
    model.train()
    outputs = model.text_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    check_tensor("logits", outputs.logits, "  ")
    print(f"  Loss: {outputs.loss.item()}")
    
    if torch.isnan(outputs.loss):
        print("\n❌ LOSS IS NaN!")
        
        # Check which logits are NaN
        nan_mask = torch.isnan(outputs.logits)
        print(f"  NaN logits count: {nan_mask.sum().item()} / {outputs.logits.numel()}")
        
        if nan_mask.any():
            nan_positions = nan_mask.nonzero(as_tuple=False)
            print(f"  First few NaN positions: {nan_positions[:5]}")
    else:
        print(f"\n✓ Loss is valid: {outputs.loss.item():.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
