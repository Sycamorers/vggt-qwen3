#!/usr/bin/env python3
"""Simple debug: Check if projector has NaN weights after initialization."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vggt_qwen3_vlm import VGGTQwen3VLM, VisionLanguageConfig
from src.models.projector_perceiver import PerceiverConfig


def check_module_weights(module, name=""):
    """Check if module has NaN weights."""
    has_nan = False
    for param_name, param in module.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ {name}.{param_name} has NaN values!")
            has_nan = True
        elif torch.isinf(param).any():
            print(f"❌ {name}.{param_name} has Inf values!")
            has_nan = True
    return has_nan


print("=" * 80)
print("Checking model initialization...")
print("=" * 80)

config = VisionLanguageConfig(
    text_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    vision_ckpt_dir="third_party/vggt",
    num_vis_tokens=128,
    geom_tokens=8,
    projector_cfg=PerceiverConfig(
        latent_dim=4096,
        num_latents=128,
        num_heads=8,
        num_layers=6,
        ffn_dim=16384,
        dropout=0.1,
    ),
    freeze_vision=True,
    dtype="bfloat16",
)

print("\nLoading model...")
model = VGGTQwen3VLM(config)

print("\nChecking projector weights...")
has_nan = check_module_weights(model.projector, "projector")

print("\nChecking geom_head weights...")
has_nan |= check_module_weights(model.geom_head, "geom_head")

if not has_nan:
    print("\n✅ All weights are valid (no NaN/Inf)")
else:
    print("\n❌ Found NaN/Inf in weights!")

# Check projector layer by layer
print("\nProjector structure:")
for name, module in model.projector.named_modules():
    if len(list(module.children())) == 0:  # Leaf modules
        print(f"  {name}: {module.__class__.__name__}")
        for param_name, param in module.named_parameters(recurse=False):
            print(f"    {param_name}: shape={tuple(param.shape)}, dtype={param.dtype}, "
                  f"mean={param.float().mean():.4f}, std={param.float().std():.4f}")

print("\n" + "=" * 80)
