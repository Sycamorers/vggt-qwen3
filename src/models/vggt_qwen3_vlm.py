"""VGGT + Perceiver projector + Qwen3 text model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .projector_perceiver import PerceiverConfig, PerceiverProjector


@dataclass
class VisionLanguageConfig:
    text_model_name: str
    vision_ckpt_dir: str
    num_vis_tokens: int = 64
    geom_tokens: int = 0
    projector_cfg: Optional[PerceiverConfig] = None
    freeze_vision: bool = True
    dtype: str = "bfloat16"


class VGGTQwen3VLM(nn.Module):
    """Minimal wrapper around VGGT aggregator and Qwen3 LoRA-enabled text model."""

    def __init__(self, config: VisionLanguageConfig) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        added = 0
        if "<image>" not in self.tokenizer.get_vocab():
            added = self.tokenizer.add_tokens(["<image>"])
        dtype = getattr(torch, config.dtype, torch.bfloat16) if isinstance(config.dtype, str) else torch.bfloat16
        self.text_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        if added:
            self.text_model.resize_token_embeddings(len(self.tokenizer))
        self.vision_model = self._load_vggt(config.vision_ckpt_dir, config.num_vis_tokens)
        for p in self.vision_model.parameters():
            p.requires_grad_(not config.freeze_vision)
        self.projector = PerceiverProjector(
            config.projector_cfg or PerceiverConfig(),
            in_dim=self.vision_model.embed_dim,
            out_dim=self.text_model.config.hidden_size,
        )
        geom_feature_dim = 37  # R(9) + t(3) + K(9) + depth_hist(16)
        self.geom_head = nn.Sequential(
            nn.Linear(geom_feature_dim, self.text_model.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
        )
        self.num_vis_tokens = config.num_vis_tokens
        self.geom_tokens = config.geom_tokens

    def _load_vggt(self, ckpt_dir: str, num_vis_tokens: int) -> nn.Module:
        """Load VGGT aggregator from the external repo.

        When `ckpt_dir` is set to "mock", return a lightweight stub so CPU smoke
        tests can run without external weights or the real VGGT codebase.
        """
        from pathlib import Path
        
        if ckpt_dir == "mock":
            return self._build_mock_vggt(num_vis_tokens)

        # Import VGGT model directly
        from vggt.models.vggt import VGGT
        
        # Initialize VGGT model
        model = VGGT(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            enable_camera=True,
            enable_point=True,
            enable_depth=True,
            enable_track=True
        )
        
        # Load checkpoint
        ckpt_path = Path(ckpt_dir)
        checkpoint_file = ckpt_path / "vggt_1B_commercial.pt"
        
        if checkpoint_file.exists():
            print(f"Loading VGGT checkpoint from {checkpoint_file}")
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ VGGT checkpoint loaded successfully")
        else:
            print(f"⚠️  Warning: Checkpoint file {checkpoint_file} not found, using random initialization")
        
        # Convert model to bfloat16 to match training precision
        model = model.to(dtype=torch.bfloat16)
        model.eval()
        
        # Store embed_dim for projector
        # VGGT aggregator outputs 2 * embed_dim because it concatenates features
        model.embed_dim = 2048  # 2 * 1024, actual output dimension from aggregator
        
        return model

    def _build_mock_vggt(self, num_vis_tokens: int) -> nn.Module:
        """Tiny stand-in that emits zeroed tokens for CPU-only smoke tests."""

        class _MockVGGT(nn.Module):
            def __init__(self, tokens: int, embed_dim: int = 256) -> None:
                super().__init__()
                self.tokens = max(tokens, 1)
                self.embed_dim = embed_dim

            def aggregator(self, images: torch.Tensor) -> torch.Tensor:
                B = images.shape[0]
                return torch.zeros(B, self.tokens, self.embed_dim, device=images.device, dtype=images.dtype)

        return _MockVGGT(tokens=num_vis_tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, V, C, H, W]
        Returns:
            vision_tokens: [B, num_vis_tokens, hidden]
        """
        B, V = images.shape[:2]
        # VGGT aggregator expects [B, S, C, H, W] where S is sequence length (number of views)
        # Our images are already in the right format [B, V, C, H, W]
        # Convert to bfloat16 to match model weights
        images = images.to(dtype=torch.bfloat16)
        
        # Only use the aggregator, not the full forward pass which includes depth/point heads
        # that have dtype issues. We only need the aggregated tokens for the language model.
        aggregated_tokens_list, patch_start_idx = self.vision_model.aggregator(images)
        
        # Get the last iteration's aggregated tokens
        # aggregated_tokens_list is a list of tokens from each iteration
        agg = aggregated_tokens_list[-1]  # [B, S, num_tokens, embed_dim]
        
        # If we just get raw aggregated tokens, reshape appropriately
        if len(agg.shape) == 3:  # [B, total_tokens, dim]
            # Take first num_vis_tokens per sample
            agg = agg[:, :self.num_vis_tokens, :]
        elif len(agg.shape) == 4:  # [B, V, tokens_per_view, dim]
            # Flatten across views and take first num_vis_tokens
            agg = agg.reshape(B, -1, agg.shape[-1])[:, :self.num_vis_tokens, :]
        
        return self.projector(agg)

    def encode_geom(self, geom_token: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not geom_token or self.geom_tokens == 0:
            return None
        feats = torch.cat(
            [
                geom_token.get("R", torch.zeros_like(geom_token["t"])),
                geom_token.get("t", torch.zeros_like(geom_token["R"])),
                geom_token.get("K", torch.zeros_like(geom_token["R"])),
                geom_token.get("depth_hist", torch.zeros_like(geom_token["R"])),
            ],
            dim=-1,
        )
        geom = self.geom_head(feats.mean(dim=1))
        return geom.unsqueeze(1).expand(-1, self.geom_tokens, -1)

    def forward(
        self,
        images: torch.Tensor,
        geom_token: Optional[Dict[str, torch.Tensor]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        vis_tokens = self.encode_images(images)
        geom_tokens = self.encode_geom(geom_token)
        features = vis_tokens if geom_tokens is None else torch.cat([geom_tokens, vis_tokens], dim=1)
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        image_id = self.tokenizer.convert_tokens_to_ids("<image>")
        image_positions = (input_ids == image_id).nonzero(as_tuple=False)
        for batch_idx, pos in image_positions:
            span = features[batch_idx]
            inputs_embeds[batch_idx, pos : pos + span.size(0), :] = span
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss
