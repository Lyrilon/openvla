"""
moe_router.py

Mixture-of-Experts vision backbone that routes an image to multiple experts and combines their outputs.
This implementation computes gating weights over per-expert global features and returns a weighted sum
of patch features from all experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

from prismatic.models.backbones.vision.base_vision import ImageTransform, VisionBackbone
from prismatic.models.backbones.vision.convnext_v2 import ConvNeXtV2Backbone
from prismatic.models.backbones.vision.dinov2_vit import DinoV2ViTBackbone
from prismatic.models.backbones.vision.eva_clip import EVAClipBackbone
from prismatic.models.backbones.vision.internimage import InternImageBackbone
from prismatic.models.backbones.vision.openclip import OpenCLIPBackbone
from prismatic.models.backbones.vision.siglip_vit import SigLIPViTBackbone


@dataclass
class ExpertImageTransform:
    transforms: Dict[str, ImageTransform]
    is_prismatic: bool = True

    def __call__(self, img, **kwargs) -> Dict[str, torch.Tensor]:
        return {k: t(img, **kwargs) for k, t in self.transforms.items()}


class ExpertRouterBackbone(VisionBackbone):
    def __init__(self, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__("expert-router", image_resize_strategy, default_image_size)
        self.experts: Dict[str, VisionBackbone] = {
            "dino": DinoV2ViTBackbone("dinov2-vit-l", image_resize_strategy, default_image_size),
            "siglip": SigLIPViTBackbone("siglip-vit-so400m", image_resize_strategy, default_image_size),
            "convnext": ConvNeXtV2Backbone("convnextv2-base", image_resize_strategy, default_image_size),
            "openclip": OpenCLIPBackbone("openclip-vit-b16", image_resize_strategy, default_image_size),
            "internimage": InternImageBackbone("internimage-base", image_resize_strategy, default_image_size),
            "eva": EVAClipBackbone("eva02-b16-224", image_resize_strategy, default_image_size),
        }

        self.projections = nn.ModuleDict()
        embed_dims = []
        num_patches = []
        for name, expert in self.experts.items():
            self.projections[name] = nn.Linear(expert.embed_dim, 768, bias=False)
            embed_dims.append(expert.embed_dim)
            num_patches.append(expert.num_patches)

        self.max_patches = max(num_patches)
        self.image_transform = ExpertImageTransform({k: e.get_image_transform() for k, e in self.experts.items()})

        self.gates = nn.ModuleDict({name: nn.Linear(768, 1) for name in self.experts})
        self.softmax = nn.Softmax(dim=1)
        self.dtype = torch.float32

    def get_fsdp_wrapping_policy(self) -> Callable:
        return lambda module: False

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        expert_feats = []
        gate_logits = []
        for name, expert in self.experts.items():
            feats = expert(pixel_values[name])  # [B, N_i, C_i]
            proj = self.projections[name](feats)
            if proj.shape[1] < self.max_patches:
                pad = torch.zeros(
                    proj.size(0),
                    self.max_patches - proj.size(1),
                    proj.size(2),
                    device=proj.device,
                    dtype=proj.dtype,
                )
                proj = torch.cat([proj, pad], dim=1)
            expert_feats.append(proj)
            gate_logits.append(self.gates[name](proj.mean(dim=1)))
        stacked_feats = torch.stack(expert_feats, dim=1)  # [B, E, N, C]
        gates = self.softmax(torch.cat(gate_logits, dim=1))  # [B, E]
        gated_feats = (stacked_feats * gates[:, :, None, None]).sum(dim=1)
        return gated_feats

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return next(iter(self.experts.values())).default_image_resolution

    @property
    def embed_dim(self) -> int:
        return 768

    @property
    def num_patches(self) -> int:
        return self.max_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
