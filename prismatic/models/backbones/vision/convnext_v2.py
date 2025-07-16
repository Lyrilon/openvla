"""
convnext_v2.py

Vision backbone wrapper around ConvNeXtV2 models from TIMM.
"""

from typing import Callable, Tuple

import timm
import torch
from torch import nn
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import LetterboxPad, VisionBackbone

# Registry =>> Supported ConvNeXtV2 Backbones (from TIMM)
CONVNEXT_V2_BACKBONES = {"convnextv2-base": "convnextv2_base"}


class ConvNeXtV2Backbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)
        timm_id = CONVNEXT_V2_BACKBONES[vision_backbone_id]
        # Use features_only to get last stage features
        self.featurizer: nn.Module = timm.create_model(
            timm_id, pretrained=True, features_only=True, out_indices=(3,), num_classes=0
        )
        self.featurizer.eval()
        self.dtype = torch.float32

        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        default_transform = timm.data.create_transform(**self.data_cfg, is_training=False)

        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_transform, Compose)
            assert isinstance(default_transform.transforms[0], Resize)
            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=default_transform.transforms[0].interpolation),
                    *default_transform.transforms[1:],
                ]
            )
        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = default_transform
        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_transform, Compose)
            assert "mean" in self.data_cfg
            fill = tuple(int(x * 255) for x in self.data_cfg["mean"])
            self.image_transform = Compose([LetterboxPad(fill), *default_transform.transforms])
        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        return lambda module: False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.featurizer(pixel_values)[0]  # [B, C, H, W]
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return feats

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.featurizer.feature_info.channels()[-1]

    @property
    def num_patches(self) -> int:
        info = self.featurizer.feature_info.info[-1]
        reduction = info.get("reduction", 32)
        h = self.default_image_size // reduction
        return h * h

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
