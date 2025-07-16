"""
internimage.py

Placeholder backbone for InternImage. Requires the `InternImage` package.
"""

from typing import Callable, Tuple

import torch

from prismatic.models.backbones.vision.base_vision import VisionBackbone


class InternImageBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)
        try:
            pass
        except Exception as e:  # pragma: no cover - requires external package
            raise ImportError("InternImage package is required for this backbone") from e
        raise NotImplementedError("InternImage support is not implemented in this environment")

    def get_fsdp_wrapping_policy(self) -> Callable:
        return lambda module: False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return 0

    @property
    def num_patches(self) -> int:
        return 0

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.float32
