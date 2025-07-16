"""
openclip.py

Vision backbone using OpenCLIP.
"""

from typing import Callable, Tuple

import open_clip
import torch
from torchvision.transforms import Compose, Resize

from prismatic.models.backbones.vision.base_vision import LetterboxPad, VisionBackbone

OPENCLIP_BACKBONES = {"openclip-vit-b16": "ViT-B-16"}


class OpenCLIPBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)
        model_name = OPENCLIP_BACKBONES[vision_backbone_id]
        self.model, self.image_transform = open_clip.create_model_and_transforms(model_name, pretrained=None)
        self.model.visual.output_tokens = True
        self.model.eval()
        self.dtype = torch.float32

        if self.image_resize_strategy == "resize-naive":
            assert isinstance(self.image_transform, Compose)
            assert isinstance(self.image_transform.transforms[0], Resize)
            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose(
                [
                    Resize(target_size, interpolation=self.image_transform.transforms[0].interpolation),
                    *self.image_transform.transforms[1:],
                ]
            )
        elif self.image_resize_strategy == "letterbox":
            assert isinstance(self.image_transform, Compose)
            fill = (0, 0, 0)
            self.image_transform = Compose([LetterboxPad(fill), *self.image_transform.transforms])

    def get_fsdp_wrapping_policy(self) -> Callable:
        return lambda module: False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        global_feat, token_feat = self.model.encode_image(pixel_values)
        return token_feat

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self.model.visual.width

    @property
    def num_patches(self) -> int:
        return self.model.visual.grid_size**2

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype
