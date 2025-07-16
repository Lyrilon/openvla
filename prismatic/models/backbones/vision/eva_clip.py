"""
eva_clip.py

Vision backbone for EVA-CLIP models available in TIMM.
"""

from typing import Callable

from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

from prismatic.models.backbones.vision.base_vision import TimmViTBackbone

# Registry of EVA-CLIP models
EVA_CLIP_BACKBONES = {"eva02-b16-224": "eva02_base_patch16_clip_224"}


class EVAClipBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            EVA_CLIP_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )

    def get_fsdp_wrapping_policy(self) -> Callable:
        vit_wrap_policy = _module_wrap_policy(module_classes={VisionTransformer})
        transformer_block_policy = transformer_auto_wrap_policy(transformer_layer_cls={Block})
        return _or_policy(policies=[vit_wrap_policy, transformer_block_policy])
