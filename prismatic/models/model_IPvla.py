"""Information-Preserving VLA model.

This module defines :class:`IPOpenVLA`, an extension of :class:`OpenVLA`
that maintains a lightweight memory state across forward calls. The memory
is updated with each inference step and can be reset between episodes.
Multiple memory update mechanisms are supported and selected via the
``memory_type`` argument.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.vlms.prismatic import IGNORE_INDEX
from prismatic.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class MemoryModule(nn.Module):
    """Simple memory update module supporting different mechanisms."""

    def __init__(self, embed_dim: int, memory_dim: int, memory_type: str = "gru") -> None:
        super().__init__()
        self.memory_type = memory_type
        self.memory_dim = memory_dim

        if memory_type == "gru":
            self.cell = nn.GRUCell(embed_dim, memory_dim)
            self.to_embed = nn.Linear(memory_dim, embed_dim)
        elif memory_type == "linear":
            self.cell = nn.Linear(embed_dim, memory_dim)
            self.to_embed = nn.Linear(memory_dim, embed_dim)
        elif memory_type == "none":
            self.cell = None
            self.to_embed = None
        else:
            raise ValueError(f"Unsupported memory_type `{memory_type}`")

    def init_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.memory_type == "none":
            return None
        return torch.zeros(batch_size, self.memory_dim, device=device)

    def forward(self, hidden: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.memory_type == "gru":
            return self.cell(hidden, state)
        elif self.memory_type == "linear":
            return torch.tanh(self.cell(hidden))
        else:  # none
            return state

    def to_embedding(self, state: torch.Tensor) -> torch.Tensor:
        if self.memory_type == "none":
            raise RuntimeError("Memory type 'none' does not provide embeddings")
        return self.to_embed(state)


class IPOpenVLA(OpenVLA):
    """OpenVLA variant with a persistent memory state."""

    def __init__(
        self,
        *args,
        memory_type: str = "gru",
        memory_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory = MemoryModule(self.llm_backbone.embed_dim, memory_dim, memory_type)
        self.register_buffer("memory_state", self.memory.init_state(1, torch.device("cpu")), persistent=False)

    def reset_memory(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset the internal memory state."""
        device = device or self.device
        self.memory_state = self.memory.init_state(batch_size, device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.memory.memory_type == "none":
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                **kwargs,
            )

        if self.memory_state is None or self.memory_state.size(0) != input_ids.size(0):
            self.reset_memory(batch_size=input_ids.size(0), device=input_ids.device)
        mem_state = self.memory_state

        # === Replicate `PrismaticVLM.forward` with prepended memory embedding ===
        if input_ids.shape[1] == 1 and kwargs.get("past_key_values") is not None:
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=kwargs.get("past_key_values"),
                inputs_embeds=None,
                labels=None,
                use_cache=kwargs.get("use_cache"),
                output_attentions=kwargs.get("output_attentions"),
                output_hidden_states=kwargs.get("output_hidden_states"),
                return_dict=kwargs.get("return_dict"),
            )
            return output

        multimodal_indices = kwargs.pop("multimodal_indices", None)
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=kwargs.get("past_key_values"),
                inputs_embeds=None,
                labels=labels,
                use_cache=kwargs.get("use_cache"),
                output_attentions=kwargs.get("output_attentions"),
                output_hidden_states=kwargs.get("output_hidden_states"),
                return_dict=kwargs.get("return_dict"),
            )

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        mem_embed = self.memory.to_embedding(mem_state).unsqueeze(1)
        input_embeddings = torch.cat([mem_embed, input_embeddings], dim=1)
        if attention_mask is not None:
            mem_mask = torch.ones(len(input_ids), 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([mem_mask, attention_mask], dim=1)
        if labels is not None:
            mem_label = torch.full((len(input_ids), 1), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([mem_label, labels], dim=1)

        multimodal_embeddings = torch.cat(
            [
                input_embeddings[multimodal_indices, :1, :],
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, 1:, :],
            ],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [
                    attention_mask[multimodal_indices, :1],
                    projected_patch_attention_mask,
                    attention_mask[multimodal_indices, 1:],
                ],
                dim=1,
            )
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
            )

        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        output: CausalLMOutputWithPast = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=kwargs.get("past_key_values"),
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=kwargs.get("use_cache"),
            output_attentions=kwargs.get("output_attentions"),
            output_hidden_states=True,
            return_dict=kwargs.get("return_dict"),
        )

        last_hidden = output.hidden_states[-1][:, -1, :]
        self.memory_state = self.memory(last_hidden, mem_state).detach()

        return output

    @torch.inference_mode()
    def predict_action_with_memory(
        self,
        image: "Image.Image",
        instruction: str,
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Infer a continuous action while maintaining memory state."""
        actions = super().predict_action(image, instruction, unnorm_key=unnorm_key, **kwargs)
        return torch.tensor(actions)
