"""Training script for IP-VLA models.

This is a lightweight training script that mirrors ``vla-scripts/train.py`` but
operates on episodic datasets and resets the model memory at the beginning of
each episode. The script is intentionally simple and is meant for small scale
experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

import draccus

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load
from prismatic.overwatch import initialize_overwatch
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.models.model_IPvla import IPOpenVLA, PlanningAwareVLA


os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)


class FlatEpisodeDataset(IterableDataset):
    """Flattens an episodic dataset into step dictionaries.

    Each yielded element contains a ``episode_start`` flag that is ``True`` for
    the first step of every episode. This allows the training loop to reset the
    model memory appropriately.
    """

    def __init__(self, episodic_dataset: IterableDataset) -> None:
        self.dataset = episodic_dataset
        self.dataset_length = len(episodic_dataset)

    def __len__(self) -> int:  # pragma: no cover - length used only for logging
        return self.dataset_length

    def __iter__(self):
        for episode in self.dataset:
            for idx, step in enumerate(episode):
                step["episode_start"] = idx == 0
                yield step


@dataclass
class TrainConfig:
    """Configuration for ``train_IPvla.py``."""

    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    run_root_dir: Path = Path("runs")

    pretrained_checkpoint: Optional[Path] = None
    run_id: Optional[str] = None
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    memory_type: str = "gru"
    memory_dim: int = 128
    planning_dim: int = 128
    decoupler_type: str = "mlp"


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("IP-VLA Training :: starting")

    torch.cuda.set_device(device_id := overwatch.local_rank())

    run_dir = cfg.run_root_dir / (cfg.run_id or "ipvla-run")
    os.makedirs(run_dir / "checkpoints", exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN", None)

    base_vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)

    dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=base_vlm.vision_backbone.get_image_transform(),
        tokenizer=base_vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=base_vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=base_vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        episodic=True,
        image_aug=False,
    )

    if overwatch.is_rank_zero():
        save_dataset_statistics(dataset.dataset_statistics, run_dir)

    vlm = PlanningAwareVLA(
        base_vlm.model_id,
        base_vlm.vision_backbone,
        base_vlm.llm_backbone,
        norm_stats=dataset.dataset_statistics,
        action_tokenizer=action_tokenizer,
        memory_type=cfg.memory_type,
        memory_dim=cfg.memory_dim,
        planning_dim=cfg.planning_dim,
        decoupler_type=cfg.decoupler_type,
    )

    optimizer = torch.optim.AdamW(vlm.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    flat_dataset = FlatEpisodeDataset(dataset)
    dataloader = DataLoader(flat_dataset, batch_size=1, collate_fn=lambda x: x[0])

    vlm.train()
    global_step = 0
    for epoch in range(cfg.epochs):
        for step in dataloader:
            if step.get("episode_start", False):
                vlm.reset_memory(batch_size=1, device=device_id)
                if hasattr(vlm, "reset_planning_state"):
                    vlm.reset_planning_state(batch_size=1, device=device_id)
            batch = collator([step])
            batch = {k: (v.to(device_id) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output = vlm(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 100 == 0 and overwatch.is_rank_zero():
                overwatch.info(f"Step {global_step} :: loss {loss.item():.4f}")
            if global_step % 1000 == 0:
                torch.save({"model": vlm.state_dict()}, run_dir / "checkpoints" / f"step-{global_step:06d}.pt")
        torch.save({"model": vlm.state_dict()}, run_dir / "checkpoints" / f"epoch-{epoch:04d}.pt")

    overwatch.info("Training complete")


if __name__ == "__main__":
    train()
