import torch
import torch.nn as nn

from prismatic.models.model_IPvla import PlanningDecoupler, PlanningAwareVLA


def test_decoupler_shapes():
    hidden = torch.randn(2, 4, 16)
    for t in ["avg_pooling", "mlp", "transformer"]:
        dec = PlanningDecoupler(16, 8, decoupler_type=t)
        out = dec(hidden)
        assert out.shape == (2, 8)


def test_planning_update():
    class DummyVLA(PlanningAwareVLA):
        def __init__(self):
            super().__init__(
                "dummy",
                nn.Identity(),
                nn.Identity(),
                norm_stats={"dummy": {"action": {"q01": [0], "q99": [1]}}},
                action_tokenizer=None,
                decoupler_type="mlp",
                planning_dim=8,
                memory_dim=4,
            )

        def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None, **kwargs):
            embed = torch.randn(input_ids.size(0), input_ids.size(1), 16, device=input_ids.device)
            self.update_planning_state(embed)
            return torch.tensor(1.0)

    model = DummyVLA()
    model.reset_planning_state(batch_size=1, device=torch.device("cpu"))
    model(torch.ones(1, 2, dtype=torch.long))
    old_state = model.planning_state.clone()
    model(torch.ones(1, 2, dtype=torch.long))
    assert not torch.allclose(old_state, model.planning_state)