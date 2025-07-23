import torch
import torch.nn as nn

class MemoryModule(nn.Module):
    def __init__(self, embed_dim, memory_dim, memory_type="gru"):
        super().__init__()
        self.memory_type = memory_type
        self.memory_dim = memory_dim
        if memory_type == "gru":
            self.cell = nn.GRUCell(embed_dim, memory_dim)
            self.to_embed = nn.Linear(memory_dim, embed_dim)
        elif memory_type == "linear":
            self.cell = nn.Linear(embed_dim, memory_dim)
            self.to_embed = nn.Linear(memory_dim, embed_dim)
        else:
            self.cell = None
            self.to_embed = None

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.memory_dim, device=device)

    def forward(self, hidden, state):
        if self.memory_type == "gru":
            return self.cell(hidden, state)
        elif self.memory_type == "linear":
            return torch.tanh(self.cell(hidden))
        else:
            return state


def test_gru_memory_update():
    mem = MemoryModule(embed_dim=4, memory_dim=2, memory_type="gru")
    state = mem.init_state(1, torch.device("cpu"))
    hidden = torch.randn(1, 4)
    new_state = mem(hidden, state)
    assert new_state.shape == (1, 2)
    newer_state = mem(hidden, new_state)
    assert not torch.allclose(new_state, newer_state)


class PlanningDecoupler(nn.Module):
    def __init__(self, embed_dim, plan_dim, decoupler_type="mlp"):
        super().__init__()
        self.decoupler_type = decoupler_type
        self.plan_dim = plan_dim
        if decoupler_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        elif decoupler_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
            )
        elif decoupler_type == "avg_pooling":
            self.encoder = None
        elif decoupler_type == "none":
            self.encoder = None
            self.cell = None
            self.to_embed = None
            return
        else:
            raise ValueError
        self.cell = nn.GRUCell(embed_dim, plan_dim)
        self.to_embed = nn.Linear(plan_dim, embed_dim)

    def init_state(self, bsz, device):
        if self.decoupler_type == "none":
            return None
        return torch.zeros(bsz, self.plan_dim, device=device)

    def _encode(self, hs):
        if self.decoupler_type == "transformer":
            return self.encoder(hs).mean(dim=1)
        elif self.decoupler_type == "mlp":
            return self.encoder(hs.mean(dim=1))
        else:
            return hs.mean(dim=1)

    def forward(self, hs, state):
        if self.decoupler_type == "none":
            return state
        enc = self._encode(hs)
        return self.cell(enc, state)


def test_planning_decoupler():
    dec = PlanningDecoupler(embed_dim=8, plan_dim=4, decoupler_type="mlp")
    state = dec.init_state(2, torch.device("cpu"))
    hidden = torch.randn(2, 3, 8)
    new_state = dec(hidden, state)
    assert new_state.shape == (2, 4)
