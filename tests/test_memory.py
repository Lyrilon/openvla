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
