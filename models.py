import torch
import torch.nn as nn
from torch.distributions import Categorical

class HighLevelPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, goal_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, goal_dim)

    def forward(self, context_seq, hidden=None):
        output, hidden = self.gru(context_seq, hidden)
        goal_logits = self.fc(output[:, -1, :])
        return Categorical(logits=goal_logits), hidden

class LowLevelPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  # only once
        return torch.distributions.Categorical(logits=logits)
