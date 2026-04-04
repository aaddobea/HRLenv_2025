import torch
from models import HighLevelPolicy, LowLevelPolicy
from environment import LLMHealthcareEnv
from train import train_RLH  # place the training loop from earlier here

env = LLMHealthcareEnv()
high_policy = HighLevelPolicy(input_dim=env.state_dim, hidden_dim=128, goal_dim=env.goal_dim)
low_policy = LowLevelPolicy(state_dim=env.state_dim, goal_dim=env.goal_dim, action_dim=env.action_dim)

train_RLH(high_policy, low_policy, env, episodes=50)
