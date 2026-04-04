import torch
import torch.nn.functional as F
import numpy as np
import csv
import os

def compute_entropy(dist):
    probs = dist.probs
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()

def train_RLH(high_policy, low_policy, env, episodes=10, gamma=0.99):
    torch.autograd.set_detect_anomaly(False)

    high_optimizer = torch.optim.Adam(high_policy.parameters(), lr=1e-4)
    low_optimizer = torch.optim.Adam(low_policy.parameters(), lr=1e-4)

    log_path = "training_log.csv"
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "TotalReward", "HighLevelReward", "LowLevelReward", "ActionEntropy"])

    for ep in range(episodes):
        state_seq = []
        state = env.reset()
        hidden = None
        total_reward = 0
        low_level_reward_total = 0

        for i in range(env.high_steps):
            state_seq.append(state)
            state_array = np.array(state_seq, dtype=np.float32)
            state_tensor = torch.tensor(state_array).unsqueeze(0)

            state_tensor = state_tensor.detach()
            with torch.no_grad():
                hidden = hidden.detach() if hidden is not None else None

            goal_dist, hidden = high_policy(state_tensor, hidden)
            goal = goal_dist.sample()
            goal_logprob = goal_dist.log_prob(goal)
            goal_tensor = F.one_hot(goal, num_classes=env.goal_dim).float().view(1, -1)

            for j in range(env.low_steps_per_goal):
                st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_dist = low_policy(st_tensor, goal_tensor.detach())
                action = action_dist.sample()
                action_logprob = action_dist.log_prob(action)

                entropy = compute_entropy(action_dist)
                next_state, reward, done, _ = env.step(action.item())

                total_reward += reward
                low_level_reward_total += reward

                low_loss = -action_logprob * reward
                low_optimizer.zero_grad()
                low_loss.backward()
                low_optimizer.step()

                state = next_state
                if done:
                    break

        high_reward = env.compute_high_level_reward(state)
        high_loss = -goal_logprob * high_reward
        high_optimizer.zero_grad()
        high_loss.backward()
        high_optimizer.step()

        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep + 1, total_reward, high_reward, low_level_reward_total, entropy])

        print(f"Episode {ep + 1}/{episodes}, Total Reward: {total_reward}")