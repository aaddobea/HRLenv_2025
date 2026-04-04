import numpy as np

class LLMHealthcareEnv:
    def __init__(self):
        self.state_dim = 16
        self.goal_dim = 4  # e.g., {verify, clarify, escalate, confirm}
        self.action_dim = 6  # e.g., {ask symptom, propose diagnosis, ask follow-up, etc.}
        self.high_steps = 3
        self.low_steps_per_goal = 4

    def reset(self):
        return np.random.rand(self.state_dim)

    def step(self, action):
        # Reward can reflect safety/reliability (mocked here)
        reward = np.random.choice([1, 0, -1], p=[0.6, 0.3, 0.1])  # unsafe actions penalized
        done = np.random.rand() > 0.95
        next_state = np.random.rand(self.state_dim)
        return next_state, reward, done, {}

    def compute_high_level_reward(self, state):
        return np.random.choice([1, 0])  # success or failure of sub-goal
