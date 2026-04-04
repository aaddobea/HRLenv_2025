import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("training_log.csv")

# Plot rewards
plt.figure(figsize=(12, 6))
plt.plot(df["Episode"], df["TotalReward"], label="Total Reward")
plt.plot(df["Episode"], df["HighLevelReward"], label="High-Level Reward")
plt.plot(df["Episode"], df["LowLevelReward"], label="Low-Level Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Trends over Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_plot.png")
plt.show()

# Plot action entropy
plt.figure(figsize=(10, 4))
plt.plot(df["Episode"], df["ActionEntropy"], color="purple", label="Action Entropy")
plt.xlabel("Episode")
plt.ylabel("Entropy")
plt.title("Action Entropy (Exploration)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_plot.png")
plt.show()