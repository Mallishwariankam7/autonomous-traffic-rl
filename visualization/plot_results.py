import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure the folder exists
os.makedirs("visualization", exist_ok=True)

densities = ["Low", "Medium", "High"]
x = np.arange(len(densities))

# Average Waiting Time Data
fixed_avg_wait = [735.84, 418.39, 398.11]
dqn_avg_wait = [324.61, 159.84, 11.08]

plt.figure(figsize=(8,5))
plt.plot(x, fixed_avg_wait, marker='o', label="Fixed Timer", linewidth=2)
plt.plot(x, dqn_avg_wait, marker='o', label="DQN Agent", linewidth=2)
plt.xticks(x, densities)
plt.xlabel("Traffic Density")
plt.ylabel("Average Wait Time per Vehicle (s)")
plt.title("Average Waiting Time vs Traffic Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualization/avg_wait_vs_density.png")
plt.show()

# Throughput Data
fixed_throughput = [700.80, 1057.60, 1098.80]
dqn_throughput = [658.80, 1269.80, 1659.40]

plt.figure(figsize=(8,5))
plt.plot(x, fixed_throughput, marker='o', label="Fixed Timer", linewidth=2)
plt.plot(x, dqn_throughput, marker='o', label="DQN Agent", linewidth=2)
plt.xticks(x, densities)
plt.xlabel("Traffic Density")
plt.ylabel("Vehicles Passed")
plt.title("Throughput vs Traffic Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualization/throughput_vs_density.png")
plt.show()
