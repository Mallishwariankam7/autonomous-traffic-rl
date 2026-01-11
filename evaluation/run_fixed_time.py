import csv
import numpy as np
import matplotlib.pyplot as plt
from baselines.fixed_timer import FixedTimer
from env.traffic_env import TrafficEnv
import os

TRAFFIC_LEVELS = {
    "low":    {"N": 0.25, "S": 0.25, "E": 0.2,  "W": 0.2},
    "medium": {"N": 0.45, "S": 0.45, "E": 0.35, "W": 0.35},
    "high":   {"N": 0.75, "S": 0.75, "E": 0.6,  "W": 0.6}
}

SEEDS = [0, 1, 2, 3, 4]
LOG_CSV = "logs/fixed_timer_metrics.csv"
MAX_STEPS = 800
LANE_CAPACITY = 25
os.makedirs("logs", exist_ok=True)

def evaluate():
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level","avg_wait","max_lane_wait","emergency_time","throughput"])

    print("\nFixed Timer Results (Multi-Density, Multi-Seed)\n")
    summary = {}

    for level, spawn_prob in TRAFFIC_LEVELS.items():
        avg_waits, max_waits, emergency_times, throughputs = [], [], [], []

        for seed in SEEDS:
            env = TrafficEnv(max_steps=MAX_STEPS, lane_capacity=LANE_CAPACITY, spawn_prob=spawn_prob, seed=seed)
            controller = FixedTimer()
            state = env.reset()
            done = False
            total_passed = 0

            while not done:
                lane_counts = {"N": state[0], "S": state[1], "E": state[2], "W": state[3]}
                action = controller.select_action(lane_counts)
                state, reward, done, info = env.step(action)
                total_passed += info.get("passed_vehicles", 0)

            # Use total waiting time instead of avg_wait per vehicle
            total_wait = sum(env.waiting_time.values())
            avg_wait_norm = total_wait / MAX_STEPS

            avg_waits.append(avg_wait_norm)
            max_waits.append(max(env.waiting_time.values()))
            emergency_times.append(env.average_emergency_clear_time())
            throughputs.append(total_passed)

        summary[level] = {
            "avg_wait": (np.mean(avg_waits), np.std(avg_waits)),
            "max_wait": (np.mean(max_waits), np.std(max_waits)),
            "emergency": (np.mean(emergency_times), np.std(emergency_times)),
            "throughput": (np.mean(throughputs), np.std(throughputs))
        }

        print(f"[{level.upper()}]")
        print(f"Avg Wait / Step: {summary[level]['avg_wait'][0]:.2f} ± {summary[level]['avg_wait'][1]:.2f}")
        print(f"Max Lane Wait: {summary[level]['max_wait'][0]:.2f} ± {summary[level]['max_wait'][1]:.2f}")
        print(f"Emergency Time: {summary[level]['emergency'][0]:.2f} ± {summary[level]['emergency'][1]:.2f}")
        print(f"Throughput: {summary[level]['throughput'][0]:.2f} ± {summary[level]['throughput'][1]:.2f}\n")

    # Save plots
    levels = list(summary.keys())
    avg_wait_means = [summary[l]["avg_wait"][0] for l in levels]
    avg_wait_stds = [summary[l]["avg_wait"][1] for l in levels]
    throughput_means = [summary[l]["throughput"][0] for l in levels]
    throughput_stds = [summary[l]["throughput"][1] for l in levels]

    os.makedirs("visualization", exist_ok=True)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.bar(levels, avg_wait_means, yerr=avg_wait_stds, capsize=5, color='skyblue')
    plt.title("Fixed Timer: Avg Waiting Time per Step")
    plt.ylabel("Avg Wait (s)")

    plt.subplot(1,2,2)
    plt.bar(levels, throughput_means, yerr=throughput_stds, capsize=5, color='lightgreen')
    plt.title("Fixed Timer: Throughput per Traffic Level")
    plt.ylabel("Vehicles Passed")

    plt.tight_layout()
    plt.savefig("visualization/fixed_timer_results.png")
    plt.show()

if __name__ == "__main__":
    evaluate()