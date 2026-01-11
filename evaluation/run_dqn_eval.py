import csv
import numpy as np
import torch
from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent

TRAFFIC_LEVELS = {
    "low":    {"N": 0.25, "S": 0.25, "E": 0.2,  "W": 0.2},
    "medium": {"N": 0.45, "S": 0.45, "E": 0.35, "W": 0.35},
    "high":   {"N": 0.75, "S": 0.75, "E": 0.6,  "W": 0.6}
}

SEEDS = [0, 1, 2, 3, 4]
MODEL_PATH = "checkpoints/dqn_final.pth"
LOG_CSV = "logs/dqn_metrics.csv"
STATE_DIM = 10
ACTION_DIM = 6
MAX_STEPS = 800

def evaluate():
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "level", "seed",
            "avg_wait_per_vehicle",
            "max_lane_wait",
            "emergency_time",
            "throughput"
        ])

    print("\nDQN Results (Multi-Density, Multi-Seed)\n")

    for level, spawn_prob in TRAFFIC_LEVELS.items():
        avg_waits, max_waits, emergency_times, throughputs = [], [], [], []

        for seed in SEEDS:
            env = TrafficEnv(
                max_steps=MAX_STEPS,
                lane_capacity=25,
                spawn_prob=spawn_prob,
                seed=seed
            )

            agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
            agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            agent.q_net.eval()
            agent.epsilon = 0.0  # deterministic

            state = env.reset()
            done = False
            total_passed = 0
            total_wait_time = 0

            while not done:
                action = agent.select_action(state)
                state, _, done, info = env.step(action)

                total_passed += info.get("passed_vehicles", 0)
                total_wait_time += sum(env.waiting_time.values())

            avg_wait = total_wait_time / max(1, total_passed)
            max_wait = max(env.waiting_time.values())
            emergency_time = env.average_emergency_clear_time()
            throughput = total_passed

            avg_waits.append(avg_wait)
            max_waits.append(max_wait)
            emergency_times.append(emergency_time)
            throughputs.append(throughput)

            with open(LOG_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    level, seed, avg_wait, max_wait, emergency_time, throughput
                ])

        print(f"[{level.upper()} TRAFFIC]")
        print(f"Avg Wait / Vehicle: {np.mean(avg_waits):.2f} ± {np.std(avg_waits):.2f}")
        print(f"Max Lane Wait: {np.mean(max_waits):.2f} ± {np.std(max_waits):.2f}")
        print(f"Emergency Time: {np.mean(emergency_times):.2f} ± {np.std(emergency_times):.2f}")
        print(f"Throughput: {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f}\n")

if __name__ == "__main__":
    evaluate()
