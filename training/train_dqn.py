import os
import csv
import torch
import random
from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent

MAX_EPISODES = 800
MAX_STEPS = 800
BATCH_SIZE = 64

STATE_DIM = 10
ACTION_DIM = 6
LR = 1e-3
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.996

DEVICE = "cpu"
SEEDS = [0, 1, 2, 3, 4]

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TRAFFIC_PROBS = {
    "low":    {"N": 0.25, "S": 0.25, "E": 0.2, "W": 0.2},
    "medium": {"N": 0.45, "S": 0.45, "E": 0.35, "W": 0.35},
    "high":   {"N": 0.75, "S": 0.75, "E": 0.6, "W": 0.6}
}

env = TrafficEnv(max_steps=MAX_STEPS, lane_capacity=25)

agent = DQNAgent(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    lr=LR,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    device=DEVICE
)

log_path = f"{LOG_DIR}/train_metrics_dqn.csv"
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "traffic_level",
        "reward",
        "avg_wait",
        "max_lane_wait",
        "emergency_time",
        "throughput",
        "epsilon"
    ])

for episode in range(1, MAX_EPISODES + 1):

    traffic_level = random.choices(
        ["low", "medium", "high"],
        weights=[0.4, 0.3, 0.3]
    )[0]

    env.spawn_prob = TRAFFIC_PROBS[traffic_level]
    random.seed(random.choice(SEEDS))

    state = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        passed = info["passed_vehicles"]
        reward += 2.0 * passed  # ✅ CORRECT reward shaping

        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step(BATCH_SIZE)

        state = next_state
        episode_reward += reward

        if done:
            break

    avg_wait = env.average_waiting_time_per_vehicle()
    max_lane_wait = max(env.waiting_time.values())
    emergency_time = env.average_emergency_clear_time()
    throughput = env.total_passed_vehicles

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            episode,
            traffic_level,
            episode_reward,
            avg_wait,
            max_lane_wait,
            emergency_time,
            throughput,
            agent.epsilon
        ])

    if episode % 50 == 0:
        torch.save(agent.q_net.state_dict(), f"{CHECKPOINT_DIR}/dqn_ep{episode}.pth")
        print(
            f"Ep {episode} | {traffic_level.upper()} | "
            f"Reward {episode_reward:.1f} | "
            f"AvgWait {avg_wait:.2f} | "
            f"Throughput {throughput} | "
            f"Eps {agent.epsilon:.3f}"
        )

torch.save(agent.q_net.state_dict(), f"{CHECKPOINT_DIR}/dqn_final.pth")
print("Training complete.")
