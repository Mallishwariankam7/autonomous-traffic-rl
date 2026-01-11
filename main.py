from env.traffic_env import TrafficEnv

env = TrafficEnv()

state = env.reset()
print("Initial State:", state)

for t in range(10):
    action = t % 2
    next_state, reward, done = env.step(action)
    print(f"Step {t+1}")
    print("Action:", action)
    print("State:", next_state)
    print("Reward:", reward)
    print("-" * 30)