import numpy as np

class RewardCalculator:
    def __init__(
        self,
        w_throughput=1.0,
        w_waiting=0.7,
        w_fairness=1.0,
        w_switch=0.1,
        w_emergency=3.0
    ):
        self.w_throughput = w_throughput
        self.w_waiting = w_waiting
        self.w_fairness = w_fairness
        self.w_switch = w_switch
        self.w_emergency = w_emergency

    def compute(
        self,
        vehicles_passed,
        avg_wait,
        collisions,
        signal_switched,
        lane_waits,
        emergency_flag,
        emergency_cleared
    ):
        r_throughput = np.tanh(vehicles_passed)
        r_waiting = -np.tanh(avg_wait)
        r_fairness = -np.std(lane_waits)
        r_switch = -signal_switched
        r_emergency = self.w_emergency if emergency_cleared else 0

        starvation_penalty = -1.5 if max(lane_waits) > 100 else 0

        total = (
            self.w_throughput * r_throughput +
            self.w_waiting * r_waiting +
            self.w_fairness * r_fairness +
            self.w_switch * r_switch +
            r_emergency +
            starvation_penalty
        )

        return total, {
            "throughput": r_throughput,
            "waiting": r_waiting,
            "fairness": r_fairness,
            "switch": r_switch,
            "emergency": r_emergency,
            "starvation": starvation_penalty
        }
