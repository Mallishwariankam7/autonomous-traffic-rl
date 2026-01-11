import random
from env.reward import RewardCalculator

EMERGENCY_PROB = 0.1

PHASES = {
    0: ["N", "S"],
    1: ["E", "W"],
    2: ["N"],
    3: ["S"],
    4: ["E"],
    5: ["W"]
}

class TrafficEnv:
    def __init__(self, max_steps=800, lane_capacity=25, spawn_prob=None, seed=None):
        self.lanes = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.waiting_time = {lane: 0 for lane in self.lanes}

        self.lane_capacity = lane_capacity
        self.max_steps = max_steps
        self.spawn_prob = spawn_prob or {"N": 0.4, "S": 0.4, "E": 0.3, "W": 0.3}

        self.time_step = 0
        self.last_action = None

        self.emergency = False
        self.emergency_lane = None
        self.emergency_cleared = False
        self.total_emergency_time = 0

        self.total_passed_vehicles = 0  # ✅ CRITICAL

        self.reward_calc = RewardCalculator()

        if seed is not None:
            random.seed(seed)

    def reset(self):
        for l in self.lanes:
            self.lanes[l] = 0
            self.waiting_time[l] = 0

        self.time_step = 0
        self.last_action = None
        self.emergency = False
        self.emergency_lane = None
        self.emergency_cleared = False
        self.total_emergency_time = 0
        self.total_passed_vehicles = 0  # ✅ RESET

        return self.get_state()

    def spawn_cars(self):
        for lane, prob in self.spawn_prob.items():
            if random.random() < prob:
                self.lanes[lane] = min(self.lanes[lane] + 1, self.lane_capacity)

    def detect_emergency(self):
        if not self.emergency and random.random() < EMERGENCY_PROB:
            self.emergency = True
            self.emergency_lane = random.choice(list(self.lanes.keys()))
            self.emergency_cleared = False

    def step(self, action):
        self.time_step += 1
        self.spawn_cars()
        self.detect_emergency()

        green_lanes = PHASES[action]
        vehicles_passed = 0

        for lane in self.lanes:
            if lane in green_lanes and self.lanes[lane] > 0:
                self.lanes[lane] -= 1
                vehicles_passed += 1
                self.waiting_time[lane] = max(0, self.waiting_time[lane] - 1)
            elif self.lanes[lane] > 0:
                self.waiting_time[lane] += 1

        if self.emergency:
            if self.emergency_lane in green_lanes and self.lanes[self.emergency_lane] > 0:
                self.lanes[self.emergency_lane] -= 1
                vehicles_passed += 1
                self.emergency_cleared = True
                self.emergency = False
                self.emergency_lane = None
            else:
                self.total_emergency_time += 1
                self.emergency_cleared = False

        self.total_passed_vehicles += vehicles_passed  # ✅ TRACK

        avg_wait = sum(self.waiting_time.values()) / 4

        reward, breakdown = self.reward_calc.compute(
            vehicles_passed,
            avg_wait,
            collisions=0,
            signal_switched=int(self.last_action != action),
            lane_waits=list(self.waiting_time.values()),
            emergency_flag=self.emergency,
            emergency_cleared=self.emergency_cleared
        )

        self.last_action = action
        done = self.time_step >= self.max_steps

        return self.get_state(), reward, done, {
            "passed_vehicles": vehicles_passed  # ✅ EXPOSE
        }

    def get_state(self):
        emergency_lane_idx = ["N", "S", "E", "W"].index(self.emergency_lane) if self.emergency else -1
        return [
            self.lanes["N"], self.lanes["S"], self.lanes["E"], self.lanes["W"],
            self.waiting_time["N"], self.waiting_time["S"],
            self.waiting_time["E"], self.waiting_time["W"],
            int(self.emergency),
            emergency_lane_idx
        ]

    def average_waiting_time_per_vehicle(self):
        total = sum(self.lanes.values())
        if total == 0:
            return 0
        return sum(self.waiting_time.values()) / total

    def average_emergency_clear_time(self):
        return self.total_emergency_time
