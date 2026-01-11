class FixedTimer:
    def __init__(self, phase_duration=20):
        self.phase_duration = phase_duration
        self.timer = 0
        self.current_phase = 0
        self.phases = [0, 1, 2, 3, 4, 5]  # 6 phases

    def select_action(self, lane_counts):
        current_lanes = {
            0: ["N", "S"],
            1: ["E", "W"],
            2: ["N"],
            3: ["S"],
            4: ["E"],
            5: ["W"]
        }[self.current_phase]

        # Sum lane counts for this phase
        phase_density = sum([lane_counts[l] for l in current_lanes])

        self.timer += 1
        if self.timer >= self.phase_duration:
            self.timer = 0
            self.current_phase = (self.current_phase + 1) % len(self.phases)

        return self.current_phase
