import numpy as np
import torch

# Crossroad Configuration
cross_decision_length = 9
cross_default_decision = [1, 2, 1, 1, 3, 1]
cross_out_left = 20
cross_out_straight = 60
cross_phase_min = 1
cross_ten_second_per_tick = 1
cross_ways = 8

# Crossroad Accident Configuration
accident_after = 360
accident_duration = 180
accident_mtth = 1080

# Environment Configuration
env_samples = 1000
env_range = 100
env_avg_24 = np.array([[61, 37, 19, 13, 28, 82, 175, 256, 256, 247, 229, 217,
                        202, 217, 220, 220, 226, 229, 223, 196, 172, 166, 139, 100],
                       [15, 9, 5, 3, 7, 21, 44, 64, 64, 62, 57, 54,
                        51, 54, 55, 55, 57, 57, 56, 49, 43, 42, 35, 25],
                       [67, 38, 23, 14, 20, 65, 141, 207, 221, 217, 210, 209,
                        210, 217, 220, 236, 242, 257, 242, 233, 197, 182, 152, 110],
                       [17, 10, 6, 4, 5, 16, 35, 52, 55, 54, 53, 52,
                        53, 54, 55, 59, 61, 64, 61, 58, 49, 46, 38, 28],
                       [202, 217, 220, 220, 226, 229, 223, 196, 172, 166, 139, 100,
                        61, 37, 19, 13, 28, 82, 175, 256, 256, 247, 229, 217],
                       [51, 54, 55, 55, 57, 57, 56, 49, 43, 42, 35, 25,
                        15, 9, 5, 3, 7, 21, 44, 64, 64, 62, 57, 54],
                       [210, 217, 220, 236, 242, 257, 242, 233, 197, 182, 152, 110,
                        67, 38, 23, 14, 20, 65, 141, 207, 221, 217, 210, 209],
                       [53, 54, 55, 59, 61, 64, 61, 58, 49, 46, 38, 28,
                        17, 10, 6, 4, 5, 16, 35, 52, 55, 54, 53, 52]]) * 10 / 360 * cross_ten_second_per_tick

# Simulation Configuration
sim_accident_on = False
sim_count = 1
sim_end = 24
sim_start = 0
sim_targets = ['RL', 'A-RL']
sim_tqdm_on = True

# Graph figure Configuration
graph_decision_per_point = 6
graph_end = 24
graph_start = 0

# SMC Configuration
smc_rate_high = 1.25
smc_rate_low = 0.75
smc_samples = 100

# RL Configuration
rl_batch_size = 64
rl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rl_episodes = 960
rl_epsilon_start = 0.9
rl_epsilon_end = 0.05
rl_epsilon_decay = 200
rl_gamma = 0.8
rl_hidden_layer = 256
rl_learning_rate = 0.001
rl_memory_size = 10000

# RL-SMC configuration
rl_smc_candidates = 5

# Calculations, Assertions
cross_out_flow = np.array([[cross_out_straight, cross_out_left, 0, 0, 0, 0, 0, 0],
                           [cross_out_straight, 0, cross_out_straight, 0, 0, 0, 0, 0],
                           [0, 0, cross_out_straight, cross_out_left, 0, 0, 0, 0],
                           [0, 0, 0, 0, cross_out_straight, cross_out_left, 0, 0],
                           [0, 0, 0, 0, cross_out_straight, 0, cross_out_straight, 0],
                           [0, 0, 0, 0, 0, 0, cross_out_straight, cross_out_left]]) * cross_ten_second_per_tick
cross_tactics = []
for i in range(cross_phase_min, cross_decision_length - cross_phase_min * 4):
    for j in range(cross_phase_min, cross_decision_length - i - cross_phase_min * 3):
        for k in range(cross_phase_min, cross_decision_length - i - j - cross_phase_min * 2):
            for m in range(cross_phase_min, cross_decision_length - i - j - k - cross_phase_min):
                for n in range(cross_phase_min, cross_decision_length - i - j - k - m):
                    cross_tactics.append([i, j, k, m, n, cross_decision_length - i - j - k - m - n])
cross_one_minute = 6 / cross_ten_second_per_tick
cross_one_hour = 360 // cross_ten_second_per_tick
cross_total_tick = 8640 // cross_ten_second_per_tick
env_time = np.arange(24) * 360 / cross_ten_second_per_tick + 180 / cross_ten_second_per_tick
sim_end_tick = sim_end * 360 // cross_ten_second_per_tick
sim_start_tick = sim_start * 360 // cross_ten_second_per_tick
graph_end_tick = graph_end * 360 // cross_ten_second_per_tick
graph_start_tick = graph_start * 360 // cross_ten_second_per_tick
graph_time = np.arange(graph_start, graph_end,
                       cross_ten_second_per_tick * cross_decision_length * graph_decision_per_point / 360)


def tactic_string(tactic):
    return str(tactic[0]) + '_' + str(tactic[1]) + '_' + str(tactic[2]) + '_' + str(tactic[3]) + '_' + str(
        tactic[4]) + '_' + str(tactic[5])


assert sum(cross_default_decision) == cross_decision_length
assert 360 % cross_ten_second_per_tick == 0
assert cross_out_flow.shape[1] == cross_ways
assert env_avg_24.shape[0] == cross_ways
assert env_avg_24.shape[1] == 24
assert sim_end >= graph_end
assert sim_start <= graph_start
