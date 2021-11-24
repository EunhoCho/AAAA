import math

import numpy as np
import torch

# Crossroad configuration
cross_decision_length = 12
cross_default_decision = [2, 2, 2, 2, 2, 2]
cross_out_left = 10
cross_out_straight = 30
cross_phase_min = 1
cross_ten_second_per_tick = 1
cross_ways = 8

# Environment configuration
env_samples = 1000
env_range = 100
env_avg_24 = np.array([[29, 21, 15, 13, 18, 36, 67, 94, 94, 91, 85, 81, 76, 81, 82, 82, 84, 85, 83, 74, 66, 64, 55, 42],
                       [7, 5, 3, 3, 4, 9, 16, 23, 23, 22, 21, 20, 19, 20, 20, 20, 21, 21, 20, 18, 16, 16, 13, 10],
                       [23, 16, 12, 11, 15, 36, 64, 81, 83, 73, 73, 75, 73, 74, 75, 77, 82, 87, 89, 82, 70, 62, 52, 38],
                       [5, 4, 3, 2, 3, 9, 16, 20, 20, 18, 18, 18, 18, 18, 18, 19, 20, 21, 22, 20, 17, 15, 13, 9],
                       [8, 6, 5, 4, 5, 9, 21, 34, 37, 33, 30, 29, 26, 27, 27, 27, 26, 26, 24, 21, 19, 19, 16, 13],
                       [2, 1, 1, 1, 1, 2, 5, 8, 9, 8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 4, 4, 4, 3],
                       [10, 7, 5, 4, 5, 8, 14, 22, 26, 26, 26, 27, 26, 28, 29, 29, 30, 32, 31, 28, 24, 23, 19, 15],
                       [2, 1, 1, 1, 1, 2, 3, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 7, 7, 6, 5, 4,
                        3]]) * 30 / 360 * cross_ten_second_per_tick

# Simulation configuration
sim_count = 20
sim_end = 24
sim_start = 0
sim_targets = [
    'AD-RL',
    'ORL',
    'SMC',
    'RL-SMC',
    'DEFAULT'
]
sim_tqdm_on = True

# Graph figure configuration
graph_decision_per_point = 6
graph_end = 24
graph_start = 0

# SMC configuration
smc_sprt_alpha = 0.2
smc_sprt_beta = 0.2
smc_sprt_delta = 0.01
smc_max_tries = 10000
smc_max_prob = 95
smc_min_samples = 10
smc_max_samples = 1000
smc_flow_rate_high = 1.25
smc_flow_rate_low = 0.75

# Pytorch configuration
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL configuration
rl_batch_size = 64
rl_episodes = 500
rl_epsilon_start = 0.9
rl_epsilon_end = 0.05
rl_epsilon_decay = 200
rl_gamma = 0.8
rl_hidden_layer = 256
rl_learning_rate = 0.001
rl_memory_size = 10000

# RL SMC configuration
rl_smc_min_candidates = 5
rl_smc_threshold = 0.25

# Crossroad anomaly configuration
anomaly_after = 720
anomaly_duration = 360
anomaly_mtth = 2160

# Anomaly detector configuration
anomaly_d_episodes = 300
anomaly_d_hidden_layer = 256
anomaly_d_learning_rate = 0.001
anomaly_d_momentum = 0.9
anomaly_d_batch_size = 4

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
cross_num_decisions = cross_total_tick // cross_decision_length
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
