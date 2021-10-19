import numpy as np
import torch

# Crossroad Configuration
TEN_SECOND_PER_TICK = 1
assert 8640 % TEN_SECOND_PER_TICK == 0

DECISION_LENGTH = 9
MIN_PHASE = 0
WAYS = 8
STRAIGHT_OUT = 60
LEFT_OUT = 20
RIGHT_OUT = 0
DEFAULT_DECISION = [2, 1, 1, 2, 1, 2]
assert sum(DEFAULT_DECISION) == DECISION_LENGTH

OUTPUT_FLOW = np.array([[STRAIGHT_OUT, LEFT_OUT, 0, 0, 0, 0, 0, 0],
                        [STRAIGHT_OUT, 0, STRAIGHT_OUT, 0, 0, 0, 0, 0],
                        [0, 0, STRAIGHT_OUT, LEFT_OUT, 0, 0, 0, 0],
                        [0, 0, 0, 0, STRAIGHT_OUT, LEFT_OUT, 0, 0],
                        [0, 0, 0, 0, STRAIGHT_OUT, 0, STRAIGHT_OUT, 0],
                        [0, 0, 0, 0, 0, 0, STRAIGHT_OUT, LEFT_OUT]]) * TEN_SECOND_PER_TICK
assert OUTPUT_FLOW.shape[1] == WAYS
TOTAL_TICK = 8640 // TEN_SECOND_PER_TICK

TACTICS = []
for i in range(MIN_PHASE, DECISION_LENGTH - MIN_PHASE * 4):
    for j in range(MIN_PHASE, DECISION_LENGTH - i - MIN_PHASE * 3):
        for k in range(MIN_PHASE, DECISION_LENGTH - i - j - MIN_PHASE * 2):
            for l in range(MIN_PHASE, DECISION_LENGTH - i - j - k - MIN_PHASE):
                for m in range(MIN_PHASE, DECISION_LENGTH - i - j - k - l):
                    TACTICS.append([i, j, k, l, m, DECISION_LENGTH - i - j - k - l - m])

# Environment Configuration
ENV_SAMPLES = 1000
ENV_METHOD = 'HARD'
ENV_STDEV_RATE = 0.25
ENV_RANGE = 100
ENV_AVG_24 = np.array([[61, 37, 19, 13, 28, 82, 175, 256, 256, 247, 229, 217,
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
                        17, 10, 6, 4, 5, 16, 35, 52, 55, 54, 53, 52]]) * 10 / 360 * TEN_SECOND_PER_TICK
assert ENV_AVG_24.shape[0] == WAYS
assert ENV_AVG_24.shape[1] == 24
ENV_TIME = np.arange(24) * 360 / TEN_SECOND_PER_TICK + 180 / TEN_SECOND_PER_TICK

# Simulation Configuration
START_HOUR = 0
END_HOUR = 24
GRAPH_START = 0
GRAPH_END = 24
TICK_PER_POINT = 6

# AdaptiveNetwork Configuration
AN_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AN_EPISODES = 500
AN_EPS_START = 0.9
AN_EPS_END = 0.05
AN_EPS_DECAY = 200
AN_GAMMA = 0.8
AN_LR = 0.001
AN_BATCH_SIZE = 64
AN_MEMORY_SIZE = 10000
AN_VALUE_NET_UPDATE = 5

# SMC Configuration
SMC_SAMPLES = 100


def tactic_string(tactic):
    return str(tactic[0]) + '_' + str(tactic[1]) + '_' + str(tactic[2]) + '_' + str(tactic[3]) + '_' + str(
        tactic[4]) + '_' + str(tactic[5])
