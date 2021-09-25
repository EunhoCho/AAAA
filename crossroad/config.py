import numpy as np

ENV_SAMPLES = 100
FLOW_NUMBER = 0
TICK_PER_POINT = 12
TEN_SECOND_PER_TICK = 1
DECISION_LENGTH = 9
DEFAULT_DECISION = [2, 2, 2, 1, 1, 1]
OUTPUT_FLOW = np.array([[300, 100, 0, 0, 0, 0, 0, 0],
                        [300, 0, 300, 0, 0, 0, 0, 0],
                        [0, 0, 300, 100, 0, 0, 0, 0],
                        [0, 0, 0, 0, 300, 100, 0, 0],
                        [0, 0, 0, 0, 300, 0, 300, 0],
                        [0, 0, 0, 0, 0, 0, 300, 100]]) * TEN_SECOND_PER_TICK

SMC_SAMPLES = 100

TACTICS = []
for i in range(1, DECISION_LENGTH - 4):
    for j in range(1, DECISION_LENGTH - i - 3):
        for k in range(1, DECISION_LENGTH - i - j - 2):
            for l in range(1, DECISION_LENGTH - i - j - k - 1):
                for m in range(1, DECISION_LENGTH - i - j - k - l):
                    TACTICS.append([i, j, k, l, m, DECISION_LENGTH - i - j - k - l - m])
