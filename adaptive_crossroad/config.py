import numpy as np
import torch

# Crossroad Configuration
ENV_SAMPLES = 100
FLOW_NUMBER = 80
TICK_PER_POINT = 12
TEN_SECOND_PER_TICK = 1
DECISION_LENGTH = 9
DEFAULT_DECISION = [2, 2, 2, 1, 1, 1]
STRAIGHT_OUT = 300
LEFT_OUT = 100
OUTPUT_FLOW = np.array([[STRAIGHT_OUT, LEFT_OUT, 0, 0, 0, 0, 0, 0],
                        [STRAIGHT_OUT, 0, STRAIGHT_OUT, 0, 0, 0, 0, 0],
                        [0, 0, STRAIGHT_OUT, LEFT_OUT, 0, 0, 0, 0],
                        [0, 0, 0, 0, STRAIGHT_OUT, LEFT_OUT, 0, 0],
                        [0, 0, 0, 0, STRAIGHT_OUT, 0, STRAIGHT_OUT, 0],
                        [0, 0, 0, 0, 0, 0, STRAIGHT_OUT, LEFT_OUT]]) * TEN_SECOND_PER_TICK

# PMC Configuration
MAX_INFLOW = 100000
PRISM_PATH = "D:/Program Files/prism-4.6/bin/"

# SMC Configuration
SMC_SAMPLES = 100

# ValueNet Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VN_CLASS = 1
VN_INPUT_SIZE = 17
VN_HIDDEN_SIZE = 2
VN_LAYERS = 1

# ValueNet Training Configuration
TRAIN_RATE = 0.7
VN_EPOCHS = 10000
VN_LEARNING_RATE = 0.00001

TACTICS = []
for i in range(1, DECISION_LENGTH - 4):
    for j in range(1, DECISION_LENGTH - i - 3):
        for k in range(1, DECISION_LENGTH - i - j - 2):
            for l in range(1, DECISION_LENGTH - i - j - k - 1):
                for m in range(1, DECISION_LENGTH - i - j - k - l):
                    TACTICS.append([i, j, k, l, m, DECISION_LENGTH - i - j - k - l - m])


def tactic_string(tactic):
    return str(tactic[0]) + '_' + str(tactic[1]) + '_' + str(tactic[2]) + '_' + str(tactic[3]) + '_' + str(
        tactic[4]) + '_' + str(tactic[5])
