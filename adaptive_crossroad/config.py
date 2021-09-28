import numpy as np
import torch

# Crossroad Configuration
ENV_SAMPLES = 100
FLOW_NUMBER = 80
TICK_PER_POINT = 12
TEN_SECOND_PER_TICK = 1
TOTAL_TICK = 8640 // TEN_SECOND_PER_TICK
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

# Environment Configuration
AStraight = np.array([29, 21, 15, 13, 18, 36, 67, 94, 94, 91, 85, 81, 76, 81, 82, 82, 84, 85, 83, 74, 66, 64, 55,
                      42]) * 100 / 360 * TEN_SECOND_PER_TICK
BStraight = np.array([31, 22, 17, 14, 16, 31, 53, 75, 83, 81, 79, 79, 76, 79, 81, 84, 87, 90, 88, 83, 75, 70, 60,
                      46]) * 100 / 360 * TEN_SECOND_PER_TICK
CStraight = np.array([8, 6, 5, 4, 5, 9, 21, 34, 37, 33, 30, 29, 26, 27, 27, 27, 26, 26, 24, 21, 19, 19, 16,
                      13]) * 100 / 360 * TEN_SECOND_PER_TICK
DStraight = np.array([10, 7, 5, 4, 5, 8, 14, 22, 26, 26, 26, 27, 26, 28, 29, 29, 30, 32, 31, 28, 24, 23, 19,
                      15]) * 100 / 360 * TEN_SECOND_PER_TICK
ALeft = AStraight / 4
BLeft = BStraight / 4
CLeft = CStraight / 4
DLeft = DStraight / 4
ROADS = [AStraight, ALeft, BStraight, BLeft, CStraight, CLeft, DStraight, DLeft]

TIME = np.arange(24) * 360 / TEN_SECOND_PER_TICK + 180 / TEN_SECOND_PER_TICK
SAMPLES = 100
STDEV_RATE = 0.25
RANGE = 100
METHOD = 'HARD'

# PMC Configuration
MAX_INFLOW = 100000
PRISM_PATH = "D:/Program Files/prism-4.6/bin/"

# SMC Configuration
SMC_SAMPLES = 100

# ValueNet Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VN_INPUT_SIZE = 17
VN_HIDDEN_SIZE = 2
VN_LAYERS = 1

# ValueNet Training Configuration
TRAIN_RATE = 0.25
VN_EPOCHS = 7000
VN_LEARNING_RATE = 0.00001
VN_BIDIRECTIONAL = True

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
