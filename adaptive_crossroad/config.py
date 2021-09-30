import numpy as np
import torch

# Simulation Configuration
FLOW_NUMBER = 80
START_HOUR = 0
END_HOUR = 48
GRAPH_START = 24
GRAPH_END = 48

# Crossroad Configuration
ENV_SAMPLES = 100
TICK_PER_POINT = 12
TEN_SECOND_PER_TICK = 1
TOTAL_TICK = 8640 // TEN_SECOND_PER_TICK
DECISION_LENGTH = 9
DEFAULT_DECISION = [2, 1, 1, 1, 2, 2]
STRAIGHT_OUT = 30
LEFT_OUT = 10
OUTPUT_FLOW = np.array([[STRAIGHT_OUT, LEFT_OUT, 0, 0, 0, 0, 0, 0],
                        [STRAIGHT_OUT, 0, STRAIGHT_OUT, 0, 0, 0, 0, 0],
                        [0, 0, STRAIGHT_OUT, LEFT_OUT, 0, 0, 0, 0],
                        [0, 0, 0, 0, STRAIGHT_OUT, LEFT_OUT, 0, 0],
                        [0, 0, 0, 0, STRAIGHT_OUT, 0, STRAIGHT_OUT, 0],
                        [0, 0, 0, 0, 0, 0, STRAIGHT_OUT, LEFT_OUT]]) * TEN_SECOND_PER_TICK

# Environment Configuration
AStraight = np.array([61, 37, 19, 13, 28, 82, 175, 256, 256, 247, 229, 217,
                      202, 217, 220, 220, 226, 229, 223, 196, 172, 166, 139, 100]) * 10 / 360 * TEN_SECOND_PER_TICK
BStraight = np.array([67, 38, 23, 14, 20, 65, 141, 207, 221, 217, 210, 209,
                      210, 217, 220, 236, 242, 257, 242, 233, 197, 182, 152, 110]) * 10 / 360 * TEN_SECOND_PER_TICK
CStraight = np.array([202, 217, 220, 220, 226, 229, 223, 196, 172, 166, 139, 100,
                      61, 37, 19, 13, 28, 82, 175, 256, 256, 247, 229, 217]) * 10 / 360 * TEN_SECOND_PER_TICK
DStraight = np.array([210, 217, 220, 236, 242, 257, 242, 233, 197, 182, 152, 110,
                      67, 38, 23, 14, 20, 65, 141, 207, 221, 217, 210, 209]) * 10 / 360 * TEN_SECOND_PER_TICK
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
VN_MAX_LOSS = 0.005
VN_LEARNING_RATE = 0.00001
VN_BIDIRECTIONAL = True
VN_Y_SCALER_MIN = 0
VN_Y_SCALER_MAX = 20000

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
