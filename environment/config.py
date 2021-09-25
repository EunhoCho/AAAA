import numpy as np

SECOND_PER_TICK = 10

AStraight = np.array([29, 21, 15, 13, 18, 36, 67, 94, 94, 91, 85, 81, 76, 81, 82, 82, 84, 85, 83, 74, 66, 64, 55,
                      42]) * 100 / 3600 * SECOND_PER_TICK
BStraight = np.array([31, 22, 17, 14, 16, 31, 53, 75, 83, 81, 79, 79, 76, 79, 81, 84, 87, 90, 88, 83, 75, 70, 60,
                      46]) * 100 / 3600 * SECOND_PER_TICK
CStraight = np.array([8, 6, 5, 4, 5, 9, 21, 34, 37, 33, 30, 29, 26, 27, 27, 27, 26, 26, 24, 21, 19, 19, 16,
                      13]) * 100 / 3600 * SECOND_PER_TICK
DStraight = np.array([10, 7, 5, 4, 5, 8, 14, 22, 26, 26, 26, 27, 26, 28, 29, 29, 30, 32, 31, 28, 24, 23, 19,
                      15]) * 100 / 3600 * SECOND_PER_TICK
ALeft = AStraight / 4
BLeft = BStraight / 4
CLeft = CStraight / 4
DLeft = DStraight / 4
ROADS = [AStraight, ALeft, BStraight, BLeft, CStraight, CLeft, DStraight, DLeft]

TIME = np.arange(24) * 3600 / SECOND_PER_TICK + 1800 / SECOND_PER_TICK
SAMPLES = 100
STDEV_RATE = 0.25
