import copy
import csv

import numpy as np
from tqdm import tqdm

from environment import config


def sample_environment(start, end, avg_flow):
    sample_number = []
    for i in range(start, end):
        sample_number.append(np.random.normal(0, 1))

    sample_environment = []
    for i in range(start, end):
        data = avg_flow[i]
        value = []
        for j in range(8):
            value.append(max(0, int(sample_number[i - start] * (data[j] * config.STDEV_RATE) ** 2 + data[j])))

        sample_environment.append(value)

    return sample_environment


def sample_environment_hard(start, end, avg_flow, cur_flow, target_flow=None, value_based=0):
    if target_flow is None:
        target_flow = copy.deepcopy(avg_flow)

    zero_value = []
    for i in range(8):
        residual = 0
        k = 0
        for j in range(max(value_based, 0), start):
            residual += cur_flow[j][i]
            while target_flow[k][i] == 0:
                k += 1

            while 0 < target_flow[k][i] <= residual:
                residual -= target_flow[k][i]
                target_flow[k][i] = 0
                k += 1

            if residual > 0:
                target_flow[k][i] -= residual
                residual = 0

        zero_value.append(k)

    sample_flow = []
    for i in range(8):
        flow = np.array([0] * (end - start))
        for j in range(zero_value[i], min(end + config.STDEV, 8640 // config.TEN_SECOND_PER_TICK)):
            for k in range(target_flow[j][i]):
                if start == 0 and end == 8640 // config.TEN_SECOND_PER_TICK:
                    value = int(np.random.normal(j, config.STDEV))
                    flow[value % (8640 // config.TEN_SECOND_PER_TICK)] += 1
                else:
                    while True:
                        value = int(np.random.normal(j, config.STDEV))
                        if value >= start:
                            if value < end:
                                flow[value - start] += 1
                            break
        sample_flow.append(flow)

    return np.transpose(sample_flow), target_flow


if __name__ == "__main__":
    avg_flow = []
    for road in config.ROADS:
        z = np.poly1d(np.polyfit(config.TIME, road, 5))

        residual = 0
        data = []
        for i in range(8640 // config.TEN_SECOND_PER_TICK):
            value = 0
            residual += z(i)

            if residual >= 1:
                value += int(residual)
                residual -= int(residual)

            if residual >= 0.5:
                value += 1
                residual -= 1

            data.append(value)

        avg_flow.append(data)

    avg_flow = np.transpose(avg_flow)

    with open('../flow/avg_flow.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(avg_flow)

    sample_tqdm = tqdm(range(config.SAMPLES))
    sample_tqdm.set_description("sampling")
    for i in sample_tqdm:
        if config.METHOD == 'HARD':
            crossroad_data, erased = sample_environment_hard(0, 8640 // config.TEN_SECOND_PER_TICK, avg_flow, [])

        else:
            crossroad_data = sample_environment(0, 8640 // config.TEN_SECOND_PER_TICK, avg_flow)

        with open('../flow/flow_' + str(i) + ('_hard' if config.METHOD == 'HARD' else '') + '.csv', 'w',
                  newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(crossroad_data)
