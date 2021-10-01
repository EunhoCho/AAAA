import copy
import csv
import random

import numpy as np
from tqdm import tqdm

from adaptive_crossroad import config


def sample_environment(start, end, avg_flow):
    sample_number = []
    for i in range(start, end):
        sample_number.append(np.random.normal(0, 1))

    sample_environment = []
    for i in range(start, end):
        data = avg_flow[i]
        value = []
        for j in range(config.WAYS):
            value.append(max(0, int(sample_number[i - start] * (data[j] * config.ENV_STDEV_RATE) ** 2 + data[j])))

        sample_environment.append(value)

    return sample_environment


def sample_environment_hard(start, end, avg_flow, cur_flow, target_flow=None, started=0, value_based=0):
    if target_flow is None:
        target_flow = copy.deepcopy(avg_flow)

    if start == 0 and end == config.TOTAL_TICK:
        sample_flow = []
        for i in range(config.WAYS):
            flow = np.array([0] * (end - start))
            for j in range(0, config.TOTAL_TICK):
                for k in range(target_flow[j][i]):
                    value = int(random.randrange(j - config.ENV_RANGE, j + config.ENV_RANGE))
                    flow[value % config.TOTAL_TICK] += 1
            sample_flow.append(flow)
        return np.transpose(sample_flow), None

    for i in range(config.WAYS):
        for j in range(max(value_based, started), start):
            value = cur_flow[j % config.TOTAL_TICK][i]
            for k in range(-config.ENV_RANGE, config.ENV_RANGE):
                target_flow[(j + k) % config.TOTAL_TICK][i] -= value / (2 * config.ENV_RANGE)

    sample_flow = []
    for i in range(config.WAYS):
        flow = np.array([0] * (end - start))

        def add_flow(range_start, range_end):
            for j in range(range_start, range_end):
                for k in range(int(target_flow[j][i])):
                    value = (int(random.randrange(j - config.ENV_RANGE, j + config.ENV_RANGE))) % config.TOTAL_TICK
                    if start <= value < end:
                        flow[value % config.TOTAL_TICK - start] += 1

                value = int(random.randrange(j - config.ENV_RANGE, j + config.ENV_RANGE))
                if start <= value < end:
                    flow[value % config.TOTAL_TICK - start] += target_flow[j][i] - int(target_flow[j][i])

        if (start % config.TOTAL_TICK) < config.ENV_RANGE:
            add_flow((start - config.ENV_RANGE) % config.TOTAL_TICK, config.TOTAL_TICK)

        if (end % config.TOTAL_TICK) > config.TOTAL_TICK - config.ENV_RANGE:
            add_flow(0, (end + config.ENV_RANGE) % config.TOTAL_TICK)

        add_flow(max(0, (start - config.ENV_RANGE) % config.TOTAL_TICK),
                 min((end + config.ENV_RANGE) % config.TOTAL_TICK, config.TOTAL_TICK))
        sample_flow.append(flow)

    return np.transpose(sample_flow), target_flow


if __name__ == "__main__":
    avg_flow = []
    for road in config.ENV_AVG_24:
        z = np.poly1d(np.polyfit(config.ENV_TIME, road, 5))

        residual = 0
        data = []
        for i in range(config.TOTAL_TICK):
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

    sample_tqdm = tqdm(range(config.ENV_SAMPLES))
    sample_tqdm.set_description("Flow Sampling")
    for i in sample_tqdm:
        if config.ENV_METHOD == 'HARD':
            crossroad_data, erased = sample_environment_hard(0, config.TOTAL_TICK, avg_flow, [])

        else:
            crossroad_data = sample_environment(0, config.TOTAL_TICK, avg_flow)

        with open('../flow/flow_' + str(i) + ('_hard' if config.ENV_METHOD == 'HARD' else '') + '.csv', 'w',
                  newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(crossroad_data)
