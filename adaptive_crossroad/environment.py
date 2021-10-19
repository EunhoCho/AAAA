import csv
import random

import numpy as np
from tqdm import tqdm

from adaptive_crossroad import config


def sample_environment(avg_flow):
    sample_flow = []
    for i in range(config.cross_ways):
        flow = np.array([0] * config.cross_total_tick)
        for j in range(0, config.cross_total_tick):
            for k in range(avg_flow[j][i]):
                value = int(random.randrange(j - config.env_range, j + config.env_range))
                flow[value % config.cross_total_tick] += 1
        sample_flow.append(flow)
    return np.transpose(sample_flow)


if __name__ == "__main__":
    avg_flow = []
    for road in config.env_avg_24:
        z = np.poly1d(np.polyfit(config.env_time, road, 5))

        residual = 0
        data = []
        for i in range(config.cross_total_tick):
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

    sample_tqdm = tqdm(range(config.env_samples))
    sample_tqdm.set_description("Flow Sampling")
    for i in sample_tqdm:
        crossroad_data = sample_environment(avg_flow)

        with open('../flow/flow_' + str(i) + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(crossroad_data)
