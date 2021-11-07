import csv
import random

import numpy as np
from tqdm import tqdm

import config


def read_flow(name=''):
    if name == '':
        file_name = 'flow/avg_flow.csv'
    else:
        file_name = 'flow/' + name + '.csv'

    flow_data = []
    with open(file_name, "r", newline='') as f:
        flow_reader = csv.reader(f)
        flows = np.array([0] * config.cross_ways)
        for i, row in enumerate(flow_reader):
            flows = flows + np.array(row).astype(int)
            if i % config.cross_ten_second_per_tick == 0:
                flow_data.append(flows)
                flows = np.array([0] * config.cross_ways)

    return flow_data


def sample_environment(avg_flow=None, name=None):
    if avg_flow is None:
        avg_flow = read_flow()

    sample_flow = []
    for i in range(config.cross_ways):
        flow = np.array([0] * config.cross_total_tick)
        for j in range(0, config.cross_total_tick):
            for k in range(avg_flow[j][i]):
                value = int(random.randrange(j - config.env_range, j + config.env_range))
                flow[value % config.cross_total_tick] += 1
        sample_flow.append(flow)

    flow = np.transpose(sample_flow)
    if name is not None:
        with open('flow/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(flow)

    return flow


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
    avg_flow = np.array(avg_flow)

    with open('flow/avg_flow.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(avg_flow)
