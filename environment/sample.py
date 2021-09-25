import csv

import numpy as np
from tqdm import tqdm

from environment import config


def sample_environment(start, end, avg_flow):
    sample_environment = []
    for data in avg_flow:
        value = []
        for i in range(start, end):
            value.append(max(int(np.random.normal(data[i], (data[i] * config.STDEV_RATE) ** 2, 1)[0]), 0))

        sample_environment.append(value)

    return sample_environment


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

    with open('../flow/avg_flow.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for i in range(8640 // config.TEN_SECOND_PER_TICK):
            line = []
            for j in range(len(avg_flow)):
                line.append(avg_flow[j][i])

            csv_writer.writerow(line)

    for i in tqdm(range(config.SAMPLES)):
        crossroad_data = sample_environment(0, 8640 // config.TEN_SECOND_PER_TICK, avg_flow)

        with open('../flow/flow_' + str(i) + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            for j in range(8640):
                line = []
                for k in range(len(crossroad_data)):
                    line.append(crossroad_data[k][j])

                csv_writer.writerow(line)
