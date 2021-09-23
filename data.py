import csv

import numpy as np

TIME = np.arange(24) * 60 * 60 + 1800
AStraight = np.array([29, 21, 15, 13, 18, 36, 67, 94, 94, 91, 85, 81, 76, 81, 82, 82, 84, 85, 83, 74, 66, 64, 55, 42]) / 36
BStraight = np.array([31, 22, 17, 14, 16, 31, 53, 75, 83, 81, 79, 79, 76, 79, 81, 84, 87, 90, 88, 83, 75, 70, 60, 46]) / 36
CStraight = np.array([8, 6, 5, 4, 5, 9, 21, 34, 37, 33, 30, 29, 26, 27, 27, 27, 26, 26, 24, 21, 19, 19, 16, 13]) / 36
DStraight = np.array([10, 7, 5, 4, 5, 8, 14, 22, 26, 26, 26, 27, 26, 28, 29, 29, 30 ,32, 31, 28, 24, 23, 19, 15]) / 36
ALeft = AStraight / 4
BLeft = BStraight / 4
CLeft = CStraight / 4
DLeft = DStraight / 4
ROADS = [AStraight, ALeft,
         BStraight, BLeft,
         CStraight, CLeft,
         DStraight, DLeft]

SAMPLES = 100
STDEV = 600

if __name__ == "__main__":
    road_data = []
    for road in ROADS:
        z = np.poly1d(np.polyfit(TIME, road, 5))

        residual = 0
        data = []
        for i in range(86400):
            value = 0
            residual += z(i)

            if residual >= 1:
                value += int(residual)
                residual -= int(residual)

            if residual >= 0.5:
                value += 1
                residual -= 1

            data.append(value)

        road_data.append(data)

    with open('data/avg_data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for i in range(86400):
            line = []
            for j in range(len(road_data)):
                line.append(road_data[j][i])

            csv_writer.writerow(line)

    for i in range(SAMPLES):
        crossroad_data = []
        for data in road_data:
            value = [0] * 86400

            for j in range(86400):
                cars = np.random.normal(j, STDEV, data[j]).astype(int)
                for car in cars:
                    if car < 0:
                        car += 86400
                    if car >= 86400:
                        car -= 86400
                    value[car] += 1

            crossroad_data.append(value)

        with open('data/data_' + str(i) + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            for j in range(86400):
                line = []
                for k in range(len(crossroad_data)):
                    line.append(crossroad_data[k][j])

                csv_writer.writerow(line)
