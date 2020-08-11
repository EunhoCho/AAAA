import csv

import numpy as np


def generate_inflow(raw_flow, name=None):
    """
    Generate the discrete inflow based on the raw inflow, and store into the csv file

    :param raw_flow: Average inflow of each tick / road
    :param name: The name of the csv file that contains the inflow
    :return: List of generated inflow - tick * 4 * 4 matrix
    """
    total_ticks = len(raw_flow[0][1])
    flow = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            if i != j:
                residual = 0
                route_flow = []
                for k in range(total_ticks):
                    value = raw_flow[i][j][k]

                    residual += value

                    if residual < 0:
                        flow_val = 0
                    else:
                        flow_val = int(residual)
                        residual -= int(residual)

                        if residual > 0:
                            random_val = np.random.choice([0, 1], 1, p=[1 - residual, residual])[0]
                            residual -= random_val

                    route_flow.append(flow_val)

                flow[i].append(route_flow)
            else:
                flow[i].append(None)

    if name is not None:
        csv_file = open(name, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        for k in range(total_ticks):
            value = []
            for i in range(4):
                for j in range(4):
                    if i == j:
                        value.append(0)
                    else:
                        value.append(flow[i][j][k])
            csv_writer.writerow(value)

        csv_file.close()

    return flow


def generate_flow_data(config_file='config.txt'):
    """
    Make a flow data on the configuration file, and record it into the csv file.

    :param config_file: The configuration file of the flow data.
    :return: None.
    """
    config = {}
    with open(config_file, 'r') as config_file:
        config_lines = config_file.readlines()
        for config_line in config_lines:
            config_split = config_line.split(':')
            config_type = config_split[0]
            config_content = config_split[1].strip('\n')
            config[config_type] = config_content

    config['SECONDS_PER_TICK'] = int(config['SECONDS_PER_TICK'])
    total_ticks_per_hour = int(60 * 60 / config['SECONDS_PER_TICK'])
    total_ticks_per_day = total_ticks_per_hour * 24

    target_data = []
    for i in range(4):
        target_data.append(config['FLOW_' + str(i)].split(','))

    raw_formula = config['EQUATION'].split(',')
    formula = []
    for i in range(len(raw_formula)):
        formula.append(float(raw_formula[i]))

    for i in range(len(formula)):
        formula[i] = formula[i] / (total_ticks_per_hour ** (len(formula) - i - 1))

    data = []
    for i in range(total_ticks_per_day):
        value = 0
        for j in range(len(formula)):
            value += formula[j] * (i ** (len(formula) - j - 1))
        data.append(round(value))

    data = np.array(data)

    raw_flow = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            if i != j:
                div = sum(data) / int(target_data[i][j])
                route_data = data / div

                route_flow = []
                for k in range(total_ticks_per_day):
                    value = route_data[k]
                    route_flow.append(value)

                raw_flow[i].append(route_flow)
            else:
                raw_flow[i].append(None)

    csv_file = open('raw_' + config['FLOW_DATA'], 'w', newline='')
    csv_writer = csv.writer(csv_file)

    for k in range(total_ticks_per_day):
        value = []
        for i in range(4):
            for j in range(4):
                if i == j:
                    value.append(0)
                else:
                    value.append(raw_flow[i][j][k])
        csv_writer.writerow(value)
    csv_file.close()

    generate_inflow(raw_flow, config['FLOW_DATA'])


if __name__ == '__main__':
    generate_flow_data()
