import csv

import numpy as np

if __name__ == '__main__':
    config = {}
    with open('config.txt', 'r') as config_file:
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

    flow = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            if i != j:
                div = sum(data) / int(target_data[i][j])
                route_data = data / div

                residual = 0
                route_flow = []
                for k in range(total_ticks_per_day):
                    value = route_data[k]
                    residual += value - int(value)
                    route_flow.append(int(value) + int(residual))
                    residual -= int(residual)
                flow[i].append(route_flow)
            else:
                flow[i].append(None)

    csv_file = open(config['FLOW_DATA'], 'w', newline='')
    csv_writer = csv.writer(csv_file)
    for k in range(total_ticks_per_day):
        value = []
        for i in range(4):
            for j in range(4):
                if i == j:
                    value.append(0)
                else:
                    value.append(flow[i][j][k])
        csv_writer.writerow(value)

    csv_file.close()