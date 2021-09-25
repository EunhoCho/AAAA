import csv

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from crossroad import config
from environment.sample import sample_environment


def read_flow(number):
    if number == -1:
        file_name = '../flow/avg_flow.csv'
    else:
        file_name = '../flow/flow_' + str(number) + '.csv'

    flow_data = []
    with open(file_name, "r", newline='') as f:
        flow_reader = csv.reader(f)
        flows = np.array([0] * 8)
        for i, row in enumerate(flow_reader):
            flows = flows + np.array(row).astype(int)
            if i % config.TEN_SECOND_PER_TICK == 0:
                flow_data.append(flows)
                flows = np.array([0] * 8)

    return flow_data


def generate_out_flow(phase_length):
    phase = 0
    tick = 0

    flows = []
    for i in range(config.DECISION_LENGTH):
        flow = config.OUTPUT_FLOW[phase].copy()

        if tick == 0:
            for j in range(8):
                if config.OUTPUT_FLOW[(i - 1) % 6][j] == 0:
                    flow[j] //= 2

        elif tick == phase_length[phase]:
            for j in range(8):
                if config.OUTPUT_FLOW[(i + 1) % 6][j] == 0:
                    flow[j] //= 2

        flows.append(flow)

        tick += 1
        while phase < 6 and tick == phase_length[phase]:
            phase += 1
            tick = 0

    return flows


def sim_run_crossroad(duration, decision, target_flow):
    out_flow = generate_out_flow(decision)
    result = []
    num_cars = np.array([0] * 8)
    phase = 0
    phase_length = decision[:]
    phase_tick = 0

    for i in range(duration):
        num_cars = num_cars + target_flow[i] - out_flow[i % config.DECISION_LENGTH]

        for j in range(8):
            if num_cars[j] < 0:
                num_cars[j] = 0

        result.append(sum(num_cars))

        phase_tick += 1
        while phase_tick == phase_length[phase]:
            phase += 1
            phase %= 6
            phase_tick = 0

    return result


def decision_making(crossroad_type, avg_flow, tick, default_decision):
    if crossroad_type == 'PMC':
        pass

    elif crossroad_type == 'SMC':
        sample_flow = []
        for i in range(config.SMC_SAMPLES):
            sample_flow.append(sample_environment(tick, tick + 9, avg_flow))

        tactics = config.TACTICS[:]
        opt_tactic = []
        min_value = -1
        for tactic in tactics:
            result = 0
            for flow in sample_flow:
                result += sum(sim_run_crossroad(9, tactic, flow))

            if min_value == -1 or min_value > result:
                opt_tactic = tactic
                min_value = result

        return opt_tactic

    elif crossroad_type == 'VN':
        pass

    return default_decision


def run_crossroad(name, crossroad_type, flow_number=config.FLOW_NUMBER, default_decision=None):
    if default_decision is None:
        default_decision = config.DEFAULT_DECISION

    avg_flow = read_flow(-1)
    target_flow = read_flow(flow_number)
    out_flow = generate_out_flow(default_decision)

    result = []
    num_cars = np.array([0] * 8)
    phase = 0
    phase_length = default_decision
    phase_tick = 0
    phase_result = np.array([0] * 8)

    with open('../log/x/' + name + '_' + str(flow_number) + '.csv', 'w', newline='') as log_x_file:
        with open('../log/y/' + name + '_' + str(flow_number) + '.csv', 'w', newline='') as log_y_file:
            x_writer = csv.writer(log_x_file)
            y_writer = csv.writer(log_y_file)

            sum_decision_result = 0
            for i in tqdm(range(8640 // config.TEN_SECOND_PER_TICK)):
                num_cars = num_cars + target_flow[i] - out_flow[i % config.DECISION_LENGTH]

                for j in range(8):
                    if num_cars[j] < 0:
                        num_cars[j] = 0

                phase_result = phase_result + num_cars
                x_writer.writerow([i, *num_cars, *target_flow[i]])

                phase_tick += 1
                while phase < 6 and phase_tick == phase_length[phase]:
                    phase += 1
                    phase_tick = 0

                if phase == 6:
                    y_writer.writerow([sum_decision_result])
                    sum_decision_result = 0

                    phase = 0
                    phase_length = decision_making(crossroad_type, avg_flow, i, default_decision)
                    out_flow = generate_out_flow(phase_length)

                    result.append(sum(phase_result) / config.DECISION_LENGTH)
                    phase_result = np.array([0] * 8)

    point_result = []
    sum_result = 0
    for i in range(len(result)):
        sum_result += result[i]
        if i % config.TICK_PER_POINT == config.TICK_PER_POINT - 1:
            point_result.append(sum_result / config.TICK_PER_POINT)
            sum_result = 0

    return point_result


if __name__ == "__main__":
    name = 'SMC_TEST'
    point_result = run_crossroad(name, 'SMC', config.FLOW_NUMBER)
    time = np.array(range(8640 // config.TEN_SECOND_PER_TICK // config.DECISION_LENGTH // config.TICK_PER_POINT)) \
           / 3600 * config.TEN_SECOND_PER_TICK * config.DECISION_LENGTH * config.TICK_PER_POINT
    plt.plot(time, point_result, label='Check')

    plt.title('Time - Number of Waiting Cars')
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Number of Cars')
    plt.xticks([0, 6, 12, 18, 24], [0, 6, 12, 18, 24])
    plt.savefig('figure/' + name + '_' + config.FLOW_NUMBER + '.png', dpi=300)
    plt.show()
