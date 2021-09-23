import csv

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

FLOW_NUMBER = 0
TICK_PER_POINT = 12
SECOND_PER_TICK = 10
DECISION_LENGTH = 9
DEFAULT_DECISION = [2, 2, 2, 1, 1, 1]
OUTPUT_FLOW = np.array([[30, 10, 0, 0, 0, 0, 0, 0],
                        [30, 0, 30, 0, 0, 0, 0, 0],
                        [0, 0, 30, 10, 0, 0, 0, 0],
                        [0, 0, 0, 0, 30, 10, 0, 0],
                        [0, 0, 0, 0, 30, 0, 30, 0],
                        [0, 0, 0, 0, 0, 0, 30, 10]]) * SECOND_PER_TICK

def read_flow(number):
    if number == -1:
        file_name = 'data/avg_data.csv'
    else:
        file_name = 'data/data_' + str(number) + '.csv'

    flow_data = []
    with open(file_name, "r", newline='') as f:
        flow_reader = csv.reader(f)
        flows = np.array([0] * 8)
        for i, row in enumerate(flow_reader):
            flows = flows + np.array(row).astype(int)
            if i % SECOND_PER_TICK == 0:
                flow_data.append(flows)
                flows = np.array([0] * 8)

    return flow_data

def generate_out_flow(phase_length):
    phase = 0
    tick = 0

    flows = []
    for i in range(DECISION_LENGTH):
        flow = OUTPUT_FLOW[phase].copy()

        if tick == 0:
            for j in range(8):
                if OUTPUT_FLOW[(i - 1) % 6][j] == 0:
                    flow[j] //= 2

        elif tick == phase_length[phase]:
            for j in range(8):
                if OUTPUT_FLOW[(i + 1) % 6][j] == 0:
                    flow[j] //= 2

        flows.append(flow)

        tick += 1
        while phase < 6 and tick == phase_length[phase]:
            phase += 1
            tick = 0

    return flows

def decision_making(crossroad_type, avg_flow, default_decision):
    if crossroad_type == 'PMC':
        pass
    elif crossroad_type == 'SMC':
        pass
    elif crossroad_type == 'VN':
        pass

    return default_decision


def run_crossroad(crossroad_type, flow_number=FLOW_NUMBER, default_decision=None):
    if default_decision is None:
        default_decision = DEFAULT_DECISION

    avg_flow = read_flow(-1)
    target_flow = read_flow(flow_number)
    out_flow = generate_out_flow(default_decision)

    result = []
    num_cars = np.array([0] * 8)
    tick = 0
    phase = 0
    phase_length = default_decision
    phase_tick = 0
    phase_result = np.array([0] * 8)

    with open('log/' + crossroad_type + '_' + str(flow_number) + '.csv', 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        for i in range(86400 // SECOND_PER_TICK):
            num_cars = num_cars + target_flow[i] - out_flow[i % DECISION_LENGTH]

            for j in range(8):
                if num_cars[j] < 0:
                    num_cars[j] = 0

            phase_result = phase_result + num_cars
            csv_writer.writerow(num_cars)

            tick += 1
            phase_tick += 1

            while phase < 6 and phase_tick == phase_length[phase]:
                phase += 1
                phase_tick = 0

            if phase == 6:
                phase_length = decision_making(crossroad_type, avg_flow, default_decision)
                out_flow = generate_out_flow(phase_length)
                phase = 0
                result.append(sum(phase_result) / DECISION_LENGTH)
                phase_result = np.array([0] * 8)

    point_result = []
    sum_result = 0
    for i in range(len(result)):
        sum_result += result[i]
        if i % TICK_PER_POINT == TICK_PER_POINT - 1:
            point_result.append(sum_result / TICK_PER_POINT)
            sum_result = 0

    return point_result


def run_tactic(tactic, flow_number):
    str_tactic = str(tactic[0]) + '_' + str(tactic[1]) + '_' + str(tactic[2]) + '_' + str(tactic[3]) + '_' + str(
        tactic[4]) + '_' + str(tactic[5])
    return run_crossroad(str_tactic, flow_number, tactic)

if __name__ == "__main__":
    tactics = []
    for i in range(1, DECISION_LENGTH - 4):
        for j in range(1, DECISION_LENGTH - i - 3):
            for k in range(1, DECISION_LENGTH - i - j - 2):
                for l in range(1, DECISION_LENGTH - i - j - k - 1):
                    for m in range(1, DECISION_LENGTH - i - j - k - l):
                        tactics.append([i, j, k, l, m, DECISION_LENGTH - i - j - k - l - m])

    for tactic in tqdm(tactics):
        for i in tqdm(range(100)):
            run_tactic(tactic, i)

    # point_result = run_crossroad('')
    # time = np.array(range(86400 // SECOND_PER_TICK // DECISION_LENGTH // TICK_PER_POINT))\
    #        / 3600 * SECOND_PER_TICK * DECISION_LENGTH * TICK_PER_POINT
    # plt.plot(time, point_result, label='Check')
    #
    # plt.title('Time - Number of Waiting Cars')
    # plt.legend(loc='upper left')
    # plt.xlabel('Hour')
    # plt.ylabel('Number of Cars')
    # plt.xticks([0, 6, 12, 18, 24], [0, 6, 12, 18, 24])
    # plt.savefig('waiting.png', dpi=300)
    # plt.show()