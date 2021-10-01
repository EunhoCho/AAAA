import csv

import numpy as np
import torch
from tqdm import tqdm

from adaptive_crossroad import config, environment


def read_flow(number):
    if number == -1:
        file_name = '../flow/avg_flow.csv'
    elif config.ENV_METHOD == 'HARD':
        file_name = '../flow/flow_' + str(number) + '_hard.csv'
    else:
        file_name = '../flow/flow_' + str(number) + '.csv'

    flow_data = []
    with open(file_name, "r", newline='') as f:
        flow_reader = csv.reader(f)
        flows = np.array([0] * config.WAYS)
        for i, row in enumerate(flow_reader):
            flows = flows + np.array(row).astype(int)
            if i % config.TEN_SECOND_PER_TICK == 0:
                flow_data.append(flows)
                flows = np.array([0] * config.WAYS)

    return flow_data


def generate_out_flow(phase_length):
    phase = 0
    tick = 0

    flows = []
    for i in range(config.DECISION_LENGTH):
        flow = config.OUTPUT_FLOW[phase].copy()
        modified = [False] * config.WAYS

        if tick == 0:
            for j in range(config.WAYS):
                if not modified[j] and config.OUTPUT_FLOW[(phase - 1) % 6][j] == 0:
                    flow[j] //= 2
                    modified[j] = True

        if tick == phase_length[phase] - 1:
            for j in range(config.WAYS):
                if not modified[j] and config.OUTPUT_FLOW[(phase + 1) % 6][j] == 0:
                    flow[j] //= 2
                    modified[j] = True

        flows.append(flow)

        tick += 1
        while phase < 6 and tick == phase_length[phase]:
            phase += 1
            tick = 0

    return flows


def sim_run_crossroad(duration, decision, target_flow, default_num_cars):
    out_flow = generate_out_flow(decision)
    result = []
    num_cars = default_num_cars.copy()
    phase = 0
    phase_length = decision[:]
    phase_tick = 0

    for i in range(duration):
        num_cars = num_cars + target_flow[i] - out_flow[i % config.DECISION_LENGTH]

        for j in range(config.WAYS):
            if num_cars[j] < 0:
                num_cars[j] = 0

        result.append(num_cars)

        phase_tick += 1
        while phase_tick == phase_length[phase]:
            phase += 1
            phase %= 6
            phase_tick = 0

    return result


def decision_making_god(tick, num_cars, real_flow):
    tick = tick % config.TOTAL_TICK
    tactics = config.TACTICS[:]
    opt_tactic = []
    min_value = -1
    for tactic in tactics:
        result = sum(sum(sim_run_crossroad(config.DECISION_LENGTH, tactic,
                                           real_flow[tick:tick + config.DECISION_LENGTH], num_cars)))

        if min_value == -1 or min_value > result:
            opt_tactic = tactic
            min_value = result

    return opt_tactic

def decision_making_SMC(tick, num_cars, avg_flow, real_flow, erased_flow, started, tactics=config.TACTICS):
    sample_flow = []
    if config.ENV_METHOD == 'HARD':
        env_result, erased_flow = environment.sample_environment_hard(tick, tick + config.DECISION_LENGTH, avg_flow,
                                                                      real_flow, erased_flow, started,
                                                                      tick - config.DECISION_LENGTH)
        for i in range(config.SMC_SAMPLES - 1):
            env_result, erased_flow_tmp = environment.sample_environment_hard(tick, tick + config.DECISION_LENGTH,
                                                                              avg_flow, real_flow, erased_flow, started,
                                                                              tick)
            sample_flow.append(env_result)
    else:
        for i in range(config.SMC_SAMPLES):
            sample_flow.append(environment.sample_environment(tick, tick + config.DECISION_LENGTH, avg_flow))

    tactics = tactics[:]
    opt_tactic = []
    min_value = -1
    for tactic in tactics:
        result = 0
        for flow in sample_flow:
            result += sum(sum(sim_run_crossroad(config.DECISION_LENGTH, tactic, flow, num_cars)))

        if min_value == -1 or min_value > result:
            opt_tactic = tactic
            min_value = result

    return opt_tactic, erased_flow


def decision_making_AN(adaptive_network, tick, num_cars):
    state = torch.FloatTensor([tick, *num_cars]).to(config.AN_DEVICE)
    tactic = adaptive_network.model(state).data.min(0)[1].view(1, 1)
    return config.TACTICS[tactic]


def run_crossroad(name, crossroad_type, flow_number, default_decision=None, start_tick=0,
                  end_tick=config.TOTAL_TICK, tqdm_off=False):
    if default_decision is None:
        default_decision = config.DEFAULT_DECISION

    if crossroad_type == 'AN':
        adaptive_network = torch.load('../adaptive_network/adaptiveNetwork.pth')
    else:
        adaptive_network = None

    with open('../log/car/' + name + '_' + str(flow_number) + ('_hard' if config.ENV_METHOD == 'HARD' else '') + '.csv',
              'w', newline='') as log_file:
        with open('../log/dm/' + name + '_' + str(flow_number) +
                  ('_hard' if config.ENV_METHOD == 'HARD' else '') + '.csv', 'w', newline='') as log_dm_file:
            x_writer = csv.writer(log_file)
            x_writer.writerow(['tick', 'cars', '', '', '', '', '', '', '', 'flow'])
            dm_writer = csv.writer(log_dm_file)
            dm_writer.writerow(['tick', 'decision'])

            # Flow Generation
            avg_flow = read_flow(-1)
            target_flow = read_flow(flow_number)
            out_flow = generate_out_flow(default_decision)

            # Simulation Configuration
            end_tick -= (end_tick - start_tick) % config.DECISION_LENGTH

            # Fundamental Variables for Simulation
            result = []
            num_cars = np.array([0] * config.WAYS)
            phase = 0
            phase_length = default_decision
            phase_tick = 0
            phase_result = np.array([0] * config.WAYS)

            # Variables for Decision Making
            erased_flow = None

            if tqdm_off:
                tick_tqdm = range(start_tick, end_tick)
            else:
                tick_tqdm = tqdm(range(start_tick, end_tick))
                tick_tqdm.set_description("Crossroad - " + name)

            for i in tick_tqdm:
                if i % config.DECISION_LENGTH == 0:
                    phase = 0
                    phase_tick = 0
                    phase_result = np.array([0] * config.WAYS)

                    if crossroad_type == 'GOD':
                        phase_length = decision_making_god(i, num_cars, target_flow)
                    elif crossroad_type == 'SMC':
                        phase_length, erased_flow = decision_making_SMC(i, num_cars, avg_flow, target_flow,
                                                                        erased_flow, start_tick)
                    elif crossroad_type == 'AN':
                        phase_length = decision_making_AN(adaptive_network, i, num_cars)

                    dm_writer.writerow([i, *phase_length])
                    out_flow = generate_out_flow(phase_length)

                num_cars = num_cars + target_flow[i % config.TOTAL_TICK] - out_flow[i % config.DECISION_LENGTH]

                for j in range(config.WAYS):
                    if num_cars[j] < 0:
                        num_cars[j] = 0

                phase_result = phase_result + num_cars
                x_writer.writerow([i, *num_cars, *target_flow[i % config.TOTAL_TICK]])

                phase_tick += 1
                while phase < 6 and phase_tick == phase_length[phase]:
                    phase += 1
                    phase_tick = 0

                if i % config.DECISION_LENGTH == config.DECISION_LENGTH - 1:
                    result.append(sum(phase_result) / config.DECISION_LENGTH)

    point_result = []
    sum_result = 0
    for i in range(len(result)):
        sum_result += result[i]
        if i % config.TICK_PER_POINT == config.TICK_PER_POINT - 1:
            point_result.append(sum_result / config.TICK_PER_POINT)
            sum_result = 0

    return point_result
