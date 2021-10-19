import csv
import random

import numpy as np
import torch
from tqdm import tqdm

from adaptive_crossroad import config, DQN


def read_flow(number):
    if number == -1:
        file_name = '../flow/avg_flow.csv'
    else:
        file_name = '../flow/flow_' + str(number) + '.csv'

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


def generate_outflow(decision):
    phase = 0
    tick = 0
    outflow = []

    for i in range(config.cross_decision_length):
        outflow.append(config.cross_out_flow[phase].copy())
        tick += 1
        while phase < 6 and tick == decision[phase]:
            phase += 1
            tick = 0

    return outflow


def run_crossroad(start: int, inflow: list[[int]], decision: list[int], state: np.ndarray):
    outflow = generate_outflow(decision)
    result = []
    state = state.copy()
    phase = 0
    phase_tick = 0

    for i in range(start, start + config.cross_decision_length):
        state = state + inflow[i % config.cross_total_tick] - outflow[i % config.cross_decision_length]

        for j in range(config.cross_ways):
            if state[j] < 0:
                state[j] = 0

        result.append(state.copy())

    phase_tick += 1
    while phase_tick == decision[phase]:
        phase += 1
        phase %= 6
        phase_tick = 0

    return result, state


def decision_making_god(tick, inflow, state):
    tactics = config.cross_tactics.copy()
    opt_tactic = None
    min_value = -1

    for tactic in tactics:
        result, _ = run_crossroad(tick, inflow, tactic, state)
        sum_result = sum(sum(result))
        if min_value == -1 or min_value > sum_result:
            opt_tactic = tactic
            min_value = sum_result

    return opt_tactic


def decision_making_SMC(tick, avg_flow, state):
    sample_flows = []
    for i in range(config.smc_samples):
        sample_flow = []
        for j in range(config.cross_decision_length):
            flow = []
            for k in range(config.cross_ways):
                avg_number = avg_flow[tick % config.cross_total_tick][k]
                low_number = int(config.smc_rate_low * avg_number)
                high_number = int(config.smc_rate_high * avg_number)

                if low_number == high_number:
                    flow.append(high_number)

                else:
                    flow.append(random.randrange(low_number, high_number + 1))

            sample_flow.append(flow)
        sample_flows.append(sample_flow)

    tactics = config.cross_tactics.copy()
    opt_tactic = None
    min_value = -1

    for tactic in tactics:
        sum_result = 0
        for sample in sample_flows:
            result, _ = run_crossroad(0, sample, tactic, state)
            sum_result += sum(sum(result))

        if min_value == -1 or min_value > sum_result:
            opt_tactic = tactic
            min_value = sum_result

    return opt_tactic


def decision_making_dqn(tick, state, dqn_model):
    state_tensor = torch.FloatTensor([tick, *state]).to(config.dqn_device)
    return config.cross_tactics[dqn_model.model(state_tensor).data.min(0)[1].view(1, 1)]


def run(name: str, cross_type: str, start: int, end: int, flow_number: int, decision: list[int] = None,
        tqdm_off: bool = False):
    if decision is None:
        decision = config.cross_default_decision

    with open('../log/car/' + name + '_' + str(flow_number) + '.csv', 'w', newline='') as car_log_file:
        with open('../log/dm/' + name + '_' + str(flow_number) + '.csv', 'w', newline='') as dm_log_file:
            car_writer = csv.writer(car_log_file)
            car_writer.writerow(['tick', 'cars'])
            dm_writer = csv.writer(dm_log_file)
            dm_writer.writerow(['tick', 'decision'])

            # Simulation Configuration
            inflow = read_flow(flow_number)
            end -= (end - start) % config.cross_decision_length
            state = np.array([0] * config.cross_ways)
            result = []

            if cross_type == 'SMC':
                avg_flow = read_flow(-1)

            if cross_type == 'DQN':
                dqn_model = DQN.DQN().to(config.dqn_device)
                dqn_model.load_state_dict(torch.load('../model/dqn.pth'))
                dqn_model.eval()

            # if cross_type == 'AD-DQN':
            #     ad_model = AD.AD()
            #     ad_model.load_state_dict(torch.load('../model/ad.pth'))
            #     ad_model.eval()
            #     dqn_model = DQN.DQN()
            #     dqn_model.load_state_dict(torch.load('../model/ad_dqn.pth'))
            #     dqn_model.eval()

            tick_tqdm = range(start, end, config.cross_decision_length)
            if not tqdm_off:
                tick_tqdm = tqdm(tick_tqdm)
                tick_tqdm.set_description("Crossroad - " + name)

            for i in tick_tqdm:
                if cross_type == 'GOD':
                    decision = decision_making_god(i, inflow, state)
                elif cross_type == 'SMC':
                    decision = decision_making_SMC(i, avg_flow, state)
                elif cross_type == 'DQN':
                    decision = decision_making_dqn(i, state, dqn_model)
                # elif cross_type == 'AD-DQN':
                #     decision_making_ad_dqn(i, state, ad_model, dqn_model)

                dm_writer.writerow([i, *decision])

                phase_result, state = run_crossroad(i, inflow, decision, state)
                for j in range(i, i + config.cross_decision_length):
                    car_writer.writerow([j, *phase_result[j - i]])

                result.append(sum(sum(phase_result)) / config.cross_decision_length)

    return result
