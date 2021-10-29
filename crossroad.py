import csv
import random

import numpy as np
import torch
from tqdm import tqdm

import car_accident
import config
import environment
import reinforcement_learning


def generate_outflow(decision, accident):
    phase = 0
    tick = 0
    outflow = []

    for i in range(config.cross_decision_length):
        outflow.append(config.cross_out_flow[phase].copy())

        if accident is not None:
            if accident.valid(tick):
                outflow[-1][accident.way * 2] //= 2
                outflow[-1][accident.way * 2 + 1] //= 2

        tick += 1
        while phase < 6 and tick == decision[phase]:
            phase += 1
            tick = 0

    return outflow


def decision_making_god(tick, inflow, state, accidents):
    tactics = config.cross_tactics.copy()
    opt_tactic = None
    min_value = -1

    for tactic in tactics:
        result, _ = run_crossroad(tick, inflow, tactic, state, accidents)
        sum_result = sum(sum(result))
        if min_value == -1 or min_value > sum_result:
            opt_tactic = tactic
            min_value = sum_result

    return opt_tactic


def decision_making_SMC(tick, avg_flow, state, tactics=None):
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

    if tactics is None:
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


def decision_making_rl(state, rl_model):
    state_tensor = torch.FloatTensor(state).to(config.rl_device)
    return config.cross_tactics[rl_model.model(state_tensor).data.min(0)[1].view(1, 1)]


def decision_making_rl_smc(tick, state, rl_model, avg_flow):
    state_tensor = torch.FloatTensor(state).to(config.rl_device)
    result = list(rl_model.model(state_tensor).data.view(56, ))
    result_index = [result.index(i) for i in sorted(result)]

    possible_tactics = []
    for i in range(config.rl_smc_candidates):
        possible_tactics.append(config.cross_tactics[result_index[i]])

    return decision_making_SMC(tick, avg_flow, state, possible_tactics)


def decision_making_a_rl(state, rl_model, accident_value):
    state_tensor = torch.FloatTensor([*state, accident_value]).to(config.rl_device)
    return config.cross_tactics[rl_model.model(state_tensor).data.min(0)[1].view(1, 1)]


def run_crossroad(start: int, inflow: list[[int]], decision: list[int], state: np.ndarray,
                  accidents: list[car_accident.Accident] = None):
    current_accident = None
    if accidents is not None:
        for accident in accidents:
            if accident.valid(start):
                current_accident = accident

    outflow = generate_outflow(decision, current_accident)
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


def run(name: str, cross_type: str, start: int, end: int, inflow: list[np.ndarray], accidents: list[car_accident.Accident],
        decision: list[int] = None, tqdm_on: bool = True):
    if decision is None:
        decision = config.cross_default_decision

    with open('log/car/' + name + '_' + cross_type + '.csv', 'w', newline='') as car_log_file:
        with open('log/dm/' + name + '_' + cross_type + '.csv', 'w', newline='') as dm_log_file:
            car_writer = csv.writer(car_log_file)
            car_writer.writerow(['tick', 'cars'])
            dm_writer = csv.writer(dm_log_file)
            dm_writer.writerow(['tick', 'decision'])

            # Simulation Configuration
            end -= (end - start) % config.cross_decision_length
            state = np.array([0] * config.cross_ways)
            result = []

            if cross_type == 'SMC' or cross_type == 'RL-SMC':
                avg_flow = environment.read_flow()

            if cross_type == 'RL' or cross_type == 'RL-SMC':
                rl_model = reinforcement_learning.DQN(path='model/rl.pth').to(config.rl_device)

            if cross_type == 'A-RL':
                rl_model = reinforcement_learning.DQN(config.cross_ways + 1, path='model/a_rl.pth').to(config.rl_device)

            # if cross_type == 'AD-DQN':
            #     ad_model = AD.AD()
            #     ad_model.load_state_dict(torch.load('model/ad.pth'))
            #     ad_model.eval()
            #     rl_model = reinforcement_learning.DQN()
            #     rl_model.load_state_dict(torch.load('model/ad_rl.pth'))
            #     rl_model.eval()

            tick_tqdm = range(start, end, config.cross_decision_length)
            if tqdm_on:
                tick_tqdm = tqdm(tick_tqdm)
                tick_tqdm.set_description(name + " - " + cross_type)

            for i in tick_tqdm:
                if cross_type == 'GOD':
                    decision = decision_making_god(i, inflow, state, accidents)
                elif cross_type == 'SMC':
                    decision = decision_making_SMC(i, avg_flow, state)
                elif cross_type == 'RL':
                    decision = decision_making_rl(state, rl_model)
                elif cross_type == 'RL-SMC':
                    decision = decision_making_rl_smc(i, state, rl_model, avg_flow)
                elif cross_type == 'A-RL':
                    accident_value = -1
                    for accident in accidents:
                        if accident.valid(i - config.cross_decision_length):
                            accident_value = accident.way
                            break
                    decision = decision_making_a_rl(state, rl_model, accident_value)

                dm_writer.writerow([i, *decision])

                phase_result, state = run_crossroad(i, inflow, decision, state, accidents)
                for j in range(i, i + config.cross_decision_length):
                    car_writer.writerow([j, *phase_result[j - i]])

                result.append(sum(sum(phase_result)) / config.cross_decision_length)

    return result
