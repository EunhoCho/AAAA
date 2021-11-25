import csv

import numpy as np
import torch
from tqdm import tqdm

import anomaly
import config
import rl
import statistical_model_checking


def generate_outflow(decision, current_anomaly=None, is_sum=False):
    phase = 0
    tick = 0
    outflow = []

    for i in range(config.cross_decision_length):
        outflow.append(config.cross_out_flow[phase].copy())

        if current_anomaly is not None:
            outflow[-1][current_anomaly.way * 2] //= 2
            outflow[-1][current_anomaly.way * 2 + 1] //= 2

        tick += 1
        while phase < 6 and tick == decision[phase]:
            phase += 1
            tick = 0

    if is_sum:
        sum_v = np.array([0] * config.cross_ways)
        for i in range(config.cross_decision_length):
            sum_v += np.array(outflow[i])

        return sum_v

    return outflow


def decision_making_SMC(tick, state, is_sprt=False, tactics=None):
    if tactics is None:
        tactics = config.cross_tactics.copy()

    if is_sprt:
        verifier = statistical_model_checking.SPRT(tick, state, tactics)
    else:
        verifier = statistical_model_checking.SimpleMonteCarloChecker(tick, state, tactics)

    return verifier.choose_decision()


def decision_making_rl(tick, state, rl_model):
    state_tensor = torch.FloatTensor([*state, tick]).to(config.cuda_device)
    return rl_model.model(state_tensor).data.min(0)[1].view(1, 1)


def decision_making_rl_smc(tick, state, rl_model, is_sprt=False):
    state_tensor = torch.FloatTensor([*state, tick]).to(config.cuda_device)
    result = rl_model.model(state_tensor).data.sort()
    threshold_value = result.values[0] * (config.rl_smc_threshold + 1)

    candidates = []
    i = 0
    while result.values[i] <= threshold_value or i < config.rl_smc_min_candidates:
        candidates.append(config.cross_tactics[result.indices[i]])
        i += 1

    return decision_making_SMC(tick, state, is_sprt, candidates)


def decision_making_a_rl(tick, state, rl_model, anomaly_value):
    state_tensor = torch.FloatTensor([*state, tick, anomaly_value]).to(config.cuda_device)
    return config.cross_tactics[rl_model.model(state_tensor).data.min(0)[1].view(1, 1)]


def run_crossroad(start, inflow, decision, state, anomalies=None):
    current_anomaly = None
    if anomalies is not None:
        for single_anomaly in anomalies:
            if single_anomaly.valid(start):
                current_anomaly = single_anomaly

    outflow = generate_outflow(decision, current_anomaly)
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


def run(name, cross_type, start, end, inflow, anomalies, decision=None, tqdm_on=True):
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

            if cross_type == 'RL' or cross_type == 'A-RL' or cross_type == 'ORL':
                rl_model = rl.DQN(path='model/rl.pth').to(config.cuda_device)

            if cross_type == 'AAAA':
                a_rl_models = [
                    rl.DQN(path='model/a_rl_0.pth').to(config.cuda_device),
                    rl.DQN(path='model/a_rl_1.pth').to(config.cuda_device),
                    rl.DQN(path='model/a_rl_2.pth').to(config.cuda_device),
                    rl.DQN(path='model/a_rl_3.pth').to(config.cuda_device)
                ]
                anomaly_value = 4

            tick_tqdm = range(start, end, config.cross_decision_length)
            if tqdm_on:
                tick_tqdm = tqdm(tick_tqdm)
                tick_tqdm.set_description(name + " - " + cross_type)

            for i in tick_tqdm:
                if cross_type == 'SMC':
                    decision = decision_making_SMC(i, state)
                elif cross_type == 'RL' or cross_type == 'ORL':
                    tactic = decision_making_rl(i, state, rl_model)
                    decision = config.cross_tactics[tactic]
                elif cross_type == 'AAAA':
                    if anomaly_value != 4:
                        target_model = a_rl_models[anomaly_value]
                    else:
                        target_model = rl_model
                    tactic = decision_making_rl(i, state, target_model)
                    decision = config.cross_tactics[tactic]

                dm_writer.writerow([i, *decision])

                phase_result, next_state = run_crossroad(i, inflow, decision, state, anomalies)
                for j in range(i, i + config.cross_decision_length):
                    car_writer.writerow([j, *phase_result[j - i]])

                result.append(sum(sum(phase_result)) / config.cross_decision_length)

                if cross_type == 'ORL':
                    rl_model.push_optimize([*state, i], tactic, sum(sum(phase_result)),
                                           [*next_state, i + config.cross_decision_length])
                elif cross_type == 'AAAA':
                    anomaly_value = 4
                    for single_anomaly in anomalies:
                        if single_anomaly.valid(i - config.cross_decision_length):
                            anomaly_value = single_anomaly.way
                            break

                state = next_state

    return result
