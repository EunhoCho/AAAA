import csv

import numpy as np
import torch
from tqdm import tqdm

import SPRT
import anomaly
import config
import reinforcement_learning


def generate_outflow(decision, current_anomaly):
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

    return outflow


def decision_making_god(tick, inflow, state, anomalies):
    tactics = config.cross_tactics.copy()
    opt_tactic = None
    min_value = -1

    for tactic in tactics:
        result, _ = run_crossroad(tick, inflow, tactic, state, anomalies)
        sum_result = sum(sum(result))
        if min_value == -1 or min_value > sum_result:
            opt_tactic = tactic
            min_value = sum_result

    return opt_tactic


def decision_making_SMC(tick, state, tactics=None):
    if tactics is None:
        tactics = config.cross_tactics.copy()

    sprt_verifier = SPRT.SPRT(tick)

    target_num = max(sum(state), config.cross_decision_length * 10)
    var = max(sum(state), config.cross_decision_length * 10)
    plus_trend = None

    while True:
        results = []
        for tactic in tactics:
            prob = sprt_verifier.verify_simulation(state, tactic, target_num)

            results.append((prob, tactic))

        results.sort(key=lambda x: -x[0])
        i = 0
        while i < len(results):
            if results[i][0] >= 0.95:
                i += 1
            else:
                break

        if i == 0:
            if plus_trend is not None and not plus_trend:
                var /= 2
            target_num += var
            plus_trend = True

        elif i == 1:
            return results[i][1]

        else:
            if target_num <= 0:
                return results[0][1]

            tactics = []
            for j in range(i):
                tactics.append(results[j][1])
            if plus_trend is not None and plus_trend:
                var /= 2
            target_num -= var


def decision_making_rl(tick, state, rl_model):
    state_tensor = torch.FloatTensor([*state, tick]).to(config.cuda_device)
    return rl_model.model(state_tensor).data.min(0)[1].view(1, 1)


def decision_making_rl_smc(tick, state, rl_model):
    state_tensor = torch.FloatTensor([*state, tick]).to(config.cuda_device)
    result = rl_model.model(state_tensor).data.sort()
    threshold_value = result.values[0] * (config.rl_smc_threshold + 1)

    candidates = []
    i = 0
    while result.values[i] <= threshold_value or i < config.rl_smc_min_candidates:
        candidates.append(config.cross_tactics[result.indices[i]])
        i += 1

    return decision_making_SMC(tick, state, candidates)


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

            if cross_type == 'RL' or cross_type == 'RL-SMC' or cross_type == 'ORL':
                rl_model = reinforcement_learning.DQN(path='model/rl.pth').to(config.cuda_device)

            if cross_type == 'A-RL' or cross_type == 'AD-RL':
                rl_model = reinforcement_learning.DQN(True, path='model/a_rl.pth').to(
                    config.cuda_device)

            if cross_type == 'AD-RL':
                ad_model = anomaly.CarAccidentDetector(path='model/ad.pth').to(config.cuda_device)
                anomaly_value = 4

            tick_tqdm = range(start, end, config.cross_decision_length)
            if tqdm_on:
                tick_tqdm = tqdm(tick_tqdm)
                tick_tqdm.set_description(name + " - " + cross_type)

            for i in tick_tqdm:
                if cross_type == 'GOD':
                    decision = decision_making_god(i, inflow, state, anomalies)
                elif cross_type == 'SMC':
                    decision = decision_making_SMC(i, state)
                elif cross_type == 'RL' or cross_type == 'ORL':
                    tactic = decision_making_rl(i, state, rl_model)
                    decision = config.cross_tactics[tactic]
                elif cross_type == 'RL-SMC':
                    decision = decision_making_rl_smc(i, state, rl_model)
                elif cross_type == 'A-RL':
                    anomaly_value = 4
                    for single_anomaly in anomalies:
                        if single_anomaly.valid(i - config.cross_decision_length):
                            anomaly_value = single_anomaly.way
                            break
                    decision = decision_making_a_rl(i, state, rl_model, anomaly_value)
                elif cross_type == 'AD-RL':
                    decision = decision_making_a_rl(i, state, rl_model, anomaly_value)

                dm_writer.writerow([i, *decision])

                phase_result, next_state = run_crossroad(i, inflow, decision, state, anomalies)
                for j in range(i, i + config.cross_decision_length):
                    car_writer.writerow([j, *phase_result[j - i]])

                result.append(sum(sum(phase_result)) / config.cross_decision_length)

                if cross_type == 'ORL':
                    rl_model.push_optimize([*state, i], tactic, sum(sum(phase_result)),
                                           [*next_state, i + config.cross_decision_length])
                if cross_type == 'AD-RL':
                    data = [i]
                    for j in range(config.cross_decision_length):
                        data.extend(phase_result[j])
                    anomaly_tensor = ad_model(torch.FloatTensor(data).to(config.cuda_device))
                    anomaly_value = torch.argmax(anomaly_tensor)

                state = next_state

    return result
