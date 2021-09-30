import csv

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from tqdm import tqdm

from adaptive_crossroad.value_net import ValueNet
from environment.sample import sample_environment, sample_environment_hard
from adaptive_crossroad import config

VALUE_NETS = {}


def read_flow(number):
    if number == -1:
        file_name = '../flow/avg_flow.csv'
    elif config.METHOD == 'HARD':
        file_name = '../flow/flow_' + str(number) + '_hard.csv'
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


def sim_run_crossroad(duration, decision, target_flow, default_num_cars):
    out_flow = generate_out_flow(decision)
    result = []
    num_cars = default_num_cars.copy()
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


def decision_making_god(tick, num_cars, real_flow):
    tick = tick % config.TOTAL_TICK
    tactics = config.TACTICS[:]
    opt_tactic = []
    min_value = -1
    for tactic in tactics:
        result = sum(sim_run_crossroad(9, tactic, real_flow[tick:tick + 9], num_cars))

        if min_value == -1 or min_value > result:
            opt_tactic = tactic
            min_value = result

    return opt_tactic

def decision_making_PMC(tick, num_cars, avg_flow, default_decision):
    for tactic in config.TACTICS:
        with open('../prism/crossroad.sm', 'w') as prism_file:
            prism_file.write('const int decision_length = ' + str(config.DECISION_LENGTH) + ';\n')
            clk_module = ["module clk\n", "\ttime : [0..decision_length + 1] init 0;\n",
                          "\treadyToTick : bool init true;\n", "\treadyToTack : bool init false;\n",
                          "\t[tick] readyToTick & time < decision_length + 1 -> 1 : (time' = time+ 1) & (readyToTick' = false) & (readyToTack' = true);\n",
                          "\t[tack] readyToTack -> 1 : (readyToTack' = false);\n"
                          "\t[toe] !readyToTick & !readyToTack -> 1 : (readyToTick'= true);\n", "endmodule\n\n",
                          'rewards "cars"\n',
                          "\t[toe] true : cars0 + cars1 + cars2 + cars3 + cars4 + cars5 + cars6 + cars7;\n",
                          "endrewards\n\n"]
            prism_file.writelines(clk_module)

            env_module = ["module env\n", "\tenv_state : [0..2] init 0;\n",
                          "\t[tick] true -> 0.5 : (env_state' = 0) + 0.5 : (env_state' = 1);\n",
                          "endmodule\n\n"]
            prism_file.writelines(env_module)

            for i in range(8):
                formula_module = ["formula inflow" + str(i) + " =\n"]

                for j in range(2):
                    if j != 0:
                        inflow = "+ (env_state = " + str(j) + " ? "
                    else:
                        inflow = "(env_state = " + str(j) + " ? "

                    for k in range(1, config.DECISION_LENGTH):
                        target_inflow = avg_flow[tick + k + 1][i]
                        if j == 0:
                            target_inflow = target_inflow - int(target_inflow * config.STDEV_RATE)
                        else:
                            target_inflow = target_inflow + int(target_inflow * config.STDEV_RATE)
                        inflow += "(time = " + str(k) + " ? " + str(target_inflow) + " : "

                    inflow += "0" + ")" * (config.DECISION_LENGTH - 1) + " : 0)"
                    if j == 1:
                        inflow += ";"
                    inflow += "\n"
                    formula_module.append(inflow)
                formula_module.append("\n")
                prism_file.writelines(formula_module)

            sys_module = ["module sys\n", "\treadyToInflow : bool init false;\n",
                          "\treadyToOutflow : bool init false;\n"]
            for i in range(8):
                sys_module.append(
                    "\tcars" + str(i) + " : [0.." + str(config.DECISION_LENGTH * config.MAX_INFLOW) + "] init " + str(
                        num_cars[i]) + ";\n")
            sys_module.append("\n\t[tick] !readyToInflow & !readyToOutflow -> 1 : (readyToInflow' = true);\n")

            inflow_line = "\t[sys_tactic] readyToInflow -> 1 : (readyToInflow' = false) & (readyToOutflow' = true)"
            for i in range(8):
                inflow_line += " & (cars" + str(i) + "' = cars" + str(i) + " + inflow" + str(i) + ")"
            inflow_line += ";\n"

            outflow_line = "\t[tack] readyToOutflow -> 1 : (readyToOutflow' = false)"
            for i in range(8):
                outflow_line += " & (cars" + str(i) + "' = cars" + str(i) + " - outflow" + str(i) + ")"
            outflow_line += ";\n"

            sys_module.append(inflow_line)
            sys_module.append(outflow_line)
            sys_module.append("endmodule\n\n")
            prism_file.writelines(sys_module)

            tactic_module = ["module tactic\n", "\tphase : [0..5] init 0;\n",
                             "\treadyToAdvance : bool init false;\n", "\treadyToStart : bool init false;\n",
                             "\t[sys_tactic] !readyToAdvance & !readyToStart -> 1 : (readyToAdvance' = true);\n\n"]

            phase = 0
            phase_tick = 0
            for i in range(config.DECISION_LENGTH):
                tactic_module.append("\t[] readyToAdvance & time = " + str(i) + " -> (phase' = " +
                                     str(phase) + ") & (readyToAdvance' = false) & (readyToStart' = true);\n")
                phase_tick += 1
                while phase < 6 and tactic[phase] == phase_tick:
                    phase += 1
                    phase_tick = 0

            tactic_module.append("\t[tack] readyToStart -> 1 : (readyToStart' = false);\n")
            tactic_module.append("endmodule\n\n")
            prism_file.writelines(tactic_module)

            outflows = [[config.STRAIGHT_OUT, [True, True, False, False, False, False]],
                        [config.LEFT_OUT, [True, False, False, False, False, False]],
                        [config.STRAIGHT_OUT, [False, True, True, False, False, False]],
                        [config.LEFT_OUT, [False, False, True, False, False, False]],
                        [config.STRAIGHT_OUT, [False, False, False, True, True, False]],
                        [config.LEFT_OUT, [False, False, False, True, False, False]],
                        [config.STRAIGHT_OUT, [False, False, False, False, True, True]],
                        [config.LEFT_OUT, [False, False, False, False, False, True]]]

            for i in range(8):
                single_outflow = outflows[i]
                outflow_formula = ["formula raw_outflow" + str(i) + " = "]
                for j in range(6):
                    if j != 0:
                        outflow_formula.append("+ ")
                    outflow_formula.append("(phase = " + str(j) + " ? ")
                    if single_outflow[1][j]:
                        outflow_formula.append(str(single_outflow[0]))
                    else:
                        outflow_formula.append("0")
                    outflow_formula.append(" : 0)")
                    if j == 5:
                        outflow_formula.append(";")
                    outflow_formula.append("\n")
                outflow_formula.append("\n")
                prism_file.writelines(outflow_formula)

            for i in range(8):
                outflow_formula = [
                    "formula outflow" + str(i) + " = (raw_outflow" + str(i) + " > cars" + str(i) + " ? cars" + str(
                        i) + " : raw_outflow" + str(i) + ");\n\n"]
                prism_file.writelines(outflow_formula)

    return default_decision


def decision_making_SMC(tick, num_cars, avg_flow, real_flow, erased_flow, started, tactics=config.TACTICS):
    sample_flow = []
    if config.METHOD == 'HARD':
        env_result, erased_flow = sample_environment_hard(tick, tick + config.DECISION_LENGTH, avg_flow,
                                                          real_flow, erased_flow, started, tick - config.DECISION_LENGTH)
        for i in range(config.SMC_SAMPLES - 1):
            env_result, erased_flow_tmp = sample_environment_hard(tick, tick + config.DECISION_LENGTH, avg_flow,
                                                                  real_flow, erased_flow, started, tick)
            sample_flow.append(env_result)
    else:
        for i in range(config.SMC_SAMPLES):
            sample_flow.append(sample_environment(tick, tick + config.DECISION_LENGTH, avg_flow))

    tactics = tactics[:]
    opt_tactic = []
    min_value = -1
    for tactic in tactics:
        result = 0
        for flow in sample_flow:
            result += sum(sim_run_crossroad(9, tactic, flow, num_cars))

        if min_value == -1 or min_value > result:
            opt_tactic = tactic
            min_value = result

    return opt_tactic, erased_flow


def decision_making_VN(vn_data):
    if len(VALUE_NETS.keys()) == 0:
        for tactic in config.TACTICS:
            str_tactic = config.tactic_string(tactic)
            valueNet = ValueNet(config.VN_INPUT_SIZE, config.VN_HIDDEN_SIZE, config.VN_LAYERS, config.DECISION_LENGTH,
                                '../valueNet/valueNet/' + str_tactic + '.torch').to(config.DEVICE)
            ss = joblib.load('../valueNet/scaler/' + str_tactic + '.sc')
            VALUE_NETS[str_tactic] = [valueNet, ss]

    ms = MinMaxScaler()
    ms.fit([[config.VN_Y_SCALER_MIN], [config.VN_Y_SCALER_MAX]])

    tactic_result = []
    for tactic in config.TACTICS:
        str_tactic = config.tactic_string(tactic)
        valueNet = VALUE_NETS[str_tactic][0]
        ss = VALUE_NETS[str_tactic][1]

        x_ss = ss.transform(vn_data)
        x_tensor = Variable(torch.Tensor(x_ss))
        x_tensor_reshaped = torch.reshape(x_tensor, (int(x_tensor.shape[0] / config.DECISION_LENGTH),
                                                     config.DECISION_LENGTH, x_tensor.shape[1])).to(config.DEVICE)

        predicted_value = valueNet(x_tensor_reshaped.to(config.DEVICE)).data.detach().cpu().numpy()
        predicted_value = ms.inverse_transform(predicted_value)
        tactic_result.append([predicted_value, tactic])

    tactic_result.sort(key=lambda x: x[0])

    return tactic_result[0][1]


def run_crossroad(name, crossroad_type, flow_number=config.FLOW_NUMBER, default_decision=None, start_tick=0,
                  end_tick=config.TOTAL_TICK, tqdm_off=False):
    if default_decision is None:
        default_decision = config.DEFAULT_DECISION

    with open('../log/x/' + name + '_' + str(flow_number) + ('_hard' if config.METHOD == 'HARD' else '') + '.csv',
              'w', newline='') as log_x_file:
        with open('../log/y/' + name + '_' + str(flow_number) +
                  ('_hard' if config.METHOD == 'HARD' else '') + '.csv', 'w', newline='') as log_y_file:
            with open('../log/dm/' + name + '_' + str(flow_number) +
                      ('_hard' if config.METHOD == 'HARD' else '') + '.csv', 'w', newline='') as log_dm_file:
                x_writer = csv.writer(log_x_file)
                y_writer = csv.writer(log_y_file)
                dm_writer = csv.writer(log_dm_file)

                # Flow Generation
                avg_flow = read_flow(-1)
                target_flow = read_flow(flow_number)
                out_flow = generate_out_flow(default_decision)

                # Simulation Configuration
                end_tick -= (end_tick - start_tick) % config.DECISION_LENGTH

                # Fundamental Variables for Simulation
                result = []
                num_cars = np.array([0] * 8)
                phase = 0
                phase_length = default_decision
                phase_tick = 0
                phase_result = np.array([0] * 8)

                # Variables for Decision Making
                erased_flow = None
                vn_data = []

                if tqdm_off:
                    tick_tqdm = range(start_tick, end_tick)
                else:
                    tick_tqdm = tqdm(range(start_tick, end_tick))
                    tick_tqdm.set_description("Crossroad - " + name)

                for i in tick_tqdm:
                    if i % config.DECISION_LENGTH == 0:
                        phase = 0
                        phase_tick = 0
                        phase_result = np.array([0] * 8)

                        if crossroad_type == 'GOD':
                            phase_length = decision_making_god(i, num_cars, target_flow)
                        elif crossroad_type == 'PMC':
                            phase_length = decision_making_PMC(i, num_cars, avg_flow, default_decision)
                        elif crossroad_type == 'SMC':
                            phase_length, erased_flow = decision_making_SMC(i, num_cars, avg_flow, target_flow,
                                                                            erased_flow, start_tick)
                        elif crossroad_type == 'VN':
                            if i == start_tick:
                                phase_length = default_decision
                            else:
                                phase_length = decision_making_VN(vn_data)
                        god_length = decision_making_god(i, num_cars, target_flow)

                        dm_writer.writerow([*phase_length, *god_length])
                        out_flow = generate_out_flow(phase_length)
                        vn_data = []

                    num_cars = num_cars + target_flow[i % config.TOTAL_TICK] - out_flow[i % config.DECISION_LENGTH]

                    for j in range(8):
                        if num_cars[j] < 0:
                            num_cars[j] = 0

                    phase_result = phase_result + num_cars
                    if crossroad_type == 'VN':
                        vn_data.append([i % config.TOTAL_TICK, *num_cars, *target_flow[i % config.TOTAL_TICK]])
                    x_writer.writerow([i % config.TOTAL_TICK, *num_cars, *target_flow[i % config.TOTAL_TICK]])

                    phase_tick += 1
                    while phase < 6 and phase_tick == phase_length[phase]:
                        phase += 1
                        phase_tick = 0

                    if i % config.DECISION_LENGTH == config.DECISION_LENGTH - 1:
                        y_writer.writerow([sum(phase_result)])
                        result.append(sum(phase_result) / config.DECISION_LENGTH)

    point_result = []
    sum_result = 0
    for i in range(len(result)):
        sum_result += result[i]
        if i % config.TICK_PER_POINT == config.TICK_PER_POINT - 1:
            point_result.append(sum_result / config.TICK_PER_POINT)
            sum_result = 0

    return point_result
