import csv

import joblib
import numpy as np
import scipy.stats
import torch
from torch.autograd import Variable

from adaptive_crossroad import config
from adaptive_crossroad.value_net import ValueNet
from environment import config as env_config
from environment.sample import sample_environment

VALUE_NETS = {}


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


def decision_making(crossroad_type, avg_flow, tick, num_cars, vn_data, default_decision):
    if crossroad_type == 'PMC':
        with open('crossroad.sm', 'w') as prism_file:
            prism_file.write('const int decision_length = ' + str(config.DECISION_LENGTH) + ';\n')
            clk_module = ["module clk\n", "\ttime : [-1..decision_length + 1] init 0;\n",
                          "\treadyToTick : bool init true;\n", "\treadyToTack : bool init false;\n",
                          "\t[tick] readyToTick & time < decision_length + 1 -> 1 : (time' = time+ 1) & (readyToTick' = false) & (readyToTack' = true);\n",
                          "\t[tack] readyToTack -> 1 : (readyToTack' = false);"
                          "\t[toe] !readyToTick & !readyToTack -> 1 : (readyToTick'= true);\n", "endmodule\n\n",
                          'rewards "cars"\n',
                          "\t[toe] true : cars0 + cars1 + cars2 + cars3 + cars4 + cars5 + cars6 + cars7;\n",
                          "endrewards\n\n"]
            prism_file.writelines(clk_module)

            for i in range(8):
                class EnvNode:
                    def __init__(self, time, value):
                        self.time = time
                        self.value = value
                        self.prob1 = scipy.stats.norm(0, 1).cdf(-0.5)
                        self.child1 = -1
                        self.prob2 = 1 - 2 * scipy.stats.norm(0, 1).cdf(-0.5)
                        self.child2 = -1
                        self.prob3 = scipy.stats.norm(0, 1).cdf(-0.5)
                        self.child3 = -1

                env_nodes = [EnvNode(0, 0)]

                k = 0
                while k != len(env_nodes):
                    target_node = env_nodes[k]
                    time = target_node.time + 1

                    if time == config.DECISION_LENGTH:
                        break

                    target_inflow = avg_flow[tick + time + 1][i]
                    target_node.child2 = target_inflow
                    target_node.child1 = target_inflow - int(target_inflow * env_config.STDEV_RATE)
                    target_node.child3 = target_inflow + int(target_inflow * env_config.STDEV_RATE)

                    if target_node.child1 == target_node.child2:
                        target_node.prob1 += target_node.prob2
                        target_node.prob2 = 0

                        if target_node.child1 == target_node.child3:
                            target_node.prob1 = 1
                            target_node.prob3 = 0

                    elif target_node.child2 == target_node.child3:
                        target_node.prob2 += target_node.prob3
                        target_node.prob3 = 0

                    k += 1

                env_module = ["module env" + str(i) + "\n",
                              "\ts" + str(i) + " : [0.." + str(len(env_nodes) - 1) + "] init 0;\n"]
                for j in range(len(env_nodes)):
                    node = env_nodes[j]
                    if node.time == config.DECISION_LENGTH - 1:
                        break

                    if node.prob2 == 0:
                        if node.prob3 == 0:
                            env_module.append(
                                "\t[tick] s" + str(i) + " = " + str(j) + " -> 1 : (s" + str(i) + "' = " + str(
                                    node.child1) + ");\n")
                        else:
                            env_module.append(
                                "\t[tick] s" + str(i) + " = " + str(j) + " -> " + str(node.prob1) + " : (s" + str(
                                    i) + "' = " + str(node.child1) + ") + " + str(node.prob3) + " : (s" + str(
                                    i) + "' = " + str(node.child3) + ");\n")
                    elif node.prob3 == 0:
                        env_module.append(
                            "\t[tick] s" + str(i) + " = " + str(j) + " -> " + str(node.prob1) + " : (s" + str(
                                i) + "' = " + str(node.child1) + ") + " + str(node.prob2) + " : (s" + str(
                                i) + "' = " + str(node.child2) + ");\n")
                    else:
                        env_module.append(
                            "\t[tick] s" + str(i) + " = " + str(j) + " -> " + str(node.prob1) + " : (s" + str(
                                i) + "' = " + str(node.child1) + ") + " + str(node.prob2) + " : (s" + str(
                                i) + "' = " + str(node.child2) + ") + " + str(node.prob3) + " : (s" + str(
                                i) + "' = " + str(node.child3) + ");\n")
                env_module.append("endmodule\n\n")

                env_module.append(
                    "formula inflow" + str(i) + " = (s" + str(i) + " = 0 ? 0 : 0)\n")
                for j in range(1, len(env_nodes)):
                    node = env_nodes[j]
                    env_module.append(
                        " + (s" + str(i) + " = " + str(j) + " ? " + str(node.value) + " : 0)")
                    if k == len(env_nodes) - 1:
                        env_module.append(";\n")
                    env_module.append("\n")

                prism_file.writelines(env_module)

            sys_module = ["module sys\n", "\treadyToInflow : bool init false;\n",
                          "\treadyToOutflow : bool init false;\n"]
            for i in range(8):
                sys_module.append(
                    "\tcars" + str(i) + " : [0.." + str(config.DECISION_LENGTH * config.MAX_INFLOW) + "] init " + str(
                        num_cars[i]) + ";\n")
            sys_module.append("\n\t[tick] !readyToInflow & !readyToOutflow -> 1 : (readyToInflow' = true);\n")

            inflow_line = "\t[] readyToInflow -> 1 : (readyToInflow' = false) & (readyToOutflow' = true)"
            for i in range(4):
                for j in range(4):
                    if i != j:
                        inflow_line += " & (cars" + str(i) + "' = cars" + str(i) + " + inflow" + str(i) + ")"
            inflow_line += ";\n"

            outflow_line = "\t[tack] readyToOutflow -> 1 : (readyToOutflow' = false)"
            for i in range(8):
                outflow_line += " & (cars" + str(i) + str(j) + "' = cars" + str(i) + " - outflow" + str(i) + ")"
            outflow_line += ";\n"

            sys_module.append(inflow_line)
            sys_module.append(outflow_line)
            sys_module.append("endmodule\n\n")
            prism_file.writelines(sys_module)

            tactic_module = ["module tactic\n", "\tphase : [0..5] init 0;\n",
                             "\treadyToAdvance : bool init false;\n", "\treadyToStart : bool init false;\n"
                                                                      "\t[tick] !readyToAdvance & !readyToStart -> 1 : (readyToAdvance' = true);\n\n"
                                                                      "\t[] readyToAdvance & lastTick = 0 -> 1 : (phase' = phase + 1) & (readyToAdvance' = false) & (readyToStart' = true);\n",
                             "\t[] readyToAdvance & lastTick > 0 -> (phase' = phase + 1) & (readyToAdvance' = false) & (readyToStart' = true);\n",
                             "\t[] readyToAdvance & lastTick > 0 -> (phase' = phase) & (readyToAdvance' = false) & (readyToStart' = true);\n",
                             "\t[tack] readyToStart -> 1 : (readyToStart' = false);\n"
                             "endmodule\n\n", "formula lastTick = decision_length - time + phase - 5;\n\n"]
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
                    outflow_formula.append("(phase = " + str(j) + "? ")
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

    elif crossroad_type == 'SMC':
        sample_flow = []
        for i in range(config.SMC_SAMPLES):
            sample_flow.append(sample_environment(tick + 1, tick + 10, avg_flow))

        tactics = config.TACTICS[:]
        opt_tactic = []
        min_value = -1
        for tactic in tactics:
            result = 0
            for flow in sample_flow:
                result += sum(sim_run_crossroad(9, tactic, flow, num_cars))

            if min_value == -1 or min_value > result:
                opt_tactic = tactic
                min_value = result

        return opt_tactic

    elif crossroad_type == 'VN':
        if len(VALUE_NETS.keys()) == 0:
            for tactic in config.TACTICS:
                str_tactic = config.tactic_string(tactic)
                valueNet = ValueNet(config.VN_CLASS, config.VN_INPUT_SIZE, config.VN_HIDDEN_SIZE, config.VN_LAYERS,
                                    config.DECISION_LENGTH, '../valueNet/valueNet/' + str_tactic + '.torch').to(
                    config.DEVICE)
                ss = joblib.load('../valueNet/scaler/standard/' + str_tactic + '.sc')
                ms = joblib.load('../valueNet/scaler/minmax/' + str_tactic + '.sc')
                VALUE_NETS[str_tactic] = [valueNet, ss, ms]

        min_value = -1
        opt_tactic = []
        for tactic in config.TACTICS:
            str_tactic = config.tactic_string(tactic)
            valueNet = VALUE_NETS[str_tactic][0]
            ss = VALUE_NETS[str_tactic][1]
            ms = VALUE_NETS[str_tactic][2]

            x_ss = ss.fit_transform(vn_data)
            x_tensor = Variable(torch.Tensor(x_ss))
            x_tensor_reshaped = torch.reshape(x_tensor, (int(x_tensor.shape[0] / config.DECISION_LENGTH),
                                                         config.DECISION_LENGTH, x_tensor.shape[1])).to(config.DEVICE)

            predicted_value = valueNet(x_tensor_reshaped.to(config.DEVICE)).data.detach().cpu().numpy()
            predicted_value = ms.inverse_transform(predicted_value)
            if min_value == -1 or min_value > predicted_value:
                opt_tactic = tactic
                min_value = predicted_value

        return opt_tactic

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

            vn_data = []
            for i in range(8640 // config.TEN_SECOND_PER_TICK):
                num_cars = num_cars + target_flow[i] - out_flow[i % config.DECISION_LENGTH]

                for j in range(8):
                    if num_cars[j] < 0:
                        num_cars[j] = 0

                phase_result = phase_result + num_cars
                vn_data.append([i, *num_cars, *target_flow[i]])
                x_writer.writerow([i, *num_cars, *target_flow[i]])

                phase_tick += 1
                while phase < 6 and phase_tick == phase_length[phase]:
                    phase += 1
                    phase_tick = 0

                if phase == 6 and i != 8639:
                    y_writer.writerow([sum(phase_result)])

                    result.append(sum(phase_result) / config.DECISION_LENGTH)
                    phase_result = np.array([0] * 8)

                    if i != 8639:
                        phase = 0
                        phase_length = decision_making(crossroad_type, avg_flow, i, num_cars, vn_data, default_decision)
                        out_flow = generate_out_flow(phase_length)
                        vn_data = []

    point_result = []
    sum_result = 0
    for i in range(len(result)):
        sum_result += result[i]
        if i % config.TICK_PER_POINT == config.TICK_PER_POINT - 1:
            point_result.append(sum_result / config.TICK_PER_POINT)
            sum_result = 0

    return point_result
