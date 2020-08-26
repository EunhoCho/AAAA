import copy
import csv

import numpy as np

from Car import Car


def setup_outflow(phase):
    """
    Setup outflow based on the given phase

    :param phase: The phase of the outflow
    :return: List of tuples that defined the outflow roads on the tick
    """
    outflow_list = []
    if phase == 0:
        outflow_list.append(('STRAIGHT', 0, 1))
        outflow_list.append(('LEFT', 0, 2))
        outflow_list.append(('RIGHT', 0, 3))
        outflow_list.append(('RIGHT', 2, 0))
    elif phase == 1:
        outflow_list.append(('STRAIGHT', 0, 1))
        outflow_list.append(('STRAIGHT', 1, 0))
        outflow_list.append(('RIGHT', 0, 3))
        outflow_list.append(('RIGHT', 1, 2))
    elif phase == 2:
        outflow_list.append(('STRAIGHT', 1, 0))
        outflow_list.append(('LEFT', 1, 3))
        outflow_list.append(('RIGHT', 1, 2))
        outflow_list.append(('RIGHT', 3, 1))
    elif phase == 3:
        outflow_list.append(('STRAIGHT', 2, 3))
        outflow_list.append(('LEFT', 2, 1))
        outflow_list.append(('RIGHT', 2, 0))
        outflow_list.append(('RIGHT', 1, 2))
    elif phase == 4:
        outflow_list.append(('STRAIGHT', 2, 3))
        outflow_list.append(('STRAIGHT', 3, 2))
        outflow_list.append(('RIGHT', 2, 0))
        outflow_list.append(('RIGHT', 3, 1))
    elif phase == 5:
        outflow_list.append(('STRAIGHT', 3, 2))
        outflow_list.append(('LEFT', 3, 0))
        outflow_list.append(('RIGHT', 3, 1))
        outflow_list.append(('RIGHT', 0, 3))

    return outflow_list


class Crossroad:
    """
    A class for representing the crossroad.

    Attributes
    ----------
    cross_type : int
        Decision making style of the crossroad. 0 = Reactive, 1 = Proactive, 2 = Omniscient.
    tick : int
        Current tick in the simulation.
    phase : int
        Current phase in the simulation.
    total_delay : int
        Total delayed ticks of passed cars through the simulation.
    total_cars : int
        Total passed cars through the simulation.
    config : dict
        Configuration of the simulation.
    cars : list[list[Car]]
        Queue of the cars that waiting in the crossroad.
    num_cars : list[list[int]]
        Number of cars that waiting in the crossroad.
    flow_data : list[list[int]]
        Flow data for using the simulation. These values are considered as the inflow of the past for cross_type 0.
    raw_flow_data : list[list[float]]
        Raw flow data that was used for making the flow data. In the simulation, it is considered as the average inflow of the cars.

    Methods
    ----------
    read_flow(file)
        Change the inflow based on the file
    inflow(start_tick=-1, length=1, raw=False)
        Get the inflow data from start_tick to start_tick+length.
    update()
        Update the crossroad for a tick.
    decision_making()
        Makes a decision for controlling the length of each phase based on the cross_type.
    run()
        Make a simulation based on the configuration and make a csv file for logging the situation.
    """

    def __init__(self, config, cross_type=1):
        """
        :param config: Configuration file of the crossroad.
        :param cross_type: Decision making style. 0 = Reactive, 1 = Proactive, 2 = Omniscient
        """
        self.cross_type = cross_type

        # Initialize the Crossroad.
        self.phase = 0
        self.total_delay = 0
        self.total_cars = 0
        self.config = {}
        self.tick = 0
        self.cars = [[None, [], [], []], [[], None, [], []], [[], [], None, []], [[], [], [], None]]
        self.num_cars = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.residual = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        # Read the configuration.
        with open(config, 'r') as config_file:
            config_lines = config_file.readlines()
            for config_line in config_lines:
                config_split = config_line.split(':')
                config_type = config_split[0]
                config_content = config_split[1].strip('\n')
                self.config[config_type] = config_content

        # Setup the configuration.
        self.tick = int(self.config['START_TICK'])
        self.config['DECISION_LENGTH'] = int(self.config['DECISION_LENGTH'])
        self.config['SECONDS_PER_TICK'] = int(self.config['SECONDS_PER_TICK'])
        self.config['TOTAL_TICKS_PER_HOUR'] = int(60 * 60 / self.config['SECONDS_PER_TICK'])
        self.config['TOTAL_TICKS_PER_DAY'] = self.config['TOTAL_TICKS_PER_HOUR'] * 24

        self.config['OUTFLOW_STRAIGHT'] = float(self.config['OUTFLOW_STRAIGHT_PER_MIN']) / 60 * self.config[
            'SECONDS_PER_TICK']
        self.config['OUTFLOW_LEFT'] = float(self.config['OUTFLOW_LEFT_PER_MIN']) / 60 * self.config['SECONDS_PER_TICK']
        self.config['OUTFLOW_RIGHT'] = float(self.config['OUTFLOW_RIGHT_PER_MIN']) / 60 * self.config[
            'SECONDS_PER_TICK']

        # Read the flow data
        self.flow_data = []
        self.read_flow(self.config['FLOW_DATA'])

        # Read the raw flow data
        with open('raw_' + self.config['FLOW_DATA'], "r") as f:
            flow_reader = csv.reader(f)
            self.raw_flow_data = []
            for row in flow_reader:
                new_row = []
                for i in range(len(row)):
                    new_row.append(float(row[i]))
                self.raw_flow_data.append(new_row)

    def read_flow(self, file):
        """
        Change the inflow based on the file

        :param file: The inflow data file.
        :return: None.
        """
        with open(file, "r") as f:
            flow_reader = csv.reader(f)
            for i, row in enumerate(flow_reader):
                if i != 0:
                    new_row = []
                    for j in range(len(row)):
                        new_row.append(int(row[j]))
                    self.flow_data.append(new_row)

    def inflow(self, start_tick=-1, length=1, raw=False):
        """
        Get the inflow data from start_tick to start_tick+length.

        :param start_tick: the tick of the inflow
        :param length: The length of the inflow
        :param raw: Boolean value for checking the inflow type. If true, returns the raw_flow_data, else, return flow_data
        :return: Returns the inflow list[list[int]] of given parameters
        """
        if start_tick == -1:
            start_tick = self.tick
        tick = start_tick % self.config['TOTAL_TICKS_PER_DAY']

        ans = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for i in range(tick, tick + length):
            if raw:
                flow_tick = self.raw_flow_data[i]
            else:
                flow_tick = self.flow_data[i]
            for j in range(4):
                for k in range(4):
                    ans[j][k] += flow_tick[j * 4 + k]

        return ans

    def update(self):
        """
        Update the crossroad for a tick.

        :return: None
        """
        # Inflow the cars based on the inflow data
        current_raw_inflow = self.inflow(raw=True)
        current_inflow = self.inflow(raw=False)
        for i in range(4):
            for j in range(4):
                self.residual[i][j] += current_raw_inflow[i][j]
                self.residual[i][j] -= current_inflow[i][j]
                for k in range(current_inflow[i][j]):
                    self.cars[i][j].append(Car(self.tick))
                    self.num_cars[i][j] += 1

        # Setup the outflow based on the phase.
        outflow_list = setup_outflow(self.phase)

        # Make an outflow and remove the cars in the queue.
        for outflow in outflow_list:
            outflow_value = self.config['OUTFLOW_' + outflow[0]]
            while outflow_value >= 1:
                target_queue = self.cars[outflow[1]][outflow[2]]
                if len(target_queue) == 0:
                    break
                target = target_queue.pop(0)
                outflow_value -= 1
                self.num_cars[outflow[1]][outflow[2]] -= 1
                self.total_delay += self.tick - target.tick
                self.total_cars += 1

    def check_decision(self, decision, start_tick):
        """
        Evaluate the decision based on the cross_type

        :param decision: A list of integer that the sum is decision_length. Each element represents the length of phase.
        :param start_tick: The tick that starts the evaluate decision.
        :return: Returns the value of evaluation metric - sum of remained cars on each tick in one decision length.
        """
        decision_length = int(self.config['DECISION_LENGTH'])

        remain_cars = copy.deepcopy(self.num_cars)
        num_cars = 0
        i = decision[0]
        j = decision[1]
        k = decision[2]
        l = decision[3]
        m = decision[4]

        for t in range(decision_length):

            # Make a inflow
            if self.cross_type == 1:
                phase_inflow = self.inflow(start_tick=start_tick + t, raw=True)
            else:
                phase_inflow = self.inflow(start_tick=start_tick + t, raw=False)

            for p in range(4):
                for q in range(4):
                    remain_cars[p][q] += phase_inflow[p][q]

            # Setup the outflow
            if t < i:
                phase = 0
            elif t < i + j:
                phase = 1
            elif t < i + j + k:
                phase = 2
            elif t < i + j + k + l:
                phase = 3
            elif t < i + j + k + l + m:
                phase = 4
            else:
                phase = 5
            outflow_list = setup_outflow(phase)

            # Make an outflow
            for outflow in outflow_list:
                outflow_value = self.config['OUTFLOW_' + outflow[0]]
                if self.cross_type == 1:
                    if remain_cars[outflow[1]][outflow[2]] > outflow_value:
                        remain_cars[outflow[1]][outflow[2]] -= outflow_value
                    else:
                        remain_cars[outflow[1]][outflow[2]] = 0
                else:
                    while outflow_value >= 1:
                        if remain_cars[outflow[1]][outflow[2]] == 0:
                            break
                        outflow_value -= 1
                        remain_cars[outflow[1]][outflow[2]] -= 1

            for p in range(4):
                for q in range(4):
                    num_cars += remain_cars[p][q]

        return num_cars

    def decision_making(self):
        """
        Makes a decision for controlling the length of each phase based on the cross_type.

        :return: The list of the integer with 6 elements. The each value represents the length of each phase.
        """
        if self.cross_type == 3:
            return self.prism_decision_making()

        decision_length = self.config['DECISION_LENGTH']

        if self.cross_type != 0:
            start_tick = self.tick
        elif self.tick == 0:
            # Make a default decision as each phase has the same value
            div = decision_length / 6
            ans = []
            residual = 0
            for i in range(6):
                residual += div

                if residual > 0:
                    value = int(residual)
                    residual -= value

                    random_val = np.random.choice([0, 1], 1, p=[1 - residual, residual])[0]
                    residual -= random_val

                else:
                    value = 0

                ans.append(value)

            print(ans)

            return ans

        else:
            start_tick = self.tick - decision_length

        ans = [0, 0, 0, 0, 0, 0]
        minimum = -1
        for i in range(1, decision_length - 4):
            for j in range(1, decision_length - i - 3):
                for k in range(1, decision_length - i - j - 2):
                    for l in range(1, decision_length - i - j - k - 1):
                        for m in range(1, decision_length - i - j - k - l):
                            decision = [i, j, k, l, m, decision_length - i - j - k - l - m]
                            num_cars = self.check_decision(decision, start_tick)

                            if minimum == -1 or num_cars < minimum:
                                ans = decision
                                minimum = num_cars
        print(ans)
        return ans

    def run(self):
        """
        Make a simulation based on the configuration and make a csv file for logging the situation.

        :return: total delayed time and total number of cars during the simulation.
        """
        max_frame = int(self.config['MAX_FRAME'])
        decision_length = int(self.config['DECISION_LENGTH'])
        phase_length = []
        phase_tick = 0

        log_file = open('log_' + str(self.cross_type) + '.csv', 'w', newline='')
        log_writer = csv.writer(log_file)
        header = ['tick', 'phase', 'delayed', 'passed']
        for i in range(4):
            for j in range(4):
                header.append(str(i) + '->' + str(j))
        log_writer.writerow(header)

        dm_file = open('dm_' + str(self.cross_type) + '.csv', 'w', newline='')
        dm_writer = csv.writer(dm_file)
        log_writer.writerow(['tick', 'phase0', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5'])

        while self.tick < max_frame:
            if self.tick % decision_length == 0:
                phase_length = self.decision_making()
                dm_writer.writerow([self.tick] + phase_length)
                self.phase = 0
                phase_tick = 0

            while phase_tick == phase_length[self.phase]:
                self.phase += 1
                phase_tick = 0

            self.update()

            write_row = [self.tick, self.phase, self.total_delay, self.total_cars]
            for i in range(4):
                for j in range(4):
                    write_row.append(self.num_cars[i][j])
            log_writer.writerow(write_row)
            print(self.tick, self.phase, self.total_delay, self.total_cars, self.num_cars)

            phase_tick += 1
            self.tick += 1

        log_file.close()
        dm_file.close()

        return self.total_delay, self.total_cars

    def prism_decision_making(self):
        decision_length = self.config['DECISION_LENGTH']
        prism_model_file = open('crossroad.sm', 'w')
        prism_model_file.write('const int decision_length = ' + str(self.config['DECISION_LENGTH']) + ';\n')

        clk_module = ["module clk\n", "\ttime : [-1..decision_length + 1] init 0;\n", "\treadyToTick : bool init true;\n",
                      "\t[tick] readyToTick & time < decision_length + 1 -> 1 : (time' = time+ 1) & (readyToTick' = false);\n",
                      "\t[toe] !readyToTick -> 1 : (readyToTick'=true);\n", "endmodule\n\n", 'rewards "cars"\n',
                      "\t[toe] true : cars01 + cars02 + cars03 + cars10 + cars12 + cars13 + cars20 + cars21 + cars23 + cars30 + cars31 + cars32;\n",
                      "endrewards\n\n"]
        prism_model_file.writelines(clk_module)

        cur_residual = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for i in range(4):
            for j in range(4):
                cur_residual[i][j] = self.residual[i][j]

        raw_inflows = []
        for time in range(decision_length):
            raw_inflows.append(self.inflow(start_tick=self.tick + time, raw=True))

        for i in range(4):
            for j in range(4):
                if i != j:
                    class env_node:
                        def __init__(self, residual, time, value):
                            self.residual = residual
                            self.time = time
                            self.value = value
                            self.prob1 = 1
                            self.child1 = -1
                            self.prob2 = 0
                            self.child2 = -1

                    env_nodes = [env_node(cur_residual[i][j], -1, 0)]

                    k = 0
                    while k != len(env_nodes):
                        target_node = env_nodes[k]
                        time = target_node.time + 1
                        residual = target_node.residual

                        if time == decision_length:
                            break

                        target_inflow = raw_inflows[time][i][j]
                        residual += target_inflow

                        if residual < 0:
                            env_nodes.append(env_node(residual, time, 0))
                            target_node.child1 = len(env_nodes) - 1
                        else:
                            value = int(residual)
                            residual -= int(residual)

                            env_nodes.append(env_node(residual, time, value))
                            target_node.prob1 = 1 - residual
                            target_node.child1 = len(env_nodes) - 1

                            env_nodes.append(env_node(residual - 1, time, value + 1))
                            target_node.prob2 = residual
                            target_node.child2 = len(env_nodes) - 1

                        k += 1

                    env_module = ["module env" + str(i) + str(j) + "\n", "\ts" + str(i) + str(j) + " : [0.." + str(len(env_nodes) - 1) + "] init 0;\n"]
                    for k in range(len(env_nodes)):
                        node = env_nodes[k]
                        if node.time == decision_length - 1:
                            break

                        if node.prob2 == 0:
                            env_module.append("\t[tick] s" + str(i) + str(j) + " = " + str(k) + " -> 1 : (s" + str(i) + str(j) + "' = " + str(node.child1) + ");\n")
                        else:
                            env_module.append("\t[tick] s" + str(i) + str(j) + " = " + str(k) + " -> " + str(node.prob1) + " : (s" + str(i) + str(j) + "' = " + str(node.child1) + ") + " + str(node.prob2) + " : (s" + str(i) + str(j) + "' = " + str(node.child2) + ");\n")

                    env_module.append("endmodule\n\n")
                    env_module.append("formula inflow" + str(i) + str(j) + " = (s" + str(i) + str(j) + " = 0 ? 0 : 0)\n")

                    for k in range(1, len(env_nodes) - 1):
                        node = env_nodes[k]
                        env_module.append(" + (s" + str(i) + str(j) + " = " + str(k) + " ? " + str(node.value) + " : 0)\n")

                    node = env_nodes[-1]
                    env_module.append(" + (s" + str(i) + str(j) + " = " + str(k) + " ? " + str(node.value) + " : 0);\n\n")

                    prism_model_file.writelines(env_module)

        sys_module = ["module sys\n", "\treadyToInflow : bool init false;\n", "\treadyToOutflow : bool init false;\n"]
        for i in range(4):
            for j in range(4):
                if i != j:
                    sys_module.append("\tcars" + str(i) + str(j) + " : int init " + str(self.num_cars[i][j]) + ";\n")

        sys_module.append("\n\t[tick] !readyToInflow & !readyToOutflow -> 1 : (readyToInflow' = true);\n")

        inflow_line = "\t[] readyToInflow -> 1 : (readyToInflow' = false) & (readyToOutflow' = true)"
        for i in range(4):
            for j in range(4):
                if i != j:
                    inflow_line += " & (cars" + str(i) + str(j) + "' = cars" + str(i) + str(j) + " + inflow" + str(i) + str(j) + ")"
        inflow_line += ";\n"

        outflow_line = "\t[tack] readyToOutflow -> 1 : (readyToOutflow' = false)"
        for i in range(4):
            for j in range(4):
                if i != j:
                    outflow_line += " & (cars" + str(i) + str(j) + "' = cars" + str(i) + str(j) + " - outflow" + str(i) + str(j) + ")"
        outflow_line += ";\n"

        sys_module.append(inflow_line)
        sys_module.append(outflow_line)
        sys_module.append("endmodule\n\n")
        prism_model_file.writelines(sys_module)

        tactic_module = ["module tactic\n", "\tphase : [0..5] init 0;\n",
                         "\treadyToAdvance : bool init false;\n", "\treadyToStart : bool init false;\n"
                         "\t[tick] !readyToAdvance & !readyToStart -> 1 : (readyToAdvance' = true);\n\n"
                         "\t[] readyToAdvance & lastTick = 0 -> 1 : (phase' = phase + 1) & (readyToAdvance' = false) & (readyToStart' = true);\n",
                         "\t[] readyToAdvance & lastTick > 0 -> (phase' = phase + 1) & (readyToAdvance' = false) & (readyToStart' = true);\n",
                         "\t[] readyToAdvance & lastTick > 0 -> (phase' = phase) & (readyToAdvance' = false) & (readyToStart' = true);\n",
                         "\t[tack] readyToStart -> 1 : (readyToStart' = false);\n"
                         "endmodule\n\n", "formula lastTick = decision_length - time + phase - 5;\n\n"]
        prism_model_file.writelines(tactic_module)

        # TODO : Outflow
        outflows = [[None,
                     ["STRAIGHT", [True, True, False, False, False, False]],
                     ["LEFT", [True, False, False, False, False, False]],
                     ["RIGHT", [True, True, False, False, False, True]]],
                    [["STRAIGHT", [False, True, True, False, False, False]],
                     None,
                     ["RIGHT", [False, True, True, True, False, False]],
                     ["LEFT", [False, False, True, False, False, False]]],
                    [["RIGHT", [True, False, False, True, True, False]],
                     ["LEFT", [False, False, False, True, False, False]],
                     None,
                     ["STRAIGHT", [False, False, False, True, True, False]]],
                    [["LEFT", [False, False, False, False, False, True]],
                     ["RIGHT", [False, False, True, False, True, True]],
                     ["STRAIGHT", [False, False, False, False, True, True]],
                     None]]
        for i in range(4):
            for j in range(4):
                if i != j:
                    single_outflow = outflows[i][j]
                    outflow_formula = ["formula outflow" + str(i) + str(j) + " = "]
                    for k in range(6):
                        if k != 0:
                            outflow_formula.append("+ ")
                        outflow_formula.append("(phase = " + str(k) + "? ")
                        if single_outflow[1][k]:
                            outflow_formula.append(str(int(self.config["OUTFLOW_" + single_outflow[0]])))
                        else:
                            outflow_formula.append("0")
                        outflow_formula.append(" : 0)")
                        if k == 5:
                            outflow_formula.append(";")
                        outflow_formula.append("\n")
                    outflow_formula.append("\n")
                    prism_model_file.writelines(outflow_formula)

        prism_model_file.close()