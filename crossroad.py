import copy
import csv

import numpy as np

from Car import Car


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
    wait_outflow : list[list[float]]
        list for Residuals of outflow values.
    flow_data : list[list[int]]
        Flow data for using the simulation. These values are considered as the inflow of the past for cross_type 0.
    raw_flow_data : list[list[float]]
        Raw flow data that was used for making the flow data. In the simulation, it is considered as the average inflow of the cars.

    Methods
    ----------
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
        self.wait_outflow = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

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
        self.config['SECONDS_PER_TICK'] = int(self.config['SECONDS_PER_TICK'])
        self.config['TOTAL_TICKS_PER_HOUR'] = int(60 * 60 / self.config['SECONDS_PER_TICK'])
        self.config['TOTAL_TICKS_PER_DAY'] = self.config['TOTAL_TICKS_PER_HOUR'] * 24

        self.config['OUTFLOW_STRAIGHT'] = float(self.config['OUTFLOW_STRAIGHT_PER_MIN']) / 60 * self.config[
            'SECONDS_PER_TICK']
        self.config['OUTFLOW_LEFT'] = float(self.config['OUTFLOW_LEFT_PER_MIN']) / 60 * self.config['SECONDS_PER_TICK']
        self.config['OUTFLOW_RIGHT'] = float(self.config['OUTFLOW_RIGHT_PER_MIN']) / 60 * self.config[
            'SECONDS_PER_TICK']

        # Read the flow data
        with open(self.config['FLOW_DATA'], "r") as f:
            flow_reader = csv.reader(f)
            self.flow_data = []
            for row in flow_reader:
                new_row = []
                for i in range(len(row)):
                    new_row.append(int(row[i]))
                self.flow_data.append(new_row)

        # Read the raw flow data
        with open('raw_' + self.config['FLOW_DATA'], "r") as f:
            flow_reader = csv.reader(f)
            self.raw_flow_data = []
            for row in flow_reader:
                new_row = []
                for i in range(len(row)):
                    new_row.append(float(row[i]))
                self.raw_flow_data.append(new_row)

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
        current_inflow = self.inflow(raw=False)
        for i in range(4):
            for j in range(4):
                for k in range(current_inflow[i][j]):
                    self.cars[i][j].append(Car(self.tick))
                    self.num_cars[i][j] += 1

        # Setup the outflow based on the phase.
        outflow_list = []
        if self.phase == 0:
            outflow_list.append(('STRAIGHT', 0, 1))
            outflow_list.append(('LEFT', 0, 2))
            outflow_list.append(('RIGHT', 0, 3))
            outflow_list.append(('RIGHT', 2, 0))
        elif self.phase == 1:
            outflow_list.append(('STRAIGHT', 0, 1))
            outflow_list.append(('STRAIGHT', 1, 0))
            outflow_list.append(('RIGHT', 0, 3))
            outflow_list.append(('RIGHT', 1, 2))
        elif self.phase == 2:
            outflow_list.append(('STRAIGHT', 1, 0))
            outflow_list.append(('LEFT', 1, 3))
            outflow_list.append(('RIGHT', 1, 2))
            outflow_list.append(('RIGHT', 3, 1))
        elif self.phase == 3:
            outflow_list.append(('STRAIGHT', 2, 3))
            outflow_list.append(('LEFT', 2, 1))
            outflow_list.append(('RIGHT', 2, 0))
            outflow_list.append(('RIGHT', 1, 2))
        elif self.phase == 4:
            outflow_list.append(('STRAIGHT', 2, 3))
            outflow_list.append(('STRAIGHT', 3, 2))
            outflow_list.append(('RIGHT', 2, 0))
            outflow_list.append(('RIGHT', 3, 1))
        elif self.phase == 5:
            outflow_list.append(('STRAIGHT', 3, 2))
            outflow_list.append(('LEFT', 3, 0))
            outflow_list.append(('RIGHT', 3, 1))
            outflow_list.append(('RIGHT', 0, 3))

        # Make an outflow and remove the cars in the queue.
        new_wait_outflow = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        for outflow in outflow_list:
            outflow_value = self.config['OUTFLOW_' + outflow[0]]
            outflow_value += self.wait_outflow[outflow[1]][outflow[2]]
            while outflow_value >= 1:
                target_queue = self.cars[outflow[1]][outflow[2]]
                if len(target_queue) == 0:
                    break
                target = target_queue.pop(0)
                outflow_value -= 1
                self.num_cars[outflow[1]][outflow[2]] -= 1
                self.total_delay += self.tick - target.tick
                self.total_cars += 1

            if outflow_value > 0:
                new_wait_outflow[outflow[1]][outflow[2]] = outflow_value
        self.wait_outflow = new_wait_outflow

    def check_decision(self, decision, start_tick):
        """
        Evaluate the decision based on the cross_type

        :param decision: A list of integer that the sum is decision_length. Each element represents the length of phase.
        :param start_tick: The tick that starts the evaluate decision.
        :return: Returns the value of evaluation metric - sum of remained cars on each tick in one decision length.
        """
        decision_length = self.config['decision_length']

        remain_cars = copy.deepcopy(self.num_cars)
        wait_outflow = copy.deepcopy(self.wait_outflow)

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
            outflow_list = []
            if t < i:
                outflow_list.append(('STRAIGHT', 0, 1))
                outflow_list.append(('LEFT', 0, 2))
                outflow_list.append(('RIGHT', 0, 3))
                outflow_list.append(('RIGHT', 2, 0))
            elif t < i + j:
                outflow_list.append(('STRAIGHT', 0, 1))
                outflow_list.append(('STRAIGHT', 1, 0))
                outflow_list.append(('RIGHT', 0, 3))
                outflow_list.append(('RIGHT', 1, 2))
            elif t < i + j + k:
                outflow_list.append(('STRAIGHT', 1, 0))
                outflow_list.append(('LEFT', 1, 3))
                outflow_list.append(('RIGHT', 1, 2))
                outflow_list.append(('RIGHT', 3, 1))
            elif t < i + j + k + l:
                outflow_list.append(('STRAIGHT', 2, 3))
                outflow_list.append(('LEFT', 2, 1))
                outflow_list.append(('RIGHT', 2, 0))
                outflow_list.append(('RIGHT', 1, 2))
            elif t < i + j + k + l + m:
                outflow_list.append(('STRAIGHT', 2, 3))
                outflow_list.append(('STRAIGHT', 3, 2))
                outflow_list.append(('RIGHT', 2, 0))
                outflow_list.append(('RIGHT', 3, 1))
            else:
                outflow_list.append(('STRAIGHT', 3, 2))
                outflow_list.append(('LEFT', 3, 0))
                outflow_list.append(('RIGHT', 3, 1))
                outflow_list.append(('RIGHT', 0, 3))

            # Make an outflow
            new_wait_outflow = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]
            for outflow in outflow_list:
                outflow_value = self.config['OUTFLOW_' + outflow[0]]
                outflow_value += wait_outflow[outflow[1]][outflow[2]]

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

                    if outflow_value > 0:
                        new_wait_outflow[outflow[1]][outflow[2]] = outflow_value
                wait_outflow = new_wait_outflow

            for p in range(4):
                for q in range(4):
                    num_cars += remain_cars[p][q]

        return num_cars

    def decision_making(self):
        """
        Makes a decision for controlling the length of each phase based on the cross_type.

        :return: The list of the integer with 6 elements. The each value represents the length of each phase.
        """
        decision_length = int(self.config['DECISION_LENGTH'])

        if self.cross_type != 0:
            start_tick = self.tick
        elif self.tick == 0:
            # Make a default decision as each phase has the same value
            div = decision_length / 6
            ans = []
            residual = 0
            for i in range(6):
                value = int(div)
                residual += div - int(div)

                residual_val = int(residual)
                residual -= int(residual)

                if residual > 0:
                    random_val = np.random.choice([0, 1], 1, p=[1 - residual, residual])[0]
                    residual -= random_val
                elif residual < 0:
                    random_val = np.random.choice([-1, 0], 1, p=[- residual, 1 + residual])[0]
                    residual -= random_val
                else:
                    random_val = 0

                value += residual_val + random_val
                if value < 0:
                    value -= random_val
                    residual += random_val
                if value < 0:
                    value -= residual_val
                    residual += residual_val

                ans.append(value)

            print(ans)

            return ans

        else:
            start_tick = self.tick - decision_length

        ans = [0, 0, 0, 0, 0, 0]
        minimum = -1
        for i in range(0, decision_length + 1):
            for j in range(0, decision_length - i + 1):
                for k in range(0, decision_length - i - j + 1):
                    for l in range(0, decision_length - i - j - k + 1):
                        for m in range(0, decision_length - i - j - k - l + 1):
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

        :return: None
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
