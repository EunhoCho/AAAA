import copy
import csv

import numpy as np

from Car import Car


class Crossroad:
    cross_type = 1
    config = {}
    tick = 0
    cars = [[None, [], [], []], [[], None, [], []], [[], [], None, []], [[], [], [], None]]
    num_cars = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    wait_outflow = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    phase = 0
    total_delay = 0
    total_cars = 0

    def __init__(self, config, cross_type=1):
        self.cross_type = cross_type
        with open(config, 'r') as config_file:
            config_lines = config_file.readlines()
            for config_line in config_lines:
                config_split = config_line.split(':')
                config_type = config_split[0]
                config_content = config_split[1].strip('\n')
                self.config[config_type] = config_content

        self.tick = int(self.config['START_TICK'])
        self.config['SECONDS_PER_TICK'] = int(self.config['SECONDS_PER_TICK'])
        self.config['TOTAL_TICKS_PER_HOUR'] = int(60 * 60 / self.config['SECONDS_PER_TICK'])
        self.config['TOTAL_TICKS_PER_DAY'] = self.config['TOTAL_TICKS_PER_HOUR'] * 24

        self.config['OUTFLOW_STRAIGHT'] = float(self.config['OUTFLOW_STRAIGHT_PER_MIN']) / 60 * self.config[
            'SECONDS_PER_TICK']
        self.config['OUTFLOW_LEFT'] = float(self.config['OUTFLOW_LEFT_PER_MIN']) / 60 * self.config['SECONDS_PER_TICK']
        self.config['OUTFLOW_RIGHT'] = float(self.config['OUTFLOW_RIGHT_PER_MIN']) / 60 * self.config[
            'SECONDS_PER_TICK']

        with open(self.config['FLOW_DATA'], "r") as f:
            flow_reader = csv.reader(f)
            self.flow_data = []
            for row in flow_reader:
                new_row = []
                for i in range(len(row)):
                    new_row.append(int(row[i]))
                self.flow_data.append(new_row)

        with open('raw_' + self.config['FLOW_DATA'], "r") as f:
            flow_reader = csv.reader(f)
            self.raw_flow_data = []
            for row in flow_reader:
                new_row = []
                for i in range(len(row)):
                    new_row.append(float(row[i]))
                self.raw_flow_data.append(new_row)

    def inflow(self, start_tick=-1, length=1, raw=False):
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
        current_inflow = self.inflow()
        for i in range(4):
            for j in range(4):
                for k in range(current_inflow[i][j]):
                    self.cars[i][j].append(Car(self.tick))
                    self.num_cars[i][j] += 1

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

    def decision_making(self):
        decision_length = int(self.config['DECISION_LENGTH'])

        if self.cross_type != 0:
            start_tick = self.tick
        elif self.tick == 0:
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
                            remain_cars = copy.deepcopy(self.num_cars)
                            wait_outflow = copy.deepcopy(self.wait_outflow)

                            num_cars = 0
                            for t in range(decision_length):
                                if self.cross_type == 1:
                                    phase_inflow = self.inflow(start_tick=start_tick + t, raw=True)
                                else:
                                    phase_inflow = self.inflow(start_tick=start_tick + t, raw=False)

                                for p in range(4):
                                    for q in range(4):
                                        remain_cars[p][q] += phase_inflow[p][q]

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

                                new_wait_outflow = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0, 0.0]]
                                for outflow in outflow_list:
                                    outflow_value = self.config['OUTFLOW_' + outflow[0]]
                                    outflow_value += wait_outflow[outflow[1]][outflow[2]]
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

                            if minimum == -1 or num_cars < minimum:
                                ans = [i, j, k, l, m, decision_length - i - j - k - l - m]
                                minimum = num_cars
        print(ans)
        return ans

    def run(self):
        max_frame = int(self.config['MAX_FRAME'])
        decision_length = int(self.config['DECISION_LENGTH'])
        phase_length = []
        phase_tick = 0

        while self.tick < max_frame:
            if self.tick % decision_length == 0:
                phase_length = self.decision_making()
                self.phase = 0
                phase_tick = 0

            while phase_tick == phase_length[self.phase]:
                self.phase += 1
                phase_tick = 0

            self.update()

            print(self.tick, self.phase, self.total_delay, self.total_cars, self.num_cars)

            phase_tick += 1
            self.tick += 1
