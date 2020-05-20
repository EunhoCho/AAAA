from flow import inflow
import copy


class Crossroad:
    is_proactive = True
    config = {}
    tick = 0
    cars = [[None, {}, {}, {}], [{}, None, {}, {}], [{}, {}, None, {}], [{}, {}, {}, None]]
    phase = 0
    total_delay = 0
    total_cars = 0

    def __init__(self, config, is_proactive=False):
        self.is_proactive = is_proactive
        with open(config, 'r') as config_file:
            config_lines = config_file.readlines()
            for config_line in config_lines:
                config_split = config_line.split(':')
                config_type = config_split[0]
                config_content = config_split[1].strip('\n')
                self.config[config_type] = config_content

    def update(self):
        current_inflow = inflow(self.tick)
        for i in range(4):
            for j in range(4):
                if current_inflow[i][j] != 0:
                    self.cars[i][j][self.tick] = current_inflow[i][j]

        outflow_list = []
        if self.phase == 0:
            outflow_list.append(('STRAIGHT', self.cars[0][1]))
            outflow_list.append(('LEFT', self.cars[0][2]))
            outflow_list.append(('RIGHT', self.cars[0][3]))
            outflow_list.append(('RIGHT', self.cars[2][0]))
        elif self.phase == 1:
            outflow_list.append(('STRAIGHT', self.cars[0][1]))
            outflow_list.append(('STRAIGHT', self.cars[1][0]))
            outflow_list.append(('RIGHT', self.cars[0][3]))
            outflow_list.append(('RIGHT', self.cars[1][2]))
        elif self.phase == 2:
            outflow_list.append(('STRAIGHT', self.cars[1][0]))
            outflow_list.append(('LEFT', self.cars[1][3]))
            outflow_list.append(('RIGHT', self.cars[1][2]))
            outflow_list.append(('RIGHT', self.cars[3][1]))
        elif self.phase == 3:
            outflow_list.append(('STRAIGHT', self.cars[2][3]))
            outflow_list.append(('LEFT', self.cars[2][1]))
            outflow_list.append(('RIGHT', self.cars[2][0]))
            outflow_list.append(('RIGHT', self.cars[1][2]))
        elif self.phase == 4:
            outflow_list.append(('STRAIGHT', self.cars[2][3]))
            outflow_list.append(('STRAIGHT', self.cars[3][2]))
            outflow_list.append(('RIGHT', self.cars[2][0]))
            outflow_list.append(('RIGHT', self.cars[3][1]))
        elif self.phase == 5:
            outflow_list.append(('STRAIGHT', self.cars[3][2]))
            outflow_list.append(('LEFT', self.cars[3][0]))
            outflow_list.append(('RIGHT', self.cars[3][1]))
            outflow_list.append(('RIGHT', self.cars[0][3]))

        for outflow in outflow_list:
            outflow_value = int(self.config['OUTFLOW_' + outflow[0]])
            while outflow_value > 0:
                if len(outflow[1].keys()) == 0:
                    break
                target = min(outflow[1].keys())
                if outflow[1][target] <= outflow_value:
                    outflow_value -= outflow[1][target]
                    self.total_delay += outflow[1][target] * (self.tick - target)
                    self.total_cars += outflow[1][target]
                    outflow[1].pop(target)
                else:
                    outflow[1][target] -= outflow_value
                    self.total_delay += outflow_value * (self.tick - target)
                    self.total_cars += outflow_value
                    outflow_value = 0

    def decision_making(self):
        return [5, 5, 5, 5, 5, 5]

    def run(self):
        max_frame = int(self.config['MAX_FRAME'])
        decision_length = int(self.config['DECISION_LENGTH'])
        phase_length = self.decision_making()
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

            print(self.tick, self.phase, self.total_delay, self.total_cars, self.cars)

            phase_tick += 1
            self.tick += 1
