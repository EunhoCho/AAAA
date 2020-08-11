import csv
from datetime import datetime
from generate_flow_data import generate_inflow


FLOW_DATA = 'raw_inflow.csv'


class SPRT:
    def __init__(self, crossroad, property, property_checker):
        self.alpha = 0.05
        self.beta = 0.05
        self.delta = 0.01
        self.minimum_sample = 2
        self.max_repeat = 400

        self.crossroad = crossroad
        self.property = property
        self.property_checker = property_checker

        with open(FLOW_DATA, "r") as f:
            flow_reader = csv.reader(f)
            self.raw_flow_data = []
            for row in flow_reader:
                new_row = [[], [], [], []]
                for i in range(4):
                    for j in range(4):
                        new_row[i].append(float(row[4 * i + j]))
                self.raw_flow_data.append(new_row)

    def verify_simulation(self):
        satisfaction = True
        satisfaction_prob = 0.0

        program_start_time = datetime.now()
        for i in range(100):
            theta = i * 0.01
            
            theta_start_time = datetime.now()
            verification_result = self.verify_theta(theta)
            theta_end_time = datetime.now()
            print(i / 100, " theta verification running time: ", theta_end_time - theta_start_time, " sec")
            
            if satisfaction and not verification_result:
                satisfaction_prob = theta
                satisfaction = False
                
        print("Verification property satisfaction probability: ", satisfaction_prob)
        program_end_time = datetime.now()
        print("=== Total Program running time: ", program_end_time - program_start_time, " sec")

    def verify_theta(self, theta):
        ret = True
        num_samples = 0
        num_true = 0

        if theta == 1.00:
            ret = False

        while self.is_sample_needed(num_samples, num_true, theta):
            sample = generate_inflow(self.raw_flow_data)
            
            if self.property_checker.check(sample, self.property):
                num_true += 1
            
            num_samples += 1

        return ret

    def is_sample_needed(self, num_sample, num_true, theta):
        if num_sample < self.minimum_sample:
            return True

        h0_threshold = self.beta / (1-self.alpha)
        h1_threshold = (1-self.beta) / self.alpha

        v = self.get_v(num_sample, num_true, theta)

        if v <= h0_threshold:
            return False
        else:
            return not v >= h1_threshold

    def get_v(self, num_sample, num_true, theta):
        p0 = theta + self.delta
        p1 = theta - self.delta

        num_false = num_sample - num_true

        p1m = p1 ** num_true * (1 - p1) ** num_false
        p0m = p0 ** num_true * (1 - p0) ** num_false

        if p0m == 0:
            p1m = p1m + float('-inf')
            p0m = p0m + float('-inf')

        return p1m / p0m
