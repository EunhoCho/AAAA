import random

import config
import crossroad
import environment


class StatisticalModelChecker:
    def __init__(self, tick, state, tactics=None):
        self.tick = tick
        self.state = state
        self.tactics = tactics

        if tactics is None:
            self.tactics = config.cross_tactics.copy()

        self.avg_flow = environment.read_flow()

    def generate_sample(self):
        sample_flow = []
        for j in range(config.cross_decision_length):
            flow = []
            for k in range(config.cross_ways):
                avg_number = self.avg_flow[self.tick % config.cross_total_tick][k]
                low_number = int(config.smc_flow_rate_low * avg_number)
                high_number = int(config.smc_flow_rate_high * avg_number)

                if low_number == high_number:
                    flow.append(high_number)

                else:
                    flow.append(random.randrange(low_number, high_number + 1))

            sample_flow.append(flow)
        return sample_flow

    def choose_decision(self):
        target_num = max(sum(self.state), config.cross_decision_length * 10)
        var = max(sum(self.state), config.cross_decision_length * 10)
        plus_trend = None

        while True:
            results = []
            for tactic in self.tactics:
                prob = self.get_probability(tactic, target_num)
                results.append((prob, tactic))

            results.sort(key=lambda x: -x[0])
            i = 0
            while i < len(results):
                if results[i][0] >= config.smc_max_prob * 0.01:
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

                self.tactics = []
                for j in range(i):
                    self.tactics.append(results[j][1])
                if plus_trend is not None and plus_trend:
                    var /= 2
                target_num -= var

            if var == 0:
                return self.tactics[0]

    def get_probability(self, decision, target):
        return 0


class SimpleMonteCarloChecker(StatisticalModelChecker):
    def __init__(self, tick, state, tactics=None):
        super().__init__(tick, state, tactics)

        self.samples = []
        for i in range(config.smc_max_samples):
            self.samples.append(self.generate_sample())

        self.sample_result = {}
        for tactic in tactics:
            tactic_result = []
            for sample in self.samples:
                result, _ = crossroad.run_crossroad(0, sample, tactic, self.state)
                tactic_result.append(sum(sum(result)) / config.cross_decision_length)

            tactic_result.sort()
            self.sample_result[config.tactic_string(tactic)] = tactic_result

    def get_probability(self, decision, target):
        for i, number in enumerate(self.sample_result[config.tactic_string(decision)]):
            if number > target:
                return i / len(self.samples)
        return 1


class SPRT(StatisticalModelChecker):
    def get_probability(self, decision, target):
        for i in range(config.smc_max_prob + 1):
            theta = i * 0.01

            if not self.verify_theta(theta, decision, target):
                return theta

        return config.smc_max_prob * 0.01

    def verify_theta(self, theta, decision, target):
        def get_v(num_sample, num_true, theta):
            p0 = theta + config.smc_sprt_delta
            p1 = theta - config.smc_sprt_delta

            num_false = num_sample - num_true

            p1m = p1 ** num_true * (1 - p1) ** num_false
            p0m = p0 ** num_true * (1 - p0) ** num_false

            if p0m == 0:
                p1m = p1m + float('-inf')
                p0m = p0m + float('-inf')

            return p1m / p0m

        def is_sample_needed(num_sample, num_true, theta):
            if num_sample < config.smc_min_samples:
                return True

            h0_threshold = config.smc_sprt_beta / (1 - config.smc_sprt_alpha)
            h1_threshold = (1 - config.smc_sprt_beta) / config.smc_sprt_alpha

            v = get_v(num_sample, num_true, theta)
            if v <= h0_threshold:
                return False
            else:
                return not v >= h1_threshold

        def is_satisfied(num_sample, num_true, theta):
            h0_threshold = config.smc_sprt_beta / (1 - config.smc_sprt_alpha)
            v = get_v(num_sample, num_true, theta)
            if v <= h0_threshold:
                return True
            return False

        num_samples = 0
        num_true = 0

        if theta == 1.00:
            return False

        while num_samples < config.smc_max_samples and is_sample_needed(num_samples, num_true, theta):
            sample = self.generate_sample()

            _, result = crossroad.run_crossroad(0, sample, decision, self.state)
            result = sum(result)

            if result <= target:
                num_true += 1

            num_samples += 1

        return is_satisfied(num_samples, num_true, theta)
