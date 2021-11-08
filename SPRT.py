import random

import config
import crossroad
import environment


class SPRT:
    def __init__(self, tick):
        self.avg_flow = environment.read_flow()
        self.tick = tick

    def verify_simulation(self, state, decision, target):
        for i in range(config.smc_max_prob + 1):
            theta = i * 0.01

            if not self.verify_theta(theta, state, decision, target):
                return theta

        return config.smc_max_prob * 0.01

    def verify_theta(self, theta, state, decision, target):
        def generate_sample(tick):
            sample_flow = []
            for j in range(config.cross_decision_length):
                flow = []
                for k in range(config.cross_ways):
                    avg_number = self.avg_flow[tick % config.cross_total_tick][k]
                    low_number = int(config.smc_flow_rate_low * avg_number)
                    high_number = int(config.smc_flow_rate_high * avg_number)

                    if low_number == high_number:
                        flow.append(high_number)

                    else:
                        flow.append(random.randrange(low_number, high_number + 1))

                sample_flow.append(flow)
            return sample_flow

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
            sample = generate_sample(self.tick)

            _, result = crossroad.run_crossroad(0, sample, decision, state)
            result = sum(result)

            if result <= target:
                num_true += 1

            num_samples += 1

        return is_satisfied(num_samples, num_true, theta)
