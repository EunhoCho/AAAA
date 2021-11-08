import datetime

from tqdm import tqdm

import anomaly
import config
import crossroad
import environment

if __name__ == "__main__":
    times = [[] for _ in range(len(config.sim_targets))]

    count = 20
    for _ in tqdm(range(count)):
        experiment_flow = environment.sample_environment()
        experiment_anomalies = anomaly.generate_anomaly(config.sim_start_tick, config.sim_end_tick)

        for i in range(len(config.sim_targets)):
            target = config.sim_targets[i]
            start = datetime.datetime.now()
            crossroad.run(target, target, config.sim_start_tick, config.sim_end_tick, experiment_flow,
                          experiment_anomalies, tqdm_on=False)
            end = datetime.datetime.now()
            times[i].append(end - start)

    for i in range(len(config.sim_targets)):
        print(config.sim_targets[i], ': ', sum(times[i], datetime.timedelta()) / count / config.cross_num_decisions)
