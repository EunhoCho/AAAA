import datetime

import numpy as np
from matplotlib import pyplot as plt

import anomaly
import config
import crossroad
import environment


def graphize(result):
    graphize_result = []

    for i in range(0, len(result), config.graph_decision_per_point):
        if len(result) < i + config.graph_decision_per_point:
            break

        sum_result = 0
        for j in range(config.graph_decision_per_point):
            sum_result += result[i + j]
        graphize_result.append(sum_result / config.graph_decision_per_point)

    return graphize_result


def plot_base(anomalies=None):
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Number of Cars')
    plt.xticks(list(range(config.graph_start, config.graph_end + 1, 6)),
               list(range(config.graph_start, config.graph_end + 1, 6)))

    if anomalies is not None:
        for single_anomaly in anomalies:
            hour = single_anomaly.tick / config.cross_one_hour
            duration = config.anomaly_duration / config.cross_one_hour
            plt.axvspan(hour, hour + duration, facecolor='r', alpha=0.1)


def run_simulation(name, flow, has_anomaly=True):
    final_result = []

    if has_anomaly:
        experiment_anomalies = anomaly.generate_anomaly(config.sim_start_tick, config.sim_end_tick, name=name)
    else:
        experiment_anomalies = []

    graphized_result = []
    for target in config.sim_targets:
        result = crossroad.run(name, target, config.sim_start_tick, config.sim_end_tick, flow, experiment_anomalies)
        graphized_target_result = np.array(graphize(result))
        graphized_result.append(graphized_target_result)
        plt.plot(config.graph_time, graphized_target_result, label=target)
        final_result.append(sum(result))

    plt.title('Time - Number of Waiting Cars')
    plot_base(experiment_anomalies)
    plt.savefig('figure/' + name + '.png', dpi=300)
    plt.show()
    plt.close()

    return np.array(final_result)


if __name__ == "__main__":
    result = np.array([0.0] * (len(config.sim_targets) + 1))
    time = np.array([datetime.timedelta] * (len(config.sim_targets) + 1))
    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_flow = environment.sample_environment(name=experiment_name)

        # single_result = run_simulation(experiment_name + '_clean', experiment_flow, False)
        single_result, single_time = run_simulation(experiment_name, experiment_flow)
        result += single_result
        time += single_time

        print('run ', i)
        print(single_result)
        print(single_time)

    result /= config.sim_count
    time /= config.sim_count
    result /= config.cross_num_decisions

    print(result)
