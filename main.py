import datetime

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


def run_simulation(name, flow, has_anomaly=True):
    if has_anomaly:
        experiment_anomalies = anomaly.generate_anomaly(config.sim_start_tick, config.sim_end_tick, name)
    else:
        experiment_anomalies = []

    god_result = crossroad.run(name, 'GOD', config.sim_start_tick, config.sim_end_tick, flow, experiment_anomalies)
    plt.plot(config.graph_time, graphize(god_result), label='GOD')

    for target in config.sim_targets:
        result = crossroad.run(name, target, config.sim_start_tick, config.sim_end_tick, flow, experiment_anomalies)
        plt.plot(config.graph_time, graphize(result), label=target)

    plt.title('Time - Number of Waiting Cars')
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Number of Cars')
    plt.xticks(list(range(config.graph_start, config.graph_end + 1, 6)),
               list(range(config.graph_start, config.graph_end + 1, 6)))
    plt.savefig('figure/' + name + '.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_flow = environment.sample_environment(name=experiment_name)

        # run_simulation(experiment_name, experiment_flow, False)
        run_simulation(experiment_name + '_anomaly', experiment_flow)
