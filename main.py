import datetime

from matplotlib import pyplot as plt

import car_accident
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


if __name__ == "__main__":
    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        experiment_flow = environment.sample_environment(name=experiment_name)
        experiment_accidents = []

        god_result = crossroad.run(experiment_name, 'GOD', config.sim_start_tick, config.sim_end_tick, experiment_flow,
                                   experiment_accidents)
        plt.plot(config.graph_time, graphize(god_result), label='GOD')

        for target in config.sim_targets:
            result = crossroad.run(experiment_name, target, config.sim_start_tick, config.sim_end_tick,
                                   experiment_flow, experiment_accidents)
            plt.plot(config.graph_time, graphize(result), label=target)

        plt.title('Time - Number of Waiting Cars')
        plt.legend(loc='upper left')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.xticks(list(range(config.graph_start, config.graph_end + 1, 6)),
                   list(range(config.graph_start, config.graph_end + 1, 6)))
        plt.savefig('figure/' + experiment_name + '.png', dpi=300)
        plt.show()
        plt.close()

        experiment_name = experiment_name + '_accident'
        experiment_accidents = car_accident.generate_accident(config.sim_start_tick, config.sim_end_tick,
                                                              experiment_name)
        god_result = crossroad.run(experiment_name, 'GOD', config.sim_start_tick, config.sim_end_tick, experiment_flow,
                                   experiment_accidents)
        plt.plot(config.graph_time, graphize(god_result), label='GOD')

        for target in config.sim_targets:
            result = crossroad.run(experiment_name, target, config.sim_start_tick, config.sim_end_tick,
                                   experiment_flow, experiment_accidents)
            plt.plot(config.graph_time, graphize(result), label=target)

        plt.title('Time - Number of Waiting Cars')
        plt.legend(loc='upper left')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.xticks(list(range(config.graph_start, config.graph_end + 1, 6)),
                   list(range(config.graph_start, config.graph_end + 1, 6)))
        plt.savefig('figure/' + experiment_name + '.png', dpi=300)
        plt.show()
        plt.close()
