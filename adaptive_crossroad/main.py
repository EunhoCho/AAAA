from matplotlib import pyplot as plt

from adaptive_crossroad import config, crossroad


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
    for i in config.sim_target_flow:
        targets = ['DQN', 'DEFAULT']
        # targets = ['AD-DQN']

        god_result = crossroad.run('GOD_' + str(i), 'GOD', config.sim_start_tick, config.sim_end_tick, i)
        plt.plot(config.graph_time, graphize(god_result), label='GOD')

        for target in targets:
            result = crossroad.run(target + '_' + str(i), target, config.sim_start_tick, config.sim_end_tick, i)
            plt.plot(config.graph_time, graphize(result), label=target)

        plt.title('Time - Number of Waiting Cars')
        plt.legend(loc='upper left')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.xticks(list(range(config.graph_start, config.graph_end + 1, 6)),
                   list(range(config.graph_start, config.graph_end + 1, 6)))
        plt.savefig('../figure/' + str(i) + '.png', dpi=300)
        plt.show()
        plt.close()
