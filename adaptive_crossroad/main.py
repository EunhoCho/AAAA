import numpy as np
from matplotlib import pyplot as plt

from adaptive_crossroad import crossroad
from adaptive_crossroad import config

FIGURE_NAME = 'SMC_PMC_' + str(config.FLOW_NUMBER)

if __name__ == "__main__":
    for i in [85]:
        targets = [('SMC', 'SMC'),
                   ('VN', 'VN'),
                   ('DEFAULT', '')]
        # targets = [('SMC', 'SMC'),
        #            ('VN', 'VN'),
        #            ('DEFAULT', '')]

        time = np.array(range(
            config.TOTAL_TICK // config.DECISION_LENGTH // config.TICK_PER_POINT)) / 360 * config.TEN_SECOND_PER_TICK * config.DECISION_LENGTH * config.TICK_PER_POINT

        point_result_god = np.array(crossroad.run_crossroad('GOD', 'GOD', i))
        plt.plot(time, point_result_god, label='GOD')

        target_result = []
        for target in targets:
            point_result = np.array(crossroad.run_crossroad(target[0], target[1], i))
            plt.plot(time, point_result, label=target[0])
            target_result.append(point_result - point_result_god)

        plt.title('Time - Number of Waiting Cars')
        plt.legend(loc='upper left')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.xticks([0, 6, 12, 18, 24], [0, 6, 12, 18, 24])
        plt.savefig('../figure/' + FIGURE_NAME + '.png', dpi=300)
        plt.show()
        plt.close()

        if len(targets) > 0:
            for j in range(len(target_result)):
                plt.plot(time, target_result[j], label=targets[j][0])

            plt.title('Time - Diff of Number of Waiting Cars')
            plt.legend(loc='upper left')
            plt.xlabel('Hour')
            plt.ylabel('Number of Cars')
            plt.xticks([0, 6, 12, 18, 24], [0, 6, 12, 18, 24])
            plt.savefig('../figure/' + FIGURE_NAME + '_diff.png', dpi=300)
            plt.show()
