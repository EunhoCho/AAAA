import numpy as np
from matplotlib import pyplot as plt

from adaptive_crossroad import crossroad, config

FIGURE_NAME = 'SMC_PMC_' + str(config.FLOW_NUMBER)

if __name__ == "__main__":
    targets = [('PMC', 'PMC', config.FLOW_NUMBER)]
    # targets = [('DEFAULT', '', config.FLOW_NUMBER),
    #            ('VN_TEST', 'VN', config.FLOW_NUMBER),
    #            ('SMC', 'SMC', config.FLOW_NUMBER),
    #            ('PMC', 'PMC', config.FLOW_NUMBER)]

    time = np.array(range(8640 // config.TEN_SECOND_PER_TICK // config.DECISION_LENGTH // config.TICK_PER_POINT)) \
           / 360 * config.TEN_SECOND_PER_TICK * config.DECISION_LENGTH * config.TICK_PER_POINT
    for target in targets:
        point_result = crossroad.run_crossroad(target[0], target[1], target[2])
        plt.plot(time, [0, *point_result], label=target[0])

    plt.title('Time - Number of Waiting Cars')
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Number of Cars')
    plt.xticks([0, 6, 12, 18, 24], [0, 6, 12, 18, 24])
    plt.savefig('../figure/' + FIGURE_NAME + '.png', dpi=300)
    plt.show()
