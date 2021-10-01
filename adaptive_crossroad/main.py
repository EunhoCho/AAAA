import numpy as np
from matplotlib import pyplot as plt

from adaptive_crossroad import config
from adaptive_crossroad import crossroad

if __name__ == "__main__":
    start_tick = config.START_HOUR * 360 // config.TEN_SECOND_PER_TICK
    end_tick = config.END_HOUR * 360 // config.TEN_SECOND_PER_TICK

    graph_start = config.GRAPH_START * 360 // config.TEN_SECOND_PER_TICK // config.DECISION_LENGTH // config.TICK_PER_POINT
    graph_end = config.GRAPH_END * 360 // config.TEN_SECOND_PER_TICK // config.DECISION_LENGTH // config.TICK_PER_POINT

    for i in [750]:
        # targets = []
        # for target in config.TACTICS:
        #     targets.append((config.tactic_string(target), '', target))
        targets = [('AN', 'AN'),
                   ('DEFAULT', '')]
        # targets = [('SMC', 'SMC'),
        #            ('AN', 'AN'),
        #            ('DEFAULT', '')]

        time = np.arange(config.GRAPH_START, config.GRAPH_END,
                         config.TEN_SECOND_PER_TICK * config.DECISION_LENGTH * config.TICK_PER_POINT / 360)

        point_result_god = np.array(crossroad.run_crossroad('GOD', 'GOD', i, start_tick=start_tick, end_tick=end_tick))[
                           graph_start:graph_end]
        plt.plot(time, point_result_god, label='GOD')

        for target in targets:
            point_result = np.array(crossroad.run_crossroad(target[0], target[1], i, start_tick=start_tick,
                                                            end_tick=end_tick))[graph_start:graph_end]
            # point_result = np.array(crossroad.run_crossroad(target[0], target[1], start_tick=start_tick,
            #                                                 end_tick=end_tick, default_decision=target[2]))[graph_start:graph_end]
            plt.plot(time, point_result, label=target[0])

        plt.title('Time - Number of Waiting Cars')
        plt.legend(loc='upper left')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.xticks(list(range(config.GRAPH_START, config.GRAPH_END + 1, 6)),
                   list(range(config.GRAPH_START, config.GRAPH_END + 1, 6)))
        plt.savefig('../figure/' + str(i) + '.png', dpi=300)
        plt.show()
        plt.close()
