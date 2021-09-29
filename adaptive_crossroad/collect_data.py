import multiprocessing

from tqdm import tqdm

from adaptive_crossroad import crossroad
from adaptive_crossroad import config


def run_tactic(tactic, flow_number, start_tick):
    str_tactic = config.tactic_string(tactic)
    start_tick = start_tick * (360 // config.TEN_SECOND_PER_TICK)
    return crossroad.run_crossroad(str_tactic + '_' + str(start_tick), '', flow_number, tactic, start_tick,
                                   start_tick + config.TOTAL_TICK, tqdm_off=True)


if __name__ == "__main__":
    tactics = config.TACTICS[:]
    tactic_tqdm = tqdm(tactics)
    tactic_tqdm.set_description("Collect Data - Tactic")

    num_training = int(config.TRAIN_RATE * config.ENV_SAMPLES)
    for tactic in tactic_tqdm:
        samples_tqdm = tqdm(range(num_training))
        samples_tqdm.set_description("Collect Data - Samples")

        for i in samples_tqdm:
            arguments = []
            for j in range(24):
                arguments.append((tactic, i, j))

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.starmap(run_tactic, arguments)
