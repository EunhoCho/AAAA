from tqdm import tqdm

from adaptive_crossroad import config, crossroad


def run_tactic(tactic, flow_number, start_tick):
    str_tactic = config.tactic_string(tactic)
    return crossroad.run_crossroad(str_tactic + '_' + str(start_tick), '', flow_number, tactic,
                                   start_tick * (360 // config.TEN_SECOND_PER_TICK))


if __name__ == "__main__":
    tactics = config.TACTICS[:]

    tactic_tqdm = tqdm(config.TACTICS)
    tactic_tqdm.set_description("Tactic")
    for tactic in tactic_tqdm:
        samples_tqdm = tqdm(range(config.ENV_SAMPLES))
        samples_tqdm.set_description("Samples")
        for i in samples_tqdm:
            for j in range(23):
                run_tactic(tactic, i, j)
