from tqdm import tqdm

from adaptive_crossroad import config, crossroad


def run_tactic(tactic, flow_number):
    str_tactic = config.tactic_string(tactic)
    return crossroad.run_crossroad(str_tactic, '', flow_number, tactic)


if __name__ == "__main__":
    tactics = config.TACTICS[:]
    for tactic in tqdm(tactics):
        for i in tqdm(range(config.ENV_SAMPLES)):
            run_tactic(tactic, i)
