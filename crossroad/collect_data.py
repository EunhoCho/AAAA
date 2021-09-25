from tqdm import tqdm

from crossroad import config, crossroad


def run_tactic(tactic, flow_number):
    str_tactic = str(tactic[0]) + '_' + str(tactic[1]) + '_' + str(tactic[2]) + '_' + str(tactic[3]) + '_' + str(
        tactic[4]) + '_' + str(tactic[5])
    return crossroad.run_crossroad('learning/' + str_tactic, flow_number, tactic)


if __name__ == "__main__":
    tactics = config.TACTICS[:]
    for tactic in tqdm(tactics):
        for i in tqdm(range(config.ENV_SAMPLES)):
            run_tactic(tactic, i)
