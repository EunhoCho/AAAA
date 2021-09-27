import csv

import numpy as np
from tqdm import tqdm

from adaptive_crossroad import config
from environment import config as env_config

if __name__ == "__main__":
    tactic_tqdm = tqdm(config.TACTICS)
    tactic_tqdm.set_description("Tactic")
    for tactic in tactic_tqdm:
        str_tactic = config.tactic_string(tactic)
        training = int(config.TRAIN_RATE * config.ENV_SAMPLES)

        with open('../learningData/x/' + str_tactic + ('_hard' if env_config.METHOD == 'HARD' else '') + '.csv',
                  'w', newline='') as x_file:
            x_writer = csv.writer(x_file)
            with open(
                    '../learningData/y/' + str_tactic + ('_hard' if env_config.METHOD == 'HARD' else '') + '.csv',
                    'w', newline='') as y_file:
                y_writer = csv.writer(y_file)

                merge_data_tqdm = tqdm(range(training))
                merge_data_tqdm.set_description("Merge Data")
                for i in merge_data_tqdm:
                    for j in range(23):
                        with open('../log/x/' + str_tactic + '_' + str(j) + '_' + str(i) +
                                  ('_hard' if env_config.METHOD == 'HARD' else '') + '.csv', 'r', newline='') as x_file:
                            x_writer.writerows(np.array(list(csv.reader(x_file))[:-config.DECISION_LENGTH]).astype(int))
                        with open('../log/y/' + str_tactic + '_' + str(j) + '_' + str(i) +
                                  ('_hard' if env_config.METHOD == 'HARD' else '') + '.csv', 'r', newline='') as y_file:
                            y_writer.writerows(np.array(list(csv.reader(y_file))[1:]).astype(int))
