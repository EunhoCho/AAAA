import csv
import os
import re

import numpy as np
from tqdm import tqdm

from adaptive_crossroad import config

if __name__ == "__main__":
    path = '../log/x/'
    file_list = os.listdir(path)

    tactic_tqdm = tqdm(config.TACTICS)
    tactic_tqdm.set_description("Merge_Data - Tactic")
    for tactic in tactic_tqdm:
        str_tactic = config.tactic_string(tactic)
        files = [file_name for file_name in file_list if re.match(str_tactic + r'.*\.csv', file_name)]

        with open('../learningData/x/' + str_tactic + ('_hard' if config.METHOD == 'HARD' else '') + '.csv',
                  'w', newline='') as x_file:
            x_writer = csv.writer(x_file)
            with open(
                    '../learningData/y/' + str_tactic + ('_hard' if config.METHOD == 'HARD' else '') + '.csv',
                    'w', newline='') as y_file:
                y_writer = csv.writer(y_file)

                merge_data_tqdm = tqdm(files)
                merge_data_tqdm.set_description("Merge_Data - File")
                for file_name in merge_data_tqdm:
                    with open('../log/x/' + file_name, 'r', newline='') as x_log_file:
                        x_writer.writerows(
                            np.array(list(csv.reader(x_log_file))[:-config.DECISION_LENGTH]).astype(int))
                    with open('../log/y/' + file_name, 'r', newline='') as y_log_file:
                        y_writer.writerows(np.array(list(csv.reader(y_log_file))[1:]).astype(int))
