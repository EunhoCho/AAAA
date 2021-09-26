import csv

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
from tqdm import tqdm

from adaptive_crossroad import config
from adaptive_crossroad.value_net import ValueNet
from environment import config as env_config

if __name__ == "__main__":
    tactic_tqdm = tqdm(config.TACTICS)
    tactic_tqdm.set_description("Tactic")
    for tactic in tactic_tqdm:
        str_tactic = config.tactic_string(tactic)
        training = int(config.TRAIN_RATE * config.ENV_SAMPLES)

        x = np.array([])
        y = np.array([])
        merge_data_tqdm = tqdm(range(training))
        merge_data_tqdm.set_description("Merge Data")
        for i in merge_data_tqdm:
            for j in range(23):
                with open('../log/x/' + str_tactic + '_' + str(j) + '_' + str(i) +
                          ('_hard' if env_config.METHOD == 'HARD' else '') + '.csv', 'r', newline='') as x_file:
                    x = [*x, *np.array(list(csv.reader(x_file))[:-config.DECISION_LENGTH]).astype(int)]
                with open('../log/y/' + str_tactic + '_' + str(j) + '_' + str(i) +
                          ('_hard' if env_config.METHOD == 'HARD' else '') + '.csv', 'r', newline='') as y_file:
                    y = [*y, *np.array(list(csv.reader(y_file))[1:]).astype(int)]

        x = np.array(x)
        y = np.array(y)

        ss = StandardScaler()
        x_ss = ss.fit_transform(x)
        x_tensor = Variable(torch.Tensor(x_ss))
        x_tensor_reshaped = torch.reshape(x_tensor, (int(x_tensor.shape[0] / config.DECISION_LENGTH),
                                                     config.DECISION_LENGTH, x_tensor.shape[1])).to(config.DEVICE)
        joblib.dump(ss, '../valueNet/scaler/standard/' + str_tactic + '.sc', compress=True)

        ms = MinMaxScaler()
        y_ms = ms.fit_transform(y)
        y_train_tensors = Variable(torch.Tensor(y))
        joblib.dump(ms, '../valueNet/scaler/minmax/' + str_tactic + '.sc', compress=True)

        valueNet = ValueNet(config.VN_CLASS, config.VN_INPUT_SIZE, config.VN_HIDDEN_SIZE, config.VN_LAYERS,
                            config.DECISION_LENGTH).to(config.DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(valueNet.parameters(), lr=config.VN_LEARNING_RATE)

        epoch_tqdm = tqdm(range(config.VN_EPOCHS))
        epoch_tqdm.set_description("epoch tqdm")

        for epoch in epoch_tqdm:
            outputs = valueNet.forward(x_tensor_reshaped)
            optimizer.zero_grad()

            loss = loss_function(outputs, y_train_tensors.to(config.DEVICE))
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                epoch_tqdm.set_description("epoch tqdm loss %.6f" % loss)

        torch.save(valueNet.state_dict(), '../valueNet/valueNet/' + str_tactic + '.torch')
