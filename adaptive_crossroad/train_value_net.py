import csv

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
from tqdm import tqdm

from adaptive_crossroad import config
from adaptive_crossroad.value_net import ValueNet

if __name__ == "__main__":
    for tactic in tqdm(config.TACTICS):
        str_tactic = config.tactic_string(tactic)

        x = np.array([])
        y = np.array([])
        for i in range(config.ENV_SAMPLES):
            with open('../log/x/' + str_tactic + '_' + str(i) + '.csv', 'r', newline='') as x_file:
                x = [*x, *np.array(list(csv.reader(x_file))[:-config.DECISION_LENGTH]).astype(int)]
            with open('../log/y/' + str_tactic + '_' + str(i) + '.csv', 'r', newline='') as y_file:
                y = [*y, *np.array(list(csv.reader(y_file))).astype(int)]

        x = np.array(x)
        y = np.array(y)

        training = int(config.TRAIN_RATE * config.ENV_SAMPLES)
        training_data_length_x = (8640 // config.TEN_SECOND_PER_TICK - config.DECISION_LENGTH) * training
        training_data_length_y = training_data_length_x // config.DECISION_LENGTH

        x_train = x[:training_data_length_x, :]
        x_test = x[training_data_length_x:, :]

        ss = StandardScaler()
        x_ss = ss.fit_transform(x_train)
        x_tensor = Variable(torch.Tensor(x_ss))
        x_tensor_reshaped = torch.reshape(x_tensor, (int(x_tensor.shape[0] / config.DECISION_LENGTH),
                                                     config.DECISION_LENGTH, x_tensor.shape[1])).to(config.DEVICE)
        joblib.dump(ss, '../valueNet/scaler/standard/' + str_tactic + '.sc', compress=True)

        ms = MinMaxScaler()
        y_ms = ms.fit_transform(y)
        y_train = y_ms[:training_data_length_y, :]
        y_test = y_ms[training_data_length_y:, :]
        y_train_tensors = Variable(torch.Tensor(y_train))
        y_test_tensors = Variable(torch.Tensor(y_test))
        joblib.dump(ms, '../valueNet/scaler/minmax/' + str_tactic + '.sc', compress=True)

        valueNet = ValueNet(config.VN_CLASS, config.VN_INPUT_SIZE, config.VN_HIDDEN_SIZE, config.VN_LAYERS,
                            config.DECISION_LENGTH).to(config.DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(valueNet.parameters(), lr=config.VN_LEARNING_RATE)

        for epoch in tqdm(range(config.VN_EPOCHS)):
            outputs = valueNet.forward(x_tensor_reshaped)
            optimizer.zero_grad()

            loss = loss_function(outputs, y_train_tensors.to(config.DEVICE))
            loss.backward()
            optimizer.step()

        torch.save(valueNet.state_dict(), '../valueNet/' + str_tactic + '.torch')
