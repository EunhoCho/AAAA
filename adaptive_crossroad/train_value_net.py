import csv

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from tqdm import tqdm

from adaptive_crossroad import config
from adaptive_crossroad.value_net import ValueNet

if __name__ == "__main__":
    for tactic in tqdm(config.TACTICS):
        target = config.tactic_string(tactic)

        x = np.array([])
        y = np.array([])
        for i in range(config.ENV_SAMPLES):
            with open('../log/x/' + target + '_' + str(i) + '.csv', 'r', newline='') as x_file:
                x = [*x, *np.array(list(csv.reader(x_file))[:-config.DECISION_LENGTH]).astype(int)]
            with open('../log/y/' + target + '_' + str(i) + '.csv', 'r', newline='') as y_file:
                y = [*y, *np.array(list(csv.reader(y_file))).astype(int)]

        x = np.array(x)
        y = np.array(y)

        training = int(config.TRAIN_RATE * config.ENV_SAMPLES)
        training_data_length_x = (8640 // config.TEN_SECOND_PER_TICK - config.DECISION_LENGTH) * training
        training_data_length_y = training_data_length_x // config.DECISION_LENGTH

        x_train = x[:training_data_length_x, :]
        x_test = x[training_data_length_x:, :]

        mm = MinMaxScaler()
        y_mm = mm.fit_transform(y)
        y_train = y_mm[:training_data_length_y, :]
        y_test = y_mm[training_data_length_y:, :]
        y_train_tensors = Variable(torch.Tensor(y_train))
        y_test_tensors = Variable(torch.Tensor(y_test))

        valueNet = ValueNet(config.VN_CLASS, config.VN_INPUT_SIZE, config.VN_HIDDEN_SIZE, config.VN_LAYERS,
                            config.DECISION_LENGTH).to(config.DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(valueNet.parameters(), lr=config.VN_LEARNING_RATE)

        for epoch in tqdm(range(config.VN_EPOCHS)):
            outputs = valueNet.forward(x_train)
            optimizer.zero_grad()

            loss = loss_function(outputs, y_train_tensors.to(config.DEVICE))
            loss.backward()
            optimizer.step()

        torch.save(valueNet.state_dict(), '../valueNet/' + target + '.torch')
