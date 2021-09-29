import csv

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
from tqdm import tqdm

from adaptive_crossroad.value_net import ValueNet
from adaptive_crossroad import config

if __name__ == "__main__":
    tactic_tqdm = tqdm(config.TACTICS[:1])
    tactic_tqdm.set_description("Train VN - Tactic")
    for tactic in tactic_tqdm:
        str_tactic = config.tactic_string(tactic)
        x = np.array([])
        y = np.array([])

        with open('../learningData/x/' + str_tactic + ('_hard' if config.METHOD == 'HARD' else '') + '.csv', 'r',
                  newline='') as x_file:
            x = np.array(list(csv.reader(x_file))).astype(int)

        with open('../learningData/y/' + str_tactic + ('_hard' if config.METHOD == 'HARD' else '') + '.csv', 'r',
                  newline='') as y_file:
            y = np.array(list(csv.reader(y_file))).astype(int)

        ss = StandardScaler()
        x_ss = ss.fit_transform(x)
        x_tensor = Variable(torch.Tensor(x_ss))
        x_tensor_reshaped = torch.reshape(x_tensor, (int(x_tensor.shape[0] / config.DECISION_LENGTH),
                                                     config.DECISION_LENGTH, x_tensor.shape[1])).to(config.DEVICE)
        joblib.dump(ss, '../valueNet/scaler/standard/' + str_tactic + '.sc', compress=True)

        ms = MinMaxScaler()
        y_ms = ms.fit_transform(y)
        y_train_tensors = Variable(torch.Tensor(y_ms))
        joblib.dump(ms, '../valueNet/scaler/minmax/' + str_tactic + '.sc', compress=True)

        valueNet = ValueNet(config.VN_INPUT_SIZE, config.VN_HIDDEN_SIZE, config.VN_LAYERS, config.DECISION_LENGTH).to(config.DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(valueNet.parameters(), lr=config.VN_LEARNING_RATE)

        epoch_tqdm = tqdm(range(config.VN_EPOCHS))
        epoch_tqdm.set_description("Train VN")

        for epoch in epoch_tqdm:
            outputs = valueNet.forward(x_tensor_reshaped)
            optimizer.zero_grad()

            loss = loss_function(outputs, y_train_tensors.to(config.DEVICE))
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                epoch_tqdm.set_description("Train VN loss %1.5f at %d" % (loss.item(), epoch))

        torch.save(valueNet.state_dict(), '../valueNet/valueNet/' + str_tactic + '.torch')

        x_plot = x[:8631]
        y_plot = y[:959]
        x_plot_ss = Variable(torch.Tensor(ss.transform(x_plot)))
        x_plot_tensors = torch.reshape(x_plot_ss, (int(x_plot_ss.shape[0] / 9), 9, x_plot_ss.shape[1]))

        train_predict = valueNet(x_plot_tensors.to(config.DEVICE))
        data_predict = train_predict.data.detach().cpu().numpy()
        data_predict = ms.inverse_transform(data_predict)

        plt.plot(y_plot, label='Actual Data')  # actual plot
        plt.plot(data_predict, label='Predicted Data')  # predicted plot
        plt.legend()
        plt.show()
