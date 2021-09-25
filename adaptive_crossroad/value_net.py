import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from adaptive_crossroad import config


class ValueNet(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, path=''):
        super(ValueNet, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_size * seq_length, 128)
        self.linear2 = nn.Linear(128, num_classes)

        if path != '':
            self.load_state_dict(torch.load(path))
            self.eval()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(config.DEVICE)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(config.DEVICE)

        (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.reshape(-1, self.hidden_size * self.seq_length)
        out = self.relu(hn)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)

        return out


def tensorize(vn_data):
    ss = StandardScaler()
    target_data = Variable(torch.Tensor(ss.transform(vn_data)))
    target_tensor = torch.reshape(target_data, (int(target_data.shape[0] / config.DECISION_LENGTH),
                                                config.DECISION_LENGTH, target_data.shape[1]))
    return None