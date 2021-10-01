import math
import random
from collections import deque, namedtuple

import torch
from torch import nn

from adaptive_crossroad import config

Transition = namedtuple('Transition', ('state', 'tactic', 'reward', 'next_state'))


class AdaptiveNetworkMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AdaptiveNetwork(nn.Module):
    def __init__(self, path=''):
        super(AdaptiveNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config.WAYS + 1, 256),
            nn.ReLU(),
            nn.Linear(256, len(config.TACTICS))
        )
        self.steps = 0

        if path != '':
            self.load_state_dict(torch.load(path))
            self.eval()

    def select_tactic(self, state):
        sample = random.random()
        eps_threshold = config.AN_EPS_END + (config.AN_EPS_START - config.AN_EPS_END) * math.exp(
            -1. * self.steps / config.AN_EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).min(1)[1].view(1, 1)

        return torch.LongTensor([[random.randrange(len(config.TACTICS))]]).to(config.AN_DEVICE)
