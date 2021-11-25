import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn, optim

import anomaly
import config
import crossroad
import environment

Transition = namedtuple('Transition', ('state', 'tactic', 'reward', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=config.rl_memory_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, hidden_layer=config.rl_hidden_layer, path=''):
        super(DQN, self).__init__()
        self.input_size = config.cross_ways + 1

        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, len(config.cross_tactics))
        )

        self.steps = -1
        self.memory = Memory()
        self.optimizer = optim.Adam(self.parameters(), config.rl_learning_rate)

        if path != '':
            self.load_state_dict(torch.load(path, map_location=config.cuda_device))
            self.eval()

    def select_tactic(self, state):
        self.steps += 1
        sample = random.random()
        eps_threshold = config.rl_epsilon_end + (config.rl_epsilon_start - config.rl_epsilon_end) * math.exp(
            -1. * self.steps / config.rl_epsilon_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).min(1)[1].view(1, 1)

        return torch.LongTensor([[random.randrange(len(config.cross_tactics))]]).to(config.cuda_device)

    def optimize_model(self):
        if len(self.memory) < config.rl_batch_size:
            return

        transitions = self.memory.sample(config.rl_batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        selected_tactics = self.model(state_batch).gather(1, tactic_batch)
        next_state_values = self.model(next_state_batch).min(1)[0].detach()
        expected_values = (next_state_values * config.rl_gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_one_step(self, tick, state, flow, anomalies):
        state_tensor = torch.FloatTensor(np.array([state])).to(config.cuda_device)
        tactic = self.select_tactic(state_tensor)

        result, next_state = crossroad.run_crossroad(tick, flow, config.cross_tactics[tactic], state[:-1],
                                                     anomalies)
        next_state = [*next_state, tick + config.cross_decision_length]
        reward = sum(sum(result))

        self.push_optimize(np.array(state), tactic, reward, np.array(next_state))
        return next_state, reward

    def train_rl(self, anomaly_value=4):
        for j in range(config.rl_episodes if anomaly_value == 4 else config.rl_episodes_transfer):
            rewards = 0
            flow = environment.sample_environment()
            if anomaly_value != 4:
                anomalies = []
                for i in range(0, config.cross_total_tick, config.anomaly_duration):
                    anomalies.append(anomaly.CarAccident(i, anomaly_value))
            else:
                anomalies = None

            state = np.array([0] * self.input_size)

            for i in range(0, config.cross_total_tick, config.cross_decision_length):
                state, reward = self.train_one_step(i % config.cross_total_tick, state, flow, anomalies)
                rewards += reward

            print('epoch: %d, average: %.2f' % (j, rewards / config.cross_total_tick))

        if anomaly_value != 4:
            torch.save(self.state_dict(), 'model/a_rl_' + str(anomaly_value) + '.pth')
        else:
            torch.save(self.state_dict(), 'model/rl.pth')

    def push_optimize(self, state, tactic_tensor, reward, next_state):
        self.memory.push(torch.FloatTensor(np.array([state])).to(config.cuda_device), tactic_tensor,
                         torch.FloatTensor(np.array([reward])).to(config.cuda_device),
                         torch.FloatTensor(np.array([next_state])).to(config.cuda_device))

        self.optimize_model()


if __name__ == "__main__":
    rl_model = DQN().to(config.cuda_device)
    rl_model.train_rl()

    for i in range(4):
        a_rl_model = DQN(path='model/rl_new.pth').to(config.cuda_device)
        a_rl_model.train_rl(i)
