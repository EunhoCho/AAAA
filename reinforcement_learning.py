import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

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
    def __init__(self, is_anomaly_aware=False, hidden_layer=config.rl_hidden_layer, path=''):
        super(DQN, self).__init__()
        self.is_anomaly_aware = is_anomaly_aware
        self.input_size = config.cross_ways + 1
        if is_anomaly_aware:
            self.input_size += 1

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
            self.load_state_dict(torch.load(path))
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

        if self.is_anomaly_aware:
            result, next_state = crossroad.run_crossroad(tick, flow, config.cross_tactics[tactic], state[:-2],
                                                         anomalies)

            anomaly_value = 4
            for single_anomaly in anomalies:
                if single_anomaly.valid(tick):
                    anomaly_value = single_anomaly.way
                    break

            next_state = [*next_state, tick + config.cross_decision_length, anomaly_value]
        else:
            result, next_state = crossroad.run_crossroad(tick, flow, config.cross_tactics[tactic], state[:-1],
                                                         anomalies)
            next_state = [*next_state, tick + config.cross_decision_length]
        reward = sum(sum(result))

        self.push_optimize(state, tactic, reward, next_state)
        return next_state, reward

    def train_rl(self):
        train_tqdm = tqdm(range(config.rl_episodes))
        for _ in train_tqdm:
            rewards = 0
            flow = environment.sample_environment()
            if self.is_anomaly_aware:
                anomalies = anomaly.generate_anomaly(0, config.cross_total_tick)
                state = np.array([0] * (2 + config.cross_ways))
                state[-1] = 4
            else:
                anomalies = None
                state = np.array([0] * (1 + config.cross_ways))

            for i in range(0, config.cross_total_tick, config.cross_decision_length):
                state, reward = self.train_one_step(i % config.cross_total_tick, state, flow, anomalies)
                rewards += reward

            train_tqdm.set_description('average: %.2f' % (rewards / config.cross_total_tick))

        if self.is_anomaly_aware:
            torch.save(self.state_dict(), 'model/a_rl.pth')
        else:
            torch.save(self.state_dict(), 'model/rl.pth')

    def push_optimize(self, state, tactic_tensor, reward, next_state):
        self.memory.push(torch.FloatTensor(np.array([state])).to(config.cuda_device), tactic_tensor,
                         torch.FloatTensor(np.array([reward])).to(config.cuda_device),
                         torch.FloatTensor(np.array([next_state])).to(config.cuda_device))

        self.optimize_model()


if __name__ == "__main__":
    rl_model = DQN(False).to(config.cuda_device)
    rl_model.train_rl()

    a_rl_model = DQN(True).to(config.cuda_device)
    a_rl_model.train_rl()
