import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from adaptive_crossroad import config, crossroad

Transition = namedtuple('Transition', ('state', 'tactic', 'reward', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=config.dqn_memory_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, path=''):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config.cross_ways + 1, int(math.log2(len(config.cross_tactics)))),
            nn.ReLU(),
            nn.Linear(int(math.log2(len(config.cross_tactics))), 2 ** int(math.log2(len(config.cross_tactics)) + 1)),
            nn.ReLU(),
            nn.Linear(2 ** int(math.log2(len(config.cross_tactics)) + 1), len(config.cross_tactics))
        )
        self.steps = 0

        if path != '':
            self.load_state_dict(torch.load(path))
            self.eval()

    def select_tactic(self, state):
        sample = random.random()
        eps_threshold = config.dqn_epsilon_end + (config.dqn_epsilon_start - config.dqn_epsilon_end) * math.exp(
            -1. * self.steps / config.dqn_epsilon_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).min(1)[1].view(1, 1)

        return torch.LongTensor([[random.randrange(len(config.cross_tactics))]]).to(config.dqn_device)


if __name__ == "__main__":
    dqn_model = DQN().to(config.dqn_device)

    optimizer = optim.Adam(dqn_model.parameters(), config.dqn_learning_rate)
    memory = Memory()
    steps = 0


    def optimize_model():
        if len(memory) < config.dqn_batch_size:
            return

        transitions = memory.sample(config.dqn_batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        selected_tactics = dqn_model.model(state_batch).gather(1, tactic_batch)
        next_state_values = dqn_model.model(next_state_batch).min(1)[0].detach()
        expected_values = (next_state_values * config.dqn_gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in dqn_model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    for i in tqdm(range(config.dqn_episodes)):
        flow = crossroad.read_flow(i % config.dqn_train_data)
        state = np.array([0] * (1 + config.cross_ways))

        for j in range(0, config.cross_total_tick, config.cross_decision_length):
            state_tensor = torch.FloatTensor([state]).to(config.dqn_device)
            tactic = dqn_model.select_tactic(state_tensor)

            result, next_state = crossroad.run_crossroad(j, flow, config.cross_tactics[tactic], state[1:])

            next_state = [(j + 1) * config.cross_decision_length, *next_state]
            reward = sum(sum(result))

            memory.push(state_tensor, tactic, torch.FloatTensor([reward]).to(config.dqn_device),
                        torch.FloatTensor([next_state]).to(config.dqn_device))
            state = next_state

            optimize_model()

    torch.save(dqn_model.state_dict(), '../model/dqn.pth')
