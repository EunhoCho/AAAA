import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import car_accident
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
    def __init__(self, input_layer=config.cross_ways, hidden_layer=config.rl_hidden_layer, path=''):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, len(config.cross_tactics))
        )
        self.steps = 0

        if path != '':
            self.load_state_dict(torch.load(path))
            self.eval()

    def select_tactic(self, state):
        sample = random.random()
        eps_threshold = config.rl_epsilon_end + (config.rl_epsilon_start - config.rl_epsilon_end) * math.exp(
            -1. * self.steps / config.rl_epsilon_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.model(state).min(1)[1].view(1, 1)

        return torch.LongTensor([[random.randrange(len(config.cross_tactics))]]).to(config.rl_device)


def train_rl():
    rl_model = DQN().to(config.rl_device)

    optimizer = optim.Adam(rl_model.parameters(), config.rl_learning_rate)
    memory = Memory()
    steps = 0

    def optimize_model():
        if len(memory) < config.rl_batch_size:
            return

        transitions = memory.sample(config.rl_batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        selected_tactics = rl_model.model(state_batch).gather(1, tactic_batch)
        next_state_values = rl_model.model(next_state_batch).min(1)[0].detach()
        expected_values = (next_state_values * config.rl_gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in rl_model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    for _ in tqdm(range(config.rl_episodes)):
        flow = environment.sample_environment()
        state = np.array([0] * config.cross_ways)

        for i in range(0, config.cross_total_tick, config.cross_decision_length):
            tick = i % config.cross_total_tick
            state_tensor = torch.FloatTensor([state]).to(config.rl_device)
            tactic = rl_model.select_tactic(state_tensor)

            result, next_state = crossroad.run_crossroad(tick, flow, config.cross_tactics[tactic], state)
            reward = sum(sum(result))

            memory.push(state_tensor, tactic, torch.FloatTensor([reward]).to(config.rl_device),
                        torch.FloatTensor([next_state]).to(config.rl_device))
            state = next_state

            optimize_model()

    torch.save(rl_model.state_dict(), 'model/rl.pth')


def train_a_rl():
    rl_model = DQN(config.cross_ways + 1).to(config.rl_device)

    optimizer = optim.Adam(rl_model.parameters(), config.rl_learning_rate)
    memory = Memory()
    steps = 0

    def optimize_model():
        if len(memory) < config.rl_batch_size:
            return

        transitions = memory.sample(config.rl_batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        selected_tactics = rl_model.model(state_batch).gather(1, tactic_batch)
        next_state_values = rl_model.model(next_state_batch).min(1)[0].detach()
        expected_values = (next_state_values * config.rl_gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in rl_model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    for _ in tqdm(range(config.rl_episodes)):
        flow = environment.sample_environment()
        accidents = car_accident.generate_accident(0, config.cross_total_tick)
        state = np.array([0] * (1 + config.cross_ways))
        state[-1] = -1

        for i in range(0, config.cross_total_tick, config.cross_decision_length):
            tick = i % config.cross_total_tick
            state_tensor = torch.FloatTensor([state]).to(config.rl_device)
            tactic = rl_model.select_tactic(state_tensor)

            result, next_state = crossroad.run_crossroad(tick, flow, config.cross_tactics[tactic], state[:-1], accidents)

            accident_value = -1
            for accident in accidents:
                if accident.valid(tick):
                    accident_value = accident.way
                    break

            next_state = [*next_state, accident_value]
            reward = sum(sum(result))

            memory.push(state_tensor, tactic, torch.FloatTensor([reward]).to(config.rl_device),
                        torch.FloatTensor([next_state]).to(config.rl_device))
            state = next_state

            optimize_model()

    torch.save(rl_model.state_dict(), 'model/a_rl.pth')


if __name__ == "__main__":
    train_rl()
    train_a_rl()
