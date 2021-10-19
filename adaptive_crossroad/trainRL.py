import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from adaptive_crossroad import config, crossroad
from adaptive_crossroad.adaptiveNetwork import AdaptiveNetwork, AdaptiveNetworkMemory, Transition

if __name__ == "__main__":
    value_net = AdaptiveNetwork().to(config.AN_DEVICE)

    optimizer = optim.Adam(value_net.parameters(), config.AN_LR)
    memory = AdaptiveNetworkMemory(config.AN_MEMORY_SIZE)
    steps = 0


    def optimize_model():
        if len(memory) < config.AN_BATCH_SIZE:
            return

        transitions = memory.sample(config.AN_BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        selected_tactics = value_net.model(state_batch).gather(1, tactic_batch)
        next_state_values = value_net.model(next_state_batch).min(1)[0].detach()
        expected_values = (next_state_values * config.AN_GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    for i in tqdm(range(config.AN_EPISODES)):
        flow = crossroad.read_flow(i)
        state = np.array([0] * (1 + config.WAYS))
        for j in range(config.TOTAL_TICK // config.DECISION_LENGTH):
            state_tensor = torch.FloatTensor([state]).to(config.AN_DEVICE)
            tactic = value_net.select_tactic(state_tensor)

            sim_result = crossroad.sim_run_crossroad(config.DECISION_LENGTH, config.TACTICS[tactic],
                                                     flow[j * config.DECISION_LENGTH:(j + 1) * config.DECISION_LENGTH],
                                                     state[1:])

            next_state = [(j + 1) * config.DECISION_LENGTH, *sim_result[-1]]
            reward = sum(sum(sim_result))

            memory.push(state_tensor, tactic, torch.FloatTensor([reward]).to(config.AN_DEVICE),
                        torch.FloatTensor([next_state]).to(config.AN_DEVICE))
            state = next_state

            optimize_model()

    torch.save(value_net, '../adaptive_network/adaptiveNetwork.pth')
