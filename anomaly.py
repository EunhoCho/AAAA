import csv
import math
import random

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import config
import crossroad
import environment


def generate_anomaly(start, end, anomaly_mtth=config.anomaly_mtth, name=None):
    anomalies = []
    recent_anomaly = CarAccident(start - config.anomaly_duration - config.anomaly_after, 0)

    for i in range(start, end, config.cross_decision_length):
        if recent_anomaly.old(i):
            prob = 1 - math.exp(-config.cross_decision_length / anomaly_mtth)

            if random.random() < prob:
                recent_anomaly = CarAccident(i, random.randrange(0, 4))
                anomalies.append(recent_anomaly)

    if name is not None:
        with open('log/anomaly/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for anomaly in anomalies:
                csv_writer.writerow([anomaly.tick, anomaly.way])

    return anomalies


class Anomaly:
    def __init__(self, tick):
        self.tick = tick

    def valid(self, tick):
        if self.tick <= tick < self.tick + config.anomaly_duration:
            return True

        return False

    def old(self, tick):
        if self.tick + config.anomaly_duration + config.anomaly_after <= tick:
            return True

        return False

    def tick_to_time(self):
        hour = self.tick // config.cross_one_hour
        rest = self.tick - hour * config.cross_one_hour

        minute = rest // config.cross_one_minute
        rest -= minute * config.cross_one_minute

        second = rest * 10 * config.cross_ten_second_per_tick
        return "%02d-%02d-%02d" % (hour, minute, second)


class CarAccident(Anomaly):
    def __init__(self, tick, way):
        super().__init__(tick)
        self.tick = tick
        self.way = way


class CarAccidentDetector(nn.Module):
    def __init__(self, path=''):
        super().__init__()
        self.input_layer = config.cross_ways * config.cross_decision_length + 1
        self.model = nn.Sequential(
            nn.Linear(self.input_layer, config.anomaly_d_hidden_layer),
            nn.ReLU(),
            nn.Linear(config.anomaly_d_hidden_layer, config.anomaly_d_hidden_layer),
            nn.ReLU(),
            nn.Linear(config.anomaly_d_hidden_layer, 5),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=config.anomaly_d_learning_rate,
                                   momentum=config.anomaly_d_momentum)

        normalize = [config.cross_total_tick * 1.0]
        for _ in range(config.cross_ways * config.cross_decision_length):
            normalize.append(100.0)
        self.normalize = np.array(normalize)

        if path != '':
            self.load_state_dict(torch.load(path, map_location=config.cuda_device))
            self.eval()

    def forward(self, data):
        return self.model(data)

    def train_ad(self):
        training_tqdm = tqdm(range(config.anomaly_d_episodes))
        for _ in training_tqdm:
            running_loss = 0.0

            flow = environment.sample_environment()
            anomalies = generate_anomaly(0, config.cross_total_tick, 200)
            state = np.array([0] * config.cross_ways)

            data = []
            labels = []
            for i in range(0, config.cross_total_tick, config.cross_decision_length):
                select = random.randrange(len(config.cross_tactics))
                result, next_state = crossroad.run_crossroad(i, flow, config.cross_tactics[select], state,
                                                             anomalies)
                single_data = [i]
                for j in range(config.cross_decision_length):
                    single_data.extend(result[j])

                anomaly_value = 4
                for single_anomaly in anomalies:
                    if single_anomaly.valid(i):
                        anomaly_value = single_anomaly.way
                        break

                data.append(single_data)
                labels.append(anomaly_value)

            self.optimizer.zero_grad()

            data = np.array(data) / self.normalize

            for i in range(0, config.cross_num_decisions, config.anomaly_d_batch_size):
                data_tensor = torch.FloatTensor(data[i:i + config.anomaly_d_batch_size]).to(config.cuda_device)
                outputs = self(data_tensor)
                loss = self.criterion(outputs, torch.LongTensor(labels[i:i + config.anomaly_d_batch_size]).to(
                    config.cuda_device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            training_tqdm.set_description(
                'loss: %.3f' % running_loss)
            running_loss = 0.0

        torch.save(self.state_dict(), 'model/ad.pth')

    def test(self):
        total = 0
        anomaly_total = 0
        correct = 0
        anomaly_correct = 0
        half_correct = 0
        test_tqdm = tqdm(range(100))
        for _ in test_tqdm:
            flow = environment.sample_environment()
            anomalies = generate_anomaly(0, config.cross_total_tick)
            state = np.array([0] * config.cross_ways)

            for i in range(0, config.cross_total_tick, config.cross_decision_length):
                total += 1

                select = random.randrange(len(config.cross_tactics))
                result, next_state = crossroad.run_crossroad(i, flow, config.cross_tactics[select], state, anomalies)
                single_data = [i]
                for j in range(config.cross_decision_length):
                    single_data.extend(result[j])

                anomaly_value = 4
                for single_anomaly in anomalies:
                    if single_anomaly.valid(i):
                        anomaly_value = single_anomaly.way
                        break

                anomaly_tensor = self.forward(torch.FloatTensor(single_data / self.normalize).to(config.cuda_device))
                predicted_value = torch.argmax(anomaly_tensor)

                if anomaly_value != 4:
                    anomaly_total += 1
                    if predicted_value != 4:
                        half_correct += 1

                if anomaly_value == predicted_value:
                    correct += 1
                    if anomaly_value != 4:
                        anomaly_correct += 1

            test_tqdm.set_description('current: %.3f, anomaly: %.3f, half: %.3f, not: %.3f' %
                                      (correct / total,
                                       anomaly_correct / anomaly_total,
                                       half_correct / anomaly_total,
                                       (correct-anomaly_correct) / (total - anomaly_total)))

        return correct / total


if __name__ == "__main__":
    ad_model = CarAccidentDetector().to(config.cuda_device)
    ad_model.train_ad()

    ad_model = CarAccidentDetector(path='model/ad.pth').to(config.cuda_device)
    print(ad_model.test())
