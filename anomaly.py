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
            value = i - recent_anomaly.tick - config.anomaly_duration - config.anomaly_after
            prob = 1 - math.exp(-value / anomaly_mtth)

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
