import csv
import math
import random

import config


def generate_accident(start, end, name=None):
    accidents = []
    recent_accident = Accident(start - config.accident_duration - config.accident_after, 0)

    for i in range(start, end, config.cross_decision_length):
        if recent_accident.old(i):
            value = i - recent_accident.tick - config.accident_duration - config.accident_after
            prob = 1 - 2 ** (-value / config.accident_mtth)

            if random.random() < prob:
                recent_accident = Accident(i, random.randrange(0, 4))
                accidents.append(recent_accident)

    if name is not None:
        with open('log/accident/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for accident in accidents:
                csv_writer.writerow([accident.tick, accident.way])

    return accidents


class Accident:
    def __init__(self, tick, way):
        self.tick = tick
        self.way = way

    def valid(self, tick):
        if self.tick + config.accident_duration > tick:
            return True

        return False

    def old(self, tick):
        if self.tick + config.accident_duration + config.accident_after <= tick:
            return True

        return False

    def tick_to_time(self):
        hour = self.tick // config.cross_one_hour
        rest = self.tick - hour * config.cross_one_hour

        minute = rest // config.cross_one_minute
        rest -= minute * config.cross_one_minute

        second = rest * 10 * config.cross_ten_second_per_tick
        return "%02d-%02d-%02d" % (hour, minute, second)
