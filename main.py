import csv
import datetime
import json
import math
import random

import requests as requests

address = 'http://169.56.76.12/api/message/'
order_address = 'http://169.56.76.12/api/order/'

order_total = 20
order_delay = 0

anomaly_mtbf = 5
anomaly_duration = 10
anomaly_wait = 3


def generate_anomaly(is_real=True, name=None):
    anomalies = [[], [], []]

    count = 1
    tick = 1
    while tick < order_total * 100:
        prob = 1 - math.exp(-count / anomaly_mtbf)
        if random.random() < prob:
            anomalies[random.choice([0, 2] if is_real else [0, 1, 2])].append(tick)
            count = 0
            tick += anomaly_duration

        count += 1
        tick += 1

    if name is not None:
        with open('log/anomaly/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(3):
                for anomaly in anomalies[i]:
                    csv_writer.writerow([anomaly, i])

    return anomalies


def generate_order_list(name=None):
    ans = []
    for i in range(0, order_total, 4):
        candidate = [1, 2, 3, 4]
        ans.extend(random.sample(candidate, 4))

    if name is not None:
        with open('log/order/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(len(ans)):
                csv_writer.writerow([i + order_delay, ans[i]])

    return ans


if __name__ == "__main__":
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    order_list = generate_order_list(experiment_name)
    anomalies = generate_anomaly(is_real=True, name=experiment_name)

    # Import Orders and Anomalies
    # experiment_name = '2021-11-20-21-58-27'
    # order_list = []
    # with open('log/order/' + experiment_name + '.csv', 'r', newline='') as csv_file:
    #     reader = csv.reader(csv_file)
    #     data = list(reader)
    #
    #     for single_data in data:
    #         order_list.append(int(single_data[0]))
    #
    # anomalies = [[], [], []]
    # with open('log/anomaly/' + experiment_name + '.csv', 'r', newline='') as csv_file:
    #     reader = csv.reader(csv_file)
    #     data = list(reader)
    #
    #     for single_data in data:
    #         if int(single_data[1]) == 0:
    #             anomalies[0].append(int(single_data[0]))
    #         else:
    #             anomalies[2].append(int(single_data[0]))

    # Experiment Start
    # AD-RL
    start_message = {'sender': 0,
                     'title': "Start",
                     'msg':
                         json.dumps({
                             "experiment_type": "SAS",
                             "dm_type": "AAAA"
                         })
                     }
    requests.post(address, data=start_message)

    # ORL
    # start_message = {'sender': 0,
    #                  'title': "Start",
    #                  'msg':
    #                      json.dumps({
    #                          "experiment_type": "SAS",
    #                          "dm_type": "ORL"
    #                      })
    #                  }
    # requests.post(address, data=start_message)

    # Random
    # start_message = {'sender': 0,
    #                  'title': "Start",
    #                  'msg':
    #                      json.dumps({
    #                          "experiment_type": "SAS",
    #                          "dm_type": "Random"
    #                      })
    #                  }
    # requests.post(address, data=start_message)
    print("STARTED")

    tick = 0
    tried = [0] * 3

    anomalies[0].append(1000000)
    anomalies[2].append(1000000)

    while True:
        input()

        if tick == anomalies[0][0]:
            anomaly_0 = True
            anomalies[0].pop(0)
        else:
            anomaly_0 = False

        if tick == anomalies[2][0]:
            anomaly_2 = True
            anomalies[2].pop(0)
        else:
            anomaly_2 = False

        process_message = {
            'sender': 0,
            'title': 'Process',
            'msg': json.dumps({
                'anomaly_0': 1 if anomaly_0 else 0,
                'anomaly_2': 1 if anomaly_2 else 0
            })
        }
        res = requests.post(address, data=process_message)
        result = res.json()

        print('tick: ', result['tick'], ', Reward: ', result['reward'])
        if 'alert' in result.keys():
            print(result['alert'])
            break

        print('request: ', result['request'])
        if int(result['c_decision']) != 3:
            print('C: ', result['recent_c'], ' => ', result['c_decision'])
        else:
            print('C: ', result['recent_c'])

        r_decision = list(result['r_decision'])
        for i in range(3):
            if r_decision[i]:
                print('R', str(i), ': ', result['inventory_' + str(i)], ' => S')
            else:
                print('R', str(i), ': ', result['inventory_' + str(i)])

        if int(result['c_decision']) != 3:
            print('S: ', result['inventory_3'], ' => ', result['s_decision'])
        else:
            print('S: ', result['inventory_3'])

        order_string = 'Remain Order: '
        for i in range(1, 5):
            order_string += str(result['order_r_' + str(i)]) + '+' + str(result['order_s_' + str(i)])
            if i != 4:
                order_string += ', '
        print(order_string)

        if tick < 20:
            order_message = {
                'item_type': order_list[tick],
                'dest': random.randrange(3)
            }
            requests.post(order_address, data=order_message)

        tick += 1

