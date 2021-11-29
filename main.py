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
    print_anomalies = []

    count = 1
    tick = 1
    while tick < order_total * 5:
        prob = 1 - math.exp(-count / anomaly_mtbf)
        if random.random() < prob:
            chosen = random.choice([0, 2] if is_real else [0, 1, 2])
            anomalies[chosen].append(tick)
            count = 0
            tick += anomaly_duration

            print_anomalies.append((tick, chosen))

        count += 1
        tick += 1

    if name is not None:
        with open('log/anomaly/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for anomaly in print_anomalies:
                csv_writer.writerow([anomaly[0], anomaly[1]])

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


def inventory_text(inventory, new=None, out=False):
    if inventory == '':
        inventory = []
    else:
        inventory = inventory.split(',')[:-1]

    if new is not None:
        inventory.extend(new)
    if out:
        inventory = inventory[1:]

    ans = ''
    for i in range(5):
        idx = 4 - i
        if len(inventory) <= idx:
            ans += ' '
        else:
            ans += inventory[idx]
        ans += ' '
    return ans


if __name__ == "__main__":
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    order_list = generate_order_list(experiment_name)
    anomalies = generate_anomaly(is_real=True, name=experiment_name)

    dm_type = "AAAA"
    # dm_type = "ORL"
    # dm_type = "Random"

    # Import Orders and Anomalies
    # experiment_name = '2021-11-20-21-58-27'
    # order_list = []
    # with open('log/order/' + experiment_name + '.csv', 'r', newline='') as csv_file:
    #     reader = csv.reader(csv_file)
    #     data = list(reader)
    #
    #     for single_data in data:
    #         order_list.append(int(single_data[1]))
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
                             "dm_type": dm_type
                         })
                     }
    requests.post(address, data=start_message)
    print("STARTED - ", dm_type)

    tick = 0

    anomalies[0].append(1000000)
    anomalies[2].append(1000000)

    request_list = []
    for i in range(4):
        for _ in range(5):
            request_list.append(order_list[i])

    with open('log/warehouse/' + experiment_name + '_' + dm_type + '.txt', 'w') as log_file:
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

            tick_reward_line = 'tick: %2d, Reward: %4d' % (result['tick'], result['reward'])
            print('tick: %2d, Reward: %4d' % (result['tick'], result['reward']))
            log_file.write(tick_reward_line + '\n')
            if 'alert' in result.keys():
                print(result['alert'])
                log_file.write(result['alert'])
                break

            # up line
            up_line = '   ┌─ '
            up_line += inventory_text(result['inventory_r_0'])

            if result['anomaly_0'] == 1:
                if result['stuck_0'] == 1:
                    up_line += '─┐ Anomaly(Stuck)      ┌─ '
                else:
                    up_line += '─┐ Anomaly             ┌─ '
            else:
                up_line += '─┐                     ┌─ '

            if result['c_decision'] == 0:
                up_line += inventory_text(result['inventory_r_0'], new=[str(result['recent_c'])],
                                          out=result['r_decision'][0])
            else:
                up_line += inventory_text(result['inventory_r_0'], out=result['r_decision'][0])

            up_line += '─┐'
            print(up_line)
            log_file.write(up_line + '\n')

            # middle line
            if result['recent_c'] != 0:
                middle_line = str(result['recent_c'])
            else:
                middle_line = ' '

            middle_line += ' ─┼─ '
            middle_line += inventory_text(result['inventory_r_1'])
            middle_line += '─┼─ '
            middle_line += inventory_text(result['inventory_r_3'])
            middle_line += '====> '

            if tick > 0 and tick - 1 < 20:
                middle_line += request_list[tick - 1]
            else:
                middle_line += ' '

            middle_line += ' ─┼─ '
            if result['c_decision'] == 1:
                middle_line += inventory_text(result['inventory_r_1'], new=[str(result['recent_c'])],
                                              out=result['r_decision'][1])
            else:
                middle_line += inventory_text(result['inventory_r_1'], out=result['r_decision'][1])

            middle_line += '─┼─ '
            r_out = []
            for i in [1, 0, 2]:
                if result['r_decision'][i]:
                    r_out.append(result['inventory_r_' + str(i)][0])

            if result['s_decision'] != 3:
                if len(r_out) != 0:
                    middle_line += inventory_text(result['inventory_r_1'], new=r_out, out=True)
                else:
                    middle_line += inventory_text(result['inventory_r_1'], out=True)
            elif len(r_out) != 0:
                middle_line += inventory_text(result['inventory_r_1'], new=r_out)
            else:
                middle_line += inventory_text(result['inventory_r_1'])

            print(middle_line)
            log_file.write(middle_line + '\n')

            # down line
            down_line = '   └─ '
            down_line += inventory_text(result['inventory_r_2'])

            if result['anomaly_2'] == 1:
                if result['stuck_2'] == 1:
                    down_line += '─┘ Anomaly(Stuck)      └─ '
                else:
                    down_line += '─┘ Anomaly             └─ '
            else:
                down_line += '─┘                     └─ '

            if result['c_decision'] == 2:
                down_line += inventory_text(result['inventory_r_2'], new=[str(result['recent_c'])],
                                            out=result['r_decision'][2])
            else:
                down_line += inventory_text(result['inventory_r_2'], out=result['r_decision'][2])

            down_line += '─┘'
            print(down_line)
            log_file.write(down_line + '\n')

            # order line
            order_line = 'Order: '
            orders = []
            for i in range(1, 5):
                orders.append(result['order_r_' + str(i)])
                order_line += str(result['order_r_' + str(i)]) + ' '
            order_line += '  +   '
            for i in range(1, 5):
                orders[i - 1] += result['order_s_' + str(i)]
                order_line += str(result['order_s_' + str(i)]) + ' '
            order_line += ' = '
            for i in range(1, 5):
                order_line += str(orders[i - 1]) + ' '
            order_line = '=> Total: %2d' % sum(orders)
            print(order_line)
            log_file.write(order_line + '\n')

            if tick < 20:
                order_message = {
                    'item_type': order_list[tick],
                    'dest': random.randrange(3)
                }
                requests.post(order_address, data=order_message)

            tick += 1
            log_file.write('\n')

