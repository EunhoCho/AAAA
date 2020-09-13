from crossroad import Crossroad
from generate_flow_data import generate_flow_data
from matplotlib import pyplot as plt

if __name__ == "__main__":
    generate_flow_data()
    reactive_y = None
    proactive_y = None
    for i in [4, 7, 8]:
        crossroad = Crossroad('config.txt', i)
        total, delay, x_numbers, y_numbers = crossroad.run()

        if i == 0:
            plt.plot(x_numbers, y_numbers, label='GOD MODE - Tick')
        elif i == 1:
            plt.plot(x_numbers, y_numbers, label='Reactive Mode - Tick')
            reactive_y = y_numbers
        elif i == 2:
            plt.plot(x_numbers, y_numbers, label='Historical Proactive Mode - Tick')
        elif i == 3:
            plt.plot(x_numbers, y_numbers, label='Proactive Mode - Tick')
            proactive_y = y_numbers
        elif i == 4:
            plt.plot(x_numbers, y_numbers, label='Consistent Mode')
        elif i == 5:
            plt.plot(x_numbers, y_numbers, label='PRISM Proactive')
        elif i == 6:
            plt.plot(x_numbers, y_numbers, label='Delayed Reactive Mode')
            reactive_y = y_numbers
        elif i == 7:
            plt.plot(x_numbers, y_numbers, label='Proactive Mode')
            proactive_y = y_numbers
        elif i == 8:
            plt.plot(x_numbers, y_numbers, label='Reactive Mode')
            reactive_y = y_numbers

    plt.title('Time - Number of Waiting Cars')
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Number of Cars')
    plt.savefig('waiting.png', dpi=300)

    plt.close()

    if proactive_y is not None and reactive_y is not None:
        plt.plot(x_numbers, proactive_y, label='Proactive Mode')
        plt.plot(x_numbers, reactive_y, label='Reactive Mode')
        plt.title('Time - Number of Waiting Cars')
        plt.legend(loc='upper left')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.savefig('waiting_2.png', dpi=300)

        plt.close()

        y_numbers = []
        for i in range(len(proactive_y)):
            y_numbers.append(reactive_y[i] - proactive_y[i])

        plt.plot(x_numbers, y_numbers, label='Reactive - Proactive')
        plt.legend(loc='upper left')
        plt.title('Time - Diff of Number of Waiting Cars')
        plt.xlabel('Hour')
        plt.ylabel('Number of Cars')
        plt.savefig('waiting_3.png', dpi=300)
        plt.close()
