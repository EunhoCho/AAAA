from crossroad import Crossroad
from generate_flow_data import generate_flow_data
from matplotlib import pyplot as plt

if __name__ == "__main__":
    generate_flow_data()
    for i in range(1, 3):
        crossroad = Crossroad('config.txt', i)
        total, delay, x_numbers, y_numbers = crossroad.run()

        if i == 0:
            plt.plot(x_numbers, y_numbers, label='GOD MODE')
        elif i == 1:
            plt.plot(x_numbers, y_numbers, label='Reactive Mode')
        elif i == 2:
            plt.plot(x_numbers, y_numbers, label='Historical Proactive Mode')
        elif i == 3:
            plt.plot(x_numbers, y_numbers, label='Proactive Mode')
        elif i == 4:
            plt.plot(x_numbers, y_numbers, label='Consistent Mode')
        elif i == 5:
            plt.plot(x_numbers, y_numbers, label='PRISM Proactive')

    plt.title('Time - Number of Waiting Cars')
    plt.legend(loc='upper left')
    plt.xlabel('Hour')
    plt.ylabel('Number of Cars')
    plt.savefig('waiting.png', dpi=300)
    plt.show()