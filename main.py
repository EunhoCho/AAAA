from crossroad import Crossroad
from generate_flow_data import generate_flow_data

if __name__ == "__main__":
    generate_flow_data()
    crossroad = Crossroad('config.txt', 0)
    crossroad.run()

    crossroad = Crossroad('config.txt', 1)
    crossroad.run()

    crossroad = Crossroad('config.txt', 2)
    crossroad.run()
