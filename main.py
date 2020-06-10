from crossroad import Crossroad
from generate_flow_data import generate_flow_data

if __name__ == "__main__":
    generate_flow_data()
    for i in range(3):
        crossroad = Crossroad('config.txt', i)
        crossroad.run()
