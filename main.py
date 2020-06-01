from crossroad import Crossroad
from generate_flow_data import generate_flow_data

if __name__ == "__main__":
    generate_flow_data()
    crossroad = Crossroad('config.txt')
    crossroad.run()
