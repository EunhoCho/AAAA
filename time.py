from crossroad import Crossroad

if __name__ == "__main__":
    for i in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        crossroad = Crossroad('config.txt', 7)
        crossroad.config['NUM_SAMPLES'] = i
        total, delay, x_numbers, y_numbers = crossroad.run()