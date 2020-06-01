class Car:
    def __init__(self, tick):
        self.tick = tick

    def __str__(self):
        return 'Car-' + str(self.tick)

    def __repr__(self):
        return 'Car-' + str(self.tick)
