class Car:
    """
    Car Class for representing the car of crossroad.

    Attributes
    ----------
    tick : int
        A tick that the car enter to the crossroad.
    """

    def __init__(self, tick):
        """
        :param tick: A tick that the car enter to the crossroad.
        """
        self.tick = tick

    def __str__(self):
        return 'Car-' + str(self.tick)

    def __repr__(self):
        return 'Car-' + str(self.tick)
