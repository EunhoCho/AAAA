from crossroad import Crossroad


class Property:
    """
    A class for representing the property that crossroad 1 makes better performance then crossroad 2.

    Attributes
    ----------
    cross_type1 : int
        Decision making style of the crossroad1. 0 = Reactive, 1 = Proactive, 2 = Omniscient.
    cross_type2 : int
        Decision making style of the crossroad2. 0 = Reactive, 1 = Proactive, 2 = Omniscient.

    Methods
    ----------
    check_property()
        Check whether the property is accomplished or not.
    """

    def __init__(self, config, inflow, cross_type1, cross_type2):
        self.cross1 = Crossroad(config, cross_type1)
        self.cross2 = Crossroad(config, cross_type2)
        self.cross1.read_flow(inflow)
        self.cross2.read_flow(inflow)

    def check_property(self):
        """
        Check whether the property is accomplished or not.

        :return: Boolean value whether the property is accomplished or not.
        """
        delay1, num_car1 = self.cross1.run()
        delay2, num_car2 = self.cross2.run()

        avg_delay1 = delay1 / num_car1
        avg_delay2 = delay2 / num_car2

        return avg_delay1 < avg_delay2
