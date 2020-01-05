from pysrc.prediction.discounts.discount import Discount


class HorizonDiscount(Discount):

    def __init__(self, horizon):
        self.discount = float(horizon)

    def gamma(self, obs):
        return self.discount
