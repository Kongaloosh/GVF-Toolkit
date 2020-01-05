from pysrc.prediction.discounts.discount import Discount


class MyopicDiscount(Discount):
    """handles discounting myopically. Myopic discounting creates one-step prediction."""

    def __init__(self):
        pass

    def gamma(self, obs):
        return 0.0