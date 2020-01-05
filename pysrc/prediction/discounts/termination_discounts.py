from pysrc.prediction.discounts.discount import Discount


class TerminationDiscount(Discount):
    """handles discounting myopically. Myopic discounting creates one-step prediction."""

    def __init__(self, termination_condition):
        """
        
        :param termination_condition: a function which takes in the observations and evaluates true or false
        """
        self.termination_condition = termination_condition

    def gamma(self, obs):
        if self.termination_condition(obs):
            return 0.0
        else:
            return 1.0