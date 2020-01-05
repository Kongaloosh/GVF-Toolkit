from pysrc.prediction.cumulants.cumulant import Cumulant


class CompositionalCumulant(Cumulant):

    def __init__(self, cumulant_index_1, cumulant_index_2, cumulant_function):
        self.cumulant_index_1 = cumulant_index_1
        self.cumulant_index_2 = cumulant_index_2
        self.cumulant_function = cumulant_function

    def cumulant(self, obs):
        return self.cumulant_function(obs[self.cumulant_index_1], obs[self.cumulant_index_2])


def addition(a, b):
    return a+b


def subtract(a, b):
    return a-b


def multiply(a, b):
    return a*b


def divide(a, b):
    return a/(b+0.0001)

