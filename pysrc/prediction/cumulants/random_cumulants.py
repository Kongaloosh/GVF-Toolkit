from pysrc.prediction.cumulants.cumulant import Cumulant
import numpy as np


class RandomCumulant(Cumulant):

    def __init__(self, **vals):
        pass

    def cumulant(self, obs):
        return np.random.normal(scale=10000)

