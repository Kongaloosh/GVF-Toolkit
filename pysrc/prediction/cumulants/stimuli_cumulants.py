from pysrc.prediction.cumulants.cumulant import Cumulant


class StimuliCumulant(Cumulant):

    def __init__(self, cumulant_index):
        self.cumulant_index = cumulant_index
        pass

    def cumulant(self, obs):
        return obs[self.cumulant_index]

