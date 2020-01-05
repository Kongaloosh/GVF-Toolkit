from pysrc.function_approximation.tiles import *
import numpy as np


class ObservationWrapper:

    def __init__(self, obs_length):
        # we use one ac
        self.numActiveFeatures = 1
        self.numPrototypes = obs_length
        self.dimensions = obs_length

    def get_features(self, observations, **vals):
        return observations