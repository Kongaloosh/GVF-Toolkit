from pysrc.function_approximation.tiles import *
import numpy as np


class TileCoderWrapper:

    def __init__(self, tilings, num_features, dimensions):
        """
        Args:
           tilings
           num_features
           dimensions
            """
        self.numActiveFeatures = tilings
        self.numPrototypes = num_features
        self.mem_size = self.numPrototypes
        self.dimensions = dimensions

    def get_features(self, observations, **vals):
        a = np.zeros(self.mem_size)
        a[getTiles(self.numActiveFeatures, self.numPrototypes, observations)] = 1
        return a

    def get_num_active(self):
        return self.numActiveFeatures