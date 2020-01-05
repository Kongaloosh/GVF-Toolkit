class RandomRepresentation:

    def __init__(self, tilings, num_features, dimensions):
        """
        Args:
           tilings
           num_features (int): the number of features to map inputs to.
           dimensions (int)
            """
        self.numActiveFeatures = tilings
        self.numPrototypes = num_features
        self.mem_size = self.numPrototypes
        self.thresholds = np.random.choice([-1,1],(self.dimensions, self.mem_size))
        self.dimensions = dimensions

    def get_features(self, observations):
        """Returns a feature vector based on """
        # each LTU has a threshold;
        features = np.zeros(self.dimensions)
        activations = np.where(np.dot(self.thresholds, observations) > self.thresholds)
        features[activations] = 1
        #

