"""


"""
# todo: explanation of what this file is.
import numpy as np
from pysrc.function_approximation.kanerva_coding import KrisKanervaCoder as KanervaCoder


class Conductor:
    """This is inspired by Mahmood 2013. This constructs a representation by performing search using generate and test.
    Instead of using one linear function approximator, it uses a concatenation of many approximators rather than a 
    single function approximator.
    todo
    """

    def __init__(self, tilings, num_features, dimensions, num_cull, function_approximator=KanervaCoder):
        """
        Args:
           tilings (int): the number of active features
           num_features (int): the number of features to map inputs to.
           dimensions (int): the number of observations received on each time-step.
           function_approximator: a reference to a class used to construct the function approximators.
           num_cull(float): the percentage of features to cull at any time-step.
            """
        self.numActiveFeatures = tilings
        self.numPrototypes = num_features
        self.mem_size = self.numPrototypes
        self.dimensions = dimensions
        self.num_cull = num_cull
        self.function_approximator = function_approximator
        self.function_approximators = self.gen_function_approximators()

    def gen_function_approximators(self):
        """Using a given function approximator,regenerates the collection of function approximators which define the 
        representation.
        Args:
            None
        Returns:
            list of function approximators which collectively produce the features from the input observations.
        """
        function_approximators = []
        generated_features = 0  # a variable which tracks how many features are created in total by all the FAs
        while generated_features < self.mem_size:
            # each function approximator has a mask which determines which of the observations which come in are inputs
            # to the function approximator. The number of obs tells us how many observations we use to construct our
            # function approximator. We limit the number of observations to 12 based on advice from Jaden Travnik about
            # the limitations of kanerva coding: he suggested max of 12 inputs with 8000 prototypes.

            num_obs = np.random.randint(1, min(self.dimensions, 12))        # the number of obs in our new FA
            mask = np.random.choice(self.dimensions, num_obs, replace=False
                                    )                                       # the mask we apply to obs for this FA
            remaining_feature_space = self.mem_size-generated_features      # how many more features we can construct
            ptypes = min(8000, remaining_feature_space)
            activation = int(ptypes * 0.02)  # Activation to 2% of features based on Jaden Travnik's suggestion
            fa = self.function_approximator(self.dimensions, ptypes, activation, mask)
            function_approximators.append(fa)                               # add the new FA to the collection
            generated_features += min(ptypes, remaining_feature_space)      # keep track of how many features used
        return function_approximators

    def regen_function_approximators(self, indices):
        """Regenerates the function approximators at a set of indices.
        
        Args:
            indices (list): the location of FAs to regenerate
        Returns:
            None
        """
        locs = []
        for i in indices:
            num_obs = np.random.randint(1, min(self.dimensions, 12))    # the number of obs in our new FA
            mask = np.random.choice(self.dimensions, num_obs, replace=False)                  # the mask we apply to obs for this FA
            n_ptypes = self.function_approximators[i].numPrototypes
            activation = int(n_ptypes * 0.02)  # 2% of features based on Jaden Travnik's suggestion
            fa = self.function_approximator(self.dimensions, n_ptypes, activation, mask)
            self.function_approximators[i] = fa                         # add the new FA to the collection
            locs += range(i*8000, i*8000 + self.function_approximators[i].numPrototypes)
        return locs

    def get_features(self, observations):
        """Returns a feature vector composed of the output of all the function approximators.
        
        Args:
            observations (list): the inputs from
            
        Returns:
            features (list): a binary feature vector with the activations of all the FAs concatenated together.
        """
        features = np.zeros(self.mem_size)
        generated_features = 0
        for fa in self.function_approximators:
            put_range = xrange(
                generated_features, generated_features+ fa.numPrototypes
                )                                                       # get the range of feats that this FA occupies
            fa_features = fa.get_features(observations)                 # get output from function approximator
            np.put(features, put_range, fa_features)                    # put the generated features into the feat vec
            generated_features += fa.numPrototypes
        return features

    def cull(self, weights, step_sizes):
        """Finds the poorest <num_cull> tilings and replaces them with new function approximators: we remove the poorest
        performing function approximators to 
        
        Args:
            weights (list): the weights associated with each feature.
            step_sizes (list): the step-sizes associated with the weights.
        Returns:
            indices (list): the indices of the of the poorest performing features
        """
        to_regen = self.find_cull_candidates(step_sizes, weights)
        return self.regen_function_approximators(to_regen)

    def find_cull_candidates(self, step_sizes, weights):
        """Using the step-sizes and the weights, finds the poorest performing predictions.
        
            Args:
                weights (list): the weights associated with each feature.
                step_sizes (list): the step-sizes associated with the weights.
            Returns:
                indices (list): location of the poorest <num_cull> features   
        """
        fa_evaluate = []                    # storage for utility of FA
        utility = weights * step_sizes
        start_feature = 0
        for fa in self.function_approximators:
            end_feature = start_feature + fa.numPrototypes
            fa_evaluate.append(np.sum(utility[start_feature:end_feature]))
            start_feature += fa.numPrototypes
        return np.argsort(fa_evaluate)[:self.num_cull]

    def get_num_active(self):
        """Calculates the number of active features produced by the collection.
        Args:
            None
            
        Returns:
            The number of active features on each time-step in the constructed feature vector
        """
        return sum([fa.numActiveFeatures for fa in self.function_approximators])