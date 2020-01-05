"""
The classes required to specify an off-policy horde.

TODO:
    - unit testing
    - documentation

"""

from pysrc.prediction.error_measures import update_rupee, update_ude
from pysrc.prediction.cumulants.stimuli_cumulants import StimuliCumulant
from pysrc.prediction.cumulants.random_cumulants import RandomCumulant
from pysrc.prediction.cumulants.compositional_cumulants import *
from pysrc.prediction.off_policy.gtd import *
import numpy as np


class HordeLayer(object):
    """A layer of GVFs which are updated sharing the same function approximator and observations.
    """

    def __init__(self, function_approx, num_predictions, step_sizes, discounts, cumulants, policies, traces_lambda,
                 protected_range=0, recurrance=False, skip=False, remove_base=False, use_step_size=False, **vals):
        """
        Initializes the Horde Layer.
        Args:
            function_approx: a function approximation method which takes observations and returns a feature vector.
            num_predictions: the number of predictions in the layer.
            step_sizes: the step-sizes for each GVF in the layer. 
            discounts: the discount functions for each GVF; presently only accepts 0 <= gamma <= 1.
            cumulants: the cumulant functions for each GVF.
            policies: the policy which each GVF makes a prediction for.
            traces_lambda: the decay value for each GVF.
            protected_range: if you wish to protect a number of the first predictions from regenning, set this value.
            recurrance: flag to allow recurrance in the observations.
            skip: flag to allow skip connections in the observations.
            remove_base: flag to remove the base observations from the construction of state.
            use_step_size: flag to determine if adaptive step-sizes should be used.
        """
        self.td_error = 0

        self.use_step_sizes = use_step_size
        self.function_approximation = function_approx
        self.last_prediction = np.zeros(num_predictions)
        self.protected_range = protected_range

        if remove_base:
            self.max_obs = np.ones(self.function_approximation.dimensions) * -np.inf
            self.min_obs = np.ones(self.function_approximation.dimensions) * np.inf
        else:
            self.max_obs = np.ones(self.function_approximation.dimensions + self.protected_range) * -np.inf
            self.min_obs = np.ones(self.function_approximation.dimensions + self.protected_range) * np.inf

        self.recurrance = recurrance
        self.skip = skip

        # GTD values
        self.discounts = discounts
        self.cumulants = cumulants
        self.weights = np.zeros((num_predictions, self.function_approximation.dimensions))
        self.step_sizes = np.ones((num_predictions, self.function_approximation.dimensions)) * step_sizes
        self.bias_correction = np.zeros(self.weights.shape)
        self.eligibility_traces = np.zeros(self.weights.shape)
        self.traces_lambda = traces_lambda
        self.policies = policies
        self.step_size_bias_correction = 0.0001 # from AW's thesis; no idea how to set it well

        # TIDBD values
        self.beta = np.log(self.step_sizes)
        self.meta_step_size = np.ones((num_predictions, 1)) * 0.01  # we use 0.01 because it is the best in general
        self.h = np.zeros(self.beta.shape)
        self.n = np.zeros(self.beta.shape)
        self.gtd_h = np.zeros(self.beta.shape)

        # Error and internal measurement
        self.avg_error = np.zeros((num_predictions))

        # Unexpected demon error
        self.ude = np.zeros((num_predictions))
        self.ude_beta = np.ones((num_predictions, 1)) * np.minimum(np.average(self.step_sizes), 0.9)
        self.delta_avg = None
        self.delta_var = None

        # RUPEE
        self.tau = np.ones((num_predictions, 1)) * 0.001
        self.rupee_beta = 0.01
        self.rupee_h_trace = np.zeros(self.eligibility_traces.shape)
        self.eligibility_avg = np.zeros(self.eligibility_traces.shape)
        self.last_phi = None
        self.rupee = np.zeros((num_predictions))


    @staticmethod
    def generate_discount(default=0.95):
        """Generates a random discount from the three possible discounts.
        Args:
            default (float): value between 0 and 1; set if you would like to constrain gamma to only one value.
        """
        if default:
            return default
        return np.random.uniform(0, 1)

    @staticmethod
    def generate_cumulant(obs, protected_range):
        """Specified on a per-experiment basis on construction of the network.
        Args:
            obs: an example observation from the environment
            protected_range: the ran
        """
        if len(obs) <= protected_range:
            protected_range = 0
        return np.random.choice(
            [
                HordeLayer.generate_composite_cumulant,
                HordeLayer.generate_stimuli_cumulant
            ],
            1
        )[0](obs, protected_range)

    @staticmethod
    def generate_composite_cumulant(obs, protected_range):
        function = np.random.choice(
            [
                addition,
                subtract,
                multiply,
                divide
            ],
            1
        )[0]
        (a, b) = np.random.randint(protected_range, len(obs), 2)
        return CompositionalCumulant(a, b, function)

    @staticmethod
    def generate_random_cumulant(**var):
        """returns a cumulant which is uniform random"""
        return RandomCumulant()

    @staticmethod
    def generate_stimuli_cumulant(obs, protected_range):
        """Generates a predictions which is of stimuli
        Args:
            obs: example observation from the environment
            protected_range: range """
        return StimuliCumulant(np.random.randint(protected_range, len(obs)))

    @staticmethod
    def generate_lambda():
        """Specified on a per-experiment basis on construction of the network."""
        # return np.random.uniform(0, 1)
        return 0

    @staticmethod
    def generate_step_size(number_of_active_features):
        """Generates a step size.
        Args:
            number_of_active_features (int): the number of active features for this layers function approximator. 
        """
        # return np.random.uniform(0, 1)/float(features)
        return 1. / (float(number_of_active_features) * 2)

    @staticmethod
    def generate_policy(num_actions):
        """Given the number of actions, come up with a policy.
        Args:
            num_actions (int): the number of actions available for the current problem.
        """
        policy = np.random.rand(num_actions)
        return policy / policy.sum()  # we divide by the sum to ensure probabilities add to one.

    def generate_prediction(self, locations, obs, num_actions):
        """Generates a specified number of predictions.
        
        Args:
            locations (list): a list of indices to regenerate new predictions. If protected, we just reset parameters.
            obs: an example observation from the environment.
            num_actions: the number of actions available for the current problem
        """
        # are the observations you're protecting against the right ones?
        for i in locations:
            self.weights[i] = np.zeros(len(self.weights[i]))
            self.eligibility_traces[i] = np.zeros(len(self.eligibility_traces[i]))
            self.last_prediction[i] = 0
            self.bias_correction[i] = np.zeros(len(self.bias_correction[i]))
            if self.recurrance:  # if this is a recurrant network, this prediction is consumed by itself
                place = len(self.max_obs) - (self.weights.shape[0] - i)
                # number of predictions, less number of protected, less current removal = how many from back in recur
                self.min_obs[place] = np.inf
                self.max_obs[place] = -np.inf

            if i > self.protected_range:
                self.cumulants[i] = self.generate_cumulant(self.min_obs, self.protected_range)
                self.traces_lambda[i] = self.generate_lambda()
                self.discounts[i] = self.generate_discount()
                self.step_sizes[i] = self.generate_step_size(self.function_approximation.numActiveFeatures)
                self.policies[:, i] = self.generate_policy(num_actions)
            # Reset Error Monitors
            self.avg_error[i] = 0

            # Unexpected demon error
            self.ude[i] = 0
            self.ude_beta[i] = min(10 * np.average(self.step_sizes[i]), 0.1)
            if type(self.delta_avg) is list:
                self.delta_avg[i] = 0
                self.delta_var[i] = 0

            # RUPEE
            self.rupee_beta = 0.01
            self.tau[i] = 0.001
            self.rupee_h_trace[i] = np.zeros(self.eligibility_traces[i].shape)
            self.eligibility_avg[i] = np.zeros(self.eligibility_traces[i].shape)

    def cull(self, percentage):
        """Culls a percentage of 
        Args:
            percentage: int or float which describes the percentage of the network to regenerate.
        
        Returns:
            indices: Indices culled.
        """
        num_to_cull = max(int((percentage / 100.) * (self.weights.shape[0] - self.protected_range)), 1)
        error = np.argsort(self.avg_error[self.protected_range:]) + self.protected_range
        return error[-num_to_cull:]

    def step(self, observations, policy, action, remove_base=False, terminal_step=False, **vals):
        """Update the Network
        Args:
            observations (list): real-valued list of observations from the environment.
            policy (list): list of length num_actions; the policy of the control policy for the given state.
        Returns:
            predictions (list): the predictions for each GVF given the observations and policy.
        """
        # get the next feature vector
        phi_next = self.function_approximation.get_features(observations)
        if type(self.last_phi) is np.ndarray:
            discounts = self.discounts
            if terminal_step:
                discounts = np.zeros(self.discounts.shape)
            # calculate importance sampling
            rho = (self.policies/policy)[:, action]
            # update the traces based on the new visitation
            self.eligibility_traces = accumulate(self.eligibility_traces, discounts, self.traces_lambda, self.last_phi, rho)
            # calculate the new cumulants
            current_cumulants = np.array([cumulant.cumulant(observations) for cumulant in self.cumulants])
            # get a vector of TD errors corresponding to the performance.
            td_error = calculate_temporal_difference_error(self.weights, current_cumulants, discounts, phi_next,
                                                           self.last_phi)
            self.td_error = td_error
            # update the weights based on the caluculated TD error
            self.weights = update_weights(td_error, self.eligibility_traces, self.weights, discounts, self.traces_lambda, self.step_sizes, self.last_phi, self.bias_correction)
            # update bias correction term
            self.bias_correction = update_gtd_h_trace(self.bias_correction, td_error, self.step_size_bias_correction
                                                      , self.eligibility_traces, self.last_phi)

            # maintain verifiers
            self.rupee, self.tau, self.eligibility_avg = \
                update_rupee(
                    beta_naught=self.rupee_beta,
                    tau=self.tau,
                    delta_e=self.eligibility_avg,
                    h=self.bias_correction,
                    e=self.eligibility_traces,
                    delta=td_error,
                    alpha=self.step_sizes,
                    phi=self.last_phi
                )
            self.ude, self.delta_avg, self.delta_var = update_ude(
                self.ude_beta,
                self.delta_avg,
                self.delta_var,
                td_error
            )
            self.avg_error = self.avg_error * 0.9 + 0.1 * np.abs(td_error)

        self.last_phi = phi_next
        self.last_prediction = np.inner(self.weights, phi_next)
        return self.last_prediction

    def predict(self, phi):
        """Generates the prediction for this GVF layer given some feature vector.
        Args:
            phi: feature vector which describes the state of the environment.
        
        Returns:
            predictions: the current predictions for the GVF layer.
            """
        return np.dot(self.weights, phi)

    def error(self):
        """The return error of the GVF layer.
        Returns:
            return_error: the difference between the expected return and the calculated return. Offset by some horizon.
        """
        return self.estimated_return - self.synced_prediction

    def initalize_episode(self):
        """Resets the predictions in the layer after an episode has finished."""
        self.eligibility_traces = np.zeros(shape=(self.eligibility_traces.shape))
        self.bias_correction = np.zeros(self.h.shape)
        self.last_phi = None
        self.delta_var = None
        self.delta_avg = None
        self.rupee_h_trace = np.zeros(shape=(self.rupee_h_trace.shape))
        self.rupee_beta = 0.01
        self.tau = np.ones(self.tau.shape) * 0.001
        self.eligibility_avg = np.zeros(shape=(self.eligibility_avg.shape))


class HordeHolder(HordeLayer):

    def __init__(self, layers, num_actions):
        """Initializes the horde.
        Args:
            layers (List: HordeLayer): a list of horde layers which form the horde structure.
            num_actions (int): the number of actions which can be taken."""
        self.layers = layers
        self.flag = True
        self.num_actions = num_actions
        print("setting up step-sizes...", [np.average(layer) for layer in self.get_step_sizes()])


    def generate_prediction(self, locations, obs):
        """
        Generates a specified number of predictions
        :param number: 
        :return: None
        """
        if obs is None:
            obs = []
            for layer in self.layers:
                obs.append(layer.min_obs[:layer.protected_range])
        for idx, layer_locations in enumerate(locations):
            self.layers[idx].generate_prediction(layer_locations, obs)

    def cull(self, percentage):
        """
        :param percentage: int or float which describes the percentage of the network to eliminate 
        :return: the indices culled
        """
        return [layer.cull(percentage) for layer in self.layers]

    @staticmethod
    def generate_state(observations, layer, predictions=None, recurrance=False, skip=False, ude=False,
                       remove_base=False, **vals):
        """Generates a state given the observations, a layer

        Args:
            Observations (list): the last report from the environment.
            layer (HordeLayer): the present layer
            """

        # if this is the first layer, we don't use the last layer's predictions, we use the observations
        if remove_base:
            state = []
        else:
            state = observations  # these states will not be included in the function approximation

        # we start the state with the observations so that all other layers have access to the underlying obs
        # for the construction of the baseline cumulants.

        if type(predictions) is type(None):
            state = np.concatenate((state, observations))
        # if this is the second layer, and a skip layer, we use the observations, and the predictions
        elif skip:
            state = np.concatenate((state, observations, predictions))
        # if it's not the first layer, and doesn't skip, just use the predictions
        else:
            state = np.concatenate((state, predictions))
        if ude:
            state = np.concatenate((state, layer.ude))
        if recurrance:
            state = np.concatenate((state, layer.last_prediction))
        return state

    def step(self, observations, policy=None, action=None, recurrance=False, skip=False, ude=False, remove_base=False, **vals):
        """Update the Network"""
        predictions = None
        for layer in self.layers:
            state = self.generate_state(observations, layer, predictions, recurrance, skip, ude, remove_base)
            predictions = layer.step(state, policy, action, remove_base)
        return predictions

    def terminal_step(self, observations, policy, action, recurrance=False, skip=False, ude=False, remove_base=False, **vals):
        """Update the Network"""
        # get the next feature vector
        predictions = None
        for layer in self.layers:
            state = self.generate_state(observations, layer, predictions, recurrance, skip, ude, remove_base)
            predictions = layer.step(state, policy, action, remove_base, terminal_step=True)
        return predictions

    def predict(self, phi):
        return [layer.predict for layer in self.layers]

    def error(self):
        return [layer.error() for layer in self.layers]

    def square_error(self):
        return [np.power(layer.error(),2) for layer in self.layers]

    def avg_error(self):
        return [layer.avg_error for layer in self.layers]

    def get_cumulants(self):
        return [layer.cumulants for layer in self.layers]

    def get_discounts(self):
        return [layer.discounts for layer in self.layers]

    def get_weights(self):
        return [layer.weights.tolist() for layer in self.layers]

    def get_rupee(self):
        return [layer.rupee.tolist() for layer in self.layers]

    def get_ude(self):
        return [layer.ude.tolist() for layer in self.layers]

    def get_step_sizes(self):
        return [layer.step_sizes.tolist() for layer in self.layers]

    def get_num_active(self):
        return np.array([layer.function_approximation.get_num_active() for layer in self.layers])

    def freeze_step_sizes(self):
        for layer in self.layers:
            layer.step_sizes *= 0
            layer.meta_step_size *= 0

    def initalize_episode(self):
        for layer in self.layers:
            layer.initalize_episode()

    def seed_predictions(self):
        for layer in self.layers:
            layer.generate_prediction(range(layer.protected_range, len(layer.weights)), layer.min_obs, self.num_actions)
