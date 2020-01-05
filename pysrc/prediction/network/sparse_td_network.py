from pysrc.prediction.error_measures import update_rupee, update_ude, update_verifier
from pysrc.prediction.cumulants.stimuli_cumulants import StimuliCumulant
from pysrc.prediction.cumulants.random_cumulants import RandomCumulant
from pysrc.prediction.cumulants.compositional_cumulants import *
from pysrc.prediction.on_policy.vector_network_td import *
from scipy.sparse.csc import csc_matrix
import numpy as np


class MatrixGeneralValueFunctionNetwork(object):
    """
    Specifies a network of general value functions. The generation methods
    """

    def __init__(self, function_approx, num_predictions, step_sizes, discounts, cumulants, traces_lambda, protected_range=0, recurrance=False, skip=False, remove_base=False, use_step_size=False, **vals):
        """
        Initialize the network with a collection of predictions
        :param predictions: a list which contains instances of predictions"""

        # learning framework values

        self.use_step_sizes = use_step_size
        self.function_approximation = function_approx
        self.last_prediction = np.zeros(num_predictions)
        self.protected_range = protected_range
        size = function_approx.dimensions
        if remove_base:
            self.max_obs = csc_matrix((np.ones(self.function_approximation.dimensions) * -np.inf), dtype=np.float64)
            self.min_obs = csc_matrix((np.ones(self.function_approximation.dimensions) * +np.inf), dtype=np.float64)
        else:
            self.max_obs = csc_matrix((np.ones(self.function_approximation.dimensions + self.protected_range) * -np.inf), dtype=np.float64)
            self.min_obs = csc_matrix((np.ones(self.function_approximation.dimensions + self.protected_range) * +np.inf), dtype=np.float64)
        self.recurrance = recurrance
        self.skip = skip

        # TD values
        self.discounts = discounts[:,None]
        self.cumulants = cumulants
        self.weights = csc_matrix(np.zeros((num_predictions, self.function_approximation.numPrototypes+1)), dtype=np.float64)
        self.step_sizes = csc_matrix(np.ones((num_predictions, self.function_approximation.numPrototypes+1)) * step_sizes[:,None], dtype=np.float64)
        self.eligibility_traces = csc_matrix(np.zeros(self.weights.shape), dtype=np.float64)
        self.traces_lambda = traces_lambda[:,None]

        # TIDBD values
        self.beta = csc_matrix(np.log(self.step_sizes), dtype=np.float64)
        self.meta_step_size = np.ones((num_predictions,1)) * 0.01 # we use 0.01 because it is the best in general
        self.h = csc_matrix(np.zeros((self.beta.shape)), dtype=np.float64)
        self.n = csc_matrix(np.zeros((self.beta.shape)), dtype=np.float64)

        # Error and internal measurement
        self.estimated_return = np.zeros((num_predictions,1))
        self.synced_prediction = np.zeros((num_predictions,1))
        self.rupee = np.zeros((num_predictions,1))
        self.ude = np.zeros((num_predictions))


        # Discounted return estimate
        horizon = int(np.log(0.001)/np.log(0.95))  # length of horizon we maintain for estimation
        self.reward_history = np.zeros((num_predictions, horizon))
        self.gamma_history = np.zeros((num_predictions, horizon))
        self.prediction_history = np.zeros((num_predictions, horizon))

        # Unexpected demon error
        self.ude = np.zeros((num_predictions))
        self.ude_beta = np.ones((num_predictions,1)) * np.minimum(np.average(self.step_sizes), 0.9)
        self.delta_avg = np.zeros((num_predictions,1))
        self.delta_var = np.zeros((num_predictions,1))

        # RUPEE
        # beta is shared between RUPEE and UDE
        # todo: does sharing beta make sense: rupee seems to use 0.1 alpha, not 10 alpha
        self.tau = np.ones((num_predictions,1)) * 0.001
        self.rupee_h_trace = csc_matrix(np.zeros(self.eligibility_traces.shape), dtype=np.float64)
        self.eligibility_avg = csc_matrix(np.zeros(self.eligibility_traces.shape), dtype=np.float64)
        self.last_phi = None
        self.rupee = np.zeros((num_predictions))


    @staticmethod
    def generate_discount():
        """Generates a random discount from the three possible discounts."""
        # gamma = np.random.uniform(0, 1)
        # while gamma == 0:
        #     gamma = np.random.uniform(0, 1)
        # return gamma
        return 0.95

    @staticmethod
    def generate_cumulant(obs, protected_range):
        """Specified on a per-experiment basis on construction of the network."""
        return np.random.choice(
                         [
                            # MatrixGeneralValueFunctionNetwork.generate_random_cumulant,
                            MatrixGeneralValueFunctionNetwork.generate_composite_cumulant,
                            MatrixGeneralValueFunctionNetwork.generate_stimuli_cumulant
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
    def generate_random_cumulant(obs):
        return RandomCumulant()

    @staticmethod
    def generate_stimuli_cumulant(obs, protected_range):
        return StimuliCumulant(np.random.randint(protected_range, len(obs)))

    @staticmethod
    def generate_lambda():
        """Specified on a per-experiment basis on construction of the network."""
        # return np.random.uniform(0, 1)
        return 0.95

    @staticmethod
    def generate_step_size(features):
        # return np.random.uniform(0, 1)/float(features)
        return 0.6/float(features)

    def generate_prediction(self, locations, obs):
        """
        Generates a specified number of predictions
        :param number: 
        :return: None
        """
        # are the observations you're protecting against the right ones?
        for i in locations:
            self.weights[i] = np.zeros(len(self.weights[i]))
            self.eligibility_traces[i] = np.zeros(len(self.eligibility_traces[i]))
            self.last_prediction[i] = 0
            self.beta[i] = np.ones(self.beta[i].shape) * (1./self.function_approximation.numActiveFeatures)
            self.n[i] = np.zeros(self.n[i].shape)
            self.h[i] = np.zeros(self.h[i].shape)
            if self.recurrance:     # if this is a recurrant network, this prediction is consumed by itself
                place = len(self.max_obs) - (self.weights.shape[0] - i)
                # number of predictions, less number of protected, less current removal = how many from back in recur
                self.min_obs[place] = np.inf
                self.max_obs[place] = -np.inf
            if i > self.protected_range:
                self.cumulants[i] = self.generate_cumulant(self.min_obs, self.protected_range)
                self.traces_lambda[i] = self.generate_lambda()
                self.discounts[i] = self.generate_discount()
                self.step_sizes[i] = self.generate_step_size(self.function_approximation.numActiveFeatures)
            # Reset Error Monitors
            horizon = int(np.log(0.001) / np.log(0.95))  # length of horizon we maintain for estimation
            self.reward_history[i] = np.zeros(horizon)
            self.gamma_history[i] = np.zeros(horizon)
            self.prediction_history[i] = np.zeros(horizon)

            # Unexpected demon error
            self.ude[i] = 0
            self.ude_beta[i] = (10 * np.average(self.step_sizes[i]))
            self.delta_avg[i] = 0
            self.delta_var[i] = 0

            # RUPEE
            # beta is shared between RUPEE and UDE
            # todo: does sharing beta make sense: rupee seems to use 0.1 alpha, not 10 alpha
            self.tau[i] = 0.001
            self.rupee_h_trace[i] = np.zeros(self.eligibility_traces[i].shape)
            self.eligibility_avg[i] = np.zeros(self.eligibility_traces[i].shape)

    def cull(self, percentage):
        """
        :param percentage: int or float which describes the percentage of the network to eliminate 
        :return: the indices culled
        """
        num_to_cull = max(int((percentage/100.) * (self.weights.shape[0]-self.protected_range)), 1)
        error = np.argsort(self.avg_error()[self.protected_range:]) + self.protected_range
        return error[-num_to_cull:]

    def step(self, observations, remove_base=False, **vals):
        """Update the Network"""
        # get the next feature vector
        add = len(self.min_obs)-len(observations)
        observations = np.concatenate((observations, np.zeros(add)))   # we don't have the predictions in the first layer, so concat zeros
        self.min_obs = np.minimum(observations, self.min_obs)
        self.max_obs = np.maximum(observations, self.max_obs)
        observations += np.abs(self.min_obs)
        observations = np.divide(observations, (np.abs(self.max_obs) + np.abs(self.min_obs)),
                                where=(np.abs(self.max_obs) + np.abs(self.min_obs)) != 0)
        observations[np.isnan(observations)] = 0
        observations[np.isinf(observations)] = 0
        # we take off the protected range, as they exist only to serve as cumulants.
        if remove_base:
            phi_next = self.function_approximation.get_features(observations)
        else:
            phi_next = self.function_approximation.get_features(observations[self.protected_range:])
        phi_next = np.concatenate((phi_next, [1]))[:,None]
        if type(self.last_phi) is np.ndarray:
            # update the traces based on the new visitation
            self.eligibility_traces = accumulate(self.eligibility_traces, self.discounts, self.traces_lambda, phi_next)
            # calculate the new cumulants
            current_cumulants = np.array([cumulant.cumulant(observations) for cumulant in self.cumulants])[:,None]
            # get a vector of TD errors corresponding to the performance.
            td_error = calculate_temporal_difference_error(self.weights, current_cumulants, self.discounts, phi_next, self.last_phi)
            # update the weights based on the caluculated TD error
            predictions = self.predict(phi_next)
            # update the running trace of maximum meta_weight updates
            if self.use_step_sizes:
                self.n = update_normalizer_accumulation(
                    self.n, self.beta, self.eligibility_traces, self.last_phi, self.h, td_error
                )
                self.beta = update_meta_weights(self.beta, self.last_phi, self.meta_step_size, td_error, self.h)
                self.beta = normalize_meta_weights(
                    self.beta, self.eligibility_traces, self.discounts, self.last_phi, phi_next
                )
                self.step_sizes = calculate_step_size(self.beta)
            # update weights
            self.weights = update_weights(td_error, self.eligibility_traces, self.weights, self.step_sizes)
            # update beta trace
            self.h = update_meta_weight_update_trace(
                self.h, self.eligibility_traces, self.last_phi, td_error, self.step_sizes
            )
            # print "begin \t", self.ude_beta.shape, \
            #     "\nele \t", self.eligibility_avg.shape, \
            #     "\nh tra \t", self.rupee_h_trace.shape, \
            #     "\nele t \t", self.eligibility_traces.shape, \
            #     "\ntd er \t", td_error.shape, \
            #     "\nss \t", self.step_sizes.shape, \
            #     "\nphi \t", self.last_phi.shape, \
            #     "\ntau \t",  self.tau.shape,\
            #     '\n'



            # maintain verifiers
            self.rupee, self.tau, self.eligibility_avg, self.rupee_h_trace =\
                update_rupee(
                    self.ude_beta,
                    self.tau,
                    self.eligibility_avg,
                    self.rupee_h_trace,
                    self.eligibility_traces,
                    td_error,
                    self.step_sizes,
                    self.last_phi
                )
            self.ude, self.delta_avg, self.delta_var = update_ude(
                self.ude_beta,
                self.delta_avg,
                self.delta_var,
                td_error
            )

            self.estimated_return, self.synced_prediction, self.reward_history, self.gamma_history, self.prediction_history = \
                update_verifier(
                    self.reward_history,
                    self.gamma_history,
                    self.prediction_history,
                    self.discounts,
                    current_cumulants,
                    predictions
                )
            if len(np.where(np.isnan(td_error))[0])> 0:
                print("regenning", np.where(np.isnan(td_error))[0])
            self.generate_prediction(np.where(np.isnan(td_error))[0], observations)
            self.generate_prediction(np.where(np.isinf(td_error))[0], observations)
            self.generate_prediction(np.where(np.isnan(self.error()))[0], observations)
            self.generate_prediction(np.where(np.isinf(self.error()))[0], observations)
            self.last_prediction = predictions[:, 0]

            # Unexpected demon error
            self.ude_beta[np.where(np.isnan(self.ude))] = (10 * np.average(self.step_sizes))
            self.ude_beta[np.where(np.isinf(self.ude))] = (10 * np.average(self.step_sizes))
            self.delta_avg[np.where(np.isnan(self.ude))] = 0
            self.delta_avg[np.where(np.isinf(self.ude))] = 0
            self.delta_var[np.where(np.isnan(self.ude))] = 0
            self.delta_var[np.where(np.isinf(self.ude))] = 0
            self.ude[np.where(np.isinf(self.ude))] = 0
            self.ude[np.where(np.isnan(self.ude))] = 0

            # RUPEE
            # beta is shared between RUPEE and UDE
            # todo: does sharing beta make sense: rupee seems to use 0.1 alpha, not 10 alpha
            # self.tau[np.where(np.isnan(self.rupee))] = 0.001
            # self.tau[np.where(np.isinf(self.rupee))] = 0.001
            # self.rupee_h_trace[:,np.where(np.isnan(self.rupee))] = np.zeros(self.eligibility_traces.shape[1])
            # self.rupee_h_trace[:,np.where(np.isinf(self.rupee))] = np.zeros(self.eligibility_traces.shape[1])
            # self.eligibility_avg[:,np.where(np.isnan(self.rupee))] = np.zeros(self.eligibility_traces.shape[1])
            # self.eligibility_avg[:,np.where(np.isinf(self.rupee))] = np.zeros(self.eligibility_traces.shape[1])

        self.last_phi = phi_next
        return self.last_prediction

    def predict(self, phi):
        return np.dot(self.weights, phi)

    def error(self):
        return self.estimated_return-self.synced_prediction

    def init_episode(self):
        self.eligibility_traces * 0
        self.last_phi = None
        self.prediction_history *= 0
        self.gamma_history *= 0
        self.reward_history *= 0
        self.delta_var *= 0
        self.delta_avg *= 0
        self.rupee_h_trace *= 0
        self.eligibility_avg *= 0


class HierarchicalMatrixNetwork(MatrixGeneralValueFunctionNetwork):
    """
    A network which has layers which prevent recurrances. Each layer is a 
    """

    def __init__(self, layers):
        """"""
        self.layers = layers
        self.flag = True

    def generate_prediction(self, locations, obs):
        """
        Generates a specified number of predictions
        :param number: 
        :return: None
        """
        if obs == None:
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
    def generate_state(observations, layer, predictions=None, recurrance=False, skip=False, ude=False, remove_base=False, **vals):
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
        # print("obs\t{0}\nude\t{1}\nstate\t{2}\n".format(np.array(observations).shape, layer.ude.shape, state.shape))
        return state

    def step(self, observations, recurrance=False, skip=False, ude=False, remove_base=False, **vals):
        """Update the Network"""
        # get the next feature vector
        predictions = None
        for layer in self.layers:
            state = self.generate_state(observations, layer, predictions, recurrance, skip, ude, remove_base)
            predictions = layer.step(state, remove_base)

    def predict(self, phi):
        return np.array([layer.predict for layer in self.layers])

    def error(self):
        return np.array([layer.error() for layer in self.layers])

    def square_error(self):
        return np.array([layer.square_error() for layer in self.layers])

    def avg_error(self):
        return np.array([layer.avg_error() for layer in self.layers])

    def get_cumulants(self):
        return np.array([layer.cumulants for layer in self.layers])

    def get_discounts(self):
        return np.array([layer.discounts for layer in self.layers])

    def get_weights(self):
        return np.array([layer.weights for layer in self.layers])

    def get_rupee(self):
        return np.array([layer.rupee for layer in self.layers])

    def get_ude(self):
        return np.array([layer.ude for layer in self.layers])

    def get_step_sizes(self):
        return [layer.step_sizes for layer in self.layers]

    def freeze_step_sizes(self):
        for layer in self.layers:
            layer.step_sizes *= 0
            layer.meta_step_size *= 0

    def init_episode(self):
        for layer in self.layers:
            layer.initalize_episode()

    def seed_predictions(self):
        for layer in self.layers:
                layer.generate_prediction(range(layer.protected_range, len(layer.weights)), layer.min_obs)

