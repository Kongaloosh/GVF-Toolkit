import numpy as np


def replace(traces, gamma, lmbda, phi):
    """Updates traces with replacing"""
    traces = gamma * lmbda * traces * (phi == 0.) + (phi != 0.) * phi
    return traces


def accumulate(traces, gamma, lmbda, phi):
    """Updates traces without replacing"""
    traces = gamma * lmbda * traces + phi
    return traces


class TDPrediction(object):
    """Base class for TD prediction"""

    def __init__(self, number_of_features, step_size, active_features, trace_decay=replace):
        self.number_of_features = number_of_features
        self.active_features = active_features
        self.step_size = step_size/active_features
        self.th = np.zeros(self.number_of_features)
        self.z = np.zeros(self.number_of_features)
        self.trace_decay = trace_decay

    def estimate(self, phi):
        """Returns a value estimate based on an input ovservation."""
        return np.dot(phi, self.th)

    def initialize_episode(self):
        """Resets the traces for a new episode"""
        self.z = np.zeros(self.number_of_features)

    def calculate_temporal_difference_error(self, reward, gamma_next, phi_next, phi):
        """Based on given inputs, calculates a TD error"""
        return reward + gamma_next * np.dot(phi_next, self.th) - np.dot(phi, self.th)

    def update_weights(self, td_error):
        """Updates the weights given a step-size, traces, and a TD error"""
        self.th += self.step_size * td_error * self.z

    def step(self):
        raise NotImplementedError("This must be implemented for each instance of a TD learner sub-class.")
