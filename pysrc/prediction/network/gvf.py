from pysrc.prediction.error_measures import OnlineVerifier, RUPEE, UDE
import numpy as np


class GeneralValueFunction(object):

    def __init__(self, learner, cumulant, discounting, function_approximation, rl_lambda):
        """
        :param learner: A TD learner which contains the functinonality to learn given some observations
        :param cumulant: The function which determines the value of the signal of interest being learned
        :param discounting: The function which determines 
        :param function_approximation: The function which specifies the 
        :param rl_lambda: The lambda function which specifies the elegibility traces
        """
        self.learner = learner
        self.cumulant = cumulant
        self.discounting = discounting
        self.funtion_approximation = function_approximation
        self.phi = None
        self.gamma = None
        self.rl_lambda = rl_lambda
        self.verifier = OnlineVerifier(self.discounting.discount)

    def step(self, obs):
        """
        Takes in the most recent observation and updates the GVF based on the observation
        :param obs: 
        :return: None
        """
        phi_next = self.funtion_approximation.get_features(obs)
        gamma_next = self.discounting.gamma(obs)
        cumulant = self.cumulant.cumulant(obs)
        self.verifier.update_all(gamma_next, cumulant, self.learner.estimate(phi_next))
        if type(self.phi) is np.ndarray and self.gamma:
            td_error = self.learner.step(cumulant, self.phi, phi_next, self.gamma, self.rl_lambda, gamma_next)
            self.funtion_approximation.update_prototypes(obs, self.learner.step_size, td_error, phi_next, self.learner.th)
        self.phi = phi_next
        self.gamma = gamma_next

    def estimate(self, obs):
        self.learner.estimate(self.funtion_approximation.get_features(obs))

    def current_error(self):
        return self.verifier.calculate_current_error()
