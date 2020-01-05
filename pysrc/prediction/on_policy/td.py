import numpy as np
from pysrc.prediction.tdprediction import TDPrediction, accumulate, replace


def updater(vals):
    """perfroms the H update for """
    h_update, h = vals
    if h_update > 0:
        return h * h_update
    else:
        return 0


class TD(TDPrediction):

    def __init__(self, number_of_features, alpha, active_features, traces=accumulate, **vals):
        """
        Initializes a TD learner.
        :param number_of_features: The size of the weights and inputs.
        :param alpha: The initial step-size value
        :param active_features: The number of active features
        :param traces: The kind of traces used. Can either be replacing or accumulating.
        :param vals: Extra Kwargs if a dict was used as a config.
        """
        super(TD, self).__init__(number_of_features, alpha, active_features, trace_decay=traces)

    def step(self, reward, phi, phi_next, gamma, lmbda, gamma_next, **vals):
        """
        :param reward: 
        :param phi: The feature-vector for the previous step.
        :param phi_next: The feature-vector for the next step.
        :param gamma: The discounting.
        :param lmbda: The lambda value for elegibility traces
        :param gamma_next: 
        :param vals: 
        :return: 
        """
        td_error = self.calculate_temporal_difference_error(reward, gamma_next, phi_next, phi)
        self.z = self.trace_decay(self.z, gamma_next, lmbda, phi)
        self.update_weights(td_error)


class TDAlphaBound(TD):
    """TD with AlphaBound"""

    def __init__(self, number_of_features, alpha, active_features, traces=accumulate, **vals):
        super(TDAlphaBound, self).__init__(number_of_features, alpha, active_features, traces)

    def update_step_size(self, gamma_next, phi_next, phi):
        self.step_size = min(self.alpha, np.abs(np.dot(self.z, (gamma_next * phi_next - phi))) ** (-1))

    def step(self, reward, phi, phi_next, gamma, lmbda, gamma_next):
        td_error = self.calculate_temporal_difference_error(reward, gamma_next, phi_next, phi)
        self.z = self.trace_decay(self.z, gamma_next, lmbda, phi)
        self.update_step_size(gamma_next, phi_next, phi)
        self.update_weights(td_error)


class TDRMSProp(TD):
    """RMS Prop for TD"""

    def __init__(self, nf, alpha, active_features, decay, **vals):
        super(TDRMSProp, self).__init__(nf, alpha, active_features)
        self.eta = self.alpha  # where we
        self.gradient_avg = np.zeros(self.nf)  # where we keep RMSprop's history of gradients
        self.decay = decay

    def update_gradient(self, gamma, phi, phi_next, **vals):
        self.gradient_avg = self.decay * self.gradient_avg + (1. - self.decay) * (gamma * phi_next - phi) ** 2

    def update_step_size(self):
        self.step_size = self.eta / (np.sqrt(self.gradient_avg) + 10. ** (-8))

    def step(self, reward, phi, phi_next, gamma, lmbda, gamma_next):
        td_error = self.calculate_temporal_difference_error(reward, gamma_next, phi_next, phi)
        self.z = self.trace_decay(self.z, gamma_next, lmbda, phi)
        self.update_gradient(phi, phi_next)
        self.update_step_size()
        self.update_weights(td_error)


class TDBD(TD):
    """ classdocs """

    def __init__(self, number_of_features, beta, active_features, meta_step_size, traces=accumulate, scalar=False, **vals):
        super(TDBD, self).__init__(number_of_features, beta, active_features, traces)
        self.scalar = scalar
        self.beta = np.zeros(self.number_of_features)
        self.h = np.zeros(self.number_of_features)
        self.meta_step_size = meta_step_size
        val = np.exp(beta) / active_features
        self.beta = np.ones(self.number_of_features) * np.log(val)
        self.step_size = np.exp(self.beta)

    def update_h(self, gamma, phi, phi_next, td_error):
        if self.scalar:
            self.h = self.h + np.dot(self.step_size * self.z * (gamma * phi_next - phi), self.h)  # what we're bounding for h update
            self.h[self.h < 0] = 0
            # todo: check this again; is it true that this is equivalent?
        else:
            h_update = np.ones(self.number_of_features) + self.step_size * self.z * (
                gamma * phi_next - phi)  # what we're bounding for h update
            self.h = map(updater, zip(h_update, self.h))    # todo: fix this bad mess
        self.h += self.step_size * td_error * self.z

    def update_meta_weights(self, gamma, phi, phi_next, td_error):
        if self.scalar:
            self.beta -= self.meta_step_size * td_error * np.dot((gamma * phi_next - phi), self.h)  # update to step-size weights
        else:
            self.beta -= self.meta_step_size * td_error * (gamma * phi_next - phi) * self.h  # update to step-size weights

    def calculate_step_size(self):
        return np.exp(self.beta)

    def initepisode(self):
        self.z = np.zeros(self.number_of_features)

    def step(self, reward, phi, phi_next, gamma, lmbda, gamma_next, **vals):
        td_error = self.calculate_temporal_difference_error(reward, gamma_next, phi_next, phi)
        self.z = self.trace_decay(self.z, gamma_next, lmbda, phi)
        self.update_meta_weights(gamma, phi, phi_next, td_error)
        self.step_size = self.calculate_step_size()
        self.update_weights(td_error)
        self.update_h(gamma, phi, phi_next, td_error)


class AutoTDBD(TDBD):
    """TD-based Incremental Delta Bar Delta using a full-gradient and adapting techniques from AutoStep."""

    def __init__(self, nf, beta, active_features, meta_step_size, traces=accumulate, scalar=False, **vals):
        """"""
        super(AutoTDBD, self).__init__(nf, beta, active_features, meta_step_size, traces, scalar)
        self.n = np.zeros(self.number_of_features)      # stores a decaying trace of the maximum weight update
        self.tau = 10.**4                               # Heat parameter; little suggestion performance is sensitive

    def initialize_episode(self):
        self.z = np.zeros(self.number_of_features)      # reset eligibility traces
        self.h = np.zeros(self.number_of_features)      # reset idbd traces
        self.n = np.zeros(self.number_of_features)      # reset the maximum weight update

    def get_effective_step_size(self, gamma, phi, phi_next):
        delta_phi = (gamma * phi_next - phi)
        return np.dot(-(np.exp(self.beta) * self.z), delta_phi)

    def update_meta_weights(self, gamma, phi, phi_next, td_error):
        """"""
        # copy the maximum weight update and protect from div by 0
        n_copy = self.n
        n_copy = np.array(n_copy)
        n_copy[n_copy == 0.] = 1.
        # update the meta-weights with which the step-size is defined
        if self.scalar:
            self.beta -= self.meta_step_size * (td_error * np.dot((gamma * phi_next - phi) , self.h)) / n_copy  # update to step-size weights
        else:
            self.beta -= self.meta_step_size * (td_error * (gamma * phi_next - phi) * self.h) / n_copy

    def update_n_accumluation(self, gamma, phi, phi_next, td_error):
        delta_phi = (gamma * phi_next - phi)
        tracker = -np.exp(self.beta) * self.z * delta_phi
        update = np.abs(td_error * delta_phi * self.h)
        self.n = np.maximum(np.abs(update), self.n + (1. / self.tau) * tracker * (np.abs(update) - self.n))

    def normalize_beta(self, gamma, phi, phi_next):
        effective_step_size = self.get_effective_step_size(gamma,phi,phi_next)
        m = np.maximum(effective_step_size, 1.)
        self.beta -= np.log(m)

    def calculate_step_size(self):
        return np.exp(self.beta)

    def step(self, reward, phi, phi_next, gamma, lmbda, gamma_next, **vals):
        td_error = self.calculate_temporal_difference_error(reward, gamma_next, phi_next, phi)
        self.z = self.trace_decay(self.z, gamma_next, lmbda, phi)
        self.update_n_accumluation(gamma, phi, phi_next, td_error)
        self.update_meta_weights(gamma, phi, phi_next, td_error)
        self.normalize_beta(gamma, phi, phi_next)
        self.step_size = self.calculate_step_size()
        self.update_weights(td_error)
        self.update_h(gamma, phi, phi_next, td_error)
        return td_error
