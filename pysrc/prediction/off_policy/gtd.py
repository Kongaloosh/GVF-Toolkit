"""
Functions which perform the necessary operations for GTD(lambda) learning.
"""

import numpy as np

__author__ = "Alex Kearney"


def replace(traces, gamma, lmbda, phi, rho):
    """Updates traces with replacing
    Args:
        traces: the eligibility traces used to assign credit to previous states for current rewards or observations.
        gamma: the discount function for the current state.
        lmbda: the eligibility trace decay value. 
        phi: the binary feature vector representing the current state.
        rho: importance sampling ratio.
        
    Returns:
        tracs: the eligibility traces updated for the current state.
    
    """
    traces = rho[:, None] * (gamma * lmbda * traces * (phi.T == 0.) + (phi.T != 0.) * phi.T)
    return traces


def accumulate(traces, gamma, lmbda, phi, rho):
    """Updates traces without replacing
    Args: 
        traces: the eligibility traces to asign credit for current observations to previous states.
        gamma: the discounting value for the current update.
        lmbda: the amount by which we decay our eligibility traces.
        phi: binary feature vector representing the current state.
        rho: importance sampling ratio.
    Returns:
        traces: the eligibility traces to asign credit for current observations to previous states.
    """
    return rho[:, None] * (traces * (gamma * lmbda)[:,None] + phi)


def calculate_temporal_difference_error(weights, cumulant, gamma_next, phi_next, phi):
    """Based on given inputs, calculates a TD error
    Args:
        weights: the learned weights of our model.
        cumulant: the currently observed cumulant signal which we are learning to predict.
        gamma_next: the discounting value for the current update.
        phi_next: binary feature vector representing the next state.
        phi: binary feature vector representing the current state.
    Returns:
        td_error: the temporal-difference error for the current observations.
    """
    return cumulant + gamma_next * np.sum(weights*phi_next, axis=1) - np.sum(weights*phi, axis=1)


def update_weights(td_error, traces, weights, gamma, lmbda, step_size, phi_next, h):
    """Updates the weights given a step-size, traces, and a TD error.
    Args:
        td_error: the temporal-difference error.
        traces: the eligibility traces to assign credit for current observations to previous states.
        weights: the learned weights of our model.
        gamma: the discounting value for the current update.
        lmbda: the amount by which we decay our eligibility traces.
        step_size: the amount by which weight updates are scaled.
        phi_next: binary feature vector representing the next state.
        h: bias correction term.
    Returns:
        weights: updated weights correcting for current TD error.
    """
    return weights + step_size * (td_error[:, None] * traces - gamma[:,None] * (1 - lmbda) * np.sum(traces*h, axis=1)[:,None] * phi_next)


def update_gtd_h_trace(h, td_error, step_size, traces, phi):
    """Updates the GTD bias correction term h
    Args:.
        h: bias correction term.
        td_error: the temporal-difference error.
        step_size: the amount by which weight updates are scaled.
        traces: the eligibility traces to assign credit for current observations to previous states
        phi: binary feature vector representing the current state.
    Returns:
        h: updated bias correction term.
    """
    return h + 0.0001 * (td_error[:,None] * traces - np.sum(h*phi, axis=1)[:, None] * phi)


def update_gtd_idbd_traces(gtd_h_trace, h, gtd_h, step_size, phi, td_error, traces):
    """Updates the meta-traces for GTD-IDBD
    Args:
        gtd_h_trace (ndarray): the meta-tracse for gtd.
        h (ndarray): the meta-traces.
        step_size (ndarray): the current step-size values.
        phi (ndarray): the last feature vector; represents s_t.
        td_error (nd-array): the temporal-difference error for the current time-step.
        traces (ndarray): the eligibility traces.
    """
    gtd_h_trace += step_size * (td_error[:,None] * traces - np.sum(h* phi,axis=1)[:, None] * traces -
                    (np.sum(gtd_h*phi,axis=1)[:, None] + np.sum(gtd_h_trace*phi,axis=1)[:, None]) * phi)
    return gtd_h_trace


def update_meta_traces(h, gtd_h, gtd_bias, step_size, phi, phi_next, td_error, traces, gamma, lmbda):
    """Updates the meta-traces for TDBID; is an accumulating trace of recent weight updates.
    Args:
        h (ndarray): the meta-traces.
        gtd_h (ndarray): the meta-traces for gtd.
        gtd_bias (ndarray) : the bias-correction for GTD.
        step_size (ndarray): the current step-size values.
        phi (ndarray): the last feature vector; represents s_t.
        phi_next (ndarray): the feature vector for the following state.
        td_error (nd-array): the temporal-difference error for the current time-step.
        traces (ndarray): the eligibility traces.
        gamma: the discounting value for the current update.
        lmbda: the decay term for elegibility traces.
    """

    h += step_size * (td_error[:,None] * traces - np.sum(h* phi, axis=1)[:, None] * traces -
                      (gamma * (1-lmbda))[:,None] * phi_next * (np.sum(traces* gtd_bias, axis=1)[:, None] + np.sum(traces* gtd_h, axis=1)[:, None]))

    return h


def update_meta_weights(phi, td_error, meta_weights, meta_trace, meta_step_size, meta_normalizer):
    """Updates the meta-weights for TIDBD; these are used to set the step-sizes.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        td_error (float): the temporal-difference error for the current time-step.
    """

    meta_normalizer_copy = np.copy(meta_normalizer)
    meta_normalizer_copy[meta_normalizer_copy == 0] = 1.
    meta_weights += meta_step_size * (phi * td_error[:, None] * meta_trace)/meta_normalizer_copy
    return meta_weights


def update_normalizer_accumulation(phi, td_error, tau, meta_weights, meta_trace, traces, meta_normalizer_trace):
    """Tracks the size of the meta-weight updates.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        td_error (float): the temporal-difference error for the current time-step."""
    update = np.abs(td_error[:, None] * phi * meta_trace)
    tracker = np.exp(meta_weights) * phi * traces
    meta_normalizer_trace = np.maximum(
        np.abs(update),
        meta_normalizer_trace + (10.**-4) * tracker * (np.abs(update) - meta_normalizer_trace)
    )
    return meta_normalizer_trace


def get_effective_step_size(gamma, lmbda, phi, phi_next, meta_weights, traces, gtd_bias, td_error):
    """Returns the effective step-size for a given time-step
    Args:
        gamma (float): discount factor
        phi (ndarray): the last feature vector; represents s_t
        phi_next (ndarray): feature vector for state s_{t+1}
        meta_weights (ndarray):
        traces (ndarray):

    Returns:
        effective_step_size (float): the amount by which the error was reduced on a given example.
    """
    denominator = np.copy(td_error)
    denominator[np.sign(denominator) == 0] = 1
    a = (
            np.sum((np.exp(meta_weights)*traces)*(phi - gamma[:, None] * phi_next), axis=1)[:,None] -

        (np.sum(
              (
                  (gamma * (1-lmbda))[:, None] * phi_next * np.sum(traces*gtd_bias, axis=1)[:,None])
                  *
                  (phi - gamma[:, None] * phi_next), axis=1
                  )[:,None]
         /(denominator[:,None])))
    a[np.sign(td_error) == 0] = 0
    return a


def normalize_step_size(gamma, lmbda, phi, phi_next, meta_weights, traces, gtd_bias, td_error):
    """Calculates the effective step-size and normalizes the current step-size by that amount.
    Args:
        gamma (float): discount factor
        phi (ndarray): feature vector for state s_t
        phi_next (ndarray): feature vector for state s_{t+1}"""
    effective_step_size = get_effective_step_size(gamma, lmbda, phi, phi_next, meta_weights, traces, gtd_bias, td_error)
    m = np.maximum(effective_step_size, 1.)
    meta_weights -= np.log(m)
    return meta_weights


def tdbid(phi, phi_next, gamma, lmbda, td_error, traces, meta_step_size, tau, meta_weights, idbid_meta_trace, meta_normalizer_trace, gtd_meta_traces, gtd_h):
    """Using the feature vector for s_t and the current TD error, performs TIDBD and updates step-sizes.
    Args:
        phi (ndarray): the last feature vector; represents s_t
        phi_next (ndarray): feature vector for state s_{t+1}
        gamma (float): discount factor
        td_error (float): the temporal-difference error for the current time-step.

    """
    meta_normalizer_trace = update_normalizer_accumulation(phi, td_error, tau, meta_weights, idbid_meta_trace, traces, meta_normalizer_trace)
    meta_weights = update_meta_weights(phi, td_error, meta_weights, idbid_meta_trace, meta_step_size, meta_normalizer_trace)
    meta_weights = normalize_step_size(gamma, lmbda, phi, phi_next, meta_weights, traces, gtd_h, td_error)
    gtd_meta_traces = update_gtd_idbd_traces(gtd_meta_traces, idbid_meta_trace, gtd_h, calculate_step_size(meta_weights), phi, td_error, traces)
    idbid_meta_trace = update_meta_traces(idbid_meta_trace, gtd_meta_traces, gtd_h, calculate_step_size(meta_weights), phi, phi_next, td_error, traces, gamma, lmbda)
    return (meta_normalizer_trace,meta_weights, gtd_meta_traces, idbid_meta_trace, calculate_step_size(meta_weights))


def calculate_step_size(meta_weights):
    """Calculates the current alpha value using the meta-weights
    Returns:
         None
    """
    return np.exp(meta_weights)

