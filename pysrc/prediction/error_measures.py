import numpy as np


def update_rupee(beta_naught, tau, delta_e, h, e, delta, alpha, phi):
    """Updates the memory vectors which are used to calculate RUPEE and returns a tuple containing them and
     current RUPEE.
     
     Args:
         beta_naught: user-specified averaging constant for trace of delta_e.
         tau: 
         delta_e: moving average of weight updates
         h: estimate of part of the gradient of MSPBE.
         e: eligibility traces.
         delta: Temporal-difference error.
         alpha: step-size which scales weight updates.
         phi: binary feature vector representing the current state.
     Returns:
         (rupee, tau, delta_e): the current rupee and updated memory variables.
         """
    tau = (1 - beta_naught) * tau + beta_naught
    beta = beta_naught / tau
    delta_e = (1 - beta) * delta_e + beta * e * delta[:,None]
    return np.sqrt(np.abs(np.sum(h * e, axis=1))), tau, delta_e


def update_ude(beta, delta_average, delta_variance, delta):
    """Updates an exponential mean and average; returns the updated values and the Unexpected Demon Error.
    Args:
        beta: amount by which the average is decayed.
        delta_average: memory vector storing present average.
        delta_variance: memory vector storing present variance
        delta: current Temporal-difference error.
        
    Returns:
        (ude, delta_average, delta_variance)
        """
    if delta_average is None:
        delta_average = delta.T
        delta_variance = np.zeros(delta.shape)
    else:
        err = delta.T - delta_average
        delta_average = delta_average + (err * beta).T
        delta_variance = (1-beta) * (delta_variance + beta * err**2)
    return np.abs(delta_average / (np.sqrt(delta_variance) + 0.001)), delta_average, delta_variance


def left_cycle(array, val):
    """Pop and push helper function for numpy arrays.
    Args:
        array: target array to pop and push.
        val: the value to push onto the array.
    Returns:
        array: updated array."""
    array[:, :-1] = array[:, 1:]
    array[:, -1] = val[:, 0]
    return array


def update_verifier(reward_history, gamma_history, prediction_history, gamma, reward, prediction):
    """Updates history of rewards, gammas and predictions and calculates the estimated discounted return of a signal
    of interest.
    Args:
        reward_history: a list maintaining previous observed rewards.
        gamma_history: a list maintaining previous discounting gammas.
        prediction_history: a list maintaining previous predictions.
        gamma: the current discounting factor.
        reward: the current observed reward.
        prediction: the current prediction.
        
    Returns:
        (estimated_return, synced_prediction, reward_history, gamma_history, prediction_history)
        
    """
    gamma_history = left_cycle(gamma_history, gamma)
    reward_history = left_cycle(reward_history, reward)
    prediction_history = left_cycle(prediction_history, prediction)
    estimated_return = np.sum(gamma_history ** np.array(range(gamma_history.shape[1])) * reward_history, axis=1)
    return estimated_return, prediction_history[:, 0], reward_history, gamma_history, prediction_history
