import numpy as np


class Agent(object):
    """
    The base class that an agent will be structured as.
    """

    def __init__(self, observation_space, action_space):
        """
        A wrapper for an agent
        Args:
            observation_space (int): the number of observations in the observation vector.
            action_space (int): the number of actions which can be taken.
        """
        self.observation_space = observation_space
        self.action_space = action_space

    def initialize_episode(self):
        raise NotImplementedError()

    def terminal_step(self, reward):
        raise NotImplementedError()

    def get_action(self, state_prime):
        raise NotImplementedError

    def step(self, observation, reward):
        """
        Args:
            observation (list): the float observations from the environment.
            reward (int): the reward for taking an action in the last state.
        Returns:
            action (int): an action in range (0,action_space) which indicates what actions to take.
        """
        raise NotImplementedError()


class RandomAgent(Agent):
    """
        A random agent which chooses actions equiprobably regardless of observations.
    """

    def __init__(self, observation_space, action_space, **vals):
        super(RandomAgent, self).__init__(observation_space, action_space)
        self.state_prime = None

    @staticmethod
    def __str__():
        return  "RandomAgent"

    def terminal_step(self, reward):
        pass

    def initialize_episode(self):
        pass

    def get_action(self, state_prime):
        return np.random.randint(self.action_space)

    def get_policy(self, observation):
        return np.ones(self.action_space)/self.action_space

    def step(self, observation, reward):
        pass


class QLearner(Agent):
    def __init__(self, observation_space, action_space, step_size=0.1, discount_rate=0.99, epsilon=0.1,
                 eligibility_decay=0.9):
        super(QLearner, self).__init__(observation_space, action_space)
        self.state = None
        self.state_prime = None
        self.action = None
        self.action_prime = None
        self.epsilon = epsilon
        self.elegibilty_decay = eligibility_decay
        self.q = np.random.rand(observation_space, action_space)
        self.step_size = step_size
        self.discount_rate = discount_rate

    @staticmethod
    def __str__():
        return "QLearner"

    def initialize_episode(self):
        """Resets agent for new episode"""
        self.state = None
        self.state_prime = None
        self.action = None
        self.action_prime = None

    def get_action(self, state_prime):
        """
        Args:
            state_prime (list): the state we are presently in.
        Returns:
            action_prime (int): a value which indicates what action to take in the environment for this step.
        """
        self.action = self.action_prime
        self.state = self.state_prime
        self.state_prime = state_prime

        if np.random.random() < self.epsilon:
            self.action_prime = np.random.randint(self.action_space)
        else:
            self.action_prime = np.argmax(np.dot(state_prime, self.q))
        return self.action_prime

    def estimate(self, val):
        return np.argmax(self.q[val])

    def terminal_step(self, reward):
        delta = reward - np.dot(self.state_prime, self.q)[self.action_prime]
        self.q[:, self.action_prime] += self.step_size * delta * self.state_prime

    def step(self, observation, reward):
        """
        Args:
            observation (list): an expression of the present world state.
            reward (int): the reward for the last transition.
        Returns:
            action (int): chosen action for the current state.
        """
        action_q = np.argmax(np.dot(observation, self.q))
        delta = reward + (self.discount_rate * np.dot(observation, self.q)[action_q]) \
            - np.dot(self.state_prime, self.q)[self.action_prime]
        self.q[:, self.action_prime] += self.step_size * delta * self.state_prime

    def get_policy(self, phi):
        """Given the present state returns the probability of taking any given action
        Args:
            phi (list): binary feature vector describing the state we wish to calculate the policy for.
        Returns:
            policy (list): list of length <num_actions> where each index i is the probability of taking action i."""
        num_actions = self.q1.shape[1]
        policy = np.zeros(num_actions)  # make a list with as many elements as there are actions
        policy += np.argmax(np.dot(phi, self.q)) * (1 - self.epsilon) \
            + (np.ones(num_actions) / num_actions) * self.epsilon
        return policy / policy.sum()


class DoubleQ(Agent):
    def __init__(self, observation_space, action_space, step_size=0.1, discount_rate=0.99, epsilon=0.01,
                 elegibilty_decay=0.9):
        """Initialises all of the Q Learning variables
        Args:
            observation_space (int): size of observations.
            action_space (int): size of available actions.
            step_size (float): a value greater than 0 which weights the updates.
            discount_rate (float): a value which discounts future rewards.
            epsilon (float): a value 0 < e < 1 which determines probability of taking a random action.
            elegibilty_decay (float): a value 0 < \ < 1 which determines the horizon of backup.
        """
        super(DoubleQ, self).__init__(observation_space, action_space)
        self.state = None
        self.state_prime = None
        self.action = None
        self.action_prime = None
        self.epsilon = epsilon
        self.elegibilty_decay = elegibilty_decay
        self.q1 = np.random.rand(observation_space, action_space)
        self.q2 = np.random.rand(observation_space, action_space)
        self.step_size = step_size
        self.discount_rate = discount_rate

    def initialize_episode(self):
        """Resets agent for new episode"""
        self.state = None
        self.state_prime = None
        self.action = None
        self.action_prime = None

    def get_action(self, state_prime):
        """
        Args:
            state_prime (list): the state we are presently in.
        Returns:
            action_prime (int): a value which indicates what action to take in the environment for this step.
        """
        self.action = self.action_prime
        self.state = self.state_prime
        self.state_prime = state_prime

        if np.random.random() < self.epsilon:
            self.action_prime = np.random.randint(self.action_space)
        else:
            self.action_prime = np.argmax(np.dot(state_prime, self.q1) + np.dot(state_prime, self.q2))
        return self.action_prime

    def estimate(self, val):
        return np.argmax(self.q1[val] + self.q2[val])

    def terminal_step(self, reward):
        if np.random.rand() <= 0.5:
            delta = reward - np.dot(self.state_prime, self.q1)[self.action_prime]
            self.q1[:, self.action_prime] += self.step_size * delta * self.state_prime
        else:
            delta = reward - np.dot(self.state_prime, self.q2)[self.action_prime]
            self.q2[:, self.action_prime] += self.step_size * delta * self.state_prime

    def step(self, observation, reward):
        """
        Args:
            observation (list): an expression of the present world state.
            reward (int): the reward for the last transition.
        Returns:
            action (int): chosen action for the current state.
        """
        if np.random.random() <= 0.5:
            # UPDATING Q1
            action_q = np.argmax(np.dot(observation, self.q1))
            delta = reward + (self.discount_rate * np.dot(observation, self.q2)[action_q]) \
                    - np.dot(self.state_prime, self.q1)[self.action_prime]
            self.q1[:, self.action_prime] += self.step_size * delta * self.state_prime
        else:
            # UPDATING Q2
            action_q = np.argmax(np.dot(observation, self.q2))
            delta = reward + (self.discount_rate * np.dot(observation, self.q1)[action_q]) \
                    - np.dot(self.state_prime, self.q2)[self.action_prime]
            self.q2[:, self.action_prime] += self.step_size * delta * self.state_prime

    def get_policy(self, phi):
        """Given the present state returns the probability of taking any given action
        Args:
            phi (list): binary feature vector describing the state we wish to calculate the policy for.
        Returns:
            policy (list): list of length <num_actions> where each index i is the probability of taking action i."""
        num_actions = self.q1.shape[1]
        policy = np.zeros(num_actions)  # make a list with as many elements as there are actions
        policy += np.argmax(np.dot(phi, self.q1) + np.dot(phi, self.q2)) * (1 - self.epsilon) \
                  + (np.ones(num_actions) / num_actions) * self.epsilon
        return policy / policy.sum()
