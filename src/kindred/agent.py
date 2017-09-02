import numpy as np
from numpy.random import choice

from gridworld import GridWorld


class Agent(object):
    # initialize hyperparameters to 0.
    epsilon = 0.0
    alpha = 0.0
    gamma = 0.0

    # initialize constants to represent the choice of the epsilon greedy policy.
    RANDOM = 0
    ARGMAX = 1
   
    # step threshold at which to change grids.
    STEPS = 5001

    # initialize Q matrix to None.
    Q = None
    
    def __init__(self, epsilon, alpha, gamma):
        """
        Args:
            epsilon (float): Probability for Epsilon policy.
            alpha (float): Step size. Range in [0, 1].
            gamma (float): Discount factor. Range in [0, 1].
        
        Returns:
            No explicit return value.
        """
        # initialize a GridWorld object to be used by the agent.
        self.grid = GridWorld()

        # set hyperparameters.
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        # initialize steps to 0. 
        self.steps = 0

        # intialize Q matrix.
        self.Q = np.zeros((self.grid.size, self.grid.size))

    @property
    def state(self):
        """
        Get current state on grid.

        Returns:
            tuple: (x, y) coordinates representing agent position on grid.
        """
        return self.grid.state

    @state.setter
    def state(self, state):
        """
        Update state of the agent.

        Args:
            state (tuple): (x, y) coordinates representing new state of agent.
        """
        self.grid.state = state
        self.steps += 1

        if self.steps == self.STEPS:
            self.grid.update_grid()

    def simulate_action(self, Q=None):
        """
        Uses Epsilon policy to choose the next action to be taken by agent.

        Chooses a random action with probability self.epsilon and with 
        probability 1 - self.epsilon chooses an action that will maximize
        reward.

        Returns:
            tuple[tuple, float]: Return (x, y) tuple representing new state, and
                                 a float value representing the reward associated.
        """
        # choose action to take based on epsilon.
        method = choice(
            [self.RANDOM, self.ARGMAX],
            None,
            [self.epsilon, 1 - self.epsilon]
        )

        if Q is None:
            Q = self.Q

        # if random, choose a random valid action.
        if method == self.RANDOM:
            action = choice(self.grid.get_valid_actions())
        # else choose action maximizing reward.
        elif method == self.ARGMAX:
            action, _ = self.argmax(Q=Q)
        # simulate new state based on action obtained above.
        new_state = tuple(map(sum, zip(self.state, self.grid.actions[action])))
        
        return new_state, self.get_reward(new_state)

    def argmax(self, state=None, Q=None):
        """
        Given a state, choose the action that maximizes reward.

        Args:
            state (tuple): (x, y) tuple representing position of agent on grid.
                           If not specified, will use current state.

        Returns:
            tuple[Actions, float]: Return an Enum value representing the action
                                   to be taken and a float value representing the
                                   reward associated.
        """ 
        # if state not provided explicitly, use current state.
        state = state or self.state     
       
        if Q is None:
            Q = self.Q

        # calculate action to maximize Q(state, action).
        max_Q = float('-inf') 
        for action in self.grid.get_valid_actions(state):
            new_state = tuple(map(sum, zip(state, self.grid.actions[action])))
            q_value = self.get_Q(state, new_state, Q)
            if q_value > max_Q:
                max_action = action
                max_Q = q_value

        return (max_action, max_Q)

    def get_reward(self, new_state):
        """
        Get reward value associated with 
        """
        if new_state == self.grid.goal:
            return 1.0
        
        return 0.0

    def get_Q(self, state, new_state, Q=None):
        """
        Access Q value associated with moving from one state to another.

        Args:
            state (tuple): (x, y) coordinates representing current position of agent.
            new_state (tuple): (x, y) coordinates representing future position of agent.

        Returns:
            float: Value representing reward in taking an action that results in this
                   change in states.
        """
        if Q is None:
            Q = self.Q

        return Q[(
            self.get_linear_index(state),
            self.get_linear_index(new_state),
        )]

    def update_Q(self, state, new_state, value):
        """
        Update Q value associated with moving from one state to another.

        Args:
            state (tuple): (x, y) coordinates representing current position of agent.
            new_state (tuple): (x, y) coordinates representing future position of agent.
            value (float): Value to be updated.
        """
        self.Q[(
            self.get_linear_index(state),
            self.get_linear_index(new_state),
        )] = value

    def reset_Q(self):
        """ Reset Q matrix to zeros. """
        self.Q = np.zeros((self.grid.size, self.grid.size))

    def get_linear_index(self, state):
        """
        Translate 2D coordinates into scalar integer index.

        Args:
            state (tuple): Tuple representing (x, y) coordinates.

        Returns:
            int: Value representing integral index into flattened array.
        """
        # assuming (x, y) coordinate translates to x + (rows * y)
        # in linear coordinates.
        return state[0] + (self.grid.dimensions[0] * state[1])
