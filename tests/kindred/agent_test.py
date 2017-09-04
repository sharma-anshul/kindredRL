import os
import unittest

import numpy as np

from src.kindred.agent import Agent
from src.kindred.gridworld import Actions


class TestAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # intialize default grid as list of lists.
        cls.default_grid = [
            [0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
        ]
        cls.default_start = (5, 3)
        cls.default_goal = (0, 8)
        
        # initialize random 16 X 16 Q matrix for testing (meant to represent transition values for a
        # 4 X 4 GridWorld object). 
        cls.test_Q = [
            [0.4, 0.6, 0.4, 0.7, 0.3, 0.3, 0.6, 0.9, 0.0, 0.5, 0.5, 0.9, 0.1, 0.7, 0.9, 0.6],
            [0.3, 0.0, 0.7, 0.3, 0.5, 0.1, 0.9, 0.0, 0.4, 1.0, 0.2, 0.9, 0.4, 0.1, 0.6, 0.6],
            [0.2, 0.1, 0.4, 0.1, 0.8, 0.6, 0.2, 0.2, 0.2, 0.3, 0.2, 0.9, 0.4, 0.6, 0.1, 0.9],
            [0.4, 0.9, 0.6, 0.8, 0.2, 0.6, 1.0, 0.8, 0.6, 0.7, 0.6, 0.5, 0.3, 0.0, 0.7, 0.9],
            [0.3, 0.8, 0.8, 0.5, 0.8, 0.6, 0.0, 0.8, 0.2, 0.4, 0.9, 1.0, 0.5, 0.8, 0.4, 0.6],
            [0.3, 0.3, 0.6, 0.3, 0.3, 0.1, 0.8, 0.3, 0.7, 0.1, 0.1, 0.6, 0.5, 0.7, 0.2, 0.1],
            [0.6, 0.7, 0.5, 0.0, 0.7, 0.4, 0.6, 0.6, 0.3, 0.8, 0.7, 0.4, 0.9, 0.1, 0.3, 0.8],
            [0.6, 0.8, 0.8, 0.5, 0.7, 0.4, 0.9, 0.3, 0.4, 0.9, 0.8, 0.3, 0.1, 0.2, 0.9, 0.8],
            [0.5, 0.9, 0.9, 0.5, 0.5, 0.0, 0.6, 0.2, 0.5, 0.8, 0.7, 0.8, 0.7, 0.7, 0.6, 0.7],
            [0.6, 0.6, 0.3, 0.2, 0.0, 0.7, 0.6, 0.3, 0.3, 1.0, 0.1, 0.2, 0.9, 0.9, 0.6, 0.3],
            [0.9, 0.8, 0.9, 0.8, 0.1, 0.4, 0.3, 0.1, 0.7, 0.7, 0.2, 0.8, 1.0, 0.2, 0.9, 0.7],
            [0.9, 0.9, 0.3, 1.0, 0.2, 0.9, 0.5, 1.0, 0.2, 0.3, 0.0, 0.7, 0.0, 1.0, 0.6, 1.0],
            [0.9, 0.3, 0.3, 0.9, 0.6, 0.3, 0.9, 0.7, 0.5, 0.0, 0.2, 0.1, 0.9, 0.0, 0.1, 0.5],
            [0.6, 0.9, 0.4, 0.2, 0.1, 0.1, 0.5, 0.5, 0.4, 0.9, 0.9, 0.4, 0.4, 0.9, 0.2, 0.1],
            [0.4, 0.5, 0.8, 0.9, 0.2, 0.7, 0.3, 0.4, 0.1, 0.3, 0.1, 0.0, 0.5, 0.6, 0.8, 0.9],
            [0.4, 0.7, 0.8, 0.4, 0.1, 0.9, 1.0, 0.2, 0.3, 0.9, 0.0, 0.5, 0.5, 0.3, 0.2, 0.8],
        ]

    def setUp(self):
        # intialize new Agent object before each test method invocation.
        self.agent = Agent(epsilon=0.5, alpha=0.3, gamma=0.95)

    def test_default_intialize(self):
        """ Test default __init__ method. """
        self.assertEqual(self.agent.grid.grid.tolist(), self.default_grid)
        self.assertEqual(self.agent.state, self.default_start)
        self.assertEqual(self.agent.epsilon, 0.5)
        self.assertEqual(self.agent.alpha, 0.3)
        self.assertEqual(self.agent.gamma, 0.95)
        self.assertEqual(
            self.agent.Q.tolist(),
            np.zeros((self.agent.grid.size, self.agent.grid.size)).tolist()
        )
        self.assertEqual(self.agent.steps, 0)
    
    def test_state_setter(self):
        """ Test whether setting state works as intended. """
        self.assertEqual(self.agent.state, self.default_start)
        
        self.agent.state = (2, 8)
        self.assertEqual(self.agent.state, (2, 8))
    
    def test_argmax(self):
        """ 
        Given a state and Q matrix, test whether argmax returns an action maximizing Q(s, a).
        """
    
        self.agent.grid.dimensions = (4, 4)

        Q = np.array(self.test_Q)

        action, reward = self.agent.argmax(state=(1, 1), Q=Q)
        self.assertEqual(action, Actions.DOWN)
        self.assertEqual(reward, 0.8)
    
    def test_get_linear_index(self):
        """ Test translation between state and Q matrix indices. """ 
        self.assertEqual(self.agent.get_linear_index(state=(4, 1)), 10)

        self.agent.grid.dimensions = (5, 3)
        self.assertEqual(self.agent.get_linear_index(state=(4, 1)), 9)
