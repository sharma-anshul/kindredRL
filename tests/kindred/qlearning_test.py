import os
import unittest

import numpy as np

from src.kindred.agent import Agent
from src.kindred.gridworld import Actions
from src.kindred.qlearning import learn


class TestQLearning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.test_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'fixtures/gridTest.txt',
        )
        
    def test_learn(self):
        """ Test synchronous q learning """
        # test learning with default gridworld example.
        num_steps, Q = learn(num_episodes=100, epsilon=0.5, alpha=0.3, gamma=0.95)
        agent = Agent(epsilon=0.5, alpha=0.3, gamma=0.95)

        # expected steps if agent takes less than 5000 steps.
        expected_steps = [Actions.RIGHT for _ in xrange(5)]
        expected_steps.extend([Actions.UP for _ in xrange(5)])
        
        # expected steps if agent takes more than 5000 steps.
        if num_steps > 5000:
            agent.grid.update_grid()
            expected_steps = [
                Actions.UP, Actions.LEFT, Actions.LEFT, Actions.LEFT, Actions.UP,
                Actions.UP, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT,
                Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.RIGHT, Actions.UP,
                Actions.UP,
            ]

        # get optimal policy.
        steps = get_steps(agent=agent, Q=Q)
        
        self.assertItemsEqual(steps, expected_steps)

        # test learning with test grid.
        _, Q = learn(num_episodes=100, epsilon=0.5, alpha=0.3, gamma=0.95, grids=[self.test_path])
        agent = Agent(epsilon=0.5, alpha=0.3, gamma=0.95, grids=[self.test_path])

        steps = get_steps(agent=agent, Q=Q)

        #expected steps for optimal policy.
        expected_steps = [Actions.DOWN for _ in xrange(3)]
        expected_steps.extend([Actions.RIGHT for _ in xrange(8)])
        
        self.assertItemsEqual(steps, expected_steps)


def get_steps(agent, Q):
    """ 
    Get optimal policy given an agent and Q matrix.
    
    Args:
        agent (Agent): Object representing an Agent.
        Q (numpy.Array): 2D array representing learned Q matrix.

    Returns:
        list[Actions]: List of Enums representing sequence of actions taken using optimal policy
                       based on the given Q matrix.
    """
    steps = []

    state = agent.grid.start
    while state != agent.grid.goal:
        # break if agent is stuck in a loop.
        if state not in steps:
            # retrieve optimal action to take for current state.
            action, _ = agent.argmax(state, Q=Q)
            new_state = tuple(map(sum, zip(state, agent.grid.actions[action])))
            steps.append(action)
            state = new_state
        else: break
    
    return steps
