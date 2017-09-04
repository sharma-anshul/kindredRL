from multiprocessing import Array, Lock, Manager, Pool, Process, Value 

import numpy as np

from agent import Agent
from gridworld import GridWorld


class SharedState(object):
    """ Class representing global Q matrix and T values. """
    def __init__(self, size):
        """
        Initialize Q matrix and T.

        Args:
            size (int): Size of grid (rows * cols).
        """
        manager = Manager()

        # internally represent Q matrix as a list of lists (in ProxyArray form).
        self.global_Q = manager.list(np.zeros((size, size)).tolist())
        self.T = Value('i', 0)
        
        # intialize multiprocessing lock.
        self.lock = Lock()

    def get_Q(self):
        """
        Get global Q matrix.
        
        Returns:
            numpy.Array: Global Q matrix.
        """
        with self.lock:
            # convert list of lists to numpy.Array.
            return np.array(self.global_Q)
    
    def update_Q(self, new_Q):
        """
        Update global Q matrix.
        
        Args:
            new_Q (numpy.Array): Updated global Q matrix.
        """
        new_Q = new_Q.tolist()

        with self.lock:
            for i in xrange(len(self.global_Q)):    
                self.global_Q[i] = new_Q[i]

    def get_T(self):
        """
        Get global T value.
        
        Returns:
            int: Global T value.
        """
        with self.lock:
            return self.T.value

    def increment_T(self): 
        """ Increment global T value. """
        with self.lock:
            self.T.value += 1
            

def learn(num_episodes, epsilon, alpha, gamma, grids=None):
    """
    Run greedy epsilon based Q Learning.

    Args:
        num_episodes (int): Number of episodes to run algorithm for.
        epsilon (float): Parameter to control the epsilon greedy policy.
        alpha (float): Learning parameter.
        gamma (float): Discount factor.
        grids (list[str|File]): List of files containing representation of grids.

    Returns:
        (int, numpy.Array): Integer specifying number of steps and 2D array representing
							the learned Q matrix. 
    """
    # intialize state and setup grid.
    agent = Agent(epsilon, alpha, gamma, grids=grids)

    # repeat for each episode:
    for i in xrange(num_episodes):
        # reset agent state to start position.
        agent.state = agent.grid.start

        if agent.steps > 5000:
            y.append(np.sum(agent.Q))
            x.append(i)


        # step through until the agent reaches goal.
        while agent.state != agent.grid.goal:
            current_state = agent.state

            # simulates the agent's next step using greedy epsilon policy.
            new_state, reward = agent.simulate_action()
        	
            # get Q value from the agent's Q matrix.
            current_value = agent.get_Q(current_state, new_state)

            # get future value based on simulated next state of the agent.
            _, future_value = agent.argmax(new_state)

            # calculate new Q value based on the update rule.
            expected_reward = current_value + alpha * (reward + (gamma * future_value) - current_value)

            # update agent's Q matrix with the calculated value.
            agent.update_Q(current_state, new_state, expected_reward)

            # update agent's state to new state.
            agent.state = new_state

    return (agent.steps, agent.Q)


def learn_async(num_agents, I_async_update, T_max, size, epsilon, alpha, gamma):
    """
    Wrapper function for running multiprocessing based Q Learning.

    Args:
        num_agents (int): Number of agents to spawn (controls number of processes).
        I_async_update (int): Number of steps after which to update global state.
        T_max (int): Maximum number of steps to be taken globally.
        size (int): Size of grid (rows * cols).
        epsilon (float): Parameter to control the epsilon greedy policy.
        alpha (float): Learning parameter.
        gamma (float): Discount factor.

    Returns:
        numpy.Array: 2D array representing the learned Q matrix. 
    """
    # intialize shared state object representing global Q matrix, and global step count T.
    shared_state = SharedState(size)

    # intialize processes equal to num_agents.
    procs = [
        Process(
            target=async_helper,
            args=(shared_state, I_async_update, T_max, epsilon, alpha, gamma,),
        )
        for _ in xrange(num_agents)    
    ]
    
    for proc in procs: proc.start()
    for proc in procs: proc.join()

    return shared_state.get_Q()


def async_helper(shared_state, I_async_update, T_max, epsilon, alpha, gamma):
    """
    Helper function for running multiprocessing based Q Learning.

    Args:
        shared_state (SharedState): Shared state object representing global Q and T values.
        I_async_update (int): Number of steps after which to update global state.
        T_max (int): Maximum number of steps to be taken globally.
        size (int): Size of grid (rows * cols).
        epsilon (float): Parameter to control the epsilon greedy policy.
        alpha (float): Learning parameter.
        gamma (float): Discount factor.
    """
    # intialize state and setup grid.
    agent = Agent(epsilon, alpha, gamma)
     
    # get global Q matrix.
    global_Q = shared_state.get_Q()    
    
    # step through until the global T value reaches T_max.
    while shared_state.get_T() < T_max:

        current_state = agent.state

        # simulates the agent's next step using greedy epsilon policy.
        new_state, reward = agent.simulate_action(global_Q)

        # get future value based on simulated next state of the agent.
        _, future_value = agent.argmax(new_state, global_Q)

        # get Q value from the agent's local Q matrix.
        current_delta_value = agent.get_Q(current_state, new_state)

        # get Q value from global Q matrix.
        current_value = agent.get_Q(current_state, new_state, global_Q)

        # calculate new Q value based on the update rule.
        expected_reward = current_delta_value + (reward + (gamma * future_value) - current_value)
        
        # update agent's local Q matrix with the calculated value.
        agent.update_Q(current_state, new_state, expected_reward)

        # update agent's state to new state.
        agent.state = new_state

        # increment global T value.
        shared_state.increment_T()

        # update global Q value.
        if (agent.steps % I_async_update == 0) or (agent.state == agent.grid.goal):
            # update global Q matrix with discounted local copy of agent's Q matrix.
            global_Q = np.add(shared_state.get_Q(), alpha * agent.Q)
            shared_state.update_Q(global_Q)

            # reset local Q matrix to zeros.
            agent.reset_Q()
