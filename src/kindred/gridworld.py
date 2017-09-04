from enum import Enum

import numpy as np
from numpy.random import choice


class Actions(Enum):
    """ Enumeration representing actions that can be taken in a GridWorld object. """
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class GridWorld(object):
    """
    Represents a grid object.
    
    Represented by a 2D grid, where each position that the agent can be in is one of 
    the following:

        0: This position can be occupied by the agent.
        1: Start position of the agent.
        2: Goal position of the agent.
        3: This position is blocked and can't be accessed.

    For the given grid world example, the initial grid is therefore represented as
    follows (resources/gridL.txt):

		0 0 0 0 0 0 0 0 2
		0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0
		3 3 3 3 3 3 3 3 0
		0 0 0 0 0 0 0 0 0
		0 0 0 1 0 0 0 0 0 
    """ 

    # initialize empty grid.
    grid = None
    size = 0
    state = ()
    start = ()
    goal = ()
    dimensions = ()

    # default grid(s).
    grids = ['resources/gridL.txt', 'resources/gridR.txt']

    # initialize step size to 0.
    alpha = 0.0

    # initialize discount factor to 0.
    gamma = 0.0

    # initialize action probability to 0.
    epsilon = 0.0

    # define block types in the grid.
    START = 1
    GOAL = 2
    BLOCKED = 3

    def __init__(self, grids=None):
        """
        Args:
            grids (str|File): Path to file containing grid representation.

        Returns:
            No explicit return value.
        """
        if grids:
            self.grids = grids

        # intialize grid on intial creation of object.
        self.initialize_grid()

        # define actions supported inside grid. 
        self.actions = {
            Actions.LEFT: (0, -1),
            Actions.RIGHT: (0, 1),
            Actions.UP: (-1, 0),
            Actions.DOWN: (1, 0),
        }
        
    def initialize_grid(self, grid=None):
        """
        Initialize grid.

        Args:
            grid (str|File): Path to file containing grid representation.
        """
        # use grid if provided else self.grid[0].
        grid = grid or self.grids[0]
    
        # read grid representation from text file.
        self.grid = np.loadtxt(grid, dtype=int)
        self.size = self.grid.size
        self.dimensions = self.grid.shape

        # extract coordinates for start and goal positions.
        self.start = tuple(element[0] for element in np.where(self.grid==self.START))
        self.goal = tuple(element[0] for element in np.where(self.grid==self.GOAL))

        # intialize current state to start position.
        self.state = self.start

    def update_grid(self, grid=None):
        """
        Update grid.

        Args:
            grid (str|File): Path to file containing grid representation.
        """
        # use grid if provided else self.grid[1].
        if not grid and len(self.grids) == 1:
            return
        grid = grid or self.grids[1]

        self.initialize_grid(grid)

    def get_valid_actions(self, state=None):
        """
        Given a state, get all valid actions available.

        Args:
            state (tuple): (x, y) coordinates representing agent position on grid.

        Returns:
            list[Actions]: A list of Enums representing valid actions from current
                           state.
        """
        # if state not provided explicitly, use current state.
        state = state or self.state
        
        valid_actions = []
        for action in self.actions:
            new_state = tuple(map(sum, zip(state, self.actions[action])))
            if self.is_valid(new_state):
                valid_actions.append(action)

        return valid_actions

    def is_valid(self, state):
        """
        Determine whether given state is valid for the current GridWorld object.

        Args:
            state (tuple): Tuple representing (x, y) coordinates.

        Returns:
            bool: True if the state is valid, False otherwise.
        """
        # check state is within grid bounds.
        if (
            state[0] >= 0 and
            state[1] >= 0 and 
            state[0] < self.dimensions[0] and
            state[1] < self.dimensions[1]
        ):
            # check state is not blocked.
            if self.grid[state] != self.BLOCKED:
                return True

        return False
