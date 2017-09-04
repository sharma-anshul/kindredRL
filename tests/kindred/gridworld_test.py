import os
import unittest

from src.kindred.gridworld import Actions
from src.kindred.gridworld import GridWorld


class TestGridWorld(unittest.TestCase):

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

        # initialize test grid as list of lists. 
        cls.test_grid = [
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [1, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2],
        ]
        cls.test_start = (2, 0)
        cls.test_goal = (5, 8)
        cls.test_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'fixtures/gridTest.txt',
        )

    def test_default_initialize(self):
        """ Test deafult __init__ method. """
        grid_world = GridWorld()  
 
        self.assertEqual(grid_world.grid.tolist(), self.default_grid)
        self.assertEqual(grid_world.start, self.default_start)
        self.assertEqual(grid_world.goal, self.default_goal)
    
    def test_custom_initialize(self):
        """ Test __init__ method with custom grid. """
        grid_world = GridWorld([self.test_path])
	
        self.assertEqual(grid_world.grid.tolist(), self.test_grid)
        self.assertEqual(grid_world.start, self.test_start)
        self.assertEqual(grid_world.goal, self.test_goal)

    def test_update_grid(self):
        """ Confirm update grid changes grid object. """
        grid_world = GridWorld()
        self.assertEqual(grid_world.grid.tolist(), self.default_grid)

        grid_world.update_grid(grid=self.test_path)
        self.assertEqual(grid_world.grid.tolist(), self.test_grid)
        self.assertEqual(grid_world.start, self.test_start)
        self.assertEqual(grid_world.goal, self.test_goal)

    def test_is_valid(self):
        """ Test whether is_valid returns valid values for a given grid. """
        grid_world = GridWorld()

        self.assertTrue(grid_world.is_valid(state=(0, 8)))
        self.assertFalse(grid_world.is_valid(state=(3, 0)))
        self.assertFalse(grid_world.is_valid(state=(6, 9)))
        self.assertFalse(grid_world.is_valid(state=(-1, 5)))

    def test_get_valid_actions(self):
        """ Check valid actions returned given a state. """
        grid_world = GridWorld()

        actions = grid_world.get_valid_actions()
        self.assertItemsEqual(actions, [Actions.LEFT, Actions.RIGHT, Actions.UP])

        actions = grid_world.get_valid_actions(state=(0, 8))
        self.assertItemsEqual(actions, [Actions.LEFT, Actions.DOWN])
        
        actions = grid_world.get_valid_actions(state=(1, 5))
        self.assertItemsEqual(actions, [Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN])
