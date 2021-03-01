from ..mcts import MCTS
from .sudoku_grid_alpha import SudokuGridAlpha
import numpy as np
from tensorflow.keras.models import load_model


class AlphaSudoku(MCTS):
    """ Monte Carlo Tree Search for Sudoku game.

    Terminal (leaf) nodes are actions that lead to an incorrect
    sudoku grid (with 2 same value on one line for example).
    The value of a leaf node will be the number of cells filled.

    The policy to choose random actions is a convolutional network
    trained on 1 million sudoku games.

    Note : unlike AlphaGo, I keep uct without probabilities to
    balance between exploration and exploitation. """

    def __init__(self, sudoku_grid, max_iterations=10000,
                 pathnet='Sudoku/policy_network',
                 model=None, exploration_weight=1, max_depth_tree=10):

        """ Sudoku Grid : either SudokuGridAlpha with model initialised,
        or numpy array. If it is a numpy array, pathnet or directly model must
        be provided. """

        if isinstance(sudoku_grid, np.ndarray):
            if model is None:
                model = load_model(pathnet)
            sudoku_grid = SudokuGridAlpha(sudoku_grid, model)
        super().__init__(sudoku_grid, exploration_weight, max_depth_tree,
                         max_iterations)
        self.probas = {}

    def _action_selection(self, sudoku_grid):
        # All children of node should already be expanded:
        assert all(child in self.children
                   for child in self.children[sudoku_grid])

        def to_maximise(child):
            """ Function to minimize described in original alphaGo paper. """
            return self.Q[child] / self.N[child] + child.proba_taken / \
                (1 + self.N[child])

        return max(self.children[sudoku_grid], key=to_maximise)
