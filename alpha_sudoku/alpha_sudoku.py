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

    def __init__(self, sudoku_grid,
                 pathnet='C:/Users/Hugues/Desktop/RLProject/policy_network',
                 model=None, exploration_weight=1, number_path=10):

        """ Sudoku Grid : either SudokuGridAlpha with model initialised,
        or numpy array. If it is a numpy array, pathnet or directly model must
        be provided. """

        if isinstance(sudoku_grid, np.ndarray):
            if model is None:
                model = load_model(pathnet)
            sudoku_grid = SudokuGridAlpha(sudoku_grid, model)
        super().__init__(sudoku_grid, exploration_weight, number_path)
        self.probas = {}
