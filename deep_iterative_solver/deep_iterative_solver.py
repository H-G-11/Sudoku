import numpy as np
from tensorflow.keras.models import load_model
from ..utils import SmartGrid, custom_encoder


class DeepIterativeSolver:
    """ At each step, take action with highest probability. """

    def __init__(self, grid, model=None,
                 pathnet='Sudoku/policy_network'):
        if isinstance(grid, np.ndarray):
            grid = SmartGrid.from_grid(grid.copy())
        self.grid = grid
        if model is None:
            model = load_model(pathnet)
        self.model = model
        self.iterations = 0

    def solve(self):
        while not self.grid.is_complete() and self.grid.is_correct():
            proba_dict = self._predict_probas()
            if proba_dict is None:
                print("Solver failed")
                return self.grid.grid
            selected_action = max(proba_dict, key=proba_dict.get)
            self._take_action(selected_action)
            self.iterations += 1
        return self.grid.grid

    def _predict_probas(self):
        array_of_proba = self.model.predict(
            custom_encoder(self.grid.grid)
            )
        proba_dict = {}
        for i in range(9):
            for j in range(9):
                if self.grid.grid[i, j] == 0:
                    if (i, j) not in self.grid.possibilities:
                        return None
                    pos_at_index = self.grid.possibilities[(i, j)]
                    if len(pos_at_index) == 1:
                        proba_dict = {}
                        proba_dict[((i, j), pos_at_index[0])] = 1
                        return proba_dict
                    for v in range(9):
                        proba_dict[((i, j), v + 1)] = \
                            array_of_proba[0, v, i, j]

        return proba_dict

    def _take_action(self, action):
        index = action[0]
        value = action[1]
        self.grid.fill_cell(*index, value)
