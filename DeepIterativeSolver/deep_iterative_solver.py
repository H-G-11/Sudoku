import numpy as np
from tensorflow import keras
from ..Utils import Grid
from ..MCTS import custom_encoder


class DeepSolver:
    """ At each step, take action with highest probability. """

    def __init__(self, grid: Grid,
                 pathnet='C:/Users/Hugues/Desktop/RLProject/policy_network'):
        self.grid = grid
        self.model = keras.models.load_model(pathnet)

    def run(self):
        while not self.grid.is_complete() and self.grid.is_correct():
            proba_dict = self._predict_probas()
            selected_action = max(proba_dict, key=proba_dict.get)
            self._take_action(selected_action)
        return self.grid.grid

    def _predict_probas(self):
        array_of_proba = self.model.predict(
            np.array([custom_encoder(self.grid.grid)])
            )
        proba_dict = {}
        for i in range(9):
            for j in range(9):
                if self.grid.grid[i, j] == 0:
                    for v in range(9):
                        proba_dict[((i, j), v + 1)] = \
                            array_of_proba[0, v, i, j]

        return proba_dict

    def _take_action(self, action):
        index = action[0]
        value = action[1]
        self.grid.fill_cell(*index, value)
