from ..mcts import SudokuGrid
from ..Utils import custom_encoder
import numpy as np


class SudokuGridAlpha(SudokuGrid):
    """ Overides some functions of parent SudokuGrid.

    Basically, just add changes relatively to probabilities
    used instead of randomly picking elements. """

    def __init__(self, grid, model):
        super().__init__(grid)
        self.model = model

    def find_random_child(self):
        pos = self.grid.possibilities
        min_nb_pos_ind = min([len(v) for v in pos.values()])
        pos_considered = []

        if min_nb_pos_ind > 1:
            # only calculate proba if non trivial choice at hand
            proba_dict = self._predict_proba()
            proba_pos_considered = []  # max proba of value in cells

        if len(pos) != 0:
            for k, v in pos.items():
                # just look at indeces with min pos
                if len(v) == min_nb_pos_ind:
                    pos_considered.append(k)
                    if min_nb_pos_ind > 1:
                        proba_pos_considered.append(
                            max(proba_dict[k].values()))
            if min_nb_pos_ind > 1:
                proba_pos_considered = np.array(proba_pos_considered)
                proba_pos_considered /= np.sum(proba_pos_considered)
                _index = np.random.choice(range(len(pos_considered)),
                                          p=proba_pos_considered)
                index = pos_considered[_index]
                possible_values = self.grid.possibilities[index]
                associated_proba = [proba_dict[index][i]
                                    for i in range(len(possible_values))]
                associated_proba = np.array(associated_proba)
                associated_proba /= np.sum(associated_proba)
                action = np.random.choice(possible_values,
                                          p=associated_proba)
            else:
                _index = np.random.choice(range(len(pos_considered)))
                index = pos_considered[_index]
                action = self.grid.possibilities[index][0]
            child = self.take_action(index, action)
            return child
        return None

    def take_action(self, index, action):
        new_grid = SudokuGridAlpha(self.grid.grid.copy(),
                                   self.model)
        new_grid.grid.fill_cell(*index, action)
        return new_grid

    def _predict_proba(self):
        """ Return proba of actions if actions are not in place
        of an already taken cell. """

        array_of_proba = self.model.predict(
            custom_encoder(self.grid.grid.copy())
            )

        proba_dict = {}
        for i in range(9):
            for j in range(9):
                if self.grid.grid[i, j] == 0:
                    proba_dict[(i, j)] = {}
                    for v in range(9):
                        proba_dict[(i, j)][v] = \
                            array_of_proba[0, v, i, j]
        return proba_dict
