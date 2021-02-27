from ..utils import SudokuGrid, custom_encoder
import numpy as np


class SudokuGridAlpha(SudokuGrid):
    """ Overides some functions of parent SudokuGrid.

    Basically, just add changes relatively to probabilities
    used instead of randomly picking elements. """

    def __init__(self, grid, model, proba_taken=0):
        super().__init__(grid)
        self.model = model
        self.proba_taken = proba_taken
        self.proba_dict = None

    def find_random_child(self):
        pos = self.grid.possibilities
        min_nb_pos_ind = min([len(v) for v in pos.values()])

        if min_nb_pos_ind > 1:
            # only calculate proba if non trivial choice at hand
            if self.proba_dict is None:
                self.proba_dict = self._predict_probas()
            selected_action = max(self.proba_dict, key=self.proba_dict.get)
            child = self.take_action(selected_action[0],
                                     selected_action[1],
                                     max(self.proba_dict.values()))
            return child

        if len(pos) != 0:
            pos_considered = []
            for k, v in pos.items():
                # just look at indeces with min pos
                if len(v) == min_nb_pos_ind:
                    pos_considered.append(k)
            _index = np.random.choice(range(len(pos_considered)))
            index = pos_considered[_index]
            action = self.grid.possibilities[index][0]
            return self.take_action(index, action, 1)
        return None

    def _predict_probas(self):
        array_of_proba = self.model.predict(
            custom_encoder(self.grid.grid)
            )
        proba_dict = {}
        for i in range(9):
            for j in range(9):
                if self.grid.grid[i, j] == 0:
                    pos_at_index = self.grid.possibilities[(i, j)]
                    if len(pos_at_index) == 1:
                        proba_dict = {}
                        proba_dict[((i, j), pos_at_index[0])] = 1
                        return proba_dict
                    for v in range(9):
                        proba_dict[((i, j), v + 1)] = \
                            array_of_proba[0, v, i, j]

        return proba_dict

    def find_children(self):
        if self.is_terminal():
            return set()
        else:
            if self.proba_dict is None:
                self.proba_dict = self._predict_probas()
            possible_moves = []
            pos = self.grid.possibilities
            min_nb_pos_ind = min([len(v) for v in pos.values()])
            for index in pos:
                # just look at indeces with min pos
                if len(pos[index]) == min_nb_pos_ind:
                    for value in pos[index]:
                        possible_moves.append((index, value))
                    # we only consider children on one cell --> sufficient
                    return {self.take_action(a[0], a[1], self.proba_dict[a])
                            for a in possible_moves}

    def take_action(self, index, action, proba):
        new_grid = SudokuGridAlpha(self.grid.grid.copy(),
                                   self.model,
                                   proba)
        new_grid.grid.fill_cell(*index, action)
        return new_grid
