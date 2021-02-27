from .grid import SmartGrid
import random
import numpy as np


class SudokuGrid:

    def __init__(self, grid):
        if isinstance(grid, np.ndarray):
            grid = SmartGrid.from_grid(grid.copy())
        self.grid = grid

    def find_children(self):
        if self.is_terminal():
            return set()
        else:
            possible_moves = []
            pos = self.grid.possibilities
            min_nb_pos_ind = min([len(v) for v in pos.values()])
            for index in pos:
                # just look at indeces with min pos
                if len(pos[index]) == min_nb_pos_ind:
                    for value in pos[index]:
                        possible_moves.append((index, value))
                    # we just return children on 1 cell --> sufficient
                    return {self.take_action(a[0], a[1])
                            for a in possible_moves}

    def find_random_child(self):
        pos = self.grid.possibilities
        min_nb_pos_ind = min([len(v) for v in pos.values()])
        if len(pos) == 0 or min_nb_pos_ind == 0:
            return None
        pos_considered = []
        for k, v in pos.items():
            # just look at indeces with min pos
            if len(v) == min_nb_pos_ind:
                pos_considered.append(k)
        index = random.choice(pos_considered)
        action = random.choice(self.grid.possibilities[index])
        child = self.take_action(index, action)
        return child

    def take_action(self, index, action):
        new_grid = SudokuGrid(self.grid.grid.copy())
        new_grid.grid.fill_cell(*index, action)
        return new_grid

    def is_terminal(self):
        if len(self.grid.possibilities) == 0:
            return True
        elif self.grid.is_complete() or not self.grid.is_correct() or \
                min([len(v) for v in self.grid.possibilities.values()]) == 0:
            return True
        return False

    def reward(self):
        return np.count_nonzero(self.grid.grid) / 81

    def __hash__(self):
        return hash(str(self.grid.grid))

    def __eq__(self, grid2):
        if np.array_equal(self.grid.grid, grid2.grid.grid):
            return True
        return False

    def __str__(self):
        return str(self.grid.grid)
