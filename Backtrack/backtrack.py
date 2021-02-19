import numpy as np
import random
from ..Utils.utils import SIZE, InsolvableError
from ..Utils.grid import Grid


class BacktrackSolver:

    def __init__(self, grid=np.zeros((SIZE ** 2, SIZE ** 2))):
        """
        forbidden_move : dict of dicts : {depth : {index : [forbidden values]}}
        """
        self.grid = Grid.from_grid(grid)
        self.history = []  # inside : {index: value}
        self.forbidden_move = {}  # move tested that lead to nowhere
        # we keep track of depth at which action was taken
        self.depth = 0
        self.iterations = 0

    def solve(self, return_n=0, return_unsolvable=False):
        """ Solve sudoku or return partial results.

        Inputs:
        return_n : int -- return (solvable) grid with n 0s.
        return_unsolvable : bool -- return first unsolvable
            sudoku encountered.
            You can't enter both return_n and return_unsolvable

        Output:
        Grid : if return_n specified, partial solution
               if return_unsolvable, unsovable sudoku
               else by default solution.
        """
        assert return_n == 0 or not return_unsolvable, "Do not specify both"
        while len(self.grid.possibilities) != 0:
            self.iterations += 1
            pos_index = self.grid.index_with_min_pos()

            # if cell with no possibilities, go backward
            if len(self.grid.possibilities[pos_index[0]]) == 0:
                if return_unsolvable:
                    print(self.grid.grid)
                    return self.grid.grid
                self._go_back()
                continue

            chosen_index = random.choice(pos_index)  # first random choice
            forbidden_actions = self.forbidden_move.get(self.depth, {})\
                .get(chosen_index, [])
            pos_actions = [v for v in self.grid.possibilities[chosen_index]
                           if v not in forbidden_actions]

            # if possible actions are forbidden
            if len(pos_actions) == 0:
                if return_unsolvable:
                    print(self.grid.grid)
                    return self.grid.grid
                self._go_back()
                continue

            chosen_value = random.choice(pos_actions)
            self.history.append({chosen_index: chosen_value})
            self.grid.fill_cell(*chosen_index, chosen_value)
            self.depth += 1

        if return_n:
            for i in range(min(len(self.history), return_n)):
                self._go_back()

        return self.grid.grid

    def _go_back(self):
        # first, reset forbidden move in future
        if self.depth in self.forbidden_move.keys():
            del self.forbidden_move[self.depth]

        # go back
        self.depth -= 1
        if self.depth < 0:
            print(self.depth)
            raise InsolvableError("Your grid is not solvable !")

        # add action to forbidden moves
        last_action = self.history.pop()
        _forbidden = self.forbidden_move.get(self.depth, {})
        index = list(last_action.keys())[0]
        if index in _forbidden.keys():
            _forbidden[index].append(last_action[index])
        else:
            _forbidden[index] = [last_action[index]]
        self.forbidden_move[self.depth] = _forbidden

        # erase cell
        index_action = list(last_action.keys())[0]
        self.grid.erase_cell(*index_action)
