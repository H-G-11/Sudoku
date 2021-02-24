from .sudoku_grid import SudokuGrid
import numpy as np
from math import log, sqrt

# This code is an adaptation of the code here:
# https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
# in the case of sudoku


class MCTS:

    def __init__(self, sudoku_grid, exploration_weight=1,
                 number_path=10):
        if isinstance(sudoku_grid, np.ndarray):
            sudoku_grid = SudokuGrid(sudoku_grid)
        self.sudoku_grid = sudoku_grid
        self.exploration_weight = exploration_weight
        self.Q = {}
        self.N = {}
        self.children = {}
        self.number_path = number_path

    def solve(self):
        while not self.sudoku_grid.is_terminal():
            for i in range(self.number_path):
                self.do_rollout()
            self.sudoku_grid = self.choose_best_action()
        return self.sudoku_grid

    def choose_best_action(self):
        if self.sudoku_grid.is_terminal():
            return self.sudoku_grid

        if self.sudoku_grid not in self.children:
            return self.sudoku_grid.find_random_child()

        def score(n):
            if self.N.get(n, 0) == 0:
                return -1
            return self.Q.get(n, 0) / self.N[n]

        if len(self.children[self.sudoku_grid]) == 0:
            if self.sudoku_grid.find_random_child() is not None:
                return self.sudoku_grid.find_random_child()
            else:
                return RuntimeError("Solver failed")

        return max(self.children[self.sudoku_grid],
                   key=score)

    def do_rollout(self):
        path = self._select(self.sudoku_grid)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, sudoku_grid):
        path = []
        while True:
            path.append(sudoku_grid)
            if sudoku_grid not in self.children \
                    or not self.children[sudoku_grid]:
                return path
            unexplored = self.children[sudoku_grid] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            sudoku_grid = self._uct_select(sudoku_grid)

    def _simulate(self, sudoku_grid):
        while not sudoku_grid.is_terminal():
            sudoku_grid = sudoku_grid.find_random_child()
        if sudoku_grid.grid.is_complete() and sudoku_grid.grid.is_correct():
            self.sudoku_grid = sudoku_grid
        return sudoku_grid.reward()

    def _expand(self, sudoku_grid):
        if sudoku_grid in self.children:
            return None
        self.children[sudoku_grid] = sudoku_grid.find_children()

    def _backpropagate(self, path, reward):
        for sudoku_grid in reversed(path):
            self.N[sudoku_grid] = self.N.get(sudoku_grid, 0) + 1
            self.Q[sudoku_grid] = self.Q.get(sudoku_grid, 0) + reward

    def _uct_select(self, sudoku_grid):
        # All children of node should already be expanded:
        assert all(child in self.children
                   for child in self.children[sudoku_grid])

        log_N_parent = log(self.N[sudoku_grid])

        def uct(child):
            "Upper confidence bound for trees"
            return self.Q[child] / self.N[child] + self.exploration_weight * \
                sqrt(log_N_parent / self.N[child])

        return max(self.children[sudoku_grid], key=uct)
