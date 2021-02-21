import numpy as np
from tensorflow import keras
from ..Utils import Grid
from .data_transform import custom_encoder
import time


class MCTS:
    """ Monte Carlo Tree Search for Sudoku game.

    For simplicity reason, I won't consider states but just
    actions, since in this game it is sufficient.
    Terminal (leaf) nodes are actions that lead to an incorrect
    sudoku grid (with 2 same value on one line for example).
    The value of a leaf node will be its 'distance' to the root. The
    heuristic is that bad actions will lead sooner to an incorrect grid.

    Another change is that actions are evaluated independently from
    the depth at which they are chosen. In fact, in Sudoku game,
    this does not matter. Nonetheless, depth is still used for
    UCT (for expanding the tree).

    The policy to choose action is a convolutional network trained
    on 1 million sudoku games, so that the approach is quite the
    same as for alpha Go, in a much simpler setting. """

    def __init__(self, grid: Grid, exploration_param=np.sqrt(2),
                 pathnet='C:/Users/Hugues/Desktop/RLProject/policy_network',
                 verbose=False, number_path=50, max_depth=50):
        """ Action will be stored in the following format:
        ((i, j), value). """
        self.N = {}  # number of times an action has been taken
        self.Q = {}  # mean value of action
        self.W = {}  # total value of action
        self.grid = grid
        self.number_path = number_path
        self.max_depth = max_depth
        self.exploration_param = exploration_param
        self.model = keras.models.load_model(pathnet)
        self.verbose = verbose

    def run(self):
        while self.grid.is_correct() or not self.grid.is_complete():
            print(self.grid.grid)
            best_action = self.choose_best_action()
            self.grid.fill_cell(*best_action[0], best_action[1])

        return self.grid.grid, self.grid.is_correct()

    def choose_best_action(self):
        """ Select best action from current root. """
        # we start again from nothing
        self.N, self.Q, self.W = {}, {}, {}
        self.search_tree()
        best_action = max(self.N, key=self.N.get)

        return best_action

    def search_tree(self):
        for path in range(self.number_path):
            start_time = time.time()
            new_grid = self.grid.copy()
            new_grid_history = []
            new_action = 0
            reward = 0

            while (new_grid.is_correct() or not new_grid.is_complete()) \
                    and reward < self.max_depth:
                probas_dict = self._predict_probas(new_grid)
                new_action = self._select(new_action, probas_dict)
                new_grid = self._take_action(new_grid, new_action)
                new_grid_history.append(new_action)
                reward += 1

            if new_grid.is_complete() and new_grid.is_correct():
                self.grid.grid = new_grid
                break

            # backpropagation
            for action in new_grid_history[:-1]:
                self.N[action] = self.N.get(action, 0) + 1
                self.W[action] = self.W.get(action, 0) + reward
                self.Q[action] = self.W[action] / self.N[action]
                reward -= 1  # so that last actions receive less

            if self.verbose:
                print(time.time() - start_time)

    def _predict_probas(self, new_grid):
        """ Return proba of actions if actions are not in place
        of a cell which is already taken. """
        array_of_proba = self.model.predict(
            np.array([custom_encoder(self.grid.grid)])
            )
        proba_dict = {}
        for i in range(9):
            for j in range(9):
                if new_grid.grid[i, j] == 0:
                    for v in range(9):
                        proba_dict[((i, j), v + 1)] = \
                            array_of_proba[0, v, i, j]

        return proba_dict

    def _select(self, parent_action, proba_dict):
        """ Function to balance exploitation and exploration described
        here https://web.stanford.edu/~surag/posts/alphazero.html """

        sqrt_N = np.sqrt(self.N.get(parent_action, 1))

        def upper_confidence_bound(a):
            return self.Q.get(a, 0) + self.exploration_param * sqrt_N * \
                proba_dict[a] / (1 + self.N.get(a, 0))

        return max(proba_dict, key=upper_confidence_bound)

    def _take_action(self, grid, action):
        index = action[0]
        value = action[1]
        grid.fill_cell(*index, value)
        return grid
