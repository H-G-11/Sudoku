import numpy as np
from tensorflow import keras
from ..Utils import SmartGrid
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

    The policy to choose action is a convolutional network trained
    on 1 million sudoku games, so that the approach is quite the
    same as for alpha Go, in a much simpler setting. """

    def __init__(self, grid: SmartGrid, exploration_param=1,
                 reward_increment=1,
                 pathnet='C:/Users/Hugues/Desktop/RLProject/policy_network',
                 verbose=False, number_path=100, max_depth=50):
        """ Action will be stored in the following format:
        ((i, j), value). """
        self.info_action = {}  # format is:
        # {parent: {N: N, Q: Q, W: W, childs :
        #   {child1: {}, child2: {}}}}
        self.grid = grid
        self.number_path = number_path
        self.reward_increment = reward_increment
        self.max_depth = max_depth
        self.exploration_param = exploration_param
        self.model = keras.models.load_model(pathnet)
        self.verbose = verbose

    def run(self):
        while self.grid.is_correct() and not self.grid.is_complete():
            best_action = self.choose_best_action()
            print(best_action)
            if best_action is None:
                break
            self.grid.fill_cell(*best_action[0], best_action[1])
            if len(self.info_action[best_action].get("childs", {})) == 0:
                print("Wrong path chosen")
                break
            self.info_action = self.info_action[best_action]["childs"]

        return self.grid.grid, self.grid.is_correct()

    def choose_best_action(self):
        """ Select best action from current root. """
        # we start again from nothing
        self.search_tree()
        best_action = max(self.info_action,
                          key=lambda a: self.info_action[a]["Q"],
                          default=None)
        return best_action

    def search_tree(self):
        for path in range(self.number_path):
            start_time = time.time()
            new_grid = self.grid.copy()
            new_grid_history = []
            dict_action = self.info_action
            reward = 0

            while new_grid.is_correct() and not new_grid.is_complete() \
                    and (reward < self.max_depth):
                probas_dict = self._predict_probas(new_grid)
                new_action = self._select(dict_action, probas_dict)
                new_grid = self._take_action(new_grid, new_action)
                new_grid_history.append(new_action)
                reward += self.reward_increment
                dict_action = dict_action.get(new_action, {})

            if self.verbose:
                print(time.time() - start_time, reward,
                      new_grid.grid)

            if new_grid.is_complete() and new_grid.is_correct():
                self.grid.grid = new_grid.grid
                self.info_action = {}
                return None

            # backpropagation
            for i, action in enumerate(new_grid_history):
                if i == 0:
                    to_update = self.info_action
                else:
                    if len(to_update.get("childs", {})) == 0:
                        to_update["childs"] = {}
                    to_update = to_update["childs"]
                if action not in to_update:
                    to_update[action] = {}
                to_update = to_update[action]
                to_update["N"] = to_update.get("N", 0) + 1
                to_update["W"] = to_update.get("W", 0) + reward / \
                    self.max_depth
                to_update["Q"] = to_update["W"] / to_update["N"]
                reward -= self.reward_increment

    def _predict_probas(self, new_grid):
        """ Return proba of actions if actions are not in place
        of a cell which is already taken. """
        array_of_proba = self.model.predict(
            np.array([custom_encoder(new_grid.grid)])
            )
        proba_dict = {}
        for i in range(9):
            for j in range(9):
                if new_grid.grid[i, j] == 0:
                    pos_at_index = new_grid.possibilities[(i, j)]
                    if len(pos_at_index) == 1:
                        return {((i, j), pos_at_index[0]): 1}
                    else:
                        for v in range(9):
                            proba_dict[((i, j), v + 1)] = \
                                array_of_proba[0, v, i, j] * \
                                len(pos_at_index) / 9
        if len(proba_dict) == 0:
            print(new_grid, array_of_proba)
        return proba_dict

    def _select(self, dict_action, proba_dict):
        """ Function to balance exploitation and exploration described
        here https://web.stanford.edu/~surag/posts/alphazero.html

        Input : dict_action = {"N": N, "Q": Q, "W": W, childs:{}}. """

        sqrt_N = np.sqrt(dict_action.get("N", 1))

        def upper_confidence_bound(a):
            child_info = dict_action.get("childs", {}).get(a, {})
            return child_info.get("Q", 1) + self.exploration_param * \
                sqrt_N * proba_dict[a] / (1 + child_info.get("N", 0)) \
                + np.random.rand() / 10  # random to encourage exploration

        return max(proba_dict, key=upper_confidence_bound)

    def _take_action(self, grid, action):
        index = action[0]
        value = action[1]
        grid.fill_cell(*index, value)
        return grid
