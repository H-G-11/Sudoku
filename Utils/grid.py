import numpy as np
from .utils import SIZE


class Grid:
    def __init__(self, grid=None):
        self.grid = grid if grid is not None \
            else np.zeros((SIZE ** 2, SIZE ** 2))
        self.possibilities = {}  # we will have to call pos at least once

    @classmethod
    def from_grid(cls, grid):
        if isinstance(grid, str):
            grid = np.array([int(i) for i in grid])
            grid = grid.reshape((SIZE ** 2, SIZE ** 2))
        obj = cls(grid)
        obj.possibilities = obj._pos()
        return obj

    def fill_cell(self, i, j, value):
        self.grid[i, j] = value
        del self.possibilities[(i, j)]  # raises error if not present
        for index in self._related_indeces(i, j):
            pos_at_index = self.possibilities.get(index, [])
            if value in pos_at_index:
                pos_at_index.remove(value)

    def erase_cell(self, i, j):
        self.grid[i, j] = 0
        self.possibilities = self._pos()

    def index_with_min_pos(self):
        min_pos = min(self.possibilities.values(), key=len)
        return [k for k, v in self.possibilities.items() if v == min_pos]

    def _related_indeces(self, i, j):
        """ All indeces that will be impacted by changing (i, j). """
        indices = self._indeces_in_box(i, j)
        for k in range(SIZE ** 2):
            if (i, k) not in indices:
                indices.append((i, k))
            if (k, j) not in indices:
                indices.append((k, j))
        return indices

    def _pos(self):
        """ Return a dictionary with keys being index and
        values being a list of different possibilities. """

        line, col = np.where(self.grid == 0)
        pos = {(line[i], col[i]): self._pos_at(line[i], col[i])
               for i in range(len(line))}
        return pos

    def _pos_at(self, i, j):
        """ Possibilities at index (i, j). """

        local_pos = list(range(1, SIZE ** 2 + 1))
        to_delete = []
        for pos in local_pos:
            if pos in self._values_in_box(i, j):
                to_delete.append(pos)
            elif pos in self.grid[i, :]:
                to_delete.append(pos)
            elif pos in self.grid[:, j]:
                to_delete.append(pos)
        return [pos for pos in local_pos if pos not in to_delete]

    def _indeces_in_box(self, i, j):
        idx_line, idx_col = SIZE * (i // SIZE), SIZE * (j // SIZE)
        indeces = []
        for i in range(SIZE):
            for j in range(SIZE):
                indeces.append((idx_line + i, idx_col + j))
        return indeces

    def _values_in_box(self, i, j):
        """ All values in the square SIZE * SIZE. """
        idx_line, idx_col = SIZE * (i // SIZE), SIZE * (j // SIZE)
        return self.grid[idx_line: idx_line + SIZE,
                         idx_col: idx_col + SIZE]
