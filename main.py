from .Utils import read_transform
from .alpha_sudoku import AlphaSudoku

NROWS = 20
data_X, data_Y = read_transform(NROWS=NROWS)

for i in range(NROWS):
    print(i)
    alpha_solver = AlphaSudoku(data_X[i])
    alpha_solver.solve()
    print(alpha_solver.sudoku_grid.grid.grid)
