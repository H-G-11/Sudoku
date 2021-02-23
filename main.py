from .Utils import PATH_TO_CSV, SmartGrid
from .mcts import MCTS, SudokuGrid
import pandas as pd

NROWS = 3
data = pd.read_csv(PATH_TO_CSV, nrows=NROWS)

for i in range(NROWS):
    deep_solver = MCTS(SudokuGrid(SmartGrid.from_grid(data.quizzes[i])))
    deep_solver.solve()
    print(SmartGrid(data.solutions[i]).grid,
          deep_solver.sudoku_grid.grid.grid)
