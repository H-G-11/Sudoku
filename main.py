from .Utils import PATH_TO_CSV, Grid
from .MCTS import MCTS
import pandas as pd

NROWS = 1
data = pd.read_csv(PATH_TO_CSV, nrows=NROWS)

for i in range(NROWS):
    mcts_solver = MCTS(Grid(data.puzzle[i]))
    print(mcts_solver.run())
