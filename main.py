from .Utils import PATH_TO_CSV, SmartGrid
from .MCTS import MCTS
import pandas as pd

NROWS = 1
data = pd.read_csv(PATH_TO_CSV, nrows=NROWS)

for i in range(NROWS):
    mcts_solver = MCTS(SmartGrid.from_grid(data.puzzle[i]),
                       number_path=10, max_depth=20, verbose=1)
    print(mcts_solver.run())
