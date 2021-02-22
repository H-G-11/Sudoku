from .Utils import PATH_TO_CSV, SmartGrid
from .MCTS import MCTS
import pandas as pd

NROWS = 1
data = pd.read_csv(PATH_TO_CSV, nrows=NROWS)

for i in range(NROWS):
    print(SmartGrid.from_grid(data.solutions[i]).grid)
    mcts_solver = MCTS(SmartGrid.from_grid(data.quizzes[i]),
                       number_path=50, max_depth=20)
    print(mcts_solver.run())
