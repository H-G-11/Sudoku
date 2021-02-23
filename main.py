from .Utils import PATH_TO_CSV, SmartGrid
from .deep_iterative_solver import DeepSolver
import numpy as np
import pandas as pd

NROWS = 10
data = pd.read_csv(PATH_TO_CSV, nrows=NROWS)

for i in range(NROWS):
    deep_solver = DeepSolver(SmartGrid.from_grid(data.quizzes[i]))
    deep_solver.run()
    print(i, np.array_equal(SmartGrid(data.solutions[i]).grid,
                            deep_solver.grid.grid))
