from .utils import read_transform
from .backtrack import BacktrackSolver
from .mcts import MCTS
from .deep_iterative_solver import DeepIterativeSolver
from .alpha_sudoku import AlphaSudoku
from tensorflow.keras.models import load_model
import time

NROWS = 2
data_X, data_Y = read_transform(NROWS=NROWS)

model = load_model('C:/Users/Hugues/Desktop/RLProject/policy_network')

for i in range(NROWS):
    print('---------', i)
    print('--------- Backtrack ')
    start_time = time.time()
    back_solver = BacktrackSolver(data_X[i])
    back_solver.solve()
    print(round(time.time() - start_time, 2))
    print(back_solver.iterations)

    print('--------- MCTS ')
    start_time = time.time()
    mcts_solver = MCTS(data_X[i])
    mcts_solver.solve()
    print(round(time.time() - start_time, 2))
    print(mcts_solver.iterations)

    print('--------- DeepIterativeSolver ')
    start_time = time.time()
    deep_solver = DeepIterativeSolver(data_X[i], model=model)
    deep_solver.solve()
    print(round(time.time() - start_time, 2))
    print(deep_solver.iterations)

    print('--------- AlphaSudoku ')
    start_time = time.time()
    alpha_solver = AlphaSudoku(data_X[i], model=model)
    alpha_solver.solve()
    print(round(time.time() - start_time, 2))
    print(alpha_solver.iterations)
