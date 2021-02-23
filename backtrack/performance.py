import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from .backtrack import BacktrackSolver
from ..Utils.utils import SIZE, PATH_TO_CSV

# number of sudokus to solve
NROWS = 1000

####################################
# First, test sudoku grid creation :
total_time = []
total_iterations = []

for i in range(NROWS):
    t = time.time()
    solver = BacktrackSolver(np.zeros((SIZE ** 2, SIZE ** 2)))
    solver.solve()
    iteration_time = time.time() - t
    total_time.append(iteration_time)
    total_iterations.append(solver.iterations)

fig_empty_grid_time = plt.hist(total_time, bins=30,
                               range=(0, 1))
plt.title("Time taken to create a sudoku grid")
plt.savefig("./RLProject/Backtrack/EmptyTime.png")
plt.close()

fig_empty_grid_iterations = plt.hist(total_iterations, bins=30,
                                     range=(50, 200))
plt.title("Iterations taken to create a sudoku grid")
plt.savefig("./RLProject/Backtrack/EmptyIterations.png")
plt.close()

print('Mean time to produce a grid : ', round(np.mean(total_time), 3))
print('Mean number of iterations to produce a grid : ',
      round(np.mean(total_iterations)))
print('Max time to produce a grid : ', round(max(total_time), 3))
print('Max number of iterations to produce a grid : ', max(total_iterations))
print('Min time to produce a grid : ', round(min(total_time), 3))
print('Min number of iterations to produce a grid : ', min(total_iterations))
print('Std time to produce a grid : ', round(np.std(total_time), 3))
print('Std number of iterations to produce a grid : ',
      round(np.std(total_iterations)))


####################################
# Second, test sudoku resolution :
data = pd.read_csv(PATH_TO_CSV,
                   nrows=NROWS)

total_time = []
total_iterations = []

for i in range(NROWS):
    t = time.time()
    solver = BacktrackSolver(data.quizzes[i])
    solver.solve()
    iteration_time = time.time() - t
    total_time.append(iteration_time)
    total_iterations.append(solver.iterations)

print('Mean time to solve a sudoku : ', round(np.mean(total_time), 3))
print('Mean number of iterations to solve a sudoku : ',
      round(np.mean(total_iterations)))

fig_solve_time = plt.hist(total_time, bins=30,
                          range=(0, 0.05))
plt.title("Time taken to solve an existing grid")
plt.savefig("./RLProject/Backtrack/SolveTime.png")
plt.close()

fig_solve_iterations = plt.hist(total_iterations)
plt.title("Iterations taken to solve an existing grid")
plt.savefig("./RLProject/Backtrack/SolveIterations.png")
plt.close()
