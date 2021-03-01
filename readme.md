## What is this project about ?

I studied the resolution of Sudoku Games with different techniques : 
- Backtrack algorithm
- Deep Learning technique
- Monte Carlo Tree Search
- AlphaSudoku, (simple) adaptation of AlphaGo to Sudoku

I provide a notebook (in French for the moment) to compare these methods.

### Backtrack algorithm

Simple implementation of the Backtrack algorithm, which allows to solve Sudokus in a simple way.

### DeepIterativeSolver

Deep CNN trained on 2 million grids with [this dataset](https://www.kaggle.com/radcliffe/3-million-sudoku-puzzles-with-ratings), solving Sudokus one cell at a time.

Unlike other implementations of such an algorithm, I use Sudoku grids with few clues. For example, [this dataset](https://www.kaggle.com/bryanpark/sudoku) is by far simpler, but more widely used because it gives better results.

### MCTS

Monte Carlo Tree Search applied to Sudoku. The reward function is simply the rate of non-empty cells in the terminal grids. I use UCB as the tree policy.

### AlphaSudoku

Use CNN trained before as the rollout policy. The tree policy is the same as AlphaGo's tree policy.

Unlike AlphaGo, I train my network only once. I don't have any value network.

## Conclusion of this project

Backtracking is still the best solution because it is faster (no need to go through the deep CNN) and allows 100 % accuracy.

Nonetheless, AlphaSudoku is better in terms of number of iterations, with an accuracy of 97%. I haven't tuned much the different models shown here (the reward function is the first one I came up with, I don't use the game symmetry, I didn't tune the exploration weights nor the number of tree extensions before action), so the results could easily be improved.

It would be interesting to test algorithms like AlphaGo Zero or MuZero to create new 16 * 16 Sudoku grids. In fact, in this framework, Backtracking could be very slow, whereas AlphaSudoku seems less impacted by the dimension (even if the number of parameters of the CNN would more important).

So, all in all, I had fun doing this project and I found it interesting to see that Backtracking can be challenged in terms of number of iterations versus accuracy.

## How to use ?

You can run the notebook or clone the repository, go to the parent folder and run 
`python -m Sudoku.main`
for a quick test.

