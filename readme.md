## What is this project about ?

I study the resolution of Sudoku Games with different techniques : 
- Backtrack algorithm
- Deep Learning technique
- Monte Carlo Tree Search
- AlphaSudoku, adaptation of AlphaGo to Sudoku

I use [this dataset](https://www.kaggle.com/radcliffe/3-million-sudoku-puzzles-with-ratings), which contains Sudoku grids with few clues. Other datasets like [this one](https://www.kaggle.com/bryanpark/sudoku) are by far simpler, but more widely used because they give better results (testing on this dataset gives me 100% accuracy for AlphaSudoku, against 97% for the previous one).

I provide a notebook (in French for the moment) to compare these methods.

### Backtrack algorithm

Simple implementation of the Backtrack algorithm, which allows to solve Sudokus in a simple way.

### Deep Iterative Solver

Deep CNN trained on 2 million grids solving Sudokus one cell at a time.

### Monte Carlo Tree Search

Monte Carlo Tree Search applied to Sudoku. The reward function is simply the rate of non-empty cells in the terminal grids. I use UCB as the tree policy.

### AlphaSudoku

Use CNN trained before as the rollout policy. The tree policy is the same as AlphaGo's tree policy.

Unlike AlphaGo, I train my network only once, and I don't have any value network.

## Conclusion of this project

Backtracking is still the best solution because it is faster (no need to go through the deep CNN) and allows 100 % accuracy.

Nonetheless, AlphaSudoku is better in terms of number of iterations (number of different grids considered), with an accuracy of 97%. I haven't tuned much the different models shown here (the reward function is the first one I came up with, I don't use the game symmetry, I didn't tune the exploration weights nor the number of tree extensions before action), so the results could easily be improved.

It would be interesting to test algorithms like AlphaGo Zero or MuZero to create new 16 * 16 Sudoku grids. In fact, in this framework, Backtracking could be very slow, whereas AlphaSudoku seems less impacted by the dimension (even if the number of parameters of the CNN would more important).

So, all in all, I had fun doing this project and I found it interesting to see that Backtracking can be challenged in terms of number of iterations versus accuracy.
