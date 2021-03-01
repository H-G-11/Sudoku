## What is this project about ?

I study the resolution of Sudoku Games with different techniques : 
- Backtrack algorithm
- Deep Learning technique
- Monte Carlo Tree Search
- AlphaSudoku, (simple) adaptation of AlphaGo to Sudoku

I provide a notebook (in French for the moment) to compare those methods.

### Backtrack algorithm

Simple implementation of the Backtrack algorithm, which provides a simple way of solving Sudokus.

### DeepIterativeSolver

Deep CNN trained on 2 million grids with [this dataset](#https://www.kaggle.com/radcliffe/3-million-sudoku-puzzles-with-ratings) solving Sudoku problems one cell at a time.

Unlike other implementations of such an algorithm, I use complicated Sudoku Problem with few clues. For example, [this dataset](#https://www.kaggle.com/bryanpark/sudoku) is by far more simple, but more used since it allows better results.

### MCTS

Monte Carlo Tree Search applied to Sudoku. The reward function is simply the rate of non-empty cell in terminal grids. I use UCB as the tree policy.

### AlphaSudoku

Use CNN trained before as the rollout policy. The tree policy is the following:
$$argmax \left(Q(s, a) + w * \frac{P(s, a)}{1 + N(s, a))}\right)$$

Unlike AlphaGo, I just train my network once for all. I don't have any value policy.

## Conclusion of this project

Backtracking is the best way to go because it is faster (no need to go through the deep CNN) and allows for a 100 % accuracy.

Nonetheless, AlphaSudoku is better in terms of iterations with 97% accuracy. I haven't tuned much the different models exposed here (the reward function is the first I came up with, I don't use the symmetry of the game, I haven't tuned exploration weights nor the number of extension of the trees before action).

It would be very interesting to do algorithms like AlphaGo Zero or MuZero to create new 16 * 16 Sudoku grids. In fact, in this setting, Backtracking could be really slow, while AlphaSudoku seems less impacted by the dimension (even if the number of parameters of the CNN would be greater).

So, all in all, I had fun doing this project and I found interesting to see that Backtracking can be longer than some probabilistic methods.

## How to use ?

You can run the notebook or clone the repository, go to the parent folder and run 
`python -m Sudoku.main`
for a quick test.

