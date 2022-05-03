# Basic Learning Demo with MCTX
A very basic implementation of AlphaZero style learning with the MCTX framework for Monte Carlo Tree Search in JAX. The included basic_tree_search.py script assumes the agent have access to the exact environment dynamics and does not include any model learning, but rather learns a policy and value function from tree search results as in AlphaZero.

## Usage
Using this repo requires JAX, Haiku, NumPy and wandb. You can start an experiment with 
```bash
python3 basic_tree_search.py -c config.json -o basic_tree_search -s 0
```
The flags have the following meaning:

-c specifies the configuration file, an example of which is provided in config.json.<br>
-o prefix for file names containing output data.<br> 
-s specifies the random seed.<br>

## Preliminary Results
The following plots display running average return as a function of training steps on a small 5x5 procedurally generated Maze environment (implemented as ProcMaze in jax_environments.py). Each time the environment is reset it builds a maze using randomized depth first search. The configuration used for this result is specified in config.json. It should be straightforward to apply to other environments written in jax with a similar interface, though given it uses MCTS with no chance nodes it is not really appropriate for environments with stochastic transitions (randomness in the initial state as in ProcMaze is ok).
<img align="center" src="img/learning_curve.png" width=800>