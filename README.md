# hyperparameter_search
Some algorithms for searching/tuning hyperparameters


## imdb dataset.ipynb
really simple code for a random search.

## hp_search_genetic.py
Code implementing a quick and dirty genetic algorithm for hyperparameter search
Uses the Tox21 dataset from [deepchem](https://github.com/deepchem/deepchem). For relevant packages, should run in a virtual environment as per deepchem instructions for 'Easy install with Conda'.

It's a rough proof of concept, as it is there's a lot of code redundancy for each of the different hyperparameters, which is slow and a bit error-prone. To make it more generally useful it needs an object-oriented reimplementation where each hyperparameter gets its own class, and `hyperparams` is a dictionary containing all hyperparameter classes.



