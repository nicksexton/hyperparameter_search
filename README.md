# hyperparameter_search
A quite disorganised repo containing some useful code for hyperparameter search/tuning.


## imdb dataset.ipynb
really simple code for a random search.

## hp_search_genetic.py
Code implementing a quick and dirty genetic algorithm for hyperparameter search
Uses the Tox21 dataset from [deepchem](https://github.com/deepchem/deepchem). For relevant packages, should run in a virtual environment as per deepchem instructions for 'Easy install with Conda'. NN model code for the Deepchem dataset is from [Deep Learning with Tensorflow](https://github.com/matroid/dlwithtf).

It's a rough proof of concept, as it is there's a lot of code redundancy for each of the different hyperparameters, which is slow and a bit error-prone. To make it more concise and generally useful it needs an object-oriented reimplementation where each hyperparameter gets its own class, and `hyperparams` is a dictionary containing all hyperparameter classes.

Would also probably be better (and more pythonic) if implemented as an iterable, so you could just compute another generation by calling .next()

