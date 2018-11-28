# Implements a quick and dirty genetic algorithm to search hyperparameters

# Would be better (and more general) with object-oriented re-implementation
# each hyperparameter is its own class with methods on how it varies, is randomly generated, etc
# overall hyperparameters class that has a dictionary of its hyperparameters

# Uses pandas dataframe

import pandas as pd
import numpy as np
from numpy import random
import tensorflow as tf
import deepchem as dc
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from math import ceil, log


# set global variables for min/max for each parameter
N_HIDDEN = [10, 80]
N_LAYERS = [1, 7]
LEARNING_RATE = [-3, 0]
LEARNING_RATE_TYPE = 'log_uniform'
DROPOUT_PROB = [0.2, 0.8]
N_EPOCHS = [10, 80]
BATCH_SIZE = [8, 1024]


def import_dc_data():
    """import the deepchem data, delete additional task labels, export train/validation/test sets"""
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    # Remove extra tasks
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]

    # return the data as a dictionary
    dc_data = {'train_X': train_X, 'valid_X': valid_X, 'test_X': test_X,
               'train_y': train_y, 'valid_y': valid_y, 'test_y': test_y,
               'train_w': train_w, 'valid_w': valid_w, 'test_w': test_w}

    return dc_data


def eval_tox21_hyperparams(dc_data, n_hidden=50, n_layers=1, learning_rate=.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100,
                           weight_positives=True):

    d = 1024
    graph = tf.Graph()
    with graph.as_default():

        # Generate tensorflow graph
        with tf.name_scope("placeholders"):
            x = tf.placeholder(tf.float32, (None, d))
            y = tf.placeholder(tf.float32, (None,))
            w = tf.placeholder(tf.float32, (None,))
            keep_prob = tf.placeholder(tf.float32)
        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                W = tf.Variable(tf.random_normal((d, n_hidden)))
                b = tf.Variable(tf.random_normal((n_hidden,)))
                x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
                # Apply dropout
                x_hidden = tf.nn.dropout(x_hidden, keep_prob)
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, 1)))
            b = tf.Variable(tf.random_normal((1,)))
            y_logit = tf.matmul(x_hidden, W) + b
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.round(y_one_prob)
        with tf.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            # Multiply by weights
            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy
            # Sum all contributions
            l = tf.reduce_sum(entropy)

        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()

# For tensorboard visualization
#        hyperparam_str = "d-%d-hidden-%d-lr-%f-n_epochs-%d-batch_size-%d-weight_pos-%s" % (
#            d, n_hidden, learning_rate, n_epochs, batch_size, str(weight_positives))
#        train_writer = tf.summary.FileWriter('/tmp/fcnet-func-' + hyperparam_str,
#                                             tf.get_default_graph())
        N = dc_data['train_X'].shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(n_epochs):
                pos = 0
                while pos < N:
                    batch_X = dc_data['train_X'][pos:pos+batch_size]
                    batch_y = dc_data['train_y'][pos:pos+batch_size]
                    batch_w = dc_data['train_w'][pos:pos+batch_size]
                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
           #         print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
           #         train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

            # Make Predictions (set keep_prob to 1.0 for predictions)
            valid_y_pred = sess.run(y_pred, feed_dict={x: dc_data['valid_X'], keep_prob: 1.0})
            valid_y = dc_data['valid_y']  # get labels

        acc = accuracy_score(valid_y, valid_y_pred, sample_weight=dc_data['valid_w'])
        prec = precision_score(valid_y, valid_y_pred)  # can't weight?
        recall = recall_score(valid_y, valid_y_pred)  # can't weight?
        roc_auc = roc_auc_score(valid_y, valid_y_pred)
    return (acc, prec, recall, roc_auc)


# build a dataframe with for model hyperparameters and evaluation metrics
def eval_log(name=None, n_hidden=50, n_layers=1, learning_rate=.001,
             dropout_prob=0.5, n_epochs=45, batch_size=100,
             weight_positives=True):
    """ Evaluates the model on the hyperparameters supplied as arguments,
    returns the results as a pd.Series """

    # run the model
    (acc, prec, rec, auc) = eval_tox21_hyperparams(dc_data, n_hidden=n_hidden, n_layers=n_layers,
                                                   learning_rate=learning_rate,
                                                   dropout_prob=dropout_prob, n_epochs=n_epochs,
                                                   batch_size=batch_size, weight_positives=weight_positives)

    # create a dict
    hparams = {'n_hidden': n_hidden,
               'n_layers': n_layers,
               'learning_rate': learning_rate,
               'dropout_prob': dropout_prob,
               'batch_size': batch_size,
               'weight_positives': weight_positives,
               'accuracy_score': acc,
               'precision_score': prec,
               'recall_score': rec,
               'auc': auc}

    return pd.Series(hparams, name=name, index=hparams.keys())


def get_random_hparams(n, n_hidden=N_HIDDEN, n_layers=N_LAYERS, learning_rate=LEARNING_RATE,
                       learning_rate_type=LEARNING_RATE_TYPE,
                       dropout_prob=DROPOUT_PROB, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    """ creates n sets of hyperparameters randomly. default arguments represent random bounds. weight_positives is
    probability of True"""
    arr_n_hidden = random.randint(n_hidden[0], n_hidden[1], size=n, dtype=int)
    arr_n_layers = random.randint(n_layers[0], n_layers[1], size=n, dtype=int)

    rand_lr = min(learning_rate) + ((learning_rate[1] - learning_rate[0]) * random.rand(n))
    if learning_rate_type == 'log_uniform':
        arr_learning_rate = np.power(10, rand_lr)
    else:
        arr_learning_rate = rand_lr

    arr_dropout_prob = min(dropout_prob) + ((dropout_prob[1] - dropout_prob[0]) * random.rand(n))
    arr_n_epochs = random.randint(n_epochs[0], n_epochs[1], size=n, dtype=int)
    arr_batch_size = random.randint(batch_size[0], batch_size[1], size=n, dtype=int)
    arr_weight_positives = random.choice([True, False], size=n)

    return (arr_n_hidden, arr_n_layers, arr_learning_rate, arr_dropout_prob,
            arr_n_epochs, arr_batch_size, arr_weight_positives)


def run_n_models(dc_data, hparams_tuple):
    """ takes a dictionary of hyperparameters (each is an array) and runs the model on all the params in the array.
    returns a dictionary of arrays with output metrics """

    (arr_n_hidden, arr_n_layers, arr_learning_rate,
     arr_dropout_prob, arr_n_epochs, arr_batch_size, arr_weight_positives) = hparams_tuple

    # create empty arrays for output metrics
    n = len(arr_n_hidden)

    acc = np.zeros(n)
    prec = np.zeros(n)
    rec = np.zeros(n)
    auc = np.zeros(n)

    # use a dirty for loop for now
    for i in range(n):
        (acc[i], prec[i], rec[i], auc[i]) = eval_tox21_hyperparams(dc_data, n_hidden=arr_n_hidden[i],
                                                                   n_layers=arr_n_layers[i],
                                                                   learning_rate=arr_learning_rate[i],
                                                                   dropout_prob=arr_dropout_prob[i],
                                                                   n_epochs=arr_n_epochs[i],
                                                                   batch_size=arr_batch_size[i],
                                                                   weight_positives=arr_weight_positives[i])
    # return tuple of arrays
    return (acc, prec, rec, auc)


def eval_n_models(dc_data, n=5, generation=None, init="random", hparams=None):
    """evaluates n different models. Generates hyperparameters randomly if not specified."""

    if init == 'hparams':
        params = hparams
    # default to init='random'
    else:
        params = get_random_hparams(n)

    (acc, prec, rec, auc) = run_n_models(dc_data, params)

    # if epoch is specified, write it as a column

    dict = {'generation': pd.Series(np.full(n, generation)),
            'n_hidden': pd.Series(params[0]),
            'n_layers': pd.Series(params[1]),
            'learning_rate': pd.Series(params[2]),
            'dropout_prob': pd.Series(params[3]),
            'n_epochs': pd.Series(params[4]),
            'batch_size': pd.Series(params[5]),
            'weight_positives': pd.Series(params[6]),
            'acc': pd.Series(acc),
            'prec': pd.Series(prec),
            'rec': pd.Series(rec),
            'auc': pd.Series(auc)}

    df = pd.DataFrame.from_dict(dict)
    return df


def get_hparams_from_dataframe(df):
    """ gets the hparams from the supplied dataframe and returns them as a tuple of numpy arrays """

    arr_n_hidden = np.asarray(df.loc[:, 'n_hidden'])
    arr_n_layers = np.asarray(df.loc[:, 'n_layers'])
    arr_learning_rate = np.asarray(df.loc[:, 'learning_rate'])
    arr_dropout_prob = np.asarray(df.loc[:, 'dropout_prob'])
    arr_n_epochs = np.asarray(df.loc[:, 'n_epochs'])
    arr_batch_size = np.asarray(df.loc[:, 'batch_size'])
    arr_weight_positives = np.asarray(df.loc[:, 'weight_positives'])

    return (arr_n_hidden, arr_n_layers, arr_learning_rate, arr_dropout_prob,
            arr_n_epochs, arr_batch_size, arr_weight_positives)


def ga_calc_next_gen(prev_generation, generation_size, metric='auc'):
    """ calculates the hyperparameters for the next generation"""

    # initialize empty arrays for next generation
    arr_n_hidden = np.zeros(generation_size, dtype=int)
    arr_n_layers = np.zeros(generation_size, dtype=int)
    arr_learning_rate = np.zeros(generation_size, dtype=np.float32)
    arr_dropout_prob = np.zeros(generation_size, dtype=np.float32)
    arr_n_epochs = np.zeros(generation_size, dtype=int)
    arr_batch_size = np.zeros(generation_size, dtype=int)
    arr_weight_positives = np.zeros(generation_size, dtype=bool)

    # sort the previous generation by desired metric
    sortd = prev_generation.sort_values(metric, ascending=False)

    # split the previous generaton into quartiles
    q1 = ceil(generation_size * 0.25)
    q2 = ceil(generation_size * 0.50)
    q3 = ceil(generation_size * 0.75)

    # top quartile go straight through
    arr_n_hidden[0:q1] = sortd.iloc[0:q1].loc[:, 'n_hidden']
    arr_n_layers[0:q1] = sortd.iloc[0:q1].loc[:, 'n_layers']
    arr_learning_rate[0:q1] = sortd.iloc[0:q1].loc[:, 'learning_rate']
    arr_dropout_prob[0:q1] = sortd.iloc[0:q1].loc[:, 'dropout_prob']
    arr_n_epochs[0:q1] = sortd.iloc[0:q1].loc[:, 'n_epochs']
    arr_batch_size[0:q1] = sortd.iloc[0:q1].loc[:, 'batch_size']
    arr_weight_positives[0:q1] = sortd.iloc[0:q1].loc[:, 'weight_positives']

    # second quartile are crossed from first quartile

    # get a shuffled view of the top quartile
    shuf = np.arange(q1)
    np.random.shuffle(shuf)  # shuffle in place
    shuffled = sortd.iloc[0:q1].iloc[shuf]

    for i in range(q2 - q1):
        arr_n_hidden[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['n_hidden'],
                                                            shuffled.iloc[i].loc['n_hidden']]))
        arr_n_layers[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['n_layers'],
                                                            shuffled.iloc[i].loc['n_layers']]))
        arr_learning_rate[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['learning_rate'],
                                                                 shuffled.iloc[i].loc['learning_rate']]))
        arr_dropout_prob[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['dropout_prob'],
                                                                shuffled.iloc[i].loc['dropout_prob']]))
        arr_n_epochs[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['n_epochs'],
                                                            shuffled.iloc[i].loc['n_epochs']]))
        arr_batch_size[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['batch_size'],
                                                              shuffled.iloc[i].loc['batch_size']]))
        arr_weight_positives[q1 + i] = np.random.choice(np.asarray([sortd.iloc[i].loc['weight_positives'],
                                                                    shuffled.iloc[i].loc['weight_positives']]))

    # third quartile are mutations of first quartile
    for i in range(q3 - q2):
        arr_n_hidden[q2 + i] = int(np.random.normal(loc=sortd.iloc[i].loc['n_hidden'],
                                                    scale=sortd.loc[:, 'n_hidden'].std()))
        arr_n_layers[q2 + i] = int(np.random.normal(loc=sortd.iloc[i].loc['n_layers'],
                                                    scale=sortd.loc[:, 'n_layers'].std()))
        # Learning rate is exponential so this needs to be a log normal dist
        lr_log = np.log10(sortd.iloc[i].loc['learning_rate'])  # get log10 of the learning rate
        # get std of the log10 learning rates
        lr_log_std = np.log10(sortd.loc[:, 'learning_rate']).std()
        print("log lr = ", lr_log, "lr_log_std", lr_log_std)
        arr_learning_rate[q2 + i] = np.power(10, np.random.normal(loc=lr_log,
                                                                  scale=lr_log_std))

        # arr_learning_rate[q2 + i] = np.random.normal(loc=sortd.iloc[i].loc['learning_rate'],
        #                                             scale=sortd.loc[:, 'learning_rate'].std())
        arr_dropout_prob[q2 + i] = np.random.normal(loc=sortd.iloc[i].loc['dropout_prob'],
                                                    scale=sortd.loc[:, 'dropout_prob'].std())
        arr_n_epochs[q2 + i] = int(np.random.normal(loc=sortd.iloc[i].loc['n_epochs'],
                                                    scale=sortd.loc[:, 'n_epochs'].std()))
        arr_batch_size[q2 + i] = int(np.random.normal(loc=sortd.iloc[i].loc['batch_size'],
                                                      scale=sortd.loc[:, 'batch_size'].std()))
        # randomly flip weight_positives with probability 0.25
        t = sortd.iloc[i].loc['weight_positives']
        arr_weight_positives[q2 + i] = np.random.choice([t, 1-t], p=[0.75, 0.25])

    # clip third quartile to within sensible bounds
    np.clip(arr_n_hidden[q2:q3], N_HIDDEN[0], N_HIDDEN[1], out=arr_n_hidden[q2:q3])
    np.clip(arr_n_layers[q2:q3], N_LAYERS[0], N_LAYERS[1], out=arr_n_layers[q2:q3])
    # learning rate varies as exp, clip between 0 and 1 should be fine
    np.clip(arr_learning_rate[q2:q3], 0, 1, out=arr_learning_rate[q2:q3])
    np.clip(arr_dropout_prob[q2:q3], DROPOUT_PROB[0], DROPOUT_PROB[1], out=arr_dropout_prob[q2:q3])
    np.clip(arr_n_epochs[q2:q3], N_EPOCHS[0], N_EPOCHS[1], out=arr_n_epochs[q2:q3])
    np.clip(arr_batch_size[q2:q3], BATCH_SIZE[0], BATCH_SIZE[1], out=arr_batch_size[q2:q3])

    # fourth quartile are randomly generated
    (rand_n_hidden, rand_n_layers, rand_learning_rate, rand_dropout_prob,
     rand_n_epochs, rand_batch_size, rand_weight_positives) = get_random_hparams(n=generation_size-q3)

    arr_n_hidden[q3:generation_size] = rand_n_hidden
    arr_n_layers[q3:generation_size] = rand_n_layers
    arr_learning_rate[q3:generation_size] = rand_learning_rate
    arr_dropout_prob[q3:generation_size] = rand_dropout_prob
    arr_n_epochs[q3:generation_size] = rand_n_epochs
    arr_batch_size[q3:generation_size] = rand_batch_size
    arr_weight_positives[q3:generation_size] = rand_weight_positives

    return (arr_n_hidden, arr_n_layers, arr_learning_rate, arr_dropout_prob,
            arr_n_epochs, arr_batch_size, arr_weight_positives)


def ga_run_generation(df, dc_data, generation_size=8):
    """ Run a single generation"""
    # get the generation number for the previous generation
    n_prev = max(df['generation'])

    # slice the df to get just the last generation
    prev = df.loc[df.loc[:, 'generation'] == n_prev]

    # compute the parameters for the next generation
    next_params = ga_calc_next_gen(prev, generation_size=generation_size, metric='auc')

    # run the next generation and return it
    next = eval_n_models(dc_data, n=generation_size, generation=n_prev +
                         1, init='hparams', hparams=next_params)

    return next


def ga_run(dc_data, seed=False, seed_pop=None, generation_size=20, n_generations=5):
    """ Run the genetic algorithm for n_generations"""

    # if no seed generation is supplied, create one
    if seed:
        pop = seed_pop
    else:
        pop = eval_n_models(dc_data, n=generation_size, generation=0)
    print("Generation 0 (seed):\n", pop.sort_values('auc', ascending=False))

    for g in range(1, n_generations):
        # run the next generation
        next = ga_run_generation(pop, dc_data, generation_size=generation_size)
        next = next.sort_values('auc', ascending=False)
        print("Generation ", g, ":")
        print("next:", next)
        pop = pop.append(next)

    return pop


if __name__ == "__main__":
    # Code here to execute, might be better in an ipynb?

    dc_data = import_dc_data()

    # create a dataframe by running eval_n_models
    df_genetic = ga_run(dc_data, generation_size=20, n_generations=10)
