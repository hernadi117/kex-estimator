import sys
import progressbar as pb
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import itertools
import auxmodels
from typing import Callable, Sequence
import geometricbrownianmotion as gbm


def generate_data(gen_model: Callable, aux: Callable, n_features: int = 2, mu_r: tuple = (-0.5, 0.5, 0.01),
                  sigma_r: tuple = (0, 0.5, 0.01), **kwargs: dict):
    """
    Generates observables which follows a Geometric Brownian Motion (GBM).

    Parameters
    ----------
    mu_r : tuple, default=(-0.5, 0.5, 0.01)
        The values of mu that should be used for data generation in the form (start, stop, step).
    sigma_r : tuple, default=(0, 0.5, 0.01)
        The values of sigma that should be used for data generation in the form (start, stop, step).
    gen_model : Callable
        The generative model to be used for generation.
    aux : Callable
        The auxiliary model to use on the observables.
    n_features : int
        Number of features (depends on the auxiliary model).
    **kwargs
        The keyword arguments are passed to the auxiliary model and generative model, and must match otherwise error.
        Use as follows: {"gbm_kwargs": {kwargs}, "aux_kwargs": {kwargs}}
    Returns
    ---------
    X : np.ndarray
        Generated data, where each row is the feature(s) of the observable.
    y : np.ndarray
        The actual features, i.e. the actual parameter that was used to generate corresponding row in X.
    labels: np.ndarray:
        An array of all possible features (unique).
    """
    gbm_kwargs = kwargs["kwargs"]["gbm_kwargs"]
    aux_kwargs = kwargs["kwargs"]["aux_kwargs"]
    m = gbm_kwargs["m"]
    labels = cartesian_product([np.arange(mu_r[0], mu_r[1], mu_r[2]), np.arange(sigma_r[0], sigma_r[1], sigma_r[2])])
    X = np.empty((np.shape(labels)[0] * m, n_features))
    y = np.repeat(labels, m, axis=0)
    widgets = [' [', pb.Timer(), '] ', pb.PercentageLabelBar(), ' (', pb.ETA(), ') ']
    with pb.ProgressBar(prefix="Data generation", maxval=np.shape(labels)[0], fd=sys.stdout, widgets=widgets) as bar:
        for bar_i, i, p in zip(itertools.count(), itertools.count(start=0, step=m), labels):
            mu, sigma = p
            ts = gen_model(mu, sigma, **gbm_kwargs)
            X[i:i + m, :] = aux(ts, **aux_kwargs).T
            bar.update(bar_i)
    mask = np.isnan(X).any(axis=1)
    return X[~mask], y[~mask], labels


def cartesian_product(arrays: Sequence) -> np.ndarray:
    """
    Calculates the cartesian product of the arrays provided.

    Parameters
    ----------
    arrays: Sequence
        A sequence of sequences, for example [[1, 2 , 3], [4, 5, 6]], to take cartesian product of.

    Returns
    ---------
    np.ndarray
        The cartesian product, where each row corresponds to one element in the product.
    """
    # Copied the fastest implementation I could find from SO
    la = len(arrays)
    L = *map(len, arrays), la
    arr = np.empty(L, dtype=np.result_type(*arrays))
    arrs = *itertools.accumulate(itertools.chain((arr,), itertools.repeat(0, la - 1)), np.ndarray.__getitem__),
    idx = slice(None), *itertools.repeat(None, la - 1)
    for i in range(la - 1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la - i]]
        arrs[i - 1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)


X_gen, y_gen, label = generate_data(gbm.gbm_naive, auxmodels.maximum_likelihood_naive, n_features=2,
                                    mu_r=(-0.5, 0.5, 0.01),
                                    sigma_r=(0, 0.5, 0.01),
                                    kwargs={"gbm_kwargs": {"m": 1000, "n": 253, "s0": 1},
                                            "aux_kwargs": {}})

X_train, X_test, y_train, y_test = train_test_split(X_gen, y_gen, test_size=0.33, random_state=42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim=np.shape(X_train)[1], activation="relu"))
model.add(tf.keras.layers.Dense(2))
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
model.fit(X_gen, y_gen, epochs=10, batch_size=32, verbose=1)
# loss, acc = model.evaluate(X_test, y_test, verbose=1)
# print(loss, acc, sep="\n")
model.save("ml_model")
