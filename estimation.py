from dataclasses import dataclass
import numpy as np
import auxmodels
import scipy.optimize
from typing import Callable
import progressbar as pb
import tensorflow as tf
import sys


def indirect_inference_estimate(observables: np.ndarray, gen_model: Callable, aux: Callable, method: str,
                                replications: int = 200,
                                paths: int = 1, **kwargs: dict):
    """
    Does an indirect inference parameter estimation on the observables to estimate the parameters described by the
    provided auxiliary model.

    Parameters
    ----------
    observables : np.ndarray
        The observables (data) that parameter estimation should be carried out on.
    gen_model : Callable
        The generative model to be used for simulations.
    aux : Callable
        The auxiliary model to use on the observables.
    method: str
        Use "local" for a local optimizer (Nelder-Mead) and "global" for a global optimizer (dual annealing).
        Use global optimizer with a lot of care, lol.
    replications: int, default=200
        The number of replications to carry out.
    paths: int, default=1
        The number of simulated paths to simulate for the indirect estimate.
    **kwargs
        The keyword arguments are passed to the auxiliary model and generative model, and must match otherwise error.
        Use as follows: {"gbm_kwargs": {kwargs}, "aux_kwargs": {kwargs}}
    Returns
    ---------
    param_estimate : IndirectEstimate
        Parameter estimate.
    """
    gbm_kwargs = kwargs["kwargs"]["gbm_kwargs"]
    aux_kwargs = kwargs["kwargs"]["aux_kwargs"]
    beta_real = aux(observables, **aux_kwargs)

    def objective(x):
        m, s = x
        ts = gen_model(m, s, **gbm_kwargs)
        beta_est = aux(ts, **aux_kwargs)
        differ = beta_real - beta_est
        return np.matmul(differ.T, differ)

    # This is a bit sus but whatever lol
    if method == "global":
        optimizer = lambda: scipy.optimize.dual_annealing(objective, bounds=((-1, 1), (0, 1)))
    else:
        optimizer = lambda: scipy.optimize.minimize(objective, x0=beta_real, method="Nelder-Mead")

    estimate = []
    widgets = [' ', pb.SimpleProgress(), ' [', pb.Timer(), '] ', ' (', pb.ETA(), ') ']
    with pb.ProgressBar(prefix="Replication", maxval=replications, fd=sys.stdout, widgets=widgets) as bar:
        for r in range(replications):
            sol = optimizer()
            estimate.append(sol.x)
            bar.update(r)
    estimate = np.array(estimate).T
    return IndirectEstimate(np.mean(estimate[0, :]), np.mean(estimate[1, :]), estimate[0, :], estimate[1, :],
                            replications, paths)


@dataclass
class IndirectEstimate:
    mu_mean: float
    sigma_mean: float
    mu: np.array
    sigma: np.array
    replications_used: int
    paths_used: int


def two_stage_estimate(observables: np.ndarray, model: str = "ml_model"):
    """
    Does a two stage parameter estimation on the observables to estimate the parameters.

    Parameters
    ----------
    observables : np.ndarray
        The observables (data) that parameter estimation should be carried out on.
    model : str
        The neural network model to use for the prediction. Currently, supports:
        "ml_model" : Maximum likelihood naive estimator.
    Returns
    ---------
    param_estimate : TwoStageEstimate
        Parameter estimate.
    """
    model = tf.keras.models.load_model(model)
    params = auxmodels.maximum_likelihood_naive(observables)
    mu, sigma = model.predict(params.T)[0]
    return TwoStageEstimate(mu, sigma)


@dataclass
class TwoStageEstimate:
    mu: float
    sigma: float
