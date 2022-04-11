import numpy as np
from statsmodels.tsa.ar_model import AutoReg


def maximum_likelihood_naive(series: np.ndarray) -> np.ndarray:
    """
    Solves the optimization problem and fins the mu and sqrt(sigma^2) which maximizes the likelihood function.

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.

    Returns
    ----------
    np.ndarray
        mu and sqrt(sigma^2) which maximizes the likelihood function, where each column [mu, sqrt(sigma^2)]^T
        corresponds to same column in the observables.
    """
    T = np.shape(series)[0] - 1
    diffed = series[1::] / series[:-1:]
    mu_ml = (np.sum(diffed, axis=0) / T) - 1
    sigma_ml = np.sum(np.square(diffed - (1 + mu_ml)), axis=0) / T
    return np.stack((mu_ml, np.sqrt(sigma_ml)), axis=0)


def naive_indirect_ml_estimator(series: np.ndarray) -> np.ndarray:
    """
    Finds the naive indirect maximum likelihood estimator, which is the mean of all mu and sqrt(sigma^2)
    where mu and sigma^2 are the parameters which maximizes the likelihood function of the observables.

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.

    Returns
    ----------
    np.ndarray
        The mean of all mu and sqrt(sigma^2) across all observables as a column vector,
         [mean(mu), mean(sqrt(sigma^2))]^T.
    """
    estimate = maximum_likelihood_naive(series)
    return np.mean(estimate, axis=1, keepdims=True)


def ar_fit(series: np.ndarray, lags: int) -> np.ndarray:
    """
    Fits an autoregressive model of order p, AR(p).

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.
    lags : int
        The order of the model.

    Returns
    ----------
    np.ndarray
        The p + 1 coefficients of the fit as a column vector, [c_0, c_1, ..., c_p]^T, where each column corresponds to
        same column of the observables.
    """
    return np.apply_along_axis(lambda arr: AutoReg(arr, lags).fit().params, 0, series)


def indirect_ar_estimator(series: np.ndarray, lags: int) -> np.ndarray:
    """
    Finds the indirect autoregressive model estimator of order p, which is the mean of all p parameters for each
    collection of observables.

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.
    lags : int
        The order of the model.

    Returns
    ----------
    np.ndarray
        The mean of coefficients of all fits across observables as a column vector,
        [mean(c_0), mean(c_1), ..., mean(c_p)]^T.
    """
    coefficients = ar_fit(series, lags)
    return np.mean(coefficients, axis=1, keepdims=True)


def poly_fit(series, deg: int, centering: bool = True) -> np.ndarray:
    """
    Performs a polynomial fit.

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.
    deg : int
        The degree of the polynomial fit.
    centering : bool, default=True
        Centers the observables (each column) at zero and scales it to have unit standard deviation.

    Returns
    ----------
    np.ndarray
        The coefficients of the fit, each column of coefficients corresponding to same column of observations.
    """
    series = center(series) if centering else series  # Yes we assign to itself cuz I do not care
    return np.polynomial.polynomial.polyfit(np.arange(np.shape(series)[0]), series, deg)


def indirect_poly_estimator(series, deg: int, centering: bool = True):
    """
    Finds the indirect polynomial fit estimator of desired degree, which is the mean of all parameters for each
    collection of observables.

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.
    deg : int
        The degree of the polynomial fit.
    centering : bool, default=True
        Centers the observables (each column) at zero and scales it to have unit standard deviation.

    Returns
    ----------
    np.ndarray
        The mean of coefficients of all fits across observables as a column vector,
        [mean(c_0), mean(c_1), ..., mean(c_p)]^T.
    """
    coefficients = poly_fit(series, deg, centering)
    return np.mean(coefficients, axis=1, keepdims=True)


def center(series: np.ndarray) -> np.ndarray:
    """
    Centers the observables (each column) at zero and scales it to have unit standard deviation.

    Parameters
    ----------
    series : np.ndarray
        The observations, where each column is one independent collection of observables.

    Returns
    ----------
    np.ndarray
        The centered observables, each column corresponding to same column of observations.
    """
    return (series - np.mean(series, axis=0)) / np.std(series, axis=0)
