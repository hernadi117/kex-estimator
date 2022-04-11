import numpy as np


def gbm_exact(mu: float, sigma: float, s0: float, dt: float, n: int, m: int, seed=False) -> np.ndarray:
    """Simulates GBM for `m` simulations, each of `n` steps with step size `dt`.

    Parameters
    ----------
    mu : float
        Drift for this GBM.
    sigma : float
        Volatility for this GBM.
    s0 : float
        Initial value of the randomly varying quantity.
    dt : float
        Step size used in the simulation.
    n : float
        Steps in one GBM simulation, excluding initial value.
    m : float
        Number of independent GBM simulations.
    seed : bool, default=False
        Provides a seed of 0 to the random number generator.

    Returns
    -------
    np.ndarray
        GBM evolutions, where column corresponds to an observed GBM evolution.
    """
    rng = np.random.default_rng(seed=0) if seed else np.random.default_rng()
    st = np.exp((mu - sigma ** 2 / 2) * dt
                + sigma * rng.normal(0.0, np.sqrt(dt), size=(n, m)))
    st = np.vstack((np.ones(shape=(1, m)), st))
    return s0 * st.cumprod(axis=0)


def gbm_naive(mu: float, sigma: float, s0: float, n: int, m: int, seed=False) -> np.ndarray:
    """Simulates GBM for `m` simulations, each of `n` (naive discretization yields step size 1).

    Parameters
    ----------
    mu : float
        Drift for this GBM.
    sigma : float
        Volatility for this GBM.
    s0 : float
        Initial value of the randomly varying quantity.
    n : float
        Steps in one GBM simulation, excluding initial value.
    m : float
        Number of independent GBM simulations.
    seed : bool, default=False
        Provides a seed of 0 to the random number generator.

    Returns
    -------
    np.ndarray
        GBM evolutions, where column corresponds to an observed GBM evolution.
    """
    rng = np.random.default_rng(seed=0) if seed else np.random.default_rng()
    st = 1 + mu + sigma * rng.normal(0.0, 1.0, size=(n, m))
    st = np.vstack((np.ones(shape=(1, m)), st))
    return s0 * st.cumprod(axis=0)
