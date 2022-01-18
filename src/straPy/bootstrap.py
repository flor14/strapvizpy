import numpy as np
import pandas as pd

def bootstrap_distribution(sample, rep, n="auto", estimator="mean", random_seed=None):
    """Bootstraps a sampling distribution for a sample.

    A sampling distribution of `rep` replicates is generated
    for the specified `estimator`with replacement with a
    bootstrap sample size of `n`.

    Parameters
    ----------
    sample : list or numpy.ndarray or pandas.core.series.Series 
        sample to bootstrap
    rep : int
        number of replicates of the distribution
    n : str or int, default="auto"
        bootstrap sample size, "auto" specifies using the same size as the sample
    estimator : {"mean", "median", "var", "sd"}
        sampling distributor's estimator
    random_seed : None or int, default=None
        seed for random state
    
    Returns
    -------
    numpy.ndarray
        bootstrapped sampling distribution
    
    Examples
    --------
    >>> bootstrap_distribution([1, 2, 3], 3, 3)
    array([1.66, 2, 2.66])
    """
    supported_estimators = {
        "mean": np.mean,
        "median": np.median,
        "var": np.var,
        "sd": np.std
    }

    if not (isinstance(sample, list) or
            isinstance(sample, np.ndarray) or
            isinstance(sample, pd.Series)):
        raise TypeError("sample should be one of the types"
                        "[list, numpy.ndarray, pandas.core.series.Series]")

    if not isinstance(rep, int):
        raise TypeError("rep should be of type 'int'")

    if isinstance(rep, int) and rep < 1:
        raise ValueError("Invalid value for rep")

    if not (isinstance(n, str) or isinstance(n, int)):
        raise TypeError("n should be of type 'str' or 'int'")

    if isinstance(n, str) and n != "auto":
        raise ValueError("Invalid value for n. Did you intend n='auto'?")

    if isinstance(n, int) and n < 1:
        raise ValueError("Invalid value for n")

    if not isinstance(estimator, str):
        raise TypeError("estimator should be of type 'str'")

    if estimator not in supported_estimators.keys():
        raise ValueError("Supported estimators are mean, median, var, sd")

    if not (random_seed is None or isinstance(random_seed, int)):
        raise TypeError("random_seed should be None or of type 'int'")

    if isinstance(random_seed, int) and random_seed < 0:
        raise ValueError("Invalid value for random_seed")

    if random_seed:
        np.random.seed(random_seed)

    if n == "auto":
        n = len(sample)

    return supported_estimators[estimator](
        np.random.choice(sample, size=(rep, n), replace=True),
        axis=1
    )

def calculate_boot_ci(dist, level=0.95):
    """Calculates a confidence interval for a distribution.

    A confidence interval for the provided sampling distribution
    `dist` is calculated for a confidence level `level`.

    Parameters
    ----------
    dist : numpy.ndarray
        bootstrapped sampling distribution
    level : float, default=0.95
        confidence level
    
    Returns
    -------
    numpy.ndarray
        lower and upper bounds of the confidence interval
    
    Examples
    --------
    >>> calculate_boot_ci([1, 1, 2, 3, 5, 10])
    array([1., 9.375])
    """
    pass