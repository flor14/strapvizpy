import numpy as np
import pandas as pd
import warnings

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

def calculate_boot_stats(sample, rep, n="auto", level=0.95, random_seed=None):
    """Calculates a bootstrapped confidence interval for a sample.
    A bootstrapped confidence interval for the provided sample is
    calculated for a confidence level `level`. The mean and standard
    deviation of the sample are also returned.

    Parameters
    ----------
    sample : list or numpy.ndarray or pandas.core.series.Series 
        sample to bootstrap
    rep : int
        number of replicates of the distribution
    n : str or int, default="auto"
        bootstrap sample size, "auto" specifies using the same size as the sample
    random_seed : None or int, default=None
        seed for random state
    level : float, default=0.95
        confidence level
    
    Returns
    -------
    dictionary
        dictionary containing mean and sd of sample, and lower and upper bounds of the confidence interval
    
    Examples
    --------
    >>> calculate_boot_stats([1, 2, 3, 4], 1000, level=0.95, random_seed=123)  
    {'lower': 1.5, 'upper': 3.5, 'mean': 2.5, 'sd': 1.118033988749895}
    """

    if not isinstance(level, float):
        raise TypeError("level should be of type 'float")

    if not (level > 0 and level < 1):
        raise ValueError("level should be between 0 and 1")

    if level < 0.7:
        warnings.warn("Warning: chosen level is quite low--level is a confidence level, not a signficance level")

    # get the bootstrapped mean vector
    dist = bootstrap_distribution(sample, rep, n, "mean", random_seed)

    stats_dict = {}

    stats_dict["lower"] = np.percentile(dist, 100 * (1-level)/2)
    stats_dict["upper"] = np.percentile(dist, 100 * (1-(1-level)/2))

    stats_dict["mean"] = np.mean(sample)
    stats_dict["sd"] = np.std(sample)

    return stats_dict