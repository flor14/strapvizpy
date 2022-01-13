def bootstrap_distribution(sample, rep, n, estimator="mean"):
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
    n : int
        bootstrap sample size
    estimator : {"mean", "median", "var", "sd"}
        sampling distributor's estimator
    
    Returns
    -------
    numpy.ndarray
        bootstrapped sampling distribution
    
    Examples
    --------
    >>> bootstrap_distribution([1, 2, 3], 3, 3)
    array([1.66, 2, 2.66])
    """
    pass

def calculate_boot_ci(dist, level=0.95):
    """Calculates a confidence interval for a distribution.

    A confidence interval for the provided sampling distribution
    `dist` is calculated for a confidence level `level`.

    Parameters
    ----------
    dist : numpy.ndarray
        bootstrapped sampling distribution
    level : int, default=0.95
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