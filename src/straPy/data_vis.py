import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

def histogram_ci_plot(sample, rep, bin_size = 30, n="auto", ci_level=0.95, ci_random_seed=None, title = "", x_axis = "Bootstrapped Sample Mean", y_axis = "Count"):
    
    """Makes a histogram of a bootstrapped sampling distribution 
    with its confidence interval and observed mean.
     
    Parameters
    ----------
    sample : list or numpy.ndarray or pandas.core.series.Series 
        sample to bootstrap
    rep : int
        number of replicates of the distribution
    bin_size = int
        a number of bins representing intervals of equal size over the range
    n : str or int, default="auto"
        bootstrap sample size, "auto" specifies using the same size as the sample
    ci_level : float, default=0.95
        confidence level
    ci_random_seed : None or int, default=None
        seed for random state
    title : str, default = ""
        title of the histogram
    x_axis : str, default = "Bootstrap Sample Mean"
        name of the x axis
    y_axis : str, default = "Count"
        name of the y axis
    
    Returns
    -------
    plot: histogram
        histogram of bootstrapped distribution with confidence interval and observed mean
    
    Examples
    --------
    >>> histogram_ci_plot([1, 2, 3, 4, 5, 6, 7], 1000, n=100, ci_level=0.95, ci_random_seed=123)
    """

    if not isinstance(title, str):
        raise TypeError("The value of the argument 'title' must be type of str.")
        
    if not isinstance(x_axis, str):
        raise TypeError("The value of the argument 'x_axis' must be type of str.")
        
    if not isinstance(y_axis, str):
        raise TypeError("The value of the argument 'y_axis' must be type of str.")
        
    plt.hist(calculate_boot_stats(sample, rep, level=ci_level, random_seed = ci_random_seed, pass_dist=True)[1], density=False, bins=bin_size)
    plt.axvline(calculate_boot_stats(sample, rep, level=ci_level, random_seed = ci_random_seed, pass_dist=True)[0]["lower"], color='k', linestyle='--')
    plt.axvline(calculate_boot_stats(sample, rep, level=ci_level, random_seed = ci_random_seed, pass_dist=True)[0]["sample_mean"], color='r', linestyle='-')
    plt.axvline(calculate_boot_stats(sample, rep, level=ci_level, random_seed = ci_random_seed, pass_dist=True)[0]["upper"], color='k', linestyle='--')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    
def ci_table(dist, ci, se=True, size=True, mean=True, alpha=0.05):
    """Makes a table that shows lower bound and upper bound of the confidence interval and relevant statistics

    A table that contains the provided sampling distribution`dist`'s 
    confidence interval `ci` and other relevant statistics, including
    standard error `se`, sample size `size`, sample mean `mean` and 
    selected alpha `alpha`.


    Parameters
    ----------
    dist : numpy.ndarray
        bootstrapped sampling distribution
    ci : numpy.ndarray
        confidence interval
    size : boolean, default=True
        sample size
    mean: boolean, default=True
        sample mean
    alpha : float, default=0.05
        selected alpha

    Returns
    -------
    table object
        table with lower bound and upper bound of the confidence interval and relevant statistics
    
    Examples
    --------
    >>> ci_table([1, 1, 2, 3, 5, 10]，[1., 9.375], se=True, size=True, mean=True, alpha=0.05）
    """
    pass