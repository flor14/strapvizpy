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
    >>> histogram_ci_plot([1, 2, 3, 4, 5, 6, 7], 1000 ,n=100, ci_level=0.95, ci_random_seed=123)
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


def summary_tables(stat, precision=2, estimator=True, alpha=True):
    """Makes two tables that summerize the statistics from the bootstrapped 
    samples and the parameters for creating the bootstrapped samples.


    Parameters
    ----------
    stat : dict
        summary statistics produced by the `calculate_boot_stats()` function 
    precision : int, default=2
        the precision of the table values
    estimator : boolean, default=True
        include the bootstrap estimate in the summary statistics table
    alpha: boolean, default=True
        include the significance level in the summary statistics table

    Returns
    -------
    summary statistics: table object
        table summerizing the lower bound and upper bound of the confidence interval,
        the standard error, the sampling statitic (if estimator = True), and the
        significance level (if alpha = True)
    bootstrap parameters: table object
        table  summerizing the parameters of the bootstrap sampling spficiying 
        the original sample size, number of repititions, the significance level, 
        and the number of samples in each bootstrap if its different from the 
        original sample size.
        
    Examples
    --------
    >>> st = calculate_boot_stats([1, 2, 3, 4], 1000, level=0.95, random_seed=123)
    >>> stats_table, parameter_table  = summary_tables(st)
    >>> stats_table
    >>> parameter_table
    """
    if not isinstance(stat, dict):
        raise TypeError("Input statistics must be a dictionary")
            
    if not isinstance(precision, int):
        raise TypeError("The precision parameter must be of type int.")
            
    if not (isinstance(estimator, bool) or 
            isinstance(alpha, bool)):
        raise TypeError("The estimator and alpha parameter must be of type boolean.")
    
    # define the statistics table
    df = pd.DataFrame(data=np.array([(stat["lower"], stat["upper"], stat["std_err"])]),
                      columns=["Lower Bound CI", "Upper Bound CI", "Standard Error"])

    if estimator is True:
        s_name = "Sample " + stat["estimator"]
        df[s_name] = stat["sample_" + stat["estimator"]]

    if alpha is True:
        df["Significance Level"] = 1 - stat["level"]
        stats_table = df.style.format(
            precision=precision, formatter={("Significance Level"): "{:.3f}"}
        )
    else:
        stats_table = df.style.format(precision=precision)

    # set formatting and caption for table
    stats_table.set_caption(
        "Bootstrapping sample statistics from sample with "+ 
        str(stat["sample_size"]) + " records"
    ).set_table_styles(
        [{"selector": "caption", "props": "caption-side: bottom; font-size:1.00em;"}],
        overwrite=False)

    # create bootstrapping parameter summary table
    df_bs = pd.DataFrame(
        data=np.array(
            [(stat["sample_size"], stat["rep"], (1 - stat["level"]))]),
        columns=["Sample Size", "Repetition", "Significance Level"])
    
    if stat["n"] != "auto":
        df_bs["Samples per bootstrap"] = round(stat["n"], 0)

    # set formatting and caption for table
    bs_params = df_bs.style.format(
        precision=0, formatter={("Significance Level"): "{:.3f}"})
    
    bs_params.set_caption("Parameters used for bootstrapping").set_table_styles(
        [{"selector": "caption", "props": "caption-side: bottom; font-size:1.00em;"}],
        overwrite=False)

    return stats_table, bs_params