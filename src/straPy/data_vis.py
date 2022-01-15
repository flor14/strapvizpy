def density_ci_plot(dist, ci):
    """Makes a density plot with its confidence interval

    The plot will show a density curve of a sampling distribution 
    `dist` with its confidence interval `ci`.


    Parameters
    ----------
    dist : numpy.ndarray
        bootstrapped sampling distribution
    ci : numpy.ndarray
        confidence interval
    

    Returns
    -------
    plot object
        density plot with confidence interval
    
    Examples
    --------
    >>> density_ci_plot([1, 1, 2, 3, 5, 10]，[1., 9.375]）
    """
    pass

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