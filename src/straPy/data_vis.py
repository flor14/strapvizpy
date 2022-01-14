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

def ci_table(dist, ci):
    """Makes a table that shows mean, lower bound and upper bound of the confidence interval

    A table that contains the provided sampling distribution
    `dist`'s mean and its confidence interval `ci`.


    Parameters
    ----------
   dist : numpy.ndarray
        bootstrapped sampling distribution
    ci : numpy.ndarray
        confidence interval
    
    Returns
    -------
    table object
        table with mean, lower bound and upper bound of the confidence interval
    
    Examples
    --------
    >>> ci_table([1, 1, 2, 3, 5, 10]，[1., 9.375]）
    """
    pass