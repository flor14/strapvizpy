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


def ci_table(stat_dic, precision=2, estimator=True, alpha=True):
    """Makes two tables that summerize the statistics from the bootstrapped samples and the parameters for creating the bootstrapped samples.


    Parameters
    ----------
    stat_dic : dict
        summary statistics produced by the `calculate_boot_stats()` function 
    precision : int, default= 2
        the precision of the table values
    esitmator : boolean, default=True
        include the bootstrap estimate in the summary statistics table
    alpha: boolean, default=True
        include the significance level in the summary statistics table

    Returns
    -------
    summary statistics: table object
        table summerizing the lower bound and upper bound of the confidence interval, the standard error, the sampling statitic (if estimator = True), and the significance level (if alpha = True)
    bootstrap parameters: table object
        table  summerizing the parameters of the bootstrap sampling spficiying the original sample size, number of repititions, the significance level, and the number of samples in each bootstrap if its different from the original sample size.  
    
    Examples
    --------
    >>> st = calculate_boot_stats([1, 2, 3, 4], 1000, level=0.95, random_seed=123)
    >>> stats_table, parameter_table  = ci_table(st)
    >>> stats_table
    >>> parameter_table
    """
    
    # define the statistics table
    df = pd.DataFrame(data=np.array([(stat_dic["lower"], stat_dic["upper"], stat_dic["std_err"])]),
        columns=["Lower bound CI", "Upper bound CI", "Standard Error"])

    if estimator is True:
        s_name = "Sample " + stat_dic["estimator"]
        df[s_name] = np.round(stat_dic["sample_" + stat_dic["estimator"]], rounding)

    if alpha is True:
        df["Significance Level"] = 1 - stat_dic["level"]
        stats_table = df.style.format(
            precision=precision, formatter={("Significance Level"): "{:.3f}"}
        )
    else:
        stats_table = df.style.format(precision=rounding)

    #set formatting and caption for table
    stats_table.set_caption(
        "Bootstrapping sample statistics from sample with "
        + str(st["sample_size"])
        + " records"
    ).set_table_styles(
        [{"selector": "caption", "props": "caption-side: bottom; font-size:1.00em;"}],
        overwrite=False,
    )

    df_bs = pd.DataFrame(
        data=np.array(
            [
                (
                    round(stat_dic["sample_size"], 0),
                    round(stat_dic["rep"], 0),
                    1 - stat_dic["level"],
                )
            ]
        ),
        columns=["Sample Size", "Repetition", "Significance Level"],
    )
    if stat_dic["n"] != "auto":
        df_bs["Samples per bootstrap"] = round(stat_dic["n"], 0)

    bs_params = df_bs.style.format(
        precision=0, formatter={("Significance Level"): "{:.3f}"}
    )
    bs_params.set_caption("Parameters used for bootstrapping").set_table_styles(
        [{"selector": "caption", "props": "caption-side: bottom; font-size:1.00em;"}],
        overwrite=False,
    )

    return stats_table, bs_params