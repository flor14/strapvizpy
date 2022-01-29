import numpy as np
import pandas as pd
from pytest import raises
from strappy.display import plot_ci, tabulate_stats
from strappy.bootstrap import calculate_boot_stats


def test_plot_ci():
    """
    Tests the ci_plot function to make sure the outputs are correct.
        
    Returns
    --------
    None
        The test should pass and no asserts should be displayed.
        
    9 tests in total
    """
    
    # test integration with calculate_boot_stats function
    test_stat = calculate_boot_stats([1, 2, 3, 4], 
                                    1000, 
                                    level=0.95,
                                    random_seed=123)
    assert isinstance(test_stat, dict
    )

    # tests with invalid input type of title
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                title=123)
    assert str(e.value) == (
        "The value of the argument 'title' must be type of str."
    )
    
    # tests with invalid input type of title
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                title=True)
    assert str(e.value) == (
        "The value of the argument 'title' must be type of str."
    )
    
    # tests with invalid input type of x_axis
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                x_axis=123)
    assert str(e.value) == (
        "The value of the argument 'x_axis' must be type of str."
    )
    
    # tests with invalid input type of x_axis
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                x_axis=True)
    assert str(e.value) == (
        "The value of the argument 'x_axis' must be type of str."
    )
    
    # tests with invalid input type of y_axis
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                y_axis=123)
    assert str(e.value) == (
        "The value of the argument 'y_axis' must be type of str."
    )
    
    # tests with invalid input type of y_axis
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                y_axis=True)
    assert str(e.value) == (
        "The value of the argument 'y_axis' must be type of str."
    )

    # tests with invalid input type of path
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                y_axis="", 
                path=0.5)
    assert str(e.value) == (
        "The value of the argument 'path' must be type of str or None."
    )

    #tests if a plot was drawn by the function
    histogram = plot_ci([1, 2, 3, 4, 5, 6, 7], 1000, 
                         n=100, 
                         ci_level=0.95, 
                         ci_random_seed=123,
                         title="Bootstrap",
                         path="./tests/")
    assert histogram.gcf().number > 0, "Chart was not created correctly"

    # tests with invalid input value of path
    with raises(NameError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 1000, path="Users/")
    assert str(e.value) == (
        "The folder path you specified is invalid."
    )


def test_table_outputs():
    """Tests the functionality of the create_tables function."""
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123, pass_dist=True)
    s, bs = tabulate_stats(st)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 5, "Stats table should have 5 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    s, bs = tabulate_stats(st, estimator=False, alpha=False)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 3, "Stats table should have 3 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    s, bs = tabulate_stats(st, alpha=False)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 4, "Stats table should have 4 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123, n=10)
    s, bs = tabulate_stats(st)
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 4, "Parameter table should have 4 columns"
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123, n=10,
                              estimator="median")
    s, bs = tabulate_stats(st)
    assert ("Sample median" in s.data.columns), "Test Statistic name incorrect"
    
    
def test_table_errors():
    "Tests the functionality of the create_tables Raise Error statements."
    
    with raises(TypeError) as e:
        tabulate_stats(6, precision=2, estimator=True, alpha=True)
    assert str(e.value) == (
        "The stats parameter must be created from "
        "calculate_boot_stats() function.")
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123)
    with raises(TypeError) as e:
        tabulate_stats(st, precision=2.9, estimator=True, alpha=True)
    assert str(e.value) == "The precision parameter must be of type int."
    
    with raises(TypeError) as e:
        tabulate_stats(st, precision=2, estimator="Y", alpha=True)
    assert str(e.value) == (
        "The estimator and alpha parameters must be of type boolean."
    )
    
    with raises(TypeError) as e:
        tabulate_stats(st, precision=2, alpha=9)
    assert str(e.value) == (
        "The estimator and alpha parameters must be of type boolean."
    )

    assert len(tabulate_stats(st,  path ="./tests/")) == 2, (
        "The output length is not correct"
    )
    
    del st["lower"]
    with raises(TypeError) as e:
        tabulate_stats(st, precision=2)
    assert str(e.value) == (
        "The statistics dictionary is missing a key. "
        "Please rerun calculate_boot_stats() function"
    )
    
    with raises(NameError) as e:
        tabulate_stats(st,  path ="pt/")
    assert str(e.value) == (
        "The folder path you specified is invalid."
    )
        
    with raises(TypeError) as e:
        tabulate_stats(st, path = 1)
    assert str(e.value) == (
        "The path parameter must be a character string."
    )
