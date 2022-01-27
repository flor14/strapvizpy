import pandas as pd
from strappy.display import plot_ci
from strappy.bootstrap import calculate_boot_stats, bootstrap_distribution
from pytest import raises
import numpy as np
import matplotlib.pyplot as plt


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

    # tests with invalid input type of save_result_to
    with raises(TypeError) as e:
        plot_ci([1, 2, 3, 4, 5, 6, 7], 
                1000, 
                n=100, 
                ci_level=0.95, 
                y_axis="", 
                save_result_to=123)
    assert str(e.value) == (
        "The value of the argument 'save_result_to' must be type of str."
    )


    #tests if a plot was drawn by the function
    histogram = plot_ci([1, 2, 3, 4, 5, 6, 7], 1000, 
                         n=100, 
                         ci_level=0.95, 
                         ci_random_seed=123,
                         title="Bootstrap")
    assert isinstance(histogram, tuple), "Chart was not created correctly"