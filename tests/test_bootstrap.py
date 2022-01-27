from logging import WARNING
from warnings import WarningMessage, warn_explicit
import warnings
import numpy as np
from pytest import raises
from strappy.bootstrap import bootstrap_distribution, calculate_boot_stats


def test_bootstrap_distribution():
    """
    Tests the functionality of `bootstrap_distribution()`, which includes
    checking the type of object returned, its shape and its values.

    5 tests in total.
    """

    dist = bootstrap_distribution([1, 2, 3], 3, 3, random_seed=1)
    
    # checks the return type
    assert isinstance(dist, np.ndarray)

    # checks the shape of returned object
    assert dist.shape == (3,)

    # checks the values of returned object
    assert np.array_equal(dist.tolist(),
                          [1.3333333333333333,
                           1.6666666666666667,
                           1.3333333333333333])

    dist_2 = bootstrap_distribution([1, 2, 3],
                                  3,
                                  estimator="var",
                                  random_seed=1)

    # checks with different estimator
    assert np.array_equal(dist_2.tolist(),
                          [0.2222222222222222,
                           0.2222222222222222,
                           0.2222222222222222])

    dist_3 = bootstrap_distribution([1, 2, 3],
                                  3,
                                  estimator="var",
                                  random_seed=1)

    # checks if random_seed works
    assert np.array_equal(dist_2, dist_3)


def test_bootstrap_distribution_errors():
    """
    Tests error cases and messages thrown by `bootstrap_distribution()`.

    10 tests in total.
    """

    # tests with invalid input type of sample
    with raises(TypeError) as e:
        bootstrap_distribution({1, 2, 3}, 3, 3)
    assert str(e.value) == (
        "sample should be one of the types"
        "[list, numpy.ndarray, pandas.core.series.Series]"
    )

    # tests with invalid input type of rep
    with raises(TypeError) as e:
        bootstrap_distribution([1, 2, 3], 3.3, 3)
    assert str(e.value) == "rep should be of type 'int'"

    # tests with invalid input value of rep
    with raises(ValueError) as e:
        bootstrap_distribution([1, 2, 3], -1, 3)
    assert str(e.value) == "Invalid value for rep"

    # tests with invalid input type of n
    with raises(TypeError) as e:
        bootstrap_distribution([1, 2, 3], 3, 3.3)
    assert str(e.value) == "n should be of type 'str' or 'int'"

    # tests with invalid input value of n
    with raises(ValueError) as e:
        bootstrap_distribution([1, 2, 3], 3, "catch me")
    assert str(e.value) == "Invalid value for n. Did you intend n='auto'?"

    # tests with invalid input value of n
    with raises(ValueError) as e:
        bootstrap_distribution([1, 2, 3], 3, -3)
    assert str(e.value) == "Invalid value for n"

    # tests with invalid input type of estimator
    with raises(TypeError) as e:
        bootstrap_distribution([1, 2, 3], 3, 3, estimator=9)
    assert str(e.value) == "estimator should be of type 'str'"

    # tests with invalid input value of estimator
    with raises(ValueError) as e:
        bootstrap_distribution([1, 2, 3], 3, 3, estimator="outlier")
    assert str(e.value) == "Supported estimators are mean, median, var, sd"

    # tests with invalid input type of random_seed
    with raises(TypeError) as e:
        bootstrap_distribution([1, 2, 3], 3, 3, random_seed="I'm not a seed")
    assert str(e.value) == "random_seed should be None or of type 'int'"

    # tests with invalid input value of random_seed
    with raises(ValueError) as e:
        bootstrap_distribution([1, 2, 3], 3, 3, random_seed=-3)
    assert str(e.value) == "Invalid value for random_seed"


def test_calculate_boot_stats():
    """
    Tests functionality of calculate_boot_stats()

    16 tests in total.
    """
    # test integration with bootstrap dist function
    test_dist = bootstrap_distribution(sample=[1, 2, 3],
                                       rep=100)

    assert isinstance(test_dist, np.ndarray)

    # set up test dicts
    test_dict = calculate_boot_stats(
        [1, 2, 3, 4],
        1000,
        random_seed=123)


    test_dict_2 = calculate_boot_stats(
        [1000, 2000, 3000, 4000],
        n=4,
        rep=1000,
        level=0.9,
        random_seed=1234,
        estimator='var')

    test_dict_3, test_sample_dist = calculate_boot_stats(
        [1000, 2000, 3000, 4000],
        n=5,
        rep=1000,
        level=0.9,
        random_seed=1234,
        estimator='var',
        pass_dist=True)

    # checks the return type
    assert isinstance(test_dict, dict)

    assert isinstance(test_sample_dist, np.ndarray)
    
    # check properties and values of dictionary output
    assert len(test_dict) == 9

    assert test_dict["lower"] == 1.5

    assert test_dict["upper"] == 3.5

    assert test_dict["std_err"] == 0.5414773771820943

    assert test_dict["sample_size"] == 4

    assert test_dict["n"] == 'auto'

    assert test_dict['rep'] == 1000

    assert test_dict['level'] == 0.95

    assert test_dict['estimator'] == 'mean'

    # increasing sampling number should decrease the standard error
    assert test_dict_2['std_err'] > test_dict_3['std_err']

    # changing level parameter should change its value
    assert test_dict_2['level'] != test_dict['level']

    # changing estimate parameter should change its value
    assert test_dict_3['estimator'] == 'var'


def test_calculate_boot_stats_errors():
    """
    Tests error cases and messages thrown by `calculate_boot_stats()`.

    4 tests in total.
    """

    # tests with invalid input type of sample
    with raises(TypeError) as e:
        calculate_boot_stats([1, 2, 3, 4],
        1000,
        level='ninety-five',
        estimator="mean",
        random_seed=123)
    assert str(e.value) == ("level should be of type 'float'")

    with raises(TypeError) as e:
        calculate_boot_stats([1, 2, 3, 4],
        1000,
        level=0.95,
        estimator="mean",
        random_seed=123,
        pass_dist='True')
    assert str(e.value) == ("pass_dist should be of type 'bool'")

    with raises(ValueError) as e:
        calculate_boot_stats([1, 2, 3, 4],
        1000,
        level=1.0,
        estimator="mean",
        random_seed=123)
    assert str(e.value) == ("level should be between 0 and 1")

    with raises(ValueError) as e:
        calculate_boot_stats([1, 2, 3, 4],
        1000,
        level=0.0,
        estimator="mean",
        random_seed=123)
    assert str(e.value) == ("level should be between 0 and 1")

    assert calculate_boot_stats([1, 2, 3, 4],
        1000,
        level=0.05,
        estimator="mean",
        random_seed=123)

  
