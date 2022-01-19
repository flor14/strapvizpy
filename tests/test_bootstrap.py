import numpy as np
from pytest import raises
from straPy.bootstrap import bootstrap_distribution


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