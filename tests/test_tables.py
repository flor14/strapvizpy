import numpy as np
from pytest import raises
from strapPy.bootstrap import bootstrap_distribution
from strapPy.bootstrap import calculate_boot_stats
    from strapPy.data_vis import create_tables

def test_table_outputs():
    """Tests the functionality of the create_tables function."""
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123)
    s, bs = create_tables(st)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 5, "Stats table should have 5 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    s, bs = create_tables(st, estimator=False, alpha=False)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 3, "Stats table should have 3 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    s, bs = create_tables(st, alpha=False)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 4, "Stats table should have 4 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123, n=10)
    s, bs = create_tables(st)
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 4, "Parameter table should have 4 columns"
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123, n=10,
                              estimator="median")
    s, bs = create_tables(st)
    assert ("Sample median" in s.data.columns), "Test Statistic name incorrect"
    
    
def test_table_errors():
    "Tests the functionality of the create_tables Raise Error statements."
    
    with raises(TypeError) as e:
        create_tables(6, precision=2, estimator=True, alpha=True)
    assert str(e.value) == ("The stats parameter must be created from \
calculate_boot_stats() function.")
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123)
    with raises(TypeError) as e:
        create_tables(st, precision=2.9, estimator=True, alpha=True)
    assert str(e.value) == "The precision parameter must be of type int."
    
    with raises(TypeError) as e:
        create_tables(st, precision=2, estimator="Y", alpha=True)
    assert str(e.value) == ("The estimator and alpha parameters must be \
of type boolean.")
    
    with raises(TypeError) as e:
        create_tables(st, precision=2, alpha=9)
    assert str(e.value) == ("The estimator and alpha parameters must be \
of type boolean.")
    
    del st["lower"]
    with raises(TypeError) as e:
        create_tables(st, precision=2)
    assert str(e.value) == ("The statistics dictionary is missing a key. \
Please rerun calculate_boot_stats() function")