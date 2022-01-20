import numpy as np
from pytest import raises
from straPy.bootstrap import bootstrap_distribution
from straPy.bootstrap import calculate_boot_stats
from straPy.data_viz import summary_tables


summary_tables(stat, precision=2, estimator=True, alpha=True)

def test_table_outputs()
    """
    Tests the functionality of the summary table function.
    """
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123)
    s, bs = summary_tables(st)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 5, "Stats table should have 5 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    

    s, bs = summary_tables(st, estimator=False, alpha=False)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 3, "Stats table should have 3 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    

    s, bs = summary_tables(st, alpha=False)
    assert s.data.shape[0] == 1, "Stats table should have 1 row"
    assert s.data.shape[1] == 3, "Stats table should have 4 columns"
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 3 columns"
    
    st = calculate_boot_stats(np.random.randint(1, 20, 20), 1000,
                              level=0.95, random_seed=123, n=10)
    s, bs = summary_tables(st)
    assert bs.data.shape[0] == 1, "Parameter table should have 1 row"
    assert bs.data.shape[1] == 3, "Parameter table should have 4 columns"