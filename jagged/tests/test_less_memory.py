# coding=utf-8
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
import pytest

from jagged.memory_tricks import reduce_while_freeing


def test_reduce_while_freeing():

    # Test normal behavior (how to test that memory is kept low?)
    dfs = [pd.DataFrame(np.ones((100, 10)) * i) for i in xrange(201)]
    expected = pd.concat(dfs)
    actual = reduce_while_freeing(dfs, reduce_function=pd.concat, batch_size=0.2)
    assert_frame_equal(expected, actual)
    assert len(dfs) == 0

    # Test an exception is thrown if memory cannot be freed
    dfs = [pd.DataFrame(np.ones((100, 10)) * i) for i in xrange(201)]
    df0 = dfs[0]  # refcount +1
    assert df0 is not None
    with pytest.raises(Exception) as excinfo:
        reduce_while_freeing(dfs, reduce_function=pd.concat, batch_size=0.2)
    assert 'no dataframe must be referenced outside the passed list' in str(excinfo.value)
    assert len(dfs) == 201


if __name__ == '__main__':
    pytest.main()
