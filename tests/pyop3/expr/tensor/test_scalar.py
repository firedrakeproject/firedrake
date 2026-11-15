import numpy as np
import pytest

import pyop3 as op3


@pytest.mark.parametrize("mode, expected", [("write", 2.0), ("inc", 6.5), ("max", 4.5), ("min", 2.0)])
@pytest.mark.parametrize("eager_strategy", ["array", "compile"])
@pytest.mark.parallel(nprocs=[1, 2])
def test_eager_assign_modes(mode, expected, eager_strategy):
    scalar = op3.Scalar(np.float64(4.5))

    scalar.assign(2.0, mode, eager=True, eager_strategy=eager_strategy)

    assert scalar.value == expected
