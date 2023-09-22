import pytest
import os
from firedrake import *


# See https://pytest-xdist.readthedocs.io/en/stable/how-to.html#identifying-the-worker-process-during-a-test
@pytest.mark.skipif(
    "PYTEST_XDIST_WORKER_COUNT" in os.environ.keys()
    and int(os.environ["PYTEST_XDIST_WORKER_COUNT"]) > 1,
    reason="Must be run first"
)
def test_pyop2_not_initialised():
    """Check that PyOP2 has not been initialised yet.
       The test fails if another test builds a firedrake object not in a fixture."""
    assert not op2.initialised()


def test_pyop2_custom_init():
    """PyOP2 init parameters set by the user should be retained."""
    op2.init(debug=True, log_level='CRITICAL')
    UnitIntervalMesh(2)
    import logging
    logger = logging.getLogger('pyop2')
    assert logger.getEffectiveLevel() == CRITICAL
    assert op2.configuration['debug'] is True
    op2.configuration.reset()


def test_int_type():
    import firedrake_configuration
    from firedrake.utils import IntType

    expected = firedrake_configuration.get_config()["options"]["petsc_int_type"]
    actual = {4: "int32", 8: "int64"}[IntType.itemsize]

    assert expected == actual
