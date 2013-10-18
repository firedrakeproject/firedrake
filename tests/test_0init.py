import pytest

from firedrake import *


def test_pyop2_not_initialised():
    """Check that PyOP2 has not been initialised yet."""
    assert not op2.initialised()


def test_pyop2_custom_init():
    """PyOP2 init parameters set by the user should be retained."""
    op2.init(debug=3, log_level=CRITICAL)
    UnitIntervalMesh(2)
    assert op2.logger.logger.getEffectiveLevel() == CRITICAL
    assert op2.cfg.debug == 3
    op2.init()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
