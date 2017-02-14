from __future__ import absolute_import, print_function, division
import pytest

from firedrake import *


def test_pyop2_not_initialised():
    """Check that PyOP2 has not been initialised yet."""
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


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
