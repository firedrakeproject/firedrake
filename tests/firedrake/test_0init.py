import os
from firedrake import *
from pathlib import Path


def test_pyop2_custom_init():
    """PyOP2 init parameters set by the user should be retained."""
    op2.init(debug=True, log_level='CRITICAL')
    UnitIntervalMesh(2)
    import logging
    logger = logging.getLogger('pyop2')
    assert logger.getEffectiveLevel() == CRITICAL
    assert op2.configuration['debug'] is True
    op2.configuration.reset()


def test_pyop2_cache_dir_set_correctly():
    root = Path(os.environ.get("VIRTUAL_ENV", "~")).joinpath(".cache")
    cache_dir = os.environ.get("PYOP2_CACHE_DIR", str(root.joinpath("pyop2")))
    assert op2.configuration["cache_dir"] == cache_dir
