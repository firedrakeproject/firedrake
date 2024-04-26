import os
import pytest


pytest.skip(allow_module_level=True, reason="pyop3 TODO")


def test_slate_logging():
    """This only checks that logging does not break Firedrake, it does not check for correctness."""
    path = os.path.dirname(os.path.abspath(__file__))
    pyop2_cache = os.environ['PYOP2_CACHE_DIR']
    err = os.system(f'python {path}/script_logging.py -log_view :{pyop2_cache}/test.txt:ascii_flamegraph')
    assert err == 0
