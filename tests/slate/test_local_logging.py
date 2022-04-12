import os


def test_slate_logging():
    """This only checks that logging does not break Firedrake, it does not check for correctness."""
    path = os.path.dirname(os.path.abspath(__file__))
    pyop2_cache = os.environ['PYOP2_CACHE_DIR']
    err = os.system(f'python {path}/script_logging.py -log_view :{pyop2_cache}/test.txt:ascii_flamegraph')
    assert err == 0
