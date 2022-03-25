import os
import tempfile
import firedrake_configuration

def test_slate_logging():
    """This only checks that logging does not break Firedrake, it does not check for correctness."""
    path = os.path.dirname(os.path.abspath(__file__))
    firedrake_configuration.setup_cache_dirs()
    pyop2_cache = os.environ.get('PYOP2_CACHE_DIR',
                                 os.path.join(tempfile.gettempdir(),
                                 'pyop2-cache-uid%d' % os.getuid()))
    err = os.system(f'python {path}/test_logging.py -log_view :{pyop2_cache}/test.txt:ascii_flamegraph')
    assert err == 0