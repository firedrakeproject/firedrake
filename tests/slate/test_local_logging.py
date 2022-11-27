import os


def test_slate_logging_flame():
    """This only checks that logging does not break Firedrake, it does not check for correctness."""
    path = os.path.dirname(os.path.abspath(__file__))
    pyop2_cache = os.environ['PYOP2_CACHE_DIR']
    err = os.system(f'python {path}/script_logging.py -log_view :{pyop2_cache}/test.txt:ascii_flamegraph')
    assert err == 0


def test_slate_logging_flops():
    """This only checks that flop counting does not break Firedrake, it does not check for correctness."""
    path = os.path.dirname(os.path.abspath(__file__))
    pyop2_cache = os.environ['PYOP2_CACHE_DIR']
    os.system('export PYOP2_COMPUTE_KERNEL_FLOPS=1')
    err = os.system(f'python {path}/script_logging.py -log_view :{pyop2_cache}/test.txt')
    assert err == 0
    os.system('export PYOP2_COMPUTE_KERNEL_FLOPS=0')
