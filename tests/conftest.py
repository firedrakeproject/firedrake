"""Global test configuration."""

# Insert the parent directory into the module path so we can find the common
# module whichever directory we are calling py.test from.
#
# Note that this will ONLY work when tests are run by calling py.test, not when
# calling them as a module. In that case it is required to have the Firedrake
# root directory on your PYTYHONPATH to be able to call tests from anywhere.
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from subprocess import check_call
from sys import executable
from functools import wraps
from inspect import getsourcefile
from mpi4py import MPI


def parallel(nprocs=3):
    """Run a test in parallel
    :arg nprocs: The number of processes to run.

    .. note ::
        Parallel tests need to either be in the same folder as the utils
        module or the test folder needs to be on the PYTHONPATH."""
    def _parallel_test(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if MPI.COMM_WORLD.size > 1:
                fn(*args, **kwargs)
            else:
                check_call(' '.join(['mpiexec', '-n', '%d' % nprocs, executable,
                                     '-c', '"import %s; %s.%s()"' %
                                     (fn.__module__, fn.__module__, fn.__name__)]),
                           cwd=os.path.dirname(getsourcefile(fn)), shell=True)
        return wrapper
    return _parallel_test


def pytest_addoption(parser):
    parser.addoption("--short", action="store_true", default=False,
                     help="Skip long tests")


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors")


def pytest_runtest_setup(item):
    run_parallel = item.keywords.get("parallel", None)
    if run_parallel:
        item._obj = parallel(run_parallel.kwargs.get('nprocs', 3))(item._obj)


def pytest_cmdline_preparse(config, args):
    if 'PYTEST_VERBOSE' in os.environ and '-v' not in args:
        args.insert(0, '-v')
    if 'PYTEST_EXITFIRST' in os.environ and '-x' not in args:
        args.insert(0, '-x')
    if 'PYTEST_NOCAPTURE' in os.environ and '-s' not in args:
        args.insert(0, '-s')
    if 'PYTEST_TBNATIVE' in os.environ:
        args.insert(0, '--tb=native')
    if 'PYTEST_WATCH' in os.environ and '-f' not in args:
        args.insert(0, '-f')
    try:
        import pytest_benchmark   # noqa: Checking for availability of plugin
        # Set number of warmup iteration to 1
        if "--benchmark-warmup-iterations" not in args:
            args.insert(0, "--benchmark-warmup-iterations=1")
    except ImportError:
        pass
