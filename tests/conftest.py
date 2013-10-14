"""Global test configuration."""

from subprocess import check_call
from sys import executable
from functools import wraps
import os
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
                           cwd=os.path.abspath(os.path.dirname(__file__)), shell=True)
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
