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
from functools import wraps


def parallel(item):
    """Run a test in parallel.

    :arg item: The test item to run.
    """
    from mpi4py import MPI
    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("parallel test can't be run within parallel environment")
    marker = item.get_marker("parallel")
    if marker is None:
        raise RuntimeError("Parallel test doesn't have parallel marker")
    nprocs = marker.kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise RuntimeError("Need at least two processes to run parallel test")

    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process
    zerocall = " ".join(["py.test", "--runxfail", "-s", "-q", str(item.fspath), "-k", item.name])
    restcall = " ".join(["py.test", "--runxfail", "--tb=no", "-q", str(item.fspath), "-k", item.name])
    call = "mpiexec -n 1 %s : -n %d %s" % (zerocall, nprocs - 1, restcall)
    check_call(call, shell=True)


def pytest_addoption(parser):
    parser.addoption("--short", action="store_true", default=False,
                     help="Skip long tests")


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors")


def check_src_hashes(fn):
    """Decorator that turns on PyOP2 option to check for source hashes.

    Used in parallel tests."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from firedrake.parameters import parameters
        val = parameters["pyop2_options"]["check_src_hashes"]
        try:
            parameters["pyop2_options"]["check_src_hashes"] = True
            return fn(*args, **kwargs)
        finally:
            parameters["pyop2_options"]["check_src_hashes"] = val
    return wrapper


def pytest_runtest_setup(item):
    from mpi4py import MPI
    if item.get_marker("parallel"):
        if MPI.COMM_WORLD.size > 1:
            # Ensure source hash checking is enabled.
            item.obj = check_src_hashes(item.obj)
        else:
            # Blow away function arg in "master" process, to ensure
            # this test isn't run on only one process.
            item.obj = lambda *args, **kwargs: True


def pytest_runtest_call(item):
    from mpi4py import MPI
    if item.get_marker("parallel") and MPI.COMM_WORLD.size == 1:
        # Spawn parallel processes to run test
        parallel(item)


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
