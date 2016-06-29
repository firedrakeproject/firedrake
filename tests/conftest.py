"""Global test configuration."""

import pytest
import os
from subprocess import check_call
from mpi4py import MPI


def parallel(item):
    """Run a test in parallel.

    :arg item: The test item to run.
    """
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
    config.addinivalue_line(
        "markers",
        "longtest: mark that the test is 'long' (skipped if --short is passed)")


def pytest_runtest_setup(item):
    if item.get_marker("longtest") is not None:
        if item.config.getoption("--short"):
            pytest.skip("Skipping long test")
    if item.get_marker("parallel"):
        if MPI.COMM_WORLD.size > 1:
            # Turn on source hash checking
            from firedrake import parameters
            from functools import partial

            def _reset(check):
                parameters["pyop2_options"]["check_src_hashes"] = check

            # Reset to current value when test is cleaned up
            item.addfinalizer(partial(_reset,
                                      parameters["pyop2_options"]["check_src_hashes"]))

            parameters["pyop2_options"]["check_src_hashes"] = True
        else:
            # Blow away function arg in "master" process, to ensure
            # this test isn't run on only one process.
            item.obj = lambda *args, **kwargs: True


def pytest_runtest_call(item):
    if item.get_marker("parallel") and MPI.COMM_WORLD.size == 1:
        # Spawn parallel processes to run test
        parallel(item)
