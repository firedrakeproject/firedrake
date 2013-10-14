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
                check_call(['mpiexec', '-n', '%d' % nprocs, executable,
                            '-c', 'import %s; %s.%s()' %
                            (fn.__module__, fn.__module__, fn.__name__)],
                           cwd=os.path.abspath(os.path.dirname(__file__)))
        return wrapper
    return _parallel_test
