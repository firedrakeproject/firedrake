"""A module providing progress bars."""
from pyop2.mpi import COMM_WORLD
from progress.bar import FillingSquaresBar


class _NullProgressBar:
    """A placeholder class with the same interface as a progress bar."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass

    def iter(self, iterator):
        return iterator


class ProgressBar(FillingSquaresBar):
    """A progress bar for simulation execution.

    This is a subclass of ``progress.bar.FillingSquaresBar`` which is
    configured to be suitable for tracking progress in forward and adjoint
    simulations. It is also extended to only output on rank 0 in parallel.

    Parameters
    ----------

    message : str
        An identifying string to be prepended to the progress bar. This
        defaults to an empty string.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator over which the simulation is run. Defaults to
        `COMM_WORLD`

    Notes
    -----

    Further parameters can be passed as per the `progress package documentation
    <https://github.com/verigak/progress>`_, or you can customise
    further by subclassing.

    Examples
    --------

    To apply a progress bar to a loop, wrap the loop iterator in the
    ``iter`` method of a ``ProgressBar``:

    >>> for t in ProgressBar("Timestep").iter(np.linspace(0.0, 1.0, 10)):
    ...    sleep(0.2)
    ...
    Timestep ▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣▣ 10/10 [0:00:02]

    To see progress bars for functional, adjoint and Hessian evaluations in an
    adjoint simulation, set the ``progress_bar`` attribute of the tape to
    `ProgressBar`:

    >>> get_working_tape().progress_bar = ProgressBar

    This use case is covered in the documentation for :class:`pyadjoint.Tape`.
    """

    def __new__(cls, *args, comm=COMM_WORLD, **kwargs):

        # Only print a progress bar on rank 0.
        if comm.rank == 0:
            return super().__new__(cls)
        else:
            return _NullProgressBar()

    # Sensible default suffix for timestepping or adjoint loops.
    suffix = "%(index)s/%(max)s [%(elapsed_td)s]"
    # Required in order to have output in parallel.
    check_tty = False
    width = 50
