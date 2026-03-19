from functools import wraps
import weakref
from contextlib import contextmanager
from itertools import zip_longest
from types import SimpleNamespace

from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from pyop2.mpi import MPI, internal_comm


def _ensemble_mpi_dispatch(func):
    """
    This wrapper checks if any arg or kwarg of the wrapped
    ensemble method is a Function or Cofunction, and if so
    it calls the specialised Firedrake implementation.
    Otherwise the standard mpi4py implementation is called.
    """
    @wraps(func)
    def _mpi_dispatch(self, *args, **kwargs):
        if any(isinstance(arg, (Function, Cofunction))
               for arg in [*args, *kwargs.values()]):
            return func(self, *args, **kwargs)
        else:
            mpicall = getattr(self._ensemble_comm, func.__name__)
            return mpicall(*args, **kwargs)
    return _mpi_dispatch


class Ensemble:
    def __init__(self, comm: MPI.Comm, M: int, **kwargs):
        """
        Create a set of space and ensemble subcommunicators.

        Wrapper methods around many MPI communication functions are
        provided for sending :class:`.Function` and :class:`.Cofunction`
        objects between spatial communicators.

        For non-Firedrake objects these wrappers will dispatch to the
        normal implementations on :class:`mpi4py.MPI.Comm`, which means
        that the same call site can be used for both Firedrake and
        non-Firedrake types.

        Parameters
        ----------
        comm :
            The communicator to split.
        M :
            The size of the communicators used for spatial parallelism.
            Must be an integer divisor of the size of ``comm``.
        kwargs :
            Can include an ``ensemble_name`` string used as a communicator
            name prefix, for debugging.

        Raises
        ------
        ValueError
            If ``M`` does not divide ``comm.size`` exactly.
        """
        size = comm.size

        if (size // M)*M != size:
            raise ValueError("Invalid size of subcommunicators %d does not divide %d" % (M, size))

        rank = comm.rank

        # Global comm
        self.global_comm = comm

        ensemble_name = kwargs.get("ensemble_name", "Ensemble")
        # User and internal communicator for spatial parallelism, contains a
        # contiguous chunk of M processes from `global_comm`.
        self.comm = self.global_comm.Split(color=(rank // M), key=rank)
        self.comm.name = f"{ensemble_name} spatial comm"
        weakref.finalize(self, self.comm.Free)

        # User and internal communicator for ensemble parallelism, contains all
        # processes in `global_comm` which have the same rank in `comm`.
        self.ensemble_comm = self.global_comm.Split(color=(rank % M), key=rank)
        self.ensemble_comm.name = f"{ensemble_name} ensemble comm"
        weakref.finalize(self, self.ensemble_comm.Free)
        # Keep a reference to the internal communicator because some methods return
        # non-blocking requests and we need to avoid cleaning up the communicator before
        # they complete. Note that this communicator should *never* be passed to PETSc, as
        # objects created with the communicator will never get cleaned up.
        self._ensemble_comm = internal_comm(self.ensemble_comm, self)

        if (self.comm.size != M) or (self.ensemble_comm.size != (size // M)):
            raise ValueError(f"{M=} does not exactly divide {comm.size=}")

    @property
    def ensemble_size(self) -> int:
        """The number of ensemble members.
        """
        return self.ensemble_comm.size

    @property
    def ensemble_rank(self) -> int:
        """The rank of the local ensemble member.
        """
        return self.ensemble_comm.rank

    def _check_function(self, f: Function | Cofunction,
                        g: Function | Cofunction | None = None):
        """
        Check if :class:`.Function` ``f`` (and possibly a second
        :class:`.Function` ``g``) is a valid argument for ensemble MPI routines

        Parameters
        ----------
        f :
            The :class:`.Function` to check.
        g :
            Second :class:`.Function` to check.

        Raises
        ------
        ValueError
            If ``Function`` communicators mismatch each other or the ensemble
            spatial communicator, or is the functions are in different spaces
        """
        if MPI.Comm.Compare(f.comm, self.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
            raise ValueError("Function communicator does not match space communicator")

        if g is not None:
            if MPI.Comm.Compare(f.comm, g.comm) not in {MPI.CONGRUENT, MPI.IDENT}:
                raise ValueError("Mismatching communicators for functions")
            if f.function_space() != g.function_space():
                raise ValueError("Mismatching function spaces for functions")

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def allreduce(self, f: Function | Cofunction,
                  f_reduced: Function | Cofunction | None = None,
                  op: MPI.Op = MPI.SUM
                  ) -> Function | Cofunction:
        """
        Allreduce a :class:`.Function` ``f`` into ``f_reduced``.

        Parameters
        ----------
        f :
            The :class:`.Function` to allreduce.
        f_reduced :
            The result of the reduction. Must be in the same
            :func:`~firedrake.functionspace.FunctionSpace` as ``f``.
        op :
            MPI reduction operator. Defaults to MPI.SUM.

        Returns
        -------
        Function | Cofunction :
            The result of the reduction.

        Raises
        ------
        ValueError
            If Function communicators mismatch each other or the ensemble
            spatial communicator, or if the Functions are in different spaces
        """
        f_reduced = f_reduced or Function(f.function_space())
        self._check_function(f, f_reduced)

        with f_reduced.dat.vec_wo as vout, f.dat.vec_ro as vin:
            self._ensemble_comm.Allreduce(vin.array_r, vout.array, op=op)
        return f_reduced

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def iallreduce(self, f: Function | Cofunction,
                   f_reduced: Function | Cofunction | None = None,
                   op: MPI.Op = MPI.SUM
                   ) -> list[MPI.Request]:
        """
        Allreduce (non-blocking) a :class:`.Function` ``f`` into ``f_reduced``.

        Parameters
        ----------
        f :
            The a :class:`.Function` to allreduce.
        f_reduced :
            The result of the reduction. Must be in the same
            :func:`~firedrake.functionspace.FunctionSpace` as ``f``.
        op :
            MPI reduction operator. Defaults to MPI.SUM.

        Returns
        -------
        list[mpi4py.MPI.Request] :
            Requests one for each of ``f.subfunctions``.

        Raises
        ------
        ValueError
            If Function communicators mismatch each other or the ensemble
            spatial communicator, or if the Functions are in different spaces
        """
        f_reduced = f_reduced or Function(f.function_space())
        self._check_function(f, f_reduced)

        return [self._ensemble_comm.Iallreduce(fdat.data, rdat.data, op=op)
                for fdat, rdat in zip(f.dat, f_reduced.dat)]

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def reduce(self, f: Function | Cofunction,
               f_reduced: Function | Cofunction | None = None,
               op: MPI.Op = MPI.SUM, root: int = 0
               ) -> Function | Cofunction:
        """
        Reduce a :class:`.Function` ``f`` into ``f_reduced``.

        Parameters
        ----------
        f :
            The :class:`.Function` to reduce.
        f_reduced :
            The result of the reduction. Must be in the same
            :func:`~firedrake.functionspace.FunctionSpace` as ``f``.
        op :
            MPI reduction operator. Defaults to MPI.SUM.
        root :
            The ensemble rank to reduce to.

        Returns
        -------
        Function | Cofunction :
            The result of the reduction.

        Raises
        ------
        ValueError
            If Function communicators mismatch each other or the ensemble
            spatial communicator, or if the Functions are in different spaces
        """
        f_reduced = f_reduced or Function(f.function_space())
        self._check_function(f, f_reduced)

        if self.ensemble_comm.rank == root:
            with f_reduced.dat.vec_wo as vout, f.dat.vec_ro as vin:
                self._ensemble_comm.Reduce(vin.array_r, vout.array, op=op, root=root)
        else:
            with f.dat.vec_ro as vin:
                self._ensemble_comm.Reduce(vin.array_r, None, op=op, root=root)

        return f_reduced

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def ireduce(self, f: Function | Cofunction,
                f_reduced: Function | Cofunction | None = None,
                op: MPI.Op = MPI.SUM, root: int = 0
                ) -> list[MPI.Request]:
        """
        Reduce (non-blocking) a :class:`.Function` ``f`` into ``f_reduced``.

        Parameters
        ----------
        f :
            The a :class:`.Function` to reduce.
        f_reduced :
            The result of the reduction. Must be in the same
            :func:`~firedrake.functionspace.FunctionSpace` as ``f``.
        op :
            MPI reduction operator. Defaults to MPI.SUM.
        root :
            The ensemble rank to reduce to.

        Returns
        -------
        list[mpi4py.MPI.Request]
            Requests one for each of ``f.subfunctions``.

        Raises
        ------
        ValueError
            If Function communicators mismatch each other or the ensemble
            spatial communicator, or if the Functions are in different spaces
        """
        f_reduced = f_reduced or Function(f.function_space())
        self._check_function(f, f_reduced)

        return [self._ensemble_comm.Ireduce(fdat.data_ro, rdat.data, op=op, root=root)
                for fdat, rdat in zip(f.dat, f_reduced.dat)]

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def bcast(self, f: Function | Cofunction, root: int = 0
              ) -> Function | Cofunction:
        """
        Broadcast a :class:`.Function` ``f`` over ``ensemble_comm``
        from :attr:`~.Ensemble.ensemble_rank` ``root``.

        Parameters
        ----------
        f :
            The :class:`.Function` to broadcast.
        root :
            The rank to broadcast from.

        Returns
        -------
        Function | Cofunction :
            The result of the broadcast.

        Raises
        ------
        ValueError
            If the Function communicator mismatches the ``ensemble.comm``.
        """
        self._check_function(f)
        with f.dat.vec as vec:
            self._ensemble_comm.Bcast(vec.array, root=root)

        return f

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def ibcast(self, f: Function | Cofunction, root: int = 0
               ) -> list[MPI.Request]:
        """
        Broadcast (non-blocking) a :class:`.Function` ``f`` over
        ``ensemble_comm`` :attr:`~.Ensemble.ensemble_rank` ``root``.

        Parameters
        ----------
        f :
            The :class:`.Function` to broadcast.
        root :
            The rank to broadcast from.

        Returns
        -------
        list[mpi4py.MPI.Request]
            Requests one for each of ``f.subfunctions``.

        Raises
        ------
        ValueError
            If the Function communicator mismatches the ``ensemble.comm``.
        """
        self._check_function(f)
        return [self._ensemble_comm.Ibcast(dat.data, root=root)
                for dat in f.dat]

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def send(self, f: Function | Cofunction, dest: int, tag: int = 0):
        """
        Send (blocking) a :class:`.Function` ``f`` over ``ensemble_comm``
        to another :attr:`~.Ensemble.ensemble_rank`.

        Parameters
        ----------
        f :
            The a :class:`.Function` to send.
        dest :
            The :attr:`~.Ensemble.ensemble_rank` to send ``f`` to.
        tag :
            The tag of the message.

        Raises
        ------
        ValueError
            If the Function communicator mismatches the ``ensemble.comm``.
        """
        self._check_function(f)
        for dat in f.dat:
            self._ensemble_comm.Send(dat.data_ro, dest=dest, tag=tag)

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def recv(self, f: Function | Cofunction, source: int = MPI.ANY_SOURCE,
             tag: int = MPI.ANY_TAG, statuses: list[MPI.Status] | MPI.Status = None,
             ) -> Function | Cofunction:
        """
        Receive (blocking) a :class:`.Function` ``f`` over
        ``ensemble_comm`` from another :attr:`~.Ensemble.ensemble_rank`.

        Parameters
        ----------
        f :
            The :class:`.Function` to receive into.
        source :
            The :attr:`~.Ensemble.ensemble_rank` to receive ``f`` from.
        tag :
            The tag of the message.
        statuses :
            The :class:`mpi4py.MPI.Status` of the internal recv calls
            (one for each of the ``subfunctions`` of ``f``).

        Returns
        -------
        Function | Cofunction :
            ``f`` with the received data.

        Raises
        ------
        ValueError
            If the Function communicator mismatches the ``ensemble.comm``.
        ValueError
            If the number of ``statuses`` provided is not the number of
            subfunctions of ``f``.
        """
        self._check_function(f)
        if statuses is not None and isinstance(statuses, MPI.Status):
            statuses = [statuses]
        if statuses is not None and len(statuses) != len(f.dat):
            raise ValueError("Need to provide enough status objects for all parts of the Function")
        for dat, status in zip_longest(f.dat, statuses or (), fillvalue=None):
            self._ensemble_comm.Recv(dat.data, source=source, tag=tag, status=status)
        return f

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def isend(self, f: Function | Cofunction, dest: int, tag: int = 0
              ) -> list[MPI.Request]:
        """
        Send (non-blocking) a :class:`.Function` ``f`` over ``ensemble_comm``
        to another :attr:`~.Ensemble.ensemble_rank`.

        Parameters
        ----------
        f :
            The a :class:`.Function` to send.
        dest :
            The :attr:`~.Ensemble.ensemble_rank` to send ``f`` to.
        tag :
            The tag of the message.

        Returns
        -------
        list[mpi4py.MPI.Request]
            Requests one for each of ``f.subfunctions``.

        Raises
        ------
        ValueError
            If the Function communicator mismatches the ``ensemble.comm``.
        """
        self._check_function(f)
        return [self._ensemble_comm.Isend(dat.data_ro, dest=dest, tag=tag)
                for dat in f.dat]

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def irecv(self, f: Function | Cofunction,
              source: int = MPI.ANY_SOURCE,
              tag: int = MPI.ANY_TAG
              ) -> list[MPI.Request]:
        """
        Receive (non-blocking) a :class:`.Function` ``f`` over
        ``ensemble_comm`` from another :attr:`~.Ensemble.ensemble_rank`.

        Parameters
        ----------
        f :
            The :class:`.Function` to receive into.
        source :
            The :attr:`~.Ensemble.ensemble_rank` to receive ``f`` from.
        tag :
            The tag of the message.

        Returns
        -------
        list[mpi4py.MPI.Request]
            Requests one for each of ``f.subfunctions``.

        Raises
        ------
        ValueError
            If the Function communicator mismatches the ``ensemble.comm``.
        """
        self._check_function(f)
        return [self._ensemble_comm.Irecv(dat.data, source=source, tag=tag)
                for dat in f.dat]

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def sendrecv(self, fsend: Function | Cofunction, dest: int, sendtag: int = 0,
                 frecv: Function | Cofunction | None = None, source: int = MPI.ANY_SOURCE,
                 recvtag: int = MPI.ANY_TAG, statuses: list[MPI.Status] | MPI.Status = None
                 ) -> Function | Cofunction:
        """
        Send (blocking) a :class:`.Function` ``fsend`` and receive a
        :class:`.Function` ``frecv`` over ``ensemble_comm`` to/from other
        :attr:`~.Ensemble.ensemble_rank`.

        ``fsend`` and ``frecv`` do not need to be in the same function space
        but do need to have the same number of subfunctions.

        Parameters
        ----------
        fsend :
            The a :class:`.Function` to send.
        dest :
            The :attr:`~.Ensemble.ensemble_rank` to send ``fsend`` to.
        sendtag :
            The tag of the send message.
        frecv :
            The :class:`.Function` to receive into.
        source :
            The :attr:`~.Ensemble.ensemble_rank` to receive ``frecv`` from.
        recvtag :
            The tag of the receive message.
        statuses :
            The :class:`mpi4py.MPI.Status` of the internal recv calls
            (one for each of the ``subfunctions`` of ``frecv``).

        Returns
        -------
        Function | Cofunction
            ``frecv`` with the received data.

        Raises
        ------
        ValueError
            If the Function communicators mismatches each other or the
            ``ensemble.comm``.
        ValueError
            If the number of ``statuses`` provided is not the number of
            subfunctions of ``f``.
        """
        frecv = frecv or Function(fsend.function_space())
        # functions don't necessarily have to match
        self._check_function(fsend)
        self._check_function(frecv)
        if statuses is not None and isinstance(statuses, MPI.Status):
            statuses = [statuses]
        if statuses is not None and len(statuses) != len(frecv.dat):
            raise ValueError("Need to provide enough status objects for all parts of the Function")
        with fsend.dat.vec_ro as sendvec, frecv.dat.vec_wo as recvvec:
            self._ensemble_comm.Sendrecv(sendvec, dest, sendtag=sendtag,
                                         recvbuf=recvvec, source=source, recvtag=recvtag,
                                         status=statuses)
        return frecv

    @PETSc.Log.EventDecorator()
    @_ensemble_mpi_dispatch
    def isendrecv(self, fsend: Function | Cofunction, dest: int, sendtag: int = 0,
                  frecv: Function | Cofunction | None = None,
                  source: int = MPI.ANY_SOURCE, recvtag: int = MPI.ANY_TAG
                  ) -> list[MPI.Request]:
        """
        Send (non-blocking) a :class:`.Function` ``fsend`` and receive a
        :class:`.Function` ``frecv`` over ``ensemble_comm`` to/from other
        :attr:`~.Ensemble.ensemble_rank`.

        ``fsend`` and ``frecv`` do not need to be in the same function space.

        Parameters
        ----------
        fsend :
            The a :class:`.Function` to send.
        dest :
            The :attr:`~.Ensemble.ensemble_rank` to send ``fsend`` to.
        sendtag :
            The tag of the send message.
        frecv :
            The :class:`.Function` to receive into.
        source :
            The :attr:`~.Ensemble.ensemble_rank` to receive ``frecv`` from.
        recvtag :
            The tag of the receive message.

        Returns
        -------
        list[mpi4py.MPI.Request]
            Requests one for each of ``f.subfunctions``.

        Raises
        ------
        ValueError
            If the Function communicators mismatches each other or the
            ``ensemble.comm``.
        """
        frecv = frecv or Function(fsend.function_space())
        # functions don't necessarily have to match
        self._check_function(fsend)
        self._check_function(frecv)

        requests = []
        requests.extend([self._ensemble_comm.Isend(dat.data_ro, dest=dest, tag=sendtag)
                         for dat in fsend.dat])
        requests.extend([self._ensemble_comm.Irecv(dat.data, source=source, tag=recvtag)
                         for dat in frecv.dat])
        return requests

    @contextmanager
    def sequential(self, *, synchronise: bool = False, reverse: bool = False, **kwargs):
        """
        Context manager for executing code on each ensemble
        member consecutively (ordered by increasing
        :attr:`~.Ensemble.ensemble_rank`).

        Any data in ``kwargs`` will be made available in the returned
        context and will be communicated forward after each ensemble
        member exits. :class:`.Function` or :class:`.Cofunction`
        ``kwargs`` will be sent with the corresponding Ensemble methods.

        For example:

        .. code-block:: python3

            with ensemble.sequential(index=0) as ctx:
                print(ensemble.ensemble_rank, ctx.index)
                ctx.index += 2

        Would print:

        .. code-block::

            0 0
            1 2
            2 4
            3 6
            ...

        If ``reverse is True`` then the ensemble ranks will be looped through
        in decreasing order i.e. ``ensemble_rank == (ensemble_size - 1)`` will
        run first, then ``ensemble_rank == (ensemble_size - 2)`` etc.

        Parameters
        ----------
        synchronise :
            If True then MPI_Barrier will be called on the ``global_comm``
            at the beginning and end of this method.

        reverse :
            If True then will iterate through spatial comms in order of
            decreasing ``ensemble_rank``.

        kwargs :
            Data to be passed forward by each rank and made available
            in the returned ``ctx``.
        """
        rank = self.ensemble_rank
        if reverse:  # send backwards
            src = rank + 1
            dst = rank - 1
            first_rank = (rank == self.ensemble_size - 1)
            last_rank = (rank == 0)
        else:  # send forwards
            src = rank - 1
            dst = rank + 1
            first_rank = (rank == 0)
            last_rank = (rank == self.ensemble_size - 1)

        if synchronise:
            self.global_comm.Barrier()

        if not first_rank:
            for i, (k, v) in enumerate(kwargs.items()):
                if isinstance(v, (Function, Cofunction)):
                    # Functions are sent in-place, everything else is pickled
                    recv_args = [kwargs[k]]
                else:
                    recv_args = []
                kwargs[k] = self.recv(*recv_args, source=src, tag=rank+i*100)

        ctx = SimpleNamespace(**kwargs)
        yield ctx

        if not last_rank:
            for i, v in enumerate((getattr(ctx, k)
                                   for k in kwargs.keys())):
                try:
                    self.send(v, dest=dst, tag=dst+i*100)
                except Exception as error:
                    raise TypeError(
                        "Failed to send object of type {type(v)__name__}. kwargs for"
                        " Ensemble.sequential must be Functions, Cofunctions,"
                        " or acceptable arguments to mpi4py.MPI.Comm.send."
                    ) from error

        if synchronise:
            self.global_comm.Barrier()
