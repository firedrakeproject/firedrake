import copy
import numpy as np
from collections import defaultdict
from mpi4py import MPI

import ufl

from pyop2 import op2

import utils
from solving import _assemble


class Halo(object):
    """Build a Halo associated with the appropriate FunctionSpace.

    The Halo is derived from a PetscSF object and builds the global
    to universal numbering map from the respective PetscSections."""

    def __init__(self, petscsf, global_numbering, universal_numbering):
        self._tag = utils._new_uid()
        self._comm = op2.MPI.comm
        self._nprocs = self.comm.size
        self._sends = defaultdict(list)
        self._receives = defaultdict(list)
        self._gnn2unn = None
        remote_sends = defaultdict(list)

        if op2.MPI.comm.size <= 1:
            return

        # Sort the SF by local indices
        nroots, nleaves, local, remote = petscsf.getGraph()
        local_new, remote_new = (list(x) for x in zip(*sorted(zip(local, remote), key=lambda x: x[0])))
        petscsf.setGraph(nroots, nleaves, local_new, remote_new)

        # Derive local receives and according remote sends
        nroots, nleaves, local, remote = petscsf.getGraph()
        for local, (rank, index) in zip(local, remote):
            if rank != self.comm.rank:
                self._receives[rank].append(local)
                remote_sends[rank].append(index)

        # Propagate remote send lists to the actual sender
        send_reqs = []
        for p in range(self._nprocs):
            # send sizes
            if p != self._comm.rank:
                s = np.array(len(remote_sends[p]), dtype=np.int32)
                send_reqs.append(self.comm.Isend(s, dest=p, tag=self.tag))

        recv_reqs = []
        sizes = [np.empty(1, dtype=np.int32) for _ in range(self._nprocs)]
        for p in range(self._nprocs):
            # receive sizes
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(sizes[p], source=p, tag=self.tag))

        MPI.Request.Waitall(recv_reqs)
        MPI.Request.Waitall(send_reqs)

        for p in range(self._nprocs):
            # allocate buffers
            if p != self._comm.rank:
                self._sends[p] = np.empty(sizes[p], dtype=np.int32)

        send_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                send_buf = np.array(remote_sends[p], dtype=np.int32)
                send_reqs.append(self.comm.Isend(send_buf, dest=p, tag=self.tag))

        recv_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(self._sends[p], source=p, tag=self.tag))

        MPI.Request.Waitall(send_reqs)
        MPI.Request.Waitall(recv_reqs)

        # Build Global-To-Universal mapping
        pStart, pEnd = global_numbering.getChart()
        self._gnn2unn = np.zeros(global_numbering.getStorageSize(), dtype=np.int32)
        for p in range(pStart, pEnd):
            dof = global_numbering.getDof(p)
            goff = global_numbering.getOffset(p)
            uoff = universal_numbering.getOffset(p)
            if uoff < 0:
                uoff = (-1*uoff)-1
            for c in range(dof):
                self._gnn2unn[goff+c] = uoff+c

    @utils.cached_property
    def op2_halo(self):
        if not self.sends and not self.receives:
            return None
        return op2.Halo(self.sends, self.receives,
                        comm=self.comm, gnn2unn=self.gnn2unn)

    @property
    def comm(self):
        return self._comm

    @property
    def tag(self):
        return self._tag

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def sends(self):
        return self._sends

    @property
    def receives(self):
        return self._receives

    @property
    def gnn2unn(self):
        return self._gnn2unn


class Matrix(object):
    """A representation of an assembled bilinear form.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.


    A :class:`pyop2.Mat` will be built from the remaining
    arguments, for valid values, see :class:`pyop2.Mat`.

    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """

    def __init__(self, a, bcs, *args, **kwargs):
        self._a = a
        self._M = op2.Mat(*args, **kwargs)
        self._thunk = None
        self._assembled = False
        self._bcs = set()
        self._bcs_at_point_of_assembly = None
        if bcs is not None:
            for bc in bcs:
                self._bcs.add(bc)

    def assemble(self):
        """Actually assemble this :class:`Matrix`.

        This calls the stashed assembly callback or does nothing if
        the matrix is already assembled.

        .. note::

            If the boundary conditions stashed on the :class:`Matrix` have
            changed since the last time it was assembled, this will
            necessitate reassembly.  So for example:

            .. code-block:: python

                A = assemble(a, bcs=[bc1])
                solve(A, x, b)
                bc2.apply(A)
                solve(A, x, b)

            will apply boundary conditions from `bc1` in the first
            solve, but both `bc1` and `bc2` in the second solve.
        """
        if self._assembly_callback is None:
            raise RuntimeError('Trying to assemble a Matrix, but no thunk found')
        if self._assembled:
            if self._needs_reassembly:
                _assemble(self.a, tensor=self, bcs=self.bcs)
                return self.assemble()
            return
        self._bcs_at_point_of_assembly = copy.copy(self.bcs)
        self._assembly_callback(self.bcs)
        self._assembled = True

    @property
    def _assembly_callback(self):
        """Return the callback for assembling this :class:`Matrix`."""
        return self._thunk

    @_assembly_callback.setter
    def _assembly_callback(self, thunk):
        """Set the callback for assembling this :class:`Matrix`.

        :arg thunk: the callback, this should take one argument, the
            boundary conditions to apply (pass None for no boundary
            conditions).

        Assigning to this property sets the :attr:`assembled` property
        to False, necessitating a re-assembly."""
        self._thunk = thunk
        self._assembled = False

    @property
    def assembled(self):
        """Return True if this :class:`Matrix` has been assembled."""
        return self._assembled

    @property
    def has_bcs(self):
        """Return True if this :class:`Matrix` has any boundary
        conditions attached to it."""
        return self._bcs != set()

    @property
    def bcs(self):
        """The set of boundary conditions attached to this
        :class:`Matrix` (may be empty)."""
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        """Attach some boundary conditions to this :class:`Matrix`.

        :arg bcs: a boundary condition (of type
            :class:`.DirichletBC`), or an iterable of boundary
            conditions.  If bcs is None, erase all boundary conditions
            on the :class:`Matrix`.

        """
        if bcs is None:
            self._bcs = set()
            return
        try:
            self._bcs = set(bcs)
        except TypeError:
            # BC instance, not iterable
            self._bcs = set([bcs])

    @property
    def a(self):
        """The bilinear form this :class:`Matrix` was assembled from"""
        return self._a

    @property
    def M(self):
        """The :class:`pyop2.Mat` representing the assembled form

        .. note ::

            This property forces an actual assembly of the form, if you
            just need a handle on the :class:`pyop2.Mat` object it's
            wrapping, use :attr:`_M` instead."""
        self.assemble()
        # User wants to see it, so force the evaluation.
        self._M._force_evaluation()
        return self._M

    @property
    def _needs_reassembly(self):
        """Does this :class:`Matrix` need reassembly.

        The :class:`Matrix` needs reassembling if the subdomains over
        which boundary conditions were applied the last time it was
        assembled are different from the subdomains of the current set
        of boundary conditions.
        """
        old_subdomains = set([bc.sub_domain for bc in self._bcs_at_point_of_assembly])
        new_subdomains = set([bc.sub_domain for bc in self.bcs])
        return old_subdomains != new_subdomains

    def add_bc(self, bc):
        """Add a boundary condition to this :class:`Matrix`.

        :arg bc: the :class:`.DirichletBC` to add.

        If the subdomain this boundary condition is applied over is
        the same as the subdomain of an existing boundary condition on
        the :class:`Matrix`, the existing boundary condition is
        replaced with this new one.  Otherwise, this boundary
        condition is added to the set of boundary conditions on the
        :class:`Matrix`.

        """
        new_bcs = set([bc])
        for existing_bc in self.bcs:
            # New BC doesn't override existing one, so keep it.
            if bc.sub_domain != existing_bc.sub_domain:
                new_bcs.add(existing_bc)
        self.bcs = new_bcs

    def _form_action(self, u):
        """Assemble the form action of this :class:`Matrix`' bilinear form
        onto the :class:`Function` ``u``.
        .. note::
            This is the form **without** any boundary conditions."""
        if not hasattr(self, '_a_action'):
            self._a_action = ufl.action(self._a, u)
        if hasattr(self, '_a_action_coeff'):
            self._a_action = ufl.replace(self._a_action, {self._a_action_coeff: u})
        self._a_action_coeff = u
        # Since we assemble the cached form, the kernels will already have
        # been compiled and stashed on the form the second time round
        return _assemble(self._a_action)

    def __repr__(self):
        return '%sassembled firedrake.Matrix(form=%r, bcs=%r)' % \
            ('' if self._assembled else 'un',
             self.a,
             self.bcs)

    def __str__(self):
        return '%sassembled firedrake.Matrix(form=%s, bcs=%s)' % \
            ('' if self._assembled else 'un',
             self.a,
             self.bcs)
