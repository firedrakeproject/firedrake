from __future__ import absolute_import
import copy
import ufl

from pyop2 import op2
from pyop2.utils import as_tuple, flatten


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
        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)
        self._bcs = [bc for bc in bcs] if bcs is not None else []
        self._bcs_at_point_of_assembly = []

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
                import firedrake.assemble as assemble
                assemble._assemble(self.a, tensor=self, bcs=self.bcs)
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
        return self._bcs != []

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
        self._bcs = []
        if bcs is not None:
            try:
                for bc in bcs:
                    self._bcs.append(bc)
            except TypeError:
                # BC instance, not iterable
                self._bcs.append(bcs)

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
        old_subdomains = set(flatten(as_tuple(bc.sub_domain)
                             for bc in self._bcs_at_point_of_assembly))
        new_subdomains = set(flatten(as_tuple(bc.sub_domain)
                             for bc in self.bcs))
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
        new_bcs = [bc]
        for existing_bc in self.bcs:
            # New BC doesn't override existing one, so keep it.
            if bc.sub_domain != existing_bc.sub_domain:
                new_bcs.append(existing_bc)
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
        import firedrake.assemble as assemble
        return assemble._assemble(self._a_action)

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
