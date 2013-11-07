from pyop2 import op2
from solving import _assemble
import copy


class Matrix(object):
    """A representation of an assembled bilinear form.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.


    A :class:`pyop2.op2.Mat` will be built from the remaining
    arguments, for valid values, see :class:`pyop2.op2.Mat`.

    .. note::

        This object acts to the right on an assembled :class:`Function`
        and to the left on an assembled co-Function (currently represented
        by a :class:`Function`).

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
            if self._bcs_at_point_of_assembly == self.bcs:
                return
            _assemble(self.a, tensor=self, bcs=self.bcs)
            return self.assemble()
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

    @assembled.setter
    def assembled(self, val):
        """Set the assembled state of this :class:`Matrix`.

        :arg val: the state (True or False)."""
        self._assembled = val

    @property
    def has_bcs(self):
        """Return True if this :class:`Matrix` has any boundary
        conditions attached to it."""
        return self._bcs != set()

    @property
    def bcs(self):
        """The boundary conditions attached to this :class:`Matrix`,
        or None."""
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        """Attach some boundary conditions to this :class:`Matrix`.

        :arg bcs: a boundary condition, or an iterable of boundary conditions.
        """
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
        """The :class:`pyop2.op2.Mat` representing the assembled form

        .. note ::

            This property forces an actual assembly of the form, if you
            just need a handle on the :class:`pyop2.op2.Mat` object it's
            wrapping, use :attr:`_M` instead."""
        self.assemble()
        return self._M
