import abc

from pyop2 import op2
from pyop2.utils import as_tuple, flatten
from firedrake import utils
from firedrake.petsc import PETSc


class MatrixBase(object, metaclass=abc.ABCMeta):
    """A representation of the linear operator associated with a
    bilinear form and bcs.  Explicitly assembled matrices and matrix-free
    matrix classes will derive from this

    :arg a: the bilinear form this :class:`MatrixBase` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`MatrixBase`.  May be `None` if there are no boundary
        conditions to apply.
    """
    def __init__(self, a, bcs):
        self._a = a

        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)
        self._bcs = [bc for bc in bcs] if bcs is not None else []
        self._bcs_at_point_of_assembly = []
        test, trial = a.arguments()
        self.comm = test.function_space().comm
        self.block_shape = (len(test.function_space()),
                            len(trial.function_space()))

    @abc.abstractmethod
    def assemble(self):
        """Actually assemble this matrix.

        Ensures any pending calculations needed to populate this
        matrix are queued up.

        Note that this does not guarantee that those calculations are
        executed.  If you want the latter, see :meth:`force_evaluation`.
        """
        self._bcs_at_point_of_assembly = list(self._bcs)

    @abc.abstractmethod
    def force_evaluation(self):
        """Force any pending writes to this matrix.

        Ensures that the matrix is assembled and populated with
        values, ready for sending to PETSc."""
        pass

    @property
    def has_bcs(self):
        """Return True if this :class:`MatrixBase` has any boundary
        conditions attached to it."""
        return self._bcs != []

    @property
    def bcs(self):
        """The set of boundary conditions attached to this
        :class:`.MatrixBase` (may be empty)."""
        return self._bcs

    @bcs.setter
    def bcs(self, bcs):
        """Attach some boundary conditions to this :class:`MatrixBase`.

        :arg bcs: a boundary condition (of type
            :class:`.DirichletBC`), or an iterable of boundary
            conditions.  If bcs is None, erase all boundary conditions
            on the :class:`.MatrixBase`.
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
        """The bilinear form this :class:`.MatrixBase` was assembled from"""
        return self._a

    def add_bc(self, bc):
        """Add a boundary condition to this :class:`MatrixBase`.

        :arg bc: the :class:`.DirichletBC` to add.

        If the subdomain this boundary condition is applied over is
        the same as the subdomain of an existing boundary condition on
        the :class:`MatrixBase`, the existing boundary condition is
        replaced with this new one.  Otherwise, this boundary
        condition is added to the set of boundary conditions on the
        :class:`MatrixBase`.
        """
        new_bcs = [bc]
        for existing_bc in self._bcs:
            # New BC doesn't override existing one, so keep it.
            if bc.sub_domain != existing_bc.sub_domain:
                new_bcs.append(existing_bc)
        self._bcs = new_bcs

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
                             for bc in self._bcs))
        return old_subdomains != new_subdomains

    def __repr__(self):
        return "%s(a=%r, bcs=%r)" % (type(self).__name__,
                                     self.a,
                                     self.bcs)

    def __str__(self):
        pfx = "" if self.assembled else "un"
        return "%sassembled %s(a=%s, bcs=%s)" % (pfx, type(self).__name__,
                                                 self.a, self.bcs)


class Matrix(MatrixBase):
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
        # sets self._a and self._bcs
        super(Matrix, self).__init__(a, bcs)
        options_prefix = kwargs.pop("options_prefix")
        self._M = op2.Mat(*args, **kwargs)
        self.petscmat = self._M.handle
        self.petscmat.setOptionsPrefix(options_prefix)
        self._thunk = None
        self.assembled = False

    @utils.known_pyop2_safe
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
            self.assembled = True
            return
        if self.assembled:
            if self._needs_reassembly:
                from firedrake.assemble import _assemble
                _assemble(self.a, tensor=self, bcs=self.bcs)
                return self.assemble()
            return
        self._assembly_callback(self.bcs)
        self.assembled = True
        super().assemble()

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
        self.assembled = False

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

    def force_evaluation(self):
        "Ensures that the matrix is fully assembled."
        self.assemble()
        self._M._force_evaluation()


class ImplicitMatrix(MatrixBase):
    """A representation of the action of bilinear form operating
    without explicitly assembling the associated matrix.  This class
    wraps the relevant information for Python PETSc matrix.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.


    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """
    def __init__(self, a, bcs, *args, **kwargs):
        # sets self._a and self._bcs
        super(ImplicitMatrix, self).__init__(a, bcs)

        options_prefix = kwargs.pop("options_prefix")
        appctx = kwargs.get("appctx", {})

        from firedrake.matrix_free.operators import ImplicitMatrixContext
        ctx = ImplicitMatrixContext(a,
                                    row_bcs=self.bcs,
                                    col_bcs=self.bcs,
                                    fc_params=kwargs["fc_params"],
                                    appctx=appctx)
        self.petscmat = PETSc.Mat().create(comm=self.comm)
        self.petscmat.setType("python")
        self.petscmat.setSizes((ctx.row_sizes, ctx.col_sizes),
                               bsize=ctx.block_size)
        self.petscmat.setPythonContext(ctx)
        self.petscmat.setOptionsPrefix(options_prefix)
        self.petscmat.setUp()
        self.petscmat.assemble()
        self.assembled = False

    def assemble(self):
        # Bump petsc matrix state by assembling it.
        # Ensures that if the matrix changed, the preconditioner is
        # updated if necessary.
        if self._needs_reassembly:
            ctx = self.petscmat.getPythonContext()
            ctx.row_bcs = self.bcs
            ctx.col_bcs = self.bcs
        self.petscmat.assemble()
        self.assembled = True
        super().assemble()

    force_evaluation = assemble
