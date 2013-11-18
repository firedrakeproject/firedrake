from pyop2 import op2
from solving import _assemble, assemble
from matrix_free_utils import _matrix_diagonal
from ufl_expr import action
import core_types
import copy
from petsc4py import PETSc


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


class MatrixFree(object):
    """A representation of an assembled bilinear form used for matrix free solves.

    This class implements petsc4py callbacks and can therefore be used
    for matrix free solves."""

    _count = 0

    def __init__(self, a, bcs=None):
        """
        :arg a: a bilinear form to represent
        :arg bcs: optional boundary conditions to apply when forming
            the matrix action.
        """
        self._a = a
        self._bcs = bcs
        M = MatrixFree.Mat(a, bcs)
        self._Mat = PETSc.Mat().create()
        self._Mat.setSizes([(M.nrows, None), (M.ncols, None)])
        self._Mat.setType('python')
        self._Mat.setPythonContext(M)
        self._Mat.setUp()
        self._PC = PETSc.PC().create()
        self._PC.setOperators(A=self._Mat, P=self._Mat,
                              structure=self._Mat.Structure.SAME_NZ)
        self._PC.setType('python')
        self._PC.setPythonContext(M)
        self._opt_prefix = 'firedrake_matrix_free_%d_' % MatrixFree._count
        MatrixFree._count += 1

    class Mat(object):
        """Class implementing matrix free operations

        These definitions need to be in a different class to
        :class:`MatrixFree` so that the latter does not contain
        circular references from PETSc objects back to itself
        (defeating the python garbage collector)."""
        def __init__(self, a, bcs=None):
            self._a = a
            self._bcs = bcs
            test, trial = a.compute_form_data().original_arguments
            self._test = test
            self._trial = trial
            self._nrows = test.function_space().dof_count
            self._ncols = trial.function_space().dof_count
            self._diagonal = None

        @property
        def nrows(self):
            return self._nrows

        @property
        def ncols(self):
            return self._ncols

        @property
        def diagonal(self):
            """Return the diagonal of the matrix."""
            if self._diagonal is None:
                self._diagonal = _matrix_diagonal(self._a, bcs=self._bcs)
            return self._diagonal

        def mult(self, A, x, y):
            """Compute the action of A on x.

            :arg A: a :class:`PETSc.Mat` (ignored).
            :arg x: a :class:`PETSc.Vec` to be multiplied by A.
            :arg y: a :class:`PETSc.Vec` to place the result in.

            .. note::
                `A` is ignored because the information used to construct
                the matrix-vector multiply lives in :attr:`_a`.
            """
            xx = core_types.Function(self._trial.function_space(), val=x.array)
            yy = core_types.Function(self._test.function_space(), val=y.array)
            assemble(action(self._a, xx, bcs=self._bcs), tensor=yy)
            yy.dat._force_evaluation(read=True, write=False)

        def getDiagonal(self, A, D):
            """Compute the diagonal of A and place it in D.
            :arg A: a :class:`PETSc.Mat` (ignored).
            :arg D: a :class:`PETSc.Vec` to place the result in.

            .. note::
                `A` is ignored because the information used to construct
                `D` lives in :attr:`_a`, see also :attr:`diagonal`.
            """
            D.array = self.diagonal.dat.data_ro

        def apply(self, PC, r, y):
            """Apply Jacobi preconditioning to residual r, writing the
            result into y.

            :arg PC: a :class:`PETSc.PC` (ignored).
            :arg r: a :class:`PETSc.Vec` containing the unpreconditioned
                residual.
            :arg y: a :class:`PETSc.Vec` to write the preconditioned
                residual into.

            .. note::
                `PC` is ignored because the information to construct the
                preconditioner lives elsewhere.  Specifically, the
                diagonal can be obtained by accessing the :attr:`diagonal`
                property."""
            rr = core_types.Function(self._trial.function_space(), val=r.array)
            yy = core_types.Function(self._test.function_space(), val=y.array)

            # r = b - Ax
            # Jacobi iteration is:
            # x_new = x_old + diag(A)^{-1} (b - A x_old)
            # So if we use Richardson iterations:
            # x_new = x_old + scale_factor * P (b - A x_old)
            # where P is the application of the preconditioner, then
            # Jacobi iterations just require dividing through by the diagonal.
            yy.assign(rr / self.diagonal)
            yy.dat._force_evaluation(read=True, write=False)

    def solve(self, L, x, solver_parameters=None):
        """Solve a == L placing the result in x.

        :arg L: a linear :class:`ufl.Form` defining the right hand
            side.
        :arg x: a :class:`Function` to place the result in.

        The bilinear form defining the left hand side is in
        :attr:`_a`."""
        ksp = PETSc.KSP().create()
        ksp.setOptionsPrefix(self._opt_prefix)
        opts = PETSc.Options()
        opts.prefix = self._opt_prefix
        ksp.setOperators(A=self._Mat)
        if solver_parameters is not None:
            solver_parameters.setdefault('ksp_rtol', 1e-7)
        else:
            solver_parameters = {'ksp_rtol': 1e-7}
        for k, v in solver_parameters.iteritems():
            if type(v) is bool and v:
                opts[k] = None
                continue
            opts[k] = v
        ksp.setFromOptions()
        for k, v in solver_parameters.iteritems():
            del opts[k]

        ksp.setPC(self._PC)

        if self._bcs is None:
            b = assemble(L)
        else:
            b = core_types.Function(x.function_space())
            for bc in self._bcs:
                bc.apply(b)
                b.assign(assemble(L) - assemble(action(self._a, b)))
            for bc in self._bcs:
                bc.apply(b)

        with b.dat.vec_ro as bv:
            with x.dat.vec as xv:
                ksp.solve(bv, xv)
