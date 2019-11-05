import abc
import itertools

from pyop2 import op2
from pyop2.utils import as_tuple
from firedrake.petsc import PETSc


class MatrixBase(object, metaclass=abc.ABCMeta):
    """A representation of the linear operator associated with a
    bilinear form and bcs.  Explicitly assembled matrices and matrix-free
    matrix classes will derive from this

    :arg a: the bilinear form this :class:`MatrixBase` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`MatrixBase`.  May be `None` if there are no boundary
        conditions to apply.
    :arg mat_type: matrix type of assembled matrix, or 'matfree' for matrix-free
    """
    def __init__(self, a, bcs, mat_type):
        self.a = a
        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)
        self.bcs = bcs
        test, trial = a.arguments()
        self.comm = test.function_space().comm
        self.block_shape = (len(test.function_space()),
                            len(trial.function_space()))
        self.mat_type = mat_type
        """Matrix type.

        Matrix type used in the assembly of the PETSc matrix: 'aij', 'baij', or 'nest',
        or 'matfree' for matrix-free."""

    @property
    def has_bcs(self):
        """Return True if this :class:`MatrixBase` has any boundary
        conditions attached to it."""
        return self._bcs != ()

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
        if bcs is not None:
            self._bcs = tuple(itertools.chain(*(as_tuple(bc) for bc in bcs)))
        else:
            self._bcs = ()

    def __repr__(self):
        return "%s(a=%r, bcs=%r)" % (type(self).__name__,
                                     self.a,
                                     self.bcs)

    def __str__(self):
        return "assembled %s(a=%s, bcs=%s)" % (type(self).__name__,
                                               self.a, self.bcs)


class Matrix(MatrixBase):
    """A representation of an assembled bilinear form.

    :arg a: the bilinear form this :class:`Matrix` represents.

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.

    :arg mat_type: matrix type of assembled matrix.

    A :class:`pyop2.Mat` will be built from the remaining
    arguments, for valid values, see :class:`pyop2.Mat`.

    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """

    def __init__(self, a, bcs, mat_type, *args, **kwargs):
        # sets self._a, self._bcs, and self._mat_type
        super(Matrix, self).__init__(a, bcs, mat_type)
        options_prefix = kwargs.pop("options_prefix")
        self.M = op2.Mat(*args, **kwargs)
        self.petscmat = self.M.handle
        self.petscmat.setOptionsPrefix(options_prefix)
        self.mat_type = mat_type

    def assemble(self):
        raise NotImplementedError("API compatibility to apply bcs after 'assemble(a)'\
                                  has been removed.  Use 'assemble(a, bcs=bcs)', which\
                                  now returns an assembled matrix.")


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
        # sets self._a, self._bcs, and self._mat_type
        super(ImplicitMatrix, self).__init__(a, bcs, "matfree")

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

    def assemble(self):
        # Bump petsc matrix state by assembling it.
        # Ensures that if the matrix changed, the preconditioner is
        # updated if necessary.
        self.petscmat.assemble()
