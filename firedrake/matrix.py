import itertools
import ufl

from pyop2 import op2
from pyop2.mpi import internal_comm
from pyop2.utils import as_tuple
from firedrake.petsc import PETSc
from types import SimpleNamespace


class MatrixBase(ufl.Matrix):
    """A representation of the linear operator associated with a
    bilinear form and bcs.  Explicitly assembled matrices and matrix-free
    matrix classes will derive from this

    :arg a: the bilinear form this :class:`MatrixBase` represents
            or a tuple of the arguments it represents

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`MatrixBase`.  May be `None` if there are no boundary
        conditions to apply.
    :arg mat_type: matrix type of assembled matrix, or 'matfree' for matrix-free
    """
    def __init__(self, a, bcs, mat_type):
        if isinstance(a, tuple):
            self.a = None
            test, trial = a
            arguments = a
        else:
            self.a = a
            test, trial = a.arguments()
            arguments = None
        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)

        ufl.Matrix.__init__(self, test.function_space(), trial.function_space())

        # ufl.Matrix._analyze_form_arguments sets the _arguments attribute to
        # non-Firedrake objects, which breaks things. To avoid this we overwrite
        # this property after the fact.
        self._analyze_form_arguments()
        self._arguments = arguments

        self.bcs = bcs
        self.comm = test.function_space().comm
        self._comm = internal_comm(self.comm, self)
        self.block_shape = (len(test.function_space()),
                            len(trial.function_space()))
        self.mat_type = mat_type
        """Matrix type.

        Matrix type used in the assembly of the PETSc matrix: 'aij', 'baij', 'dense' or 'nest',
        or 'matfree' for matrix-free."""

    def arguments(self):
        if self.a:
            return self.a.arguments()
        else:
            return self._arguments

    def ufl_domains(self):
        return self._domains

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

    A ``pyop2.types.mat.Mat`` will be built from the remaining
    arguments, for valid values, see ``pyop2.types.mat.Mat`` source code.

    .. note::

        This object acts to the right on an assembled :class:`.Function`
        and to the left on an assembled cofunction (currently represented
        by a :class:`.Function`).

    """

    def __init__(self, a, bcs, mat_type, *args, **kwargs):
        # sets self._a, self._bcs, and self._mat_type
        MatrixBase.__init__(self, a, bcs, mat_type)
        options_prefix = kwargs.pop("options_prefix")
        self.M = op2.Mat(*args, mat_type=mat_type, **kwargs)
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
        self.petscmat = PETSc.Mat().create(comm=self._comm)
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


class AssembledMatrix(MatrixBase):
    """A representation of a matrix that doesn't require knowing the underlying form.
     This class wraps the relevant information for Python PETSc matrix.

    :arg a: A tuple of the arguments the matrix represents

    :arg bcs: an iterable of boundary conditions to apply to this
        :class:`Matrix`.  May be `None` if there are no boundary
        conditions to apply.

    :arg petscmat: the already constructed petsc matrix this object represents.
    """
    def __init__(self, a, bcs, petscmat, *args, **kwargs):
        super(AssembledMatrix, self).__init__(a, bcs, "assembled")

        self.petscmat = petscmat

        # this allows call to self.M.handle without a new class
        self.M = SimpleNamespace(handle=self.mat())

    def mat(self):
        return self.petscmat

    def __add__(self, other):
        if isinstance(other, AssembledMatrix):
            return self.petscmat + other.petscmat
        else:
            raise TypeError("Unable to add %s to AssembledMatrix"
                            % (type(other)))
