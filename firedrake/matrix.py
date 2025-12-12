from __future__ import annotations
from typing import Any, Iterable, TYPE_CHECKING
import itertools

if TYPE_CHECKING:
    from firedrake.bcs import BCBase
    from firedrake.matrix_free.operators import ImplicitMatrixContext
    from firedrake.slate.slate import TensorBase

import ufl
from ufl.argument import BaseArgument
from pyop2 import op2
from pyop2.utils import as_tuple
from firedrake.petsc import PETSc


class DummyOP2Mat:
    """A hashable implementation of M.handle"""
    def __init__(self, handle):
        self.handle = handle


class MatrixBase(ufl.Matrix):

    def __init__(
        self,
        a: ufl.BaseForm | TensorBase | tuple[BaseArgument, BaseArgument],
        bcs: Iterable[BCBase] | None = None,
        fc_params: dict[str, Any] | None = None,
    ):
        """A representation of the linear operator associated with a bilinear form and bcs.
        Explicitly assembled matrices and matrix-free .matrix classes will derive from this.

        Parameters
        ----------
        a
            A UFL BaseForm (with two arguments) that this MatrixBase represents,
            or a tuple of the arguments it represents, or a slate TensorBase.
        fc_params
            A dictionary of form compiler parameters for this matrix.
        bcs
            An optional iterable of boundary conditions to apply to this :class:`MatrixBase`.
            None by default.
        """
        from firedrake.slate.slate import TensorBase
        if isinstance(a, tuple):
            self.a = None
            test, trial = a
            arguments = a
        else:
            assert isinstance(a, ufl.BaseForm | TensorBase)
            self.a = a
            test, trial = a.arguments()
            arguments = None
        # Iteration over bcs must be in a parallel consistent order
        # (so we can't use a set, since the iteration order may differ
        # on different processes)

        super().__init__(test.function_space(), trial.function_space())

        # ufl.Matrix._analyze_form_arguments sets the _arguments attribute to
        # non-Firedrake objects, which breaks things. To avoid this we overwrite
        # this property after the fact.
        self._analyze_form_arguments()
        self._arguments = arguments

        self.bcs = bcs or ()
        self.comm = test.function_space().comm
        self.block_shape = (len(test.function_space()),
                            len(trial.function_space()))
        self.form_compiler_parameters = fc_params or {}

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
        return f"{type(self).__name__}(a={self.a}, bcs={self.bcs})"

    def __str__(self):
        return f"assembled {type(self).__name__}(a={self.a}, bcs={self.bcs})"

    def __add__(self, other):
        if isinstance(other, MatrixBase):
            mat = self.petscmat + other.petscmat
            return AssembledMatrix(self.arguments(), (), mat)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, MatrixBase):
            mat = self.petscmat - other.petscmat
            return AssembledMatrix(self.arguments(), (), mat)
        else:
            return NotImplemented

    def assign(self, val):
        """Set matrix entries."""
        if isinstance(val, MatrixBase):
            val.petscmat.copy(self.petscmat)
        else:
            raise TypeError(f"Cannot assign a {type(val).__name__} to a {type(self).__name__}.")
        return self

    def zero(self):
        """Set all matrix entries to zero."""
        self.petscmat.zeroEntries()
        return self


class Matrix(MatrixBase):

    def __init__(
        self,
        a: ufl.BaseForm,
        mat: op2.Mat | PETSc.Mat,
        bcs: Iterable[BCBase] | None = None,
        fc_params: dict[str, Any] | None = None,
        options_prefix: str | None = None,
    ):
        """A representation of an assembled bilinear form.

        Parameters
        ----------
        a
            The bilinear form this :class:`Matrix` represents.
        mat : op2.Mat | PETSc.Mat
            The underlying matrix object. Either a PyOP2 Mat or a PETSc Mat.
        bcs : Iterable[DirichletBC] | None, optional
            An iterable of boundary conditions to apply to this :class:`Matrix`.
            May be `None` if there are no boundary conditions to apply.
            By default None.
        fc_params : dict[str, Any] | None, optional
            A dictionary of form compiler parameters for this matrix, by default None.
        options_prefix : str | None, optional
            PETSc options prefix to apply, by default None.
        """
        super().__init__(a, bcs=bcs, fc_params=fc_params)
        if isinstance(mat, op2.Mat):
            self.M = mat
        else:
            assert isinstance(mat, PETSc.Mat)
            self.M = DummyOP2Mat(mat)
        self.petscmat = self.M.handle
        if options_prefix:
            self.petscmat.setOptionsPrefix(options_prefix)
        self.mat_type = self.petscmat.getType()

    def assemble(self):
        raise NotImplementedError("API compatibility to apply bcs after 'assemble(a)'\
                                  has been removed.  Use 'assemble(a, bcs=bcs)', which\
                                  now returns an assembled matrix.")


class ImplicitMatrix(MatrixBase):

    def __init__(
        self,
        a: ufl.BaseForm,
        ctx: ImplicitMatrixContext,
        bcs: Iterable[BCBase] | None = None,
        fc_params: dict[str, Any] | None = None,
        options_prefix: str | None = None,
    ):
        """A representation of the action of bilinear form operating without
        explicitly assembling the associated matrix. This class wraps the
        relevant information for Python PETSc matrix.

        Parameters
        ----------
        a
            The bilinear form this :class:`ImplicitMatrix` represents.
        ctx
            An :class:`ImplicitMatrixContext` that defines the operations
            of the matrix.
        bcs
            An iterable of boundary conditions to apply to this :class:`Matrix`.
            May be `None` if there are no boundary conditions to apply.
            By default None.
        fc_params
            A dictionary of form compiler parameters for this matrix, by default None.
        options_prefix
            PETSc options prefix to apply, by default None.
        """
        super().__init__(a, bcs=bcs, fc_params=fc_params)

        self.petscmat = PETSc.Mat().create(comm=self.comm)
        self.petscmat.setType("python")
        self.petscmat.setSizes((ctx.row_sizes, ctx.col_sizes),
                               bsize=ctx.block_size)
        self.petscmat.setPythonContext(ctx)
        self.petscmat.setOptionsPrefix(options_prefix)
        self.petscmat.setUp()
        self.petscmat.assemble()
        self.mat_type = "matfree"

    def assemble(self):
        # Bump petsc matrix state by assembling it.
        # Ensures that if the matrix changed, the preconditioner is
        # updated if necessary.
        self.petscmat.assemble()


class AssembledMatrix(MatrixBase):

    def __init__(
        self,
        args: tuple[BaseArgument, BaseArgument],
        petscmat: PETSc.Mat,
        bcs: Iterable[BCBase] | None = None,
        options_prefix: str | None = None,
    ):
        """A representation of a matrix that doesn't require knowing the underlying form.

        Parameters
        ----------
        args
            A tuple of the arguments the matrix represents.
        petscmat
            The PETSc matrix this object wraps.
        bcs
            an iterable of boundary conditions to apply to this :class:`Matrix`.
            May be `None` if there are no boundary conditions to apply. By default None.
        options_prefix
            PETSc options prefix to apply, by default None.
        """
        super().__init__(args, bcs=bcs)

        self.petscmat = petscmat
        if options_prefix:
            self.petscmat.setOptionsPrefix(options_prefix)
        self.mat_type = self.petscmat.getType()

        # this mimics op2.Mat.handle
        self.M = DummyOP2Mat(self.petscmat)
