from __future__ import annotations

import numpy
import abc

import dataclasses
from functools import cached_property, partial
from typing import Literal, Callable, Iterable
from dataclasses import dataclass
from numbers import Number

from ufl.algorithms import extract_arguments, replace
from ufl.domain import extract_unique_domain
from ufl.classes import Expr
from ufl.constantvalue import as_ufl
from ufl.form import ZeroBaseForm, BaseForm
from ufl.core.interpolate import Interpolate as UFLInterpolate

import pyop3 as op3
from pyop3.cache import cached_on, with_heavy_caches
from pyop3.dtypes import get_mpi_dtype

from finat.ufl import TensorElement, VectorElement, MixedElement, FiniteElementBase

from tsfc import kernel_args
from tsfc.ufl_utils import RUNTIME_POINT_VARIABLE

from firedrake.utils import IntType
from firedrake.mesh import MissingPointsBehaviour, VertexOnlyMeshTopology, MeshGeometry, MeshTopology, VertexOnlyMesh
from firedrake.ufl_expr import Argument, Coargument, TrialFunction, TestFunction, action, extract_domains
from firedrake.petsc import PETSc
from firedrake.functionspaceimpl import WithGeometry
from firedrake.mesh import get_mesh_topologies, MeshLoopIndex
from firedrake.matrix import ImplicitMatrix, MatrixBase, Matrix
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.bcs import DirichletBC
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.functionspace import VectorFunctionSpace, TensorFunctionSpace, FunctionSpace
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.exceptions import (
    DofNotDefinedError, VertexOnlyMeshMissingPointsError, NonUniqueMeshSequenceError,
    DofTypeError,
)

from mpi4py import MPI

from pyadjoint.tape import no_annotations

__all__ = (
    "interpolate",
    "Interpolate",
    "get_interpolator",
    "InterpolateOptions",
    "Interpolator"
)


@dataclass(kw_only=True)
class InterpolateOptions:
    """Options for interpolation operations.

    Parameters
    ----------
    subset : pyop3.Slice or None
        An optional subset to apply the interpolation over.
        Cannot, at present, be used when interpolating across meshes unless
        the target mesh is a :func:`.VertexOnlyMesh`.
    access : pyop3.constants.Intent or None
        The pyop3 access descriptor for combining updates to shared
        DoFs. Possible values include ``WRITE``, ``MIN``, ``MAX``, and ``INC``.
        ``WRITE``, ``MIN``, and ``MAX`` are supported when interpolating across
        meshes. Only ``INC`` is supported for the matrix-free adjoint
        interpolation.
    allow_missing_dofs : bool
        For interpolation across meshes: allow degrees of freedom (aka DoFs/nodes)
        in the target mesh that cannot be defined on the source mesh.
        For example, where nodes are point evaluations, points in the target mesh
        that are not in the source mesh. When ``False`` this raises a ``ValueError``
        should this occur. When ``True`` the corresponding values are either
        (a) unchanged if some ``output`` is given to the :meth:`interpolate` method
        or (b) set to zero.
        Can be overwritten with the ``default_missing_val`` kwarg of :meth:`interpolate`.
        This does not affect adjoint interpolation. Ignored if interpolating within
        the same mesh or onto a :func:`.VertexOnlyMesh` (the behaviour of a
        :func:`.VertexOnlyMesh` in this scenario is, at present, set when it is created).
    default_missing_val : float or None
        For interpolation across meshes: the optional value to assign to DoFs
        in the target mesh that are outside the source mesh. If this is not set
        then the values are either (a) unchanged if some ``output`` is given to
        the :meth:`interpolate` method or (b) set to zero.
        Ignored if interpolating within the same mesh or onto a :func:`.VertexOnlyMesh`.
    """
    subset: op3.Slice | None = None
    access: Literal[op3.WRITE, op3.MIN_WRITE, op3.MAX_WRITE, op3.INC] | None = None
    allow_missing_dofs: bool = False
    default_missing_val: float | None = None


class Interpolate(UFLInterpolate):

    def __init__(self, expr: Expr, V: WithGeometry | BaseForm, **kwargs):
        """Symbolic representation of the interpolation operator.

        Parameters
        ----------
        expr : ufl.core.expr.Expr
               The UFL expression to interpolate.
        V : firedrake.functionspaceimpl.WithGeometry or ufl.BaseForm
            The function space to interpolate into or the coargument defined
            on the dual of the function space to interpolate into.
        **kwargs
            Additional interpolation options. See :class:`InterpolateOptions`
            for available parameters and their descriptions.
        """
        expr = as_ufl(expr)
        expr_args = expr.arguments()[1:] if isinstance(expr, BaseForm) else extract_arguments(expr)
        expr_arg_numbers = {arg.number() for arg in expr_args}
        self.is_adjoint = expr_arg_numbers == {0}
        if isinstance(V, WithGeometry):
            # Need to create a Firedrake Argument so that it has a .function_space() method
            V = Argument(V.dual(), 1 if self.is_adjoint else 0)

        self.target_space = V.arguments()[0].function_space()
        if expr.ufl_shape != self.target_space.value_shape:
            raise ValueError(f"Shape mismatch: Expression shape {expr.ufl_shape}, FunctionSpace shape {self.target_space.value_shape}.")

        super().__init__(expr, V)

        self._options = InterpolateOptions(**kwargs)

    function_space = UFLInterpolate.ufl_function_space

    def _ufl_expr_reconstruct_(
            self, expr: Expr, v: WithGeometry | BaseForm | None = None, **interp_data
    ):
        # Note that we can't use dataclasses.asdict here because we can't deepcopy
        # PETSc objects, this is the recommended workaround.
        options = {
            field.name: getattr(self.options, field.name)
            for field in dataclasses.fields(self.options)
        }
        interp_data = options | interp_data
        return UFLInterpolate._ufl_expr_reconstruct_(self, expr, v=v, **interp_data)

    @property
    def options(self) -> InterpolateOptions:
        """Access the interpolation options.

        Returns
        -------
        InterpolateOptions
            An :class:`InterpolateOptions` instance containing the interpolation options.
        """
        return self._options

    def subdomain_data(self):
        """Return the loop over the target cells that the interpolation reaches."""
        domain = self.target_space.mesh().unique()
        return {domain: {"cell": [self._interpolator._loop_index]}}

    @cached_property
    def _interpolator(self):
        """Access the numerical interpolator.

        Returns
        -------
        Interpolator
            An appropriate :class:`Interpolator` subclass for this
            interpolation expression.
        """
        arguments = self.arguments()
        has_mixed_arguments = any(len(arg.function_space()) > 1 for arg in arguments)

        operand, = self.ufl_operands
        target_mesh = self.target_space.mesh()

        try:
            source_mesh = extract_unique_domain(operand) or target_mesh
        except ValueError:
            if has_mixed_arguments or len(self.target_space) > 1:
                return MixedInterpolator(self)
            raise NotImplementedError(
                "Interpolating an expression with no arguments defined on multiple meshes is not implemented yet."
            )

        try:
            target_mesh = target_mesh.unique()
            source_mesh = source_mesh.unique()
        except NonUniqueMeshSequenceError:
            return MixedInterpolator(self)

        submesh_interp_implemented = (
            all(isinstance(m.topology, MeshTopology) for m in [target_mesh, source_mesh])
            and target_mesh.submesh_ancestors[-1] is source_mesh.submesh_ancestors[-1]
            and target_mesh.topological_dimension == source_mesh.topological_dimension
        )
        if target_mesh is source_mesh or submesh_interp_implemented:
            return SameMeshInterpolator(self, source_mesh, target_mesh)

        if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
            if isinstance(source_mesh.topology, VertexOnlyMeshTopology):
                return VomOntoVomInterpolator(self, source_mesh, target_mesh)
            if target_mesh.geometric_dimension != source_mesh.geometric_dimension:
                raise ValueError("Cannot interpolate onto a VertexOnlyMesh of a different geometric dimension.")
            return SameMeshInterpolator(self, source_mesh, target_mesh)

        if has_mixed_arguments or len(self.target_space) > 1:
            return MixedInterpolator(self)

        return CrossMeshInterpolator(self, source_mesh, target_mesh)


@PETSc.Log.EventDecorator()
@with_heavy_caches(lambda expr, *a, **kw: get_mesh_topologies(expr))
def interpolate(expr: Expr, V: WithGeometry | BaseForm, **kwargs) -> Interpolate:
    """Returns a UFL expression for the interpolation operation of ``expr`` into ``V``.

    Parameters
    ----------
    expr : ufl.core.expr.Expr
        The UFL expression to interpolate.
    V : firedrake.functionspaceimpl.WithGeometry or ufl.BaseForm
        The function space to interpolate into or the coargument defined
        on the dual of the function space to interpolate into.
    **kwargs
        Additional interpolation options. See :class:`InterpolateOptions`
        for available parameters and their descriptions.

    Returns
    -------
    Interpolate
        A symbolic :class:`Interpolate` object representing the interpolation operation.
    """
    return Interpolate(expr, V, **kwargs)


class Interpolator(abc.ABC):
    """Base class for calculating interpolation. Should not be instantiated directly; use the
    :func:`get_interpolator` function.

    Parameters
    ----------
    expr : Interpolate
        The symbolic interpolation expression.

    """
    def __init__(self, expr: Interpolate):
        dual_arg, operand = expr.argument_slots()
        self.ufl_interpolate = expr
        """The symbolic UFL Interpolate expression."""
        self.interpolate_args = expr.arguments()
        """Arguments of the Interpolate expression."""
        self.rank = len(self.interpolate_args)
        """Number of arguments in the Interpolate expression."""
        self.operand = operand
        """The primal argument slot of the Interpolate expression."""
        self.dual_arg = dual_arg
        """The dual argument slot of the Interpolate expression."""
        self.target_space = dual_arg.function_space().dual()
        """The primal space we are interpolating into."""

        # Interpolation options
        self.subset = expr.options.subset
        self.allow_missing_dofs = expr.options.allow_missing_dofs
        self.default_missing_val = expr.options.default_missing_val
        # An interpolation kernel never reads the output, it only reduces its
        # results into it. Therefore, MIN and MAX are the WRITE reductions.
        access = expr.options.access
        if access is op3.MIN_RW:
            access = op3.MIN_WRITE
        elif access is op3.MAX_RW:
            access = op3.MAX_WRITE
        self.access = access

    @abc.abstractmethod
    def _get_callable(
        self,
        tensor: Function | Cofunction | MatrixBase | None = None,
        bcs: Iterable[DirichletBC] | None = None,
        mat_type: Literal["aij", "baij", "nest", "matfree"] | None = None,
        sub_mat_type: Literal["aij", "baij"] | None = None,
        pyop3_compiler_parameters=None,
    ) -> Callable[[], Function | Cofunction | MatrixBase | Number]:
        """Return a callable to perform interpolation.

        If ``self.rank == 2``, then the callable must return a ``MatrixBase``.
        This is ``tensor`` when the interpolator can assemble into it.
        If ``self.rank == 1``, then the callable must return a ``Function``
        or ``Cofunction`` (in the forward and adjoint cases respectively).
        If ``self.rank == 0``, then the callable must return a number.

        Parameters
        ----------
        tensor
            Optional tensor to store the result in, by default None.
        bcs
            An optional list of boundary conditions to zero-out in the
            output function space. Interpolator rows or columns which are
            associated with boundary condition nodes are zeroed out when this is
            specified. By default None.
        mat_type
            The PETSc matrix type to use when assembling a rank 2 interpolation.
            For cross-mesh interpolation, only ``"aij"`` is supported. For same-mesh
            interpolation, ``"aij"`` and ``"baij"`` are supported. For same/cross mesh interpolation
            between :func:`.MixedFunctionSpace`, ``"aij"`` and ``"nest"`` are supported.
            For interpolation between input-ordering linked :func:`.VertexOnlyMesh`,
            ``"aij"``, ``"baij"``, and ``"matfree"`` are supported.
        sub_mat_type
            The PETSc sub-matrix type to use when assembling a rank 2 interpolation between
            :func:`.MixedFunctionSpace` with ``mat_type="nest"``. Only ``"aij"`` and ``"baij"``
            are supported.
        """
        pass

    @property
    @abc.abstractmethod
    def _allowed_mat_types(self) -> set[Literal["aij", "baij", "nest", "matfree"]]:
        """Returns a set of valid matrix types for assembly of two-forms.
        """
        pass

    # TODO: compiler params not universally handled
    def assemble(
        self,
        tensor: Function | Cofunction | MatrixBase | None = None,
        bcs: Iterable[DirichletBC] | None = None,
        mat_type: Literal["aij", "baij", "nest", "matfree"] | None = None,
        sub_mat_type: Literal["aij", "baij"] | None = None,
        pyop3_compiler_parameters=None,
    ) -> Function | Cofunction | MatrixBase | Number:
        """Assemble the interpolation. The result depends on the rank (number of arguments)
        of the :class:`Interpolate` expression:

        * rank 2: assemble the operator and return a matrix
        * rank 1: assemble the action and return a function or cofunction
        * rank 0: assemble the action and return a scalar by applying the dual argument

        Parameters
        ----------
        tensor
            Optional tensor to store the interpolated result. For rank 2
            expressions this is expected to be a subclass of
            :class:`~firedrake.matrix.MatrixBase`. For rank 1 expressions
            this is a :class:`~firedrake.function.Function` or :class:`~firedrake.cofunction.Cofunction`,
            for forward and adjoint interpolation respectively.
        bcs
            An optional list of boundary conditions to zero-out in the
            output function space. Interpolator rows or columns which are
            associated with boundary condition nodes are zeroed out when this is
            specified. By default None.
        mat_type
            The PETSc matrix type to use when assembling a rank 2 interpolation.
            For cross-mesh interpolation, ``"aij"`` and ``"matfree"`` are supported. For same-mesh
            interpolation, ``"aij"``, ``"baij"``, and ``"matfree"`` are supported. For same/cross mesh interpolation
            between :func:`.MixedFunctionSpace`, ``"aij"`` and ``"nest"`` are supported.
            For interpolation between input-ordering linked :func:`.VertexOnlyMesh`,
            ``"aij"``, ``"baij"``, and ``"matfree"`` are supported.
        sub_mat_type
            The PETSc sub-matrix type to use when assembling a rank 2 interpolation between
            :func:`.MixedFunctionSpace` with ``mat_type="nest"``. Only ``"aij"`` and ``"baij"``
            are supported.
        Returns
        -------
        Function | Cofunction | MatrixBase | numbers.Number
            The function, cofunction, matrix, or scalar resulting from the
            interpolation.
        """
        self._check_mat_type(mat_type)

        if mat_type == "matfree" and self.rank == 2:
            Vrow, Vcol = (arg.function_space() for arg in self.interpolate_args)
            if bcs is None:
                bcs = ()
            row_bcs = [bc for bc in bcs if bc.parent_function_space.topological == Vrow.topological]
            col_bcs = [bc for bc in bcs if bc.parent_function_space.topological == Vcol.topological]
            ctx = ImplicitMatrixContext(
                self.ufl_interpolate, row_bcs=row_bcs, col_bcs=col_bcs,
            )
            return ImplicitMatrix(self.ufl_interpolate, ctx, bcs=bcs)

        result = self._get_callable(tensor=tensor, bcs=bcs, mat_type=mat_type, sub_mat_type=sub_mat_type, pyop3_compiler_parameters=pyop3_compiler_parameters)()

        if self.rank == 2:
            # Assembling the operator
            assert isinstance(tensor, MatrixBase | None)
            assert isinstance(result, MatrixBase)
            if tensor and result is not tensor:
                result.petscmat.copy(tensor.petscmat)
                return tensor
            return result
        else:
            assert isinstance(tensor, Function | Cofunction | None)
            return tensor.assign(result) if tensor else result

    def _check_mat_type(
            self,
            mat_type: Literal["aij", "baij", "nest", "matfree"] | None,
    ) -> None:
        """Check that the given mat_type is valid for this Interpolator.

        Parameters
        ----------
        mat_type
            The PETSc matrix type to check.

        Raises
        ------
        NotImplementedError
            If the given mat_type is not supported for this Interpolator.
        """
        if self.rank == 2 and mat_type not in self._allowed_mat_types:
            raise NotImplementedError(f"Assembly of matrix type {mat_type} not implemented yet for {type(self).__name__}.")


def get_interpolator(expr: Interpolate) -> Interpolator:
    """Create an Interpolator.

    Parameters
    ----------
    expr : Interpolate
        Symbolic interpolation expression.

    Returns
    -------
    Interpolator
        An appropriate :class:`Interpolator` subclass for the given
        interpolation expression.
    """
    return expr._interpolator


class CrossMeshInterpolator(Interpolator):
    """
    Interpolate a function from one mesh and function space to another.

    For arguments, see :class:`.Interpolator`.
    """
    @no_annotations
    def __init__(self, expr: Interpolate, source_mesh, target_mesh):
        super().__init__(expr)
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh

        if self.access == op3.INC:
            raise NotImplementedError(
                "Increment access is not implemented for cross-mesh interpolation."
            )
        self.access = self.access or op3.WRITE

        if self.allow_missing_dofs:
            self.missing_points_behaviour = MissingPointsBehaviour.IGNORE
        else:
            self.missing_points_behaviour = MissingPointsBehaviour.ERROR

        if self.source_mesh.unique().geometric_dimension != self.target_mesh.unique().geometric_dimension:
            raise ValueError("Geometric dimensions of source and destination meshes must match.")

        # Interpolate into intermediate quadrature space for non-point-evaluation elements
        if into_quadrature_space := not self.target_space.finat_element.has_pointwise_dual_basis:
            self.original_target_space = self.target_space
            r"""The original target space for interpolation, as specified by the user."""
            self.target_space = self.target_space.quadrature_space
            r"""The target space for the cross-mesh interpolation. Must have point-evaluation dofs.
            If ``self.original_target_space`` does not have point-evaluation dofs, then this is
            an intermediate quadrature space."""

        self.into_quadrature_space = into_quadrature_space

    @cached_property
    def _target_space_element(self) -> FiniteElementBase:
        """The element of `self.target_space`. If `self.target_space` is tensor/vector valued,
        the base scalar element.

        Returns
        -------
        FiniteElementBase
            The base element of `self.target_space`.
        """
        dest_element = self.target_space.ufl_element()
        if isinstance(dest_element, MixedElement):
            if isinstance(dest_element, VectorElement | TensorElement):
                # In this case all sub elements are equal
                return dest_element.sub_elements[0]
            else:
                raise NotImplementedError("Interpolation with MixedFunctionSpace requires MixedInterpolator.")
        else:
            # scalar fiat/finat element
            return dest_element

    @cached_property
    def _target_space_type(self) -> Callable[..., WithGeometry]:
        """Returns a callable which returns a function space matching the type of `self.target_space`.

        Returns
        -------
        Callable
            A callable which returns a :class:`.WithGeometry` matching the type of `self.target_space`.
        """
        # Get the correct type of function space
        shape = self.target_space.value_shape
        if len(shape) == 0:
            return FunctionSpace
        elif len(shape) == 1:
            return partial(VectorFunctionSpace, dim=shape[0])
        else:
            symmetry = self.target_space.ufl_element().symmetry()
            return partial(TensorFunctionSpace, shape=shape, symmetry=symmetry)

    @cached_property
    def _symbolic_expressions(self) -> tuple[Interpolate, Interpolate]:
        """The symbolic ``Interpolate`` expressions for point evaluation of `self.target_space`s
        dofs in the source mesh, and the corresponding input-ordering interpolation.

        Returns
        -------
        tuple[Interpolate, Interpolate]
            A tuple containing the point evaluation interpolation and the
            input-ordering interpolation.

        Raises
        ------
        DoFNotDefinedError
            If any of the target spaces dofs cannot be defined in the source mesh.
        DoFTypeError
            If the target space does not have point-evaluation dofs.
        """
        from firedrake.assemble import assemble
        if not self.target_space.finat_element.has_pointwise_dual_basis:
            raise DofTypeError(f"FunctionSpace {self.target_space} must have point-evaluation dofs.")

        # Immerse coordinates of target space point evaluation dofs in src_mesh
        # If `self.into_quadrature_space` is true, then the point evaluation dofs
        # are the quadrature points of the original target space.
        target_mesh = self.target_space.mesh().unique()
        target_space_vec = VectorFunctionSpace(target_mesh, self._target_space_element)
        f_dest_node_coords = assemble(interpolate(target_mesh.coordinates, target_space_vec))
        dest_node_coords = f_dest_node_coords.dat.data_ro
        try:
            vom = VertexOnlyMesh(
                self.source_mesh.unique(),
                dest_node_coords,
                redundant=False,
                missing_points_behaviour=self.missing_points_behaviour,
            )
        except VertexOnlyMeshMissingPointsError:
            raise DofNotDefinedError(f"The given target function space on domain {target_mesh} "
                                     "contains degrees of freedom which cannot cannot be defined in the "
                                     f"source function space on domain {self.source_mesh.unique()}. "
                                     "This may be because the target mesh covers a larger domain than the "
                                     "source mesh. To disable this error, set allow_missing_dofs=True.")

        # Expression for point evaluation at the dest_node_coords
        P0DG_vom = self._target_space_type(vom, "DG", 0)
        point_eval = interpolate(self.operand, P0DG_vom)

        # Expression for interpolating into the input-ordering VOM
        P0DG_vom_input_ordering = self._target_space_type(vom.input_ordering, "DG", 0)
        arg = Argument(P0DG_vom, 0 if self.ufl_interpolate.is_adjoint else 1)
        point_eval_input_ordering = interpolate(arg, P0DG_vom_input_ordering)

        return point_eval, point_eval_input_ordering

    @cached_property
    def _interpolate_from_quadrature(self) -> Interpolate:
        """Returns symbolic expression for interpolation from the intermediate quadrature
        space into the user-provided target space. Only relevant if `self.into_quadrature_space` is True.

        Returns
        -------
        Interpolate
            A symbolic interpolate expression.
        """
        if self.rank == 2:
            if self.ufl_interpolate.is_adjoint:
                return interpolate(TestFunction(self.target_space), self.original_target_space)
            else:
                return interpolate(TrialFunction(self.target_space), self.original_target_space)
        elif self.ufl_interpolate.is_adjoint:
            return interpolate(TestFunction(self.target_space), self.dual_arg)

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None, pyop3_compiler_parameters=None):
        from firedrake.assemble import assemble
        if bcs:
            raise NotImplementedError("bcs not implemented for cross-mesh interpolation.")
        mat_type = mat_type or "aij"

        if self.into_quadrature_space:
            f = Function(self.target_space.dual() if self.ufl_interpolate.is_adjoint else self.target_space)
        else:
            f = tensor or Function(self.ufl_interpolate.function_space() or self.target_space)
        # Each target DoF receives a single value. Therefore, MIN and MAX only
        # combine that value with the current values of a supplied output.
        reduction = {op3.MIN_WRITE: numpy.minimum, op3.MAX_WRITE: numpy.maximum}.get(self.access) if f is tensor else None

        point_eval, point_eval_input_ordering = self._symbolic_expressions
        P0DG_vom_input_ordering = point_eval_input_ordering.argument_slots()[0].function_space().dual()

        if self.rank == 2:
            assert mat_type == "aij"
            # The cross-mesh interpolation matrix is the product of the
            # `self.point_eval_interpolate` and the permutation
            # given by `self.to_input_ordering_interpolate`.
            if self.ufl_interpolate.is_adjoint:
                interp_expr = action(point_eval, point_eval_input_ordering)
            else:
                interp_expr = action(point_eval_input_ordering, point_eval)

            def callable() -> Matrix:
                res = assemble(interp_expr, mat_type=mat_type).petscmat
                if self.into_quadrature_space:
                    source_space = self.operand.function_space()
                    if self.ufl_interpolate.is_adjoint:
                        I = Matrix(interpolate(TestFunction(source_space), self.target_space), res)
                        res = assemble(action(I, self._interpolate_from_quadrature)).petscmat
                    else:
                        I = Matrix(interpolate(TrialFunction(source_space), self.target_space), res)
                        res = assemble(action(self._interpolate_from_quadrature, I)).petscmat
                return Matrix(self.ufl_interpolate, res)

        elif self.ufl_interpolate.is_adjoint:
            assert self.rank == 1

            def callable() -> Cofunction:
                if self.into_quadrature_space:
                    cofunc = assemble(self._interpolate_from_quadrature)
                    f_target = Cofunction(point_eval.function_space())
                else:
                    cofunc = self.dual_arg
                    f_target = f

                assert isinstance(cofunc, Cofunction)

                # Our first adjoint operation is to assign the dat values to a
                # P0DG cofunction on our input ordering VOM.
                f_input_ordering = Cofunction(P0DG_vom_input_ordering.dual())
                f_input_ordering.dat.data_wo[:] = cofunc.dat.data_ro[:]

                # The rest of the adjoint interpolation is the composition
                # of the adjoint interpolators in the reverse direction.
                # We don't worry about skipping over missing points here
                # because we're going from the input ordering VOM to the original VOM
                # and all points from the input ordering VOM are in the original.
                f_src_at_src_node_coords = assemble(action(point_eval_input_ordering, f_input_ordering))
                assemble(action(point_eval, f_src_at_src_node_coords), tensor=f_target)
                return f_target
        else:
            assert self.rank in {0, 1}

            def callable() -> Function | Number:
                # We create the input-ordering Function before interpolating so we can
                # set default missing values if required.
                f_point_eval_input_ordering = Function(P0DG_vom_input_ordering)
                if self.default_missing_val is not None:
                    f_point_eval_input_ordering.assign(self.default_missing_val)
                elif self.allow_missing_dofs:
                    # If we allow missing points there may be points in the target
                    # mesh that are not in the source mesh. If we don't specify a
                    # default missing value we set these to NaN so we can identify
                    # them later.
                    f_point_eval_input_ordering.dat.data_wo[:] = numpy.nan

                assemble(action(point_eval_input_ordering, point_eval), tensor=f_point_eval_input_ordering)
                # We assign these values to the output function
                if self.allow_missing_dofs and self.default_missing_val is None:
                    f.dat.buffer.sync_roots()
                    indices = numpy.where(~numpy.isnan(f_point_eval_input_ordering.dat.data_ro))[0]
                else:
                    indices = slice(None)
                values = f_point_eval_input_ordering.dat.data_ro[indices]
                if reduction is not None:
                    values = reduction(f.dat.data_ro[indices], values)
                f.dat.data_wo[indices] = values

                if self.into_quadrature_space:
                    f_target = Function(self.original_target_space)
                    assemble(interpolate(f, self.original_target_space), tensor=f_target)
                else:
                    f_target = f

                if self.rank == 0:
                    # We take the action of the dual_arg on the interpolated function
                    assert isinstance(self.dual_arg, Cofunction)
                    return assemble(action(self.dual_arg, f_target))
                else:
                    return f_target
        return callable

    @property
    def _allowed_mat_types(self):
        return {"aij", "matfree", None}


class SameMeshInterpolator(Interpolator):
    """
    An interpolator for interpolation within the same mesh or onto a validly-
    defined :func:`.VertexOnlyMesh`.

    For arguments, see :class:`.Interpolator`.
    """

    @no_annotations
    def __init__(self, expr, source_mesh, target_mesh):
        super().__init__(expr)
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh

        if not isinstance(self.dual_arg, Coargument):
            # Matrix-free assembly of 0-form or 1-form requires INC access
            if self.access and self.access != op3.INC:
                raise ValueError("Matfree adjoint interpolation requires INC access")
            self.access = op3.INC
        elif self.access is None:
            # Default access for forward 1-form or 2-form (forward and adjoint)
            self.access = op3.WRITE

    @property
    def _needs_adjoint_weighting(self):
        return (isinstance(self.dual_arg, Cofunction)
                and any(not V.finat_element.is_dg() for V in self.target_space))

    @cached_property
    def _weighted_dual_arg(self):
        return Function(self.dual_arg.function_space())

    @cached_property
    def _loop_index(self) -> MeshLoopIndex:
        """The loop over the target cells that the interpolation reaches."""
        target_mesh = self.target_mesh.unique()
        source_mesh = self.source_mesh.unique()
        if source_mesh is not target_mesh and source_mesh.submesh_youngest_common_ancestor(target_mesh):
            # Only the target cells that the source mesh covers are reached.
            subdomain_id = "everywhere" if self.subset is None else self.subset
            return target_mesh.iter("cell", subdomain_id=subdomain_id, intersect_meshes=[source_mesh])
        elif self.subset is not None:
            return target_mesh.iter("cell", subdomain_id=self.subset)
        else:
            return target_mesh.iter("cell")

    def _loop_target_nodes(self, f: Function | Cofunction, value: float,
                           mode: Literal["inc", "min", "max"]) -> None:
        """Combine ``value`` with each node of ``f`` in the cells of the iteration set.

        Parameters
        ----------
        f
            The function or cofunction on the target space to update.
        value
            The value to combine with the nodes.
        mode
            How ``value`` is combined with the current node values.
        """
        target_mesh = self.target_mesh.unique()
        for fk in f.subfunctions:
            op3.loop(
                c := self._loop_index,
                fk.dat[target_mesh.closure(c)].assign(value, mode),
                eager=True,
            )

    def _initialize_minmax(self, tensor: Function | Cofunction) -> None:
        """Set the identity of the MIN or MAX reduction on the nodes that the interpolation reaches.

        The other nodes keep the zero of a newly allocated tensor.
        """
        # A node can be reached only from the halo cells of its owner. The
        # opposite reduction carries the identity from the halo copies to the
        # owner. Therefore, the MIN identity is written with a MAX reduction,
        # and the MAX identity with a MIN reduction.
        finfo = numpy.finfo(tensor.dat.dtype)
        if self.access is op3.MIN_WRITE:
            value, mode = float(finfo.max), "max"
        else:
            value, mode = float(finfo.min), "min"
        self._loop_target_nodes(tensor, value, mode)

    @cached_on(
        get_obj=lambda self: self.source_mesh.unique().topology,
        get_key=lambda self: (self.dual_arg.function_space(), self.subset),
    )
    def _adjoint_weight(self) -> op3.Dat:
        """The reciprocal of the number of cells in the iteration set whose closure contains each node of the dual argument."""
        weight = Function(self.dual_arg.function_space())
        self._loop_target_nodes(weight, 1.0, "inc")
        with weight.dat.vec_rw as weight_vec:
            weight_vec.reciprocal()
        return weight.dat

    @cached_property
    def _interpolate_to_assemble(self):
        """The interpolation that is handed to the assembler.

        This is the user's `Interpolate` carrying the assembly options, and the
        weighted copy of the dual argument when the adjoint needs one.
        """
        if not self._needs_adjoint_weighting:
            return self.ufl_interpolate
        return self.ufl_interpolate._ufl_expr_reconstruct_(self.operand, v=self._weighted_dual_arg,
                                                           subset=self.subset, access=self.access)

    def _update_weighted_dual_arg(self):
        self._weighted_dual_arg.dat.assign(self.dual_arg.dat, eager=True)
        with self._adjoint_weight().vec_ro as weight, self._weighted_dual_arg.dat.vec_rw as dual:
            dual.pointwiseMult(dual, weight)

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None,
                      pyop3_compiler_parameters=None):
        from firedrake.assemble import get_form_assembler

        # The output tensor can share a dat with an input. Then the kernel would
        # overwrite input values that it has not read yet. Therefore, the kernel
        # must write into a temporary, and we copy the temporary into the output
        # after assembly.
        output = None
        reads_output = False
        if isinstance(tensor, Function | Cofunction):
            inputs = {coefficient.dat.buffer for coefficient in self._interpolate_to_assemble.coefficients()}
            inputs.update(mesh.coordinates.dat.buffer for mesh in extract_domains(self._interpolate_to_assemble))
            if tensor.dat.buffer in inputs:
                output = tensor
                reads_output = self.access in {op3.MIN_WRITE, op3.MAX_WRITE} or self.subset is not None

        assembler = get_form_assembler(self._interpolate_to_assemble, bcs=bcs,
                                       mat_type=mat_type, sub_mat_type=sub_mat_type,
                                       pyop3_compiler_parameters=pyop3_compiler_parameters,
                                       needs_zeroing=False, access=self.access)

        copy_input = None
        copy_output = None
        if output is not None:
            tensor = assembler.allocate()
            if reads_output:
                copy_input = partial(tensor.dat.assign, output.dat, eager=True)
            copy_output = partial(output.dat.assign, tensor.dat, eager=True)
        elif tensor is None and self.access in {op3.MIN_WRITE, op3.MAX_WRITE}:
            tensor = assembler.allocate()
            self._initialize_minmax(tensor)

        # INC adds the contributions to the current output values. A zero
        # interpolation has no local kernels to write a WRITE output.
        # Therefore, we must zero the output explicitly in both cases.
        needs_zeroing = tensor is not None and (
            self.access is op3.INC or (self.access is op3.WRITE and not assembler.local_kernels))

        def callable():
            if self._needs_adjoint_weighting:
                self._update_weighted_dual_arg()
            if copy_input is not None:
                copy_input()
            if needs_zeroing:
                tensor.zero()
            result = assembler.assemble(tensor=tensor)
            if copy_output is not None:
                copy_output()
            return output if copy_output is not None else result

        return callable

    @property
    def _allowed_mat_types(self):
        return {"aij", "baij", "nest", "matfree", None}


class VomOntoVomInterpolator(SameMeshInterpolator):

    def __init__(self, expr: Interpolate, source_mesh, target_mesh):
        super().__init__(expr, source_mesh, target_mesh)

        if self.source_mesh.input_ordering is self.target_mesh:
            # The forward interpolation is a star forest reduction
            self.forward_reduce = True
            self.original_vom = self.source_mesh
        elif self.target_mesh.input_ordering is self.source_mesh:
            # The forward interpolation is a star forest broadcast
            self.forward_reduce = False
            self.original_vom = self.target_mesh
        else:
            raise ValueError(
                "The target vom and source vom must be linked by input ordering!"
            )

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None,
                      pyop3_compiler_parameters=None):
        if bcs:
            raise NotImplementedError("bcs not implemented for vom-to-vom interpolation.")
        mat_type = mat_type or "matfree"

        if self.rank == 1:
            f = tensor or Function(self.ufl_interpolate.function_space())
            self.mat = self._build_python_mat(get_mpi_dtype(f.dat.dtype, f.function_space().block_size)[0])
            if self.ufl_interpolate.is_adjoint:
                assert isinstance(self.dual_arg, Cofunction)
                assert isinstance(f, Cofunction)

                def callable() -> Cofunction:
                    with self.dual_arg.dat.vec_ro as source_vec:
                        coeff = expr_as_coeff(self.target_space, self.operand, self.ufl_interpolate.is_adjoint, self.source_mesh, source_vec)
                        with coeff.dat.vec_ro as coeff_vec, f.dat.vec_wo as target_vec:
                            self.mat.multHermitian(coeff_vec, target_vec)
                    return f
            else:
                assert isinstance(f, Function)

                def callable() -> Function:
                    coeff = expr_as_coeff(self.target_space, self.operand, self.ufl_interpolate.is_adjoint, self.source_mesh)
                    with coeff.dat.vec_ro as coeff_vec, f.dat.vec_wo as target_vec:
                        self.mat.mult(coeff_vec, target_vec)
                    return f
        elif self.rank == 2:
            if mat_type == "matfree":
                # Create a temporary function to get the correct MPI type
                temp_source_func = Function(self.interpolate_args[1].function_space())
                self.mat = self._build_python_mat(get_mpi_dtype(temp_source_func.dat.dtype, temp_source_func.function_space().block_size)[0])
            else:
                self.mat = self._create_permutation_mat(mat_type)

            def callable() -> Matrix:
                return Matrix(self.ufl_interpolate, self.mat)

        return callable

    def _build_python_mat(self, mpi_type) -> PETSc.Mat:
        # In this case mat_type="matfree", so we use the SF wrapped as a PETSc Mat
        # to perform the permutation.
        source_size, target_size, _, _ = self._get_mat_sizes()
        ctx = VomOntoVomMatContext(
            self.original_vom.input_ordering_sf,
            self.forward_reduce,
            self.target_space,
            self.source_mesh,
            self.operand,
            self.ufl_interpolate.is_adjoint,
        )
        ctx.mpi_type = mpi_type
        mat = PETSc.Mat().create(comm=self.source_mesh.comm)
        if self.forward_reduce:
            mat_size = (source_size, target_size)
        else:
            mat_size = (target_size, source_size)
        mat.setSizes(mat_size)
        mat.setType(mat.Type.PYTHON)
        mat.setPythonContext(ctx)
        mat.setUp()
        return mat

    def _create_permutation_mat(self, mat_type: Literal["aij", "baij"]) -> PETSc.Mat:
        """Create the PETSc matrix that represents the interpolation operator from a vertex-only mesh to
        its input ordering vertex-only mesh.

        Returns
        -------
        PETSc.Mat
            PETSc seqaij matrix
        """
        # Get correct local and global sizes for the matrix
        source_size, target_size, nleaves, local_sizes = self._get_mat_sizes()
        if mat_type == "baij" and self.target_space.block_size > 1:
            create = PETSc.Mat().createBAIJ
        else:
            create = PETSc.Mat().createAIJ
        mat = create(
            size=(target_size, source_size),
            bsize=self.target_space.block_size,
            nnz=1,
            comm=self.source_mesh.comm
        )
        mat.setUp()
        # To create the permutation matrix we broadcast an array of indices which are contiguous
        # across all ranks and then use these indices to set the values of the matrix directly.
        start = sum(local_sizes[:self.target_space.comm.rank])
        end = start + source_size[0]
        contiguous_indices = numpy.arange(start, end, dtype=IntType)
        perm = numpy.zeros(nleaves, dtype=IntType)  # result stored in here
        sf = self.original_vom.input_ordering_sf
        mpi_int = MPI._typedict[numpy.dtype(IntType).char]
        sf.bcastBegin(mpi_int, contiguous_indices, perm, MPI.REPLACE)
        sf.bcastEnd(mpi_int, contiguous_indices, perm, MPI.REPLACE)
        rows = numpy.arange(target_size[0] + 1, dtype=IntType)
        # Vector and Tensor valued functions are stored in a flattened array, so
        # we need to space out the column indices according to the block size
        cols = (self.target_space.block_size * perm[:, None] + numpy.arange(self.target_space.block_size, dtype=IntType)[None, :]).reshape(-1)
        mat.setValuesCSR(rows, cols, numpy.ones_like(cols, dtype=IntType))
        mat.assemble()
        if self.forward_reduce and not self.ufl_interpolate.is_adjoint:
            # The mat we have constructed thus far takes us from the input-ordering VOM to the
            # immersed VOM. If we're going the other way, then we need to transpose it,
            # unless we're doing the adjoint interpolation, since source_mesh and target_mesh
            # are defined assuming we're doing forward interpolation.
            mat.transpose()
        return mat

    def _get_mat_sizes(self):
        nroots, leaves, _ = self.original_vom.input_ordering_sf.getGraph()
        nleaves = len(leaves)
        local_sizes = self.target_space.comm.allgather(nroots)
        source_size = (self.target_space.block_size * nroots, self.target_space.block_size * sum(local_sizes))
        target_size = (
            self.target_space.block_size * nleaves,
            self.target_space.block_size * self.target_space.comm.allreduce(nleaves, op=MPI.SUM),
        )
        return source_size, target_size, nleaves, local_sizes

    @property
    def _allowed_mat_types(self):
        return {"aij", "baij", "matfree", None}


def get_runtime_tabulation_coordinates(arg: kernel_args.TabulationKernelArg,
                                       mesh: MeshGeometry) -> Function:
    """Return the reference coordinates that a kernel tabulates its runtime point at.

    Parameters
    ----------
    arg
        The kernel argument that receives the runtime points.
    mesh
        The mesh that the kernel iterates over.

    Returns
    -------
    Function
        The coordinates of the points of ``mesh`` in the reference cell of
        its parent mesh.

    Raises
    ------
    ValueError
        If ``arg`` is not the runtime tabulation argument.
    TypeError
        If ``mesh`` is not a :class:`VertexOnlyMesh`.
    """
    # FIXME: name-matching is a stopgap; get this from a coefficient map handed
    # down by compile_expression_dual_evaluation, or a Cofunction/Coargument target.
    if arg.loopy_arg.name != RUNTIME_POINT_VARIABLE:
        raise ValueError(f"Expecting the runtime tabulation argument {RUNTIME_POINT_VARIABLE}: got {arg.loopy_arg.name}")
    if not isinstance(mesh.topology, VertexOnlyMeshTopology):
        raise TypeError(f"Runtime tabulation is only supported on a VertexOnlyMesh: got {type(mesh.topology).__name__}")
    return mesh.reference_coordinates


@no_annotations
def expr_as_coeff(
        target_space: WithGeometry,
        operand: Expr,
        is_adjoint: bool,
        source_mesh: MeshGeometry,
        source_vec: PETSc.Vec | None = None,
) -> Function:
    """Return a Function that corresponds to the expression used at
    construction, where the expression has been interpolated into the P0DG
    function space on the source vertex-only mesh.

    Parameters
    ----------
    target_space : WithGeometry
        The function space on the target mesh which the expression is being
        interpolated into.
    operand : Expr
        The expression being interpolated.
    is_adjoint : bool
        Whether the interpolation is an adjoint interpolation.
    source_mesh : MeshGeometry
        The source vertex-only mesh on which the expression should be evaluated.
    source_vec : PETSc.Vec | None, optional
        Optional vector used to replace arguments in the expression.
        By default None.

    Returns
    -------
    Function
        A Function representing the expression as a coefficient on the
        source vertex-only mesh.

    """
    # Since we always output a coefficient when we don't have arguments in
    # the expression, we should evaluate the expression on the source mesh
    # so its dat can be sent to the target mesh.
    element = target_space.ufl_element()  # Could be vector/tensor valued
    # if we have any arguments in the expression we need to replace
    # them with equivalent coefficients now
    target_mesh = target_space.mesh()
    arguments = extract_arguments(operand)
    if len(arguments):
        if len(arguments) > 1:
            raise NotImplementedError("Can only interpolate expressions with one argument!")
        if source_vec is None:
            raise ValueError("Need to provide a source dat for the argument!")
        arg = arguments[0]
        source_space = arg.function_space()
        P0DG = FunctionSpace(target_mesh if is_adjoint else source_mesh, element)
        arg_coeff = Function(target_space if is_adjoint else source_space)
        arg_coeff.dat.data_wo[:] = source_vec.getArray(readonly=True).reshape(
            arg_coeff.dat.data_wo.shape
        )
        coeff_expr = replace(operand, {arg: arg_coeff})
        return Function(P0DG).interpolate(coeff_expr)
    else:
        P0DG = FunctionSpace(source_mesh, element)
        return Function(P0DG).interpolate(operand)


class VomOntoVomMatContext:
    """Object that facilitates matfree interpolation between a VertexOnlyMesh and its
    input_ordering VertexOnlyMesh. This wraps the star forest representing the permutation
    between the VOMs and advertises it as a PETSc Mat.

    Parameters
    ----------
    sf : PETSc.SF
        The PETSc Star Forest representing the permutation between the VOMs.
    forward_reduce : bool
        True if the forward interpolation is a reduction, False if it is a broadcast.
    target_space : WithGeometry
        The FunctionSpace being interpolated into.
    source_vom : MeshGeometry
        The VOM being interpolated from.
    operand : Expr
        The expression being interpolated.
    is_adjoint : bool
        Whether the interpolation is an adjoint interpolation.

    Raises
    ------
    ValueError
        If the source and target vertex-only meshes are not linked by input_ordering.
    """
    def __init__(
            self,
            sf: PETSc.SF,
            forward_reduce: bool,
            target_space: WithGeometry,
            source_vom: MeshGeometry,
            operand: Expr,
            is_adjoint: bool,
    ):

        self.forward_reduce = forward_reduce
        self.sf = sf
        self.target_space = target_space
        self.source_vom = source_vom
        self.operand = operand
        self.is_adjoint = is_adjoint

    @property
    def mpi_type(self):
        """
        The MPI type to use for the PETSc SF.

        Should correspond to the underlying data type of the PETSc Vec.
        """
        return self._mpi_type

    @mpi_type.setter
    def mpi_type(self, val):
        self._mpi_type = val

    def reduce(self, source_vec: PETSc.Vec, target_vec: PETSc.Vec) -> None:
        """Reduce data in source_vec using the PETSc SF.

        Parameters
        ----------
        source_vec : PETSc.Vec
            The vector to reduce.
        target_vec : PETSc.Vec
            The vector to store the result in.
        """
        source_arr = source_vec.getArray(readonly=True)
        target_arr = target_vec.getArray()
        self.sf.reduceBegin(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )
        self.sf.reduceEnd(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )

    def broadcast(self, source_vec: PETSc.Vec, target_vec: PETSc.Vec) -> None:
        """Broadcast data in source_vec using the PETSc SF, storing the
        result in target_vec.

        Parameters
        ----------
        source_vec : PETSc.Vec
            The vector to broadcast.
        target_vec : PETSc.Vec
            The vector to store the result in.
        """
        source_arr = source_vec.getArray(readonly=True)
        target_arr = target_vec.getArray()
        self.sf.bcastBegin(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )
        self.sf.bcastEnd(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )

    def mult(self, mat: PETSc.Mat, source_vec: PETSc.Vec, target_vec: PETSc.Vec) -> None:
        """Apply the interpolation operator to source_vec, storing the
        result in target_vec.

        Parameters
        ----------
        mat : PETSc.Mat
            Required by petsc4py but unused.
        source_vec : PETSc.Vec
            The vector to interpolate.
        target_vec : PETSc.Vec
            The vector to store the result in.
        """
        # Need to convert the expression into a coefficient
        # so that we can broadcast/reduce it
        coeff = expr_as_coeff(self.target_space, self.operand, self.is_adjoint, self.source_vom, source_vec)
        with coeff.dat.vec_ro as coeff_vec:
            if self.forward_reduce:
                self.reduce(coeff_vec, target_vec)
            else:
                self.broadcast(coeff_vec, target_vec)

    def multHermitian(self, mat: PETSc.Mat, source_vec: PETSc.Vec, target_vec: PETSc.Vec) -> None:
        """Apply the adjoint of the interpolation operator to source_vec, storing the
        result in target_vec. Since ``VomOntoVomMat`` represents a permutation, it is
        real-valued and thus the Hermitian adjoint is the transpose.

        Parameters
        ----------
        mat : PETSc.Mat
            Required by petsc4py but unused.
        source_vec : PETSc.Vec
            The vector to adjoint interpolate.
        target_vec : PETSc.Vec
            The vector to store the result in.
        """
        self.multTranspose(mat, source_vec, target_vec)

    def multTranspose(self, mat: PETSc.Mat, source_vec: PETSc.Vec, target_vec: PETSc.Vec) -> None:
        """Apply the tranpose of the interpolation operator to source_vec, storing the
        result in target_vec. Called by `self.multHermitian`.

        Parameters
        ----------
        mat : PETSc.Mat
            Required by petsc4py but unused.
        source_vec : PETSc.Vec
            The vector to transpose interpolate.
        target_vec : PETSc.Vec
            The vector to store the result in.

        """
        if self.forward_reduce:
            self.broadcast(source_vec, target_vec)
        else:
            # We need to ensure the target vec is zeroed for SF Reduce to
            # represent multHermitian in case the interpolation matrix is not
            # square (in which case it will have columns which are zero). This
            # happens when we interpolate from an input-ordering vertex-only
            # mesh to an immersed vertex-only mesh where the input ordering
            # contains points that are not in the immersed mesh. The resulting
            # interpolation matrix will have columns of zeros for the points
            # that are not in the immersed mesh. The adjoint interpolation
            # matrix will then have rows of zeros for those points.
            target_vec.zeroEntries()
            self.reduce(source_vec, target_vec)

    def duplicate(self, mat: PETSc.Mat | None = None, op: PETSc.Mat.DuplicateOption | None = None) -> PETSc.Mat:
        """Duplicate the matrix. Needed to wrap as a PETSc Python Mat.

        Parameters
        ----------
        mat : PETSc.Mat | None, optional
            Unused, by default None
        op : PETSc.Mat.DuplicateOption | None, optional
            Unused, by default None

        Returns
        -------
        PETSc.Mat
            VomOntoVomMatContext wrapped as a PETSc Mat of type python.
        """
        return self


class MixedInterpolator(Interpolator):
    """Interpolator between MixedFunctionSpaces."""

    def _get_sub_interpolators(
            self, bcs: Iterable[DirichletBC] | None = None
    ) -> dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]]:
        """Gets `Interpolator`s and boundary conditions for each sub-Interpolate
        in the mixed expression.

        Returns
        -------
        dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]]
            A map from block index tuples to `Interpolator`s and bcs.
        """
        # Get the primal spaces
        spaces = tuple(
            a.function_space().dual() if isinstance(a, Coargument) else a.function_space() for a in self.interpolate_args
        )
        # TODO consider a stricter equality test for indexed MixedFunctionSpace
        # See https://github.com/firedrakeproject/firedrake/issues/4668
        space_equals = lambda V1, V2: V1 == V2 and V1.parent == V2.parent and V1.index == V2.index

        # Interpolation onto a mixed space is the sum of the interpolations onto
        # its subspaces. The Jacobian has the target as an argument, so each of
        # its blocks interpolates onto one target subspace.
        jacobian = self.ufl_interpolate._ufl_expr_reconstruct_(self.operand, self.target_space)
        target = jacobian.argument_slots()[0].number()
        splitter = ExtractSubBlock()

        # Get sub-interpolators and sub-bcs for each block
        Isub: dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]] = {}
        for indices in numpy.ndindex(tuple(len(a.function_space()) for a in jacobian.arguments())):
            block = splitter.split(jacobian, indices)
            if not isinstance(self.dual_arg, Coargument):
                # Contract the block with the dual argument on the same target subspace
                block = action(block, splitter.split(self.dual_arg, indices[target:target + 1]))
            if isinstance(block, ZeroBaseForm):
                # Ensure block sparsity
                continue
            sub_bcs = []
            for space, index in zip(spaces, indices):
                subspace = space.sub(index)
                # filter the bcs and drop index information because the sub interpolate
                # doesn't know that it's part of a bigger expression
                sub_bcs.extend(
                    bc.reconstruct(V=subspace.collapse())
                    for bc in bcs if space_equals(bc.function_space(), subspace)
                )
            Isub[indices] = (get_interpolator(block), sub_bcs)

        return Isub

    def _build_matnest(
            self,
            Isub: dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]],
            sub_mat_type: Literal["aij", "baij"],
    ) -> PETSc.Mat:
        """Return a PETSc nested matrix built from sub-interpolator matrices."""
        spaces = tuple(a.function_space() for a in self.interpolate_args)
        shape = tuple(len(V) for V in spaces)
        blocks = numpy.full(shape, PETSc.Mat(), dtype=object)
        for indices, (interp, sub_bcs) in Isub.items():
            blocks[indices] = interp._get_callable(bcs=sub_bcs, mat_type=sub_mat_type)().petscmat
        isrows, iscols = (V.field_ises for V in spaces)
        return PETSc.Mat().createNest(blocks, isrows=isrows, iscols=iscols, comm=self.target_space.comm)

    def _build_aij(
            self,
            Isub: dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]],
    ) -> PETSc.Mat:
        """Return a PETSc AIJ matrix built from sub-interpolator matrices by converting a
        nested matrix."""
        matnest = self._build_matnest(Isub, sub_mat_type="aij")
        return matnest.convert("aij")

    def _get_callable(
        self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None,
        pyop3_compiler_parameters=None,
    ):
        mat_type = mat_type or "aij"
        sub_mat_type = sub_mat_type or "aij"
        Isub = self._get_sub_interpolators(bcs=bcs)
        V_dest = self.ufl_interpolate.function_space() or self.target_space
        f = tensor or Function(V_dest)
        if self.rank == 2:
            if mat_type == "nest":
                build = partial(self._build_matnest, Isub, sub_mat_type)
            else:
                assert mat_type == "aij"
                build = partial(self._build_aij, Isub)

            def callable() -> Matrix:
                return Matrix(self.ufl_interpolate, build(), bcs=bcs)
        elif self.rank == 1:
            def callable() -> Function | Cofunction:
                for k, sub_tensor in enumerate(f.subfunctions):
                    sub_tensor.assign(sum(
                        interp.assemble(bcs=sub_bcs) for indices, (interp, sub_bcs) in Isub.items() if indices[0] == k
                    ))
                return f
        else:
            def callable() -> Number:
                return sum(interp.assemble(bcs=sub_bcs) for (interp, sub_bcs) in Isub.values())
        return callable

    @property
    def _allowed_mat_types(self):
        return {"aij", "nest", "matfree", None}
