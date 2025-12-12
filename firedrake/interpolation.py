import numpy
import os
import tempfile
import abc

from functools import partial
from typing import Hashable, Literal, Callable, Iterable
from dataclasses import asdict, dataclass
from numbers import Number

from ufl.algorithms import extract_arguments, replace
from ufl.domain import extract_unique_domain
from ufl.classes import Expr
from ufl.duals import is_dual
from ufl.constantvalue import zero, as_ufl
from ufl.form import ZeroBaseForm, BaseForm
from ufl.core.interpolate import Interpolate as UFLInterpolate

from pyop2 import op2
from pyop2.caching import memory_and_disk_cache

from finat.ufl import TensorElement, VectorElement, MixedElement
from finat.element_factory import create_element

from tsfc.driver import compile_expression_dual_evaluation
from tsfc.ufl_utils import extract_firedrake_constants, hash_expr

from firedrake.utils import IntType, ScalarType, cached_property, known_pyop2_safe, tuplify
from firedrake.pointeval_utils import runtime_quadrature_element
from firedrake.tsfc_interface import extract_numbered_coefficients, _cachedir
from firedrake.ufl_expr import Argument, Coargument, action
from firedrake.mesh import MissingPointsBehaviour, VertexOnlyMeshTopology, MeshGeometry, MeshTopology, VertexOnlyMesh
from firedrake.petsc import PETSc
from firedrake.halo import _get_mtype
from firedrake.functionspaceimpl import WithGeometry
from firedrake.matrix import ImplicitMatrix, MatrixBase, Matrix
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.bcs import DirichletBC
from firedrake.formmanipulation import split_form
from firedrake.functionspace import VectorFunctionSpace, TensorFunctionSpace, FunctionSpace
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.exceptions import (
    DofNotDefinedError, VertexOnlyMeshMissingPointsError, NonUniqueMeshSequenceError
)

from mpi4py import MPI

from pyadjoint.tape import stop_annotating, no_annotations

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
    subset : pyop2.types.set.Subset or None
        An optional subset to apply the interpolation over.
        Cannot, at present, be used when interpolating across meshes unless
        the target mesh is a :func:`.VertexOnlyMesh`.
    access : pyop2.types.access.Access or None
        The pyop2 access descriptor for combining updates to shared
        DoFs. Possible values include ``WRITE``, ``MIN``, ``MAX``, and ``INC``.
        Only ``WRITE`` is supported at present when interpolating across meshes
        unless the target mesh is a :func:`.VertexOnlyMesh`. Only ``INC`` is
        supported for the matrix-free adjoint interpolation.
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
    subset: op2.Subset | None = None
    access: Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC] | None = None
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
        interp_data = interp_data or asdict(self.options)
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
        if len(arguments) == 2 and has_mixed_arguments:
            return MixedInterpolator(self)

        operand, = self.ufl_operands
        target_mesh = self.target_space.mesh()

        try:
            source_mesh = extract_unique_domain(operand) or target_mesh
        except ValueError:
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
            and target_mesh.submesh_ancesters[-1] is source_mesh.submesh_ancesters[-1]
            and target_mesh.topological_dimension == source_mesh.topological_dimension
        )
        if target_mesh is source_mesh or submesh_interp_implemented:
            return SameMeshInterpolator(self)

        if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
            if isinstance(source_mesh.topology, VertexOnlyMeshTopology):
                return VomOntoVomInterpolator(self)
            if target_mesh.geometric_dimension != source_mesh.geometric_dimension:
                raise ValueError("Cannot interpolate onto a VertexOnlyMesh of a different geometric dimension.")
            return SameMeshInterpolator(self)

        if has_mixed_arguments or len(self.target_space) > 1:
            return MixedInterpolator(self)

        return CrossMeshInterpolator(self)


@PETSc.Log.EventDecorator()
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
        # Delay calling .unique() because MixedInterpolator is fine with MeshSequence
        self.target_mesh = self.target_space.mesh()
        """The domain we are interpolating into."""

        try:
            source_mesh = extract_unique_domain(operand)
        except ValueError:
            source_mesh = extract_unique_domain(operand, expand_mesh_sequence=False)
        self.source_mesh = source_mesh or self.target_mesh
        """The domain we are interpolating from."""

        # Interpolation options
        self.subset = expr.options.subset
        self.allow_missing_dofs = expr.options.allow_missing_dofs
        self.default_missing_val = expr.options.default_missing_val
        self.access = expr.options.access

    @abc.abstractmethod
    def _get_callable(
        self,
        tensor: Function | Cofunction | MatrixBase | None = None,
        bcs: Iterable[DirichletBC] | None = None,
        mat_type: Literal["aij", "baij", "nest", "matfree"] | None = None,
        sub_mat_type: Literal["aij", "baij"] | None = None,
    ) -> Callable[[], Function | Cofunction | PETSc.Mat | Number]:
        """Return a callable to perform interpolation.

        If ``self.rank == 2``, then the callable must return a PETSc matrix.
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

    def assemble(
        self,
        tensor: Function | Cofunction | MatrixBase | None = None,
        bcs: Iterable[DirichletBC] | None = None,
        mat_type: Literal["aij", "baij", "nest", "matfree"] | None = None,
        sub_mat_type: Literal["aij", "baij"] | None = None,
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
            For cross-mesh interpolation, only ``"aij"`` is supported. For same-mesh
            interpolation, ``"aij"`` and ``"baij"`` are supported. For same/cross mesh interpolation
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
            ctx = ImplicitMatrixContext(
                self.ufl_interpolate, row_bcs=bcs, col_bcs=bcs,
            )
            return ImplicitMatrix(self.ufl_interpolate, ctx, bcs=bcs)

        result = self._get_callable(tensor=tensor, bcs=bcs, mat_type=mat_type, sub_mat_type=sub_mat_type)()

        if self.rank == 2:
            # Assembling the operator
            assert isinstance(tensor, MatrixBase | None)
            assert isinstance(result, PETSc.Mat)
            if tensor:
                result.copy(tensor.petscmat)
                return tensor
            else:
                return Matrix(self.ufl_interpolate, result, bcs=bcs)
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
    def __init__(self, expr: Interpolate):
        super().__init__(expr)
        if self.access and self.access != op2.WRITE:
            raise NotImplementedError(
                "Access other than op2.WRITE not implemented for cross-mesh interpolation."
            )
        else:
            self.access = op2.WRITE

        # TODO check V.finat_element.is_lagrange() once https://github.com/firedrakeproject/fiat/pull/200 is released
        target_element = self.target_space.ufl_element()
        if not ((isinstance(target_element, MixedElement)
                 and all(sub.mapping() == "identity" for sub in target_element.sub_elements))
                or target_element.mapping() == "identity"):
            # Identity mapping between reference cell and physical coordinates
            # implies point evaluation nodes.
            raise NotImplementedError(
                "Can only cross-mesh interpolate into spaces with point evaluation nodes."
            )

        if self.allow_missing_dofs:
            self.missing_points_behaviour = MissingPointsBehaviour.IGNORE
        else:
            self.missing_points_behaviour = MissingPointsBehaviour.ERROR

        if self.source_mesh.unique().geometric_dimension != self.target_mesh.unique().geometric_dimension:
            raise ValueError("Geometric dimensions of source and destination meshes must match.")

        dest_element = self.target_space.ufl_element()
        if isinstance(dest_element, MixedElement):
            if isinstance(dest_element, VectorElement | TensorElement):
                # In this case all sub elements are equal
                base_element = dest_element.sub_elements[0]
                if base_element.reference_value_shape != ():
                    raise NotImplementedError(
                        "Can't yet cross-mesh interpolate onto function spaces made from VectorElements "
                        "or TensorElements made from sub elements with value shape other than ()."
                    )
                self.dest_element = base_element
            else:
                raise NotImplementedError("Interpolation with MixedFunctionSpace requires MixedInterpolator.")
        else:
            # scalar fiat/finat element
            self.dest_element = dest_element

    def _get_symbolic_expressions(self) -> tuple[Interpolate, Interpolate]:
        """Return the symbolic ``Interpolate`` expressions for point evaluation and
        re-ordering into the input-ordering VertexOnlyMesh.

        Returns
        -------
        tuple[Interpolate, Interpolate]
            A tuple containing the point evaluation interpolation and the
            input-ordering interpolation.

        Raises
        ------
        DofNotDefinedError
            If any DoFs in the target mesh cannot be defined in the source mesh.
        """
        from firedrake.assemble import assemble
        # Immerse coordinates of target space point evaluation dofs in src_mesh
        target_space_vec = VectorFunctionSpace(self.target_mesh.unique(), self.dest_element)
        f_dest_node_coords = assemble(interpolate(self.target_mesh.unique().coordinates, target_space_vec))
        dest_node_coords = f_dest_node_coords.dat.data_ro.reshape(-1, self.target_mesh.unique().geometric_dimension)
        try:
            vom = VertexOnlyMesh(
                self.source_mesh.unique(),
                dest_node_coords,
                redundant=False,
                missing_points_behaviour=self.missing_points_behaviour,
            )
        except VertexOnlyMeshMissingPointsError:
            raise DofNotDefinedError(f"The given target function space on domain {self.target_mesh} "
                                     "contains degrees of freedom which cannot cannot be defined in the "
                                     f"source function space on domain {self.source_mesh}. "
                                     "This may be because the target mesh covers a larger domain than the "
                                     "source mesh. To disable this error, set allow_missing_dofs=True.")

        # Get the correct type of function space
        shape = self.target_space.ufl_function_space().value_shape
        if len(shape) == 0:
            fs_type = FunctionSpace
        elif len(shape) == 1:
            fs_type = partial(VectorFunctionSpace, dim=shape[0])
        else:
            symmetry = self.target_space.ufl_element().symmetry()
            fs_type = partial(TensorFunctionSpace, shape=shape, symmetry=symmetry)

        # Get expression for point evaluation at the dest_node_coords
        P0DG_vom = fs_type(vom, "DG", 0)
        point_eval = interpolate(self.operand, P0DG_vom)

        # Interpolate into the input-ordering VOM
        P0DG_vom_input_ordering = fs_type(vom.input_ordering, "DG", 0)

        arg = Argument(P0DG_vom, 0 if self.ufl_interpolate.is_adjoint else 1)
        point_eval_input_ordering = interpolate(arg, P0DG_vom_input_ordering)
        return point_eval, point_eval_input_ordering

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None):
        from firedrake.assemble import assemble
        if bcs:
            raise NotImplementedError("bcs not implemented for cross-mesh interpolation.")
        mat_type = mat_type or "aij"

        # self.ufl_interpolate.function_space() is None in the 0-form case
        V_dest = self.ufl_interpolate.function_space() or self.target_space
        f = tensor or Function(V_dest)

        point_eval, point_eval_input_ordering = self._get_symbolic_expressions()
        P0DG_vom_input_ordering = point_eval_input_ordering.argument_slots()[0].function_space().dual()

        if self.rank == 2:
            assert mat_type == "aij"
            # The cross-mesh interpolation matrix is the product of the
            # `self.point_eval_interpolate` and the permutation
            # given by `self.to_input_ordering_interpolate`.
            if self.ufl_interpolate.is_adjoint:
                symbolic = action(point_eval, point_eval_input_ordering)
            else:
                symbolic = action(point_eval_input_ordering, point_eval)

            def callable() -> PETSc.Mat:
                return assemble(symbolic, mat_type=mat_type).petscmat
        elif self.ufl_interpolate.is_adjoint:
            assert self.rank == 1
            # f_src is a cofunction on V_dest.dual
            cofunc = self.dual_arg
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
            def callable() -> Cofunction:
                f_src_at_src_node_coords = assemble(action(point_eval_input_ordering, f_input_ordering))
                assemble(action(point_eval, f_src_at_src_node_coords), tensor=f)
                return f
        else:
            assert self.rank in {0, 1}
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

            def callable() -> Function | Number:
                assemble(action(point_eval_input_ordering, point_eval), tensor=f_point_eval_input_ordering)
                # We assign these values to the output function
                if self.allow_missing_dofs and self.default_missing_val is None:
                    indices = numpy.where(~numpy.isnan(f_point_eval_input_ordering.dat.data_ro))[0]
                    f.dat.data_wo[indices] = f_point_eval_input_ordering.dat.data_ro[indices]
                else:
                    f.dat.data_wo[:] = f_point_eval_input_ordering.dat.data_ro[:]

                if self.rank == 0:
                    # We take the action of the dual_arg on the interpolated function
                    assert isinstance(self.dual_arg, Cofunction)
                    return assemble(action(self.dual_arg, f))
                else:
                    return f
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
    def __init__(self, expr):
        super().__init__(expr)
        subset = self.subset
        if subset is None:
            target = self.target_mesh.unique().topology
            source = self.source_mesh.unique().topology
            if all(isinstance(m, MeshTopology) for m in [target, source]) and target is not source:
                composed_map, result_integral_type = source.trans_mesh_entity_map(target, "cell", "everywhere", None)
                if result_integral_type != "cell":
                    raise AssertionError("Only cell-cell interpolation supported.")
                indices_active = composed_map.indices_active_with_halo
                make_subset = not indices_active.all()
                make_subset = target.comm.allreduce(make_subset, op=MPI.LOR)
                if make_subset:
                    if not self.allow_missing_dofs:
                        raise ValueError("Iteration (sub)set unclear: run with `allow_missing_dofs=True`.")
                    subset = op2.Subset(target.cell_set, numpy.where(indices_active))
                else:
                    # Do not need subset as target <= source.
                    pass
        self.subset = subset

        if not isinstance(self.dual_arg, Coargument):
            # Matrix-free assembly of 0-form or 1-form requires INC access
            if self.access and self.access != op2.INC:
                raise ValueError("Matfree adjoint interpolation requires INC access")
            self.access = op2.INC
        elif self.access is None:
            # Default access for forward 1-form or 2-form (forward and adjoint)
            self.access = op2.WRITE

    def _get_tensor(self, mat_type: Literal["aij", "baij"]) -> op2.Mat | Function | Cofunction:
        """Return a suitable tensor to interpolate into.

        Parameters
        ----------
        mat_type
            The PETSc matrix type to use when assembling a rank 2 interpolation.
            Only ``"aij"`` and ``"baij"`` are currently allowed.

        Returns
        -------
        op2.Mat | Function | Cofunction
            The tensor to interpolate into.
        """
        if self.rank == 0:
            R = FunctionSpace(self.target_mesh.unique(), "Real", 0)
            f = Function(R, dtype=ScalarType)
        elif self.rank == 1:
            f = Function(self.ufl_interpolate.function_space())
            if self.access in {op2.MIN, op2.MAX}:
                finfo = numpy.finfo(f.dat.dtype)
                if self.access == op2.MIN:
                    val = Constant(finfo.max)
                else:
                    val = Constant(finfo.min)
                f.assign(val)
        elif self.rank == 2:
            sparsity = self._get_monolithic_sparsity(mat_type)
            f = op2.Mat(sparsity)
        else:
            raise ValueError(f"Cannot interpolate an expression with {self.rank} arguments")
        return f

    def _get_monolithic_sparsity(self, mat_type: Literal["aij", "baij"]) -> op2.Sparsity:
        """Returns op2.Sparsity for the interpolation matrix. Only mat_type 'aij' and 'baij'
        are currently supported.

        Parameters
        ----------
        mat_type
            The PETSc matrix type to use when assembling a rank 2 interpolation.
            Only ``"aij"`` and ``"baij"`` are currently allowed.

        Returns
        -------
        op2.Sparsity
            The sparsity pattern for the interpolation matrix.
        """
        Vrow = self.interpolate_args[0].function_space()
        Vcol = self.interpolate_args[1].function_space()
        if len(Vrow) > 1 or len(Vcol) > 1:
            raise NotImplementedError("Interpolation matrix with MixedFunctionSpace requires MixedInterpolator")
        Vrow_map = get_interp_node_map(self.source_mesh.unique(), self.target_mesh.unique(), Vrow)
        Vcol_map = get_interp_node_map(self.source_mesh.unique(), self.target_mesh.unique(), Vcol)
        sparsity = op2.Sparsity((Vrow.dof_dset, Vcol.dof_dset),
                                [(Vrow_map, Vcol_map, None)],  # non-mixed
                                name=f"{Vrow.name}_{Vcol.name}_sparsity",
                                nest=False,
                                block_sparse=(mat_type == "baij"))
        return sparsity

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None):
        mat_type = mat_type or "aij"
        if (isinstance(tensor, Cofunction) and isinstance(self.dual_arg, Cofunction)) and set(tensor.dat).intersection(set(self.dual_arg.dat)):
            # adjoint one-form case: we need an empty tensor, so if it shares dats with
            # the dual_arg we cannot use it directly, so we store it
            f = self._get_tensor(mat_type)
            copyout = (partial(f.dat.copy, tensor.dat),)
        else:
            f = tensor or self._get_tensor(mat_type)
            copyout = ()

        op2_tensor = f if isinstance(f, op2.Mat) else f.dat
        loops = []
        if self.access is op2.INC:
            loops.append(op2_tensor.zero)

        # Arguments in the operand are allowed to be from a MixedFunctionSpace
        # We need to split the target space V and generate separate kernels
        if self.rank == 2:
            expressions = {(0,): self.ufl_interpolate}
        elif isinstance(self.dual_arg, Coargument):
            # Split in the coargument
            expressions = dict(split_form(self.ufl_interpolate))
        else:
            assert isinstance(self.dual_arg, Cofunction)
            # Split in the cofunction: split_form can only split in the coargument
            # Replace the cofunction with a coargument to construct the Jacobian
            interp = self.ufl_interpolate._ufl_expr_reconstruct_(self.operand, self.target_space)
            # Split the Jacobian into blocks
            interp_split = dict(split_form(interp))
            # Split the cofunction
            dual_split = dict(split_form(self.dual_arg))
            # Combine the splits by taking their action
            expressions = {i: action(interp_split[i], dual_split[i[-1:]]) for i in interp_split}

        # Interpolate each sub expression into each function space
        for indices, sub_expr in expressions.items():
            sub_op2_tensor = op2_tensor[indices[0]] if self.rank == 1 else op2_tensor
            loops.extend(_build_interpolation_callables(sub_expr, sub_op2_tensor, self.access, self.subset, bcs))

        if bcs and self.rank == 1:
            loops.extend(partial(bc.apply, f) for bc in bcs)

        loops.extend(copyout)

        def callable() -> Function | Cofunction | PETSc.Mat | Number:
            for l in loops:
                l()
            if self.rank == 0:
                return f.dat.data.item()
            elif self.rank == 2:
                return f.handle  # In this case f is an op2.Mat
            else:
                return f

        return callable

    @property
    def _allowed_mat_types(self):
        return {"aij", "baij", "matfree", None}


class VomOntoVomInterpolator(SameMeshInterpolator):

    def __init__(self, expr: Interpolate):
        super().__init__(expr)

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None):
        if bcs:
            raise NotImplementedError("bcs not implemented for vom-to-vom interpolation.")
        mat_type = mat_type or "matfree"
        self.mat = VomOntoVomMat(self, mat_type=mat_type)
        if self.rank == 1:
            f = tensor or self._get_tensor(mat_type)
            # NOTE: get_dat_mpi_type ensures we get the correct MPI type for the
            # data, including the correct data size and dimensional information
            # (so for vector function spaces in 2 dimensions we might need a
            # concatenation of 2 MPI.DOUBLE types when we are in real mode)
            self.mat.mpi_type = _get_mtype(f.dat)[0]
            if self.ufl_interpolate.is_adjoint:
                assert isinstance(self.dual_arg, Cofunction)
                assert isinstance(f, Cofunction)

                def callable() -> Cofunction:
                    with self.dual_arg.dat.vec_ro as source_vec:
                        coeff = self.mat.expr_as_coeff(source_vec)
                        with coeff.dat.vec_ro as coeff_vec, f.dat.vec_wo as target_vec:
                            self.mat.handle.multHermitian(coeff_vec, target_vec)
                    return f
            else:
                assert isinstance(f, Function)

                def callable() -> Function:
                    coeff = self.mat.expr_as_coeff()
                    with coeff.dat.vec_ro as coeff_vec, f.dat.vec_wo as target_vec:
                        self.mat.handle.mult(coeff_vec, target_vec)
                    return f
        elif self.rank == 2:
            # Create a temporary function to get the correct MPI type
            temp_source_func = Function(self.interpolate_args[1].function_space())
            self.mat.mpi_type = _get_mtype(temp_source_func.dat)[0]

            def callable() -> PETSc.Mat:
                return self.mat.handle

        return callable

    @property
    def _allowed_mat_types(self):
        return {"aij", "baij", "matfree", None}


@known_pyop2_safe
def _build_interpolation_callables(
    expr: Interpolate | ZeroBaseForm,
    tensor: op2.Dat | op2.Mat | op2.Global,
    access: Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC],
    subset: op2.Subset | None = None,
    bcs: Iterable[DirichletBC] | None = None
) -> tuple[Callable, ...]:
    """Return a tuple of callables which calculate the interpolation.

    Parameters
    ----------
    expr : ufl.Interpolate | ufl.ZeroBaseForm
        The symbolic interpolation expression, or a ZeroBaseForm. ZeroBaseForms
        are simplified here to avoid code generation when access is WRITE or INC.
    tensor : op2.Dat | op2.Mat | op2.Global
        Object to hold the result of the interpolation.
    access : Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC]
        op2 access descriptor
    subset : op2.Subset | None
        An optional subset to apply the interpolation over, by default None.
    bcs : Iterable[DirichletBC] | None
        An optional list of boundary conditions to zero-out in the
        output function space. Interpolator rows or columns which are
        associated with boundary condition nodes are zeroed out when this is
        specified. By default None, by default None.

    Returns
    -------
    tuple[Callable, ...]
        Tuple of callables which perform the interpolation.
    """
    if isinstance(expr, ZeroBaseForm):
        # Zero simplification, avoid code-generation
        if access is op2.INC:
            return ()
        elif access is op2.WRITE:
            return (partial(tensor.zero, subset=subset),)
        # Unclear how to avoid codegen for MIN and MAX
        # Reconstruct the expression as an Interpolate
        V = expr.arguments()[-1].function_space().dual()
        expr = interpolate(zero(V.value_shape), V)

    if not isinstance(expr, Interpolate):
        raise ValueError("Expecting to interpolate a symbolic Interpolate expression.")

    dual_arg, operand = expr.argument_slots()
    assert isinstance(dual_arg, Cofunction | Coargument)
    V = dual_arg.function_space().dual()

    if access is op2.READ:
        raise ValueError("Can't have READ access for output function")

    # NOTE: The par_loop is always over the target mesh cells.
    target_mesh = V.mesh()
    source_mesh = extract_unique_domain(operand) or target_mesh
    target_element = V.ufl_element()
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        # For interpolation onto a VOM, we use a FInAT QuadratureElement as the
        # target element with runtime point set expressions as their
        # quadrature rule point set.
        rt_var_name = "rt_X"
        target_element = runtime_quadrature_element(source_mesh, target_element,
                                                    rt_var_name=rt_var_name)

    cell_set = target_mesh.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset

    parameters = {}
    parameters['scalar_type'] = ScalarType

    copyin = ()
    copyout = ()

    # For the matfree adjoint 1-form and the 0-form, the cellwise kernel will add multiple
    # contributions from the facet DOFs of the dual argument.
    # The incoming Cofunction needs to be weighted by the reciprocal of the DOF multiplicity.
    if isinstance(dual_arg, Cofunction) and not create_element(target_element).is_dg():
        # Create a buffer for the weighted Cofunction
        W = dual_arg.function_space()
        v = Function(W)
        expr = expr._ufl_expr_reconstruct_(operand, v=v)
        copyin += (partial(dual_arg.dat.copy, v.dat),)

        # Compute the reciprocal of the DOF multiplicity
        wdat = W.make_dat()
        m_ = get_interp_node_map(source_mesh, target_mesh, W)
        wsize = W.finat_element.space_dimension() * W.block_size
        kernel_code = f"""
        void multiplicity(PetscScalar *restrict w) {{
            for (PetscInt i=0; i<{wsize}; i++) w[i] += 1;
        }}"""
        kernel = op2.Kernel(kernel_code, "multiplicity")
        op2.par_loop(kernel, cell_set, wdat(op2.INC, m_))
        with wdat.vec as w:
            w.reciprocal()

        # Create a callable to apply the weight
        with wdat.vec_ro as w, v.dat.vec as y:
            copyin += (partial(y.pointwiseMult, y, w),)

    kernel = compile_expression(cell_set.comm, expr, target_element,
                                domain=source_mesh, parameters=parameters)
    ast = kernel.ast
    oriented = kernel.oriented
    needs_cell_sizes = kernel.needs_cell_sizes
    coefficient_numbers = kernel.coefficient_numbers
    needs_external_coords = kernel.needs_external_coords
    name = kernel.name
    kernel = op2.Kernel(ast, name, requires_zeroed_output_arguments=(access is not op2.INC),
                        flop_count=kernel.flop_count, events=(kernel.event,))

    parloop_args = [kernel, cell_set]

    coefficients = extract_numbered_coefficients(expr, coefficient_numbers)
    if needs_external_coords:
        coefficients = [source_mesh.coordinates] + coefficients

    if any(c.dat == tensor for c in coefficients):
        output = tensor
        tensor = op2.Dat(tensor.dataset)
        if access is not op2.WRITE:
            copyin += (partial(output.copy, tensor), )
        copyout += (partial(tensor.copy, output), )

    arguments = expr.arguments()
    if isinstance(tensor, op2.Global):
        parloop_args.append(tensor(access))
    elif isinstance(tensor, op2.Dat):
        V_dest = arguments[-1].function_space()
        m_ = get_interp_node_map(source_mesh, target_mesh, V_dest)
        parloop_args.append(tensor(access, m_))
    else:
        assert access == op2.WRITE  # Other access descriptors not done for Matrices.
        Vrow = arguments[0].function_space()
        Vcol = arguments[1].function_space()
        assert tensor.handle.getSize() == (Vrow.dim(), Vcol.dim())
        rows_map = get_interp_node_map(source_mesh, target_mesh, Vrow)
        columns_map = get_interp_node_map(source_mesh, target_mesh, Vcol)
        lgmaps = None
        if bcs:
            if is_dual(Vrow):
                Vrow = Vrow.dual()
            if is_dual(Vcol):
                Vcol = Vcol.dual()
            bc_rows = [bc for bc in bcs if bc.function_space() == Vrow]
            bc_cols = [bc for bc in bcs if bc.function_space() == Vcol]
            lgmaps = [(Vrow.local_to_global_map(bc_rows), Vcol.local_to_global_map(bc_cols))]
        parloop_args.append(tensor(access, (rows_map, columns_map), lgmaps=lgmaps))

    if oriented:
        co = target_mesh.cell_orientations()
        parloop_args.append(co.dat(op2.READ, co.cell_node_map()))

    if needs_cell_sizes:
        cs = source_mesh.cell_sizes
        parloop_args.append(cs.dat(op2.READ, cs.cell_node_map()))

    for coefficient in coefficients:
        m_ = get_interp_node_map(source_mesh, target_mesh, coefficient.function_space())
        parloop_args.append(coefficient.dat(op2.READ, m_))

    for const in extract_firedrake_constants(expr):
        parloop_args.append(const.dat(op2.READ))

    # Finally, add the target mesh reference coordinates if they appear in the kernel
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        if target_mesh is not source_mesh:
            # NOTE: TSFC will sometimes drop run-time arguments in generated
            # kernels if they are deemed not-necessary.
            # FIXME: Checking for argument name in the inner kernel to decide
            # whether to add an extra coefficient is a stopgap until
            # compile_expression_dual_evaluation
            #   (a) outputs a coefficient map to indicate argument ordering in
            #       parloops as `compile_form` does and
            #   (b) allows the dual evaluation related coefficients to be supplied to
            #       them rather than having to be added post-hoc (likely by
            #       replacing `to_element` with a CoFunction/CoArgument as the
            #       target `dual` which would contain `dual` related
            #       coefficient(s))
            if any(arg.name == rt_var_name for arg in kernel.code[name].args):
                # Add the coordinates of the target mesh quadrature points in the
                # source mesh's reference cell as an extra argument for the inner
                # loop. (With a vertex only mesh this is a single point for each
                # vertex cell.)
                target_ref_coords = target_mesh.reference_coordinates
                m_ = target_ref_coords.cell_node_map()
                parloop_args.append(target_ref_coords.dat(op2.READ, m_))

    parloop = op2.ParLoop(*parloop_args)
    if isinstance(tensor, op2.Mat):
        return parloop, tensor.assemble
    else:
        return copyin + (parloop, ) + copyout


def get_interp_node_map(source_mesh: MeshGeometry, target_mesh: MeshGeometry, fs: WithGeometry) -> op2.Map | None:
    """Return the map between cells of the target mesh and nodes of the function space.

    If the function space is defined on the source mesh then the node map is composed
    with a map between target and source cells.
    """
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        coeff_mesh = fs.mesh()
        m_ = fs.cell_node_map()
        if coeff_mesh is target_mesh or not coeff_mesh:
            # NOTE: coeff_mesh is None is allowed e.g. when interpolating from
            # a Real space
            pass
        elif coeff_mesh is source_mesh:
            if m_:
                # Since the par_loop is over the target mesh cells we need to
                # compose a map that takes us from target mesh cells to the
                # function space nodes on the source mesh.
                if source_mesh.extruded:
                    # ExtrudedSet cannot be a map target so we need to build
                    # this ourselves
                    m_ = vom_cell_parent_node_map_extruded(target_mesh, m_)
                else:
                    m_ = compose_map_and_cache(target_mesh.cell_parent_cell_map, m_)
            else:
                # m_ is allowed to be None when interpolating from a Real space,
                # even in the trans-mesh case.
                pass
        else:
            raise ValueError("Have coefficient with unexpected mesh")
    else:
        m_ = fs.entity_node_map(target_mesh.topology, "cell", "everywhere", None)
    return m_


try:
    _expr_cachedir = os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"]
except KeyError:
    _expr_cachedir = os.path.join(tempfile.gettempdir(),
                                  f"firedrake-tsfc-expression-kernel-cache-uid{os.getuid()}")


def _compile_expression_key(comm, expr, ufl_element, domain, parameters) -> tuple[Hashable, ...]:
    """Generate a cache key suitable for :func:`tsfc.compile_expression_dual_evaluation`."""
    dual_arg, operand = expr.argument_slots()
    return (hash_expr(operand), type(dual_arg), hash(ufl_element), tuplify(parameters))


@memory_and_disk_cache(
    hashkey=_compile_expression_key,
    cachedir=_cachedir
)
@PETSc.Log.EventDecorator()
def compile_expression(comm, *args, **kwargs):
    return compile_expression_dual_evaluation(*args, **kwargs)


def compose_map_and_cache(map1: op2.Map, map2: op2.Map | None) -> op2.ComposedMap | None:
    """
    Retrieve a :class:`pyop2.ComposedMap` map from the cache of map1
    using map2 as the cache key. The composed map maps from the iterset
    of map1 to the toset of map2. Makes :class:`pyop2.ComposedMap` and
    caches the result on map1 if the composed map is not found.

    :arg map1: The map with the desired iterset from which the result is
        retrieved or cached
    :arg map2: The map with the desired toset

    :returns:  The composed map
    """
    cache_key = hash((map2, "composed"))
    try:
        cmap = map1._cache[cache_key]
    except KeyError:
        # Real function space case separately
        cmap = None if map2 is None else op2.ComposedMap(map2, map1)
        map1._cache[cache_key] = cmap
    return cmap


def vom_cell_parent_node_map_extruded(vertex_only_mesh: MeshGeometry, extruded_cell_node_map: op2.Map) -> op2.Map:
    """Build a map from the cells of a vertex only mesh to the nodes of the
    nodes on the source mesh where the source mesh is extruded.

    Parameters
    ----------
    vertex_only_mesh : :class:`mesh.MeshGeometry`
        The ``mesh.VertexOnlyMesh`` whose cells we iterate over.
    extruded_cell_node_map : :class:`pyop2.Map`
        The cell node map of the function space on the extruded mesh within
        which the ``mesh.VertexOnlyMesh`` is immersed.

    Returns
    -------
    :class:`pyop2.Map`
        The map from the cells of the vertex only mesh to the nodes of the
        source mesh's cell node map. The map iterset is the
        ``vertex_only_mesh.cell_set`` and the map toset is the
        ``extruded_cell_node_map.toset``.

    Notes
    -----

    For an extruded mesh the cell node map is a map from a
    :class:`pyop2.ExtrudedSet` (the cells of the extruded mesh) to a
    :class:`pyop2.Set` (the nodes of the extruded mesh).

    Take for example

    ``mx = ExtrudedMesh(UnitIntervalMesh(2), 3)`` with
    ``mx.layers = 4``

    which looks like

    .. code-block:: text

        -------------------layer 4-------------------
        | parent_cell_num =  2 | parent_cell_num =  5 |
        |                      |                      |
        | extrusion_height = 2 | extrusion_height = 2 |
        -------------------layer 3-------------------
        | parent_cell_num =  1 | parent_cell_num =  4 |
        |                      |                      |
        | extrusion_height = 1 | extrusion_height = 1 |
        -------------------layer 2-------------------
        | parent_cell_num =  0 | parent_cell_num =  3 |
        |                      |                      |
        | extrusion_height = 0 | extrusion_height = 0 |
        -------------------layer 1-------------------
          base_cell_num = 0      base_cell_num = 1


    If we declare ``FunctionSpace(mx, "CG", 2)`` then the node numbering (i.e.
    Degree of Freedom/DoF numbering) is

    .. code-block:: text

        6 ---------13----------20---------27---------34
        |                       |                     |
        5          12          19         26         33
        |                       |                     |
        4 ---------11----------18---------25---------32
        |                       |                     |
        3          10          17         24         31
        |                       |                     |
        2 ---------9-----------16---------23---------30
        |                       |                     |
        1          8           15         22         29
        |                       |                     |
        0 ---------7-----------14---------21---------28
          base_cell_num = 0       base_cell_num = 1

    Cell node map values for an extruded mesh are indexed by the base cell
    number (rows) and the degree of freedom (DoF) index (columns). So
    ``extruded_cell_node_map.values[0] = [14, 15, 16,  0,  1,  2,  7,  8,  9]``
    are all the DoF/node numbers for the ``base_cell_num = 0``.
    Similarly
    ``extruded_cell_node_map.values[1] = [28, 29, 30, 21, 22, 23, 14, 15, 16]``
    contain all 9 of the DoFs for ``base_cell_num = 1``.
    To get the DoFs/nodes for the rest of the  cells we need to include the
    ``extruded_cell_node_map.offset``, which tells us how far each cell's
    DoFs/nodes are translated up from the first layer to the second, and
    multiply these by the the given ``extrusion_height``. So in our example
    ``extruded_cell_node_map.offset = [2, 2, 2, 2, 2, 2, 2, 2, 2]`` (we index
    this with the DoF/node index - it's an array because each DoF/node in the
    extruded mesh cell, in principal, can be offset upwards by a different
    amount).
    For ``base_cell_num = 0`` with ``extrusion_height = 1``
    (``parent_cell_num = 1``) we add ``1*2 = 2`` to each of the DoFs/nodes in
    ``extruded_cell_node_map.values[0]`` to get
    ``extruded_cell_node_map.values[0] + 1 * extruded_cell_node_map.offset[0] =
    [16, 17, 18,  2,  3,  4,  9, 10, 11]`` where ``0`` is the DoF/node index.

    For each cell (vertex) of a vertex only mesh immersed in a parent
    extruded mesh, we can can get the corresponding ``base_cell_num`` and
    ``extrusion_height`` of the parent extruded mesh. Armed with this
    information we use the above to work out the corresponding DoFs/nodes on
    the parent extruded mesh.

    """
    if not isinstance(vertex_only_mesh.topology, VertexOnlyMeshTopology):
        raise TypeError("The input mesh must be a VertexOnlyMesh")
    cnm = extruded_cell_node_map
    vmx = vertex_only_mesh
    dofs_per_target_cell = cnm.arity
    base_cells = vmx.cell_parent_base_cell_list
    heights = vmx.cell_parent_extrusion_height_list
    assert cnm.values_with_halo.shape[1] == dofs_per_target_cell
    assert len(cnm.offset) == dofs_per_target_cell
    target_cell_parent_node_list = [
        cnm.values_with_halo[base_cell, :] + height * cnm.offset[:]
        for base_cell, height in zip(base_cells, heights)
    ]
    return op2.Map(
        vmx.cell_set, cnm.toset, dofs_per_target_cell, target_cell_parent_node_list
    )


class VomOntoVomMat:
    """
    Object that facilitates interpolation between a VertexOnlyMesh and its
    input_ordering VertexOnlyMesh. This is either a PETSc Star Forest wrapped
    as a PETSc Mat, or a concrete PETSc Mat, depending on whether
    `mat_type='matfree` is passed to assemble.
    """
    def __init__(
            self,
            interpolator: VomOntoVomInterpolator,
            mat_type: Literal["aij", "baij", "matfree"],
    ):
        """Initialise the VomOntoVomMat.

        Parameters
        ----------
        interpolator : VomOntoVomInterpolator
            A :class:`VomOntoVomInterpolator` object.
        mat_type : Literal["aij", "baij", "matfree"] | None, optional
            The type of PETSc Mat to create. If 'matfree', a
            matfree PETSc Mat wrapping the SF is created. If 'aij' or 'baij',
            a concrete PETSc Mat is created.

        Raises
        ------
        ValueError
            If the source and target vertex-only meshes are not linked by input_ordering.
        """
        if interpolator.source_mesh.input_ordering is interpolator.target_mesh:
            self.forward_reduce = True
            """True if the forward interpolation is a star forest reduction, False if broadcast."""
            self.original_vom = interpolator.source_mesh
            """The original VOM from which the SF is constructed."""
        elif interpolator.target_mesh.input_ordering is interpolator.source_mesh:
            self.forward_reduce = False
            self.original_vom = interpolator.target_mesh
        else:
            raise ValueError(
                "The target vom and source vom must be linked by input ordering!"
            )
        self.sf = self.original_vom.input_ordering_without_halos_sf
        """The PETSc Star Forest representing the permutation between the VOMs."""
        self.target_space = interpolator.target_space
        """The FunctionSpace being interpolated into."""
        self.target_vom = interpolator.target_mesh
        """The VOM being interpolated to."""
        self.source_vom = interpolator.source_mesh
        """The VOM being interpolated from."""
        self.operand = interpolator.operand
        """The expression in the primal slot of the Interpolate."""
        self.arguments = extract_arguments(self.operand)
        """The arguments of the expression being interpolated."""
        self.is_adjoint = interpolator.ufl_interpolate.is_adjoint
        """Are we doing the adjoint interpolation?"""

        # Calculate correct local and global sizes for the matrix
        nroots, leaves, _ = self.sf.getGraph()
        self.nleaves = len(leaves)
        """The local number of leaves in the SF."""
        self._local_sizes = self.target_space.comm.allgather(nroots)
        """List of local number of roots on each process."""
        self.source_size = (self.target_space.block_size * nroots, self.target_space.block_size * sum(self._local_sizes))
        """Tuple containing the local and global size of the source space."""
        self.target_size = (
            self.target_space.block_size * self.nleaves,
            self.target_space.block_size * self.target_space.comm.allreduce(self.nleaves, op=MPI.SUM),
        )
        """Tuple containing the local and global size of the target space."""

        if mat_type == "matfree":
            # If matfree, we use the SF wrapped as a PETSc Mat
            # to perform the permutation.
            self.handle = self._wrap_python_mat()
        else:
            # Otherwise we build the concrete permutation
            # matrix as a PETSc Mat. This is used to build the
            # cross-mesh interpolation matrix.
            self.handle = self._create_permutation_mat(mat_type)

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

    def expr_as_coeff(self, source_vec: PETSc.Vec | None = None) -> Function:
        """Return a Function that corresponds to the expression used at
        construction, where the expression has been interpolated into the P0DG
        function space on the source vertex-only mesh.

        Will fail if there are no arguments.

        Parameters
        ----------
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
        with stop_annotating():
            element = self.target_space.ufl_element()  # Could be vector/tensor valued
            # if we have any arguments in the expression we need to replace
            # them with equivalent coefficients now
            if len(self.arguments):
                if len(self.arguments) > 1:
                    raise NotImplementedError("Can only interpolate expressions with one argument!")
                if source_vec is None:
                    raise ValueError("Need to provide a source dat for the argument!")

                arg = self.arguments[0]
                source_space = arg.function_space()
                P0DG = FunctionSpace(self.target_vom if self.is_adjoint else self.source_vom, element)
                arg_coeff = Function(self.target_space if self.is_adjoint else source_space)
                arg_coeff.dat.data_wo[:] = source_vec.getArray(readonly=True).reshape(
                    arg_coeff.dat.data_wo.shape
                )
                coeff_expr = replace(self.operand, {arg: arg_coeff})
                coeff = Function(P0DG).interpolate(coeff_expr)
            else:
                P0DG = FunctionSpace(self.source_vom, element)
                coeff = Function(P0DG).interpolate(self.operand)
        return coeff

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
        coeff = self.expr_as_coeff(source_vec)
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

    def _create_permutation_mat(self, mat_type: Literal["aij", "baij"]) -> PETSc.Mat:
        """Create the PETSc matrix that represents the interpolation operator from a vertex-only mesh to
        its input ordering vertex-only mesh.

        Returns
        -------
        PETSc.Mat
            PETSc seqaij matrix
        """
        if mat_type == "baij" and self.target_space.block_size > 1:
            create = PETSc.Mat().createBAIJ
        else:
            create = PETSc.Mat().createAIJ
        mat = create(
            size=(self.target_size, self.source_size),
            bsize=self.target_space.block_size,
            nnz=1,
            comm=self.target_space.comm
        )
        mat.setUp()
        # To create the permutation matrix we broadcast an array of indices which are contiguous
        # across all ranks and then use these indices to set the values of the matrix directly.
        start = sum(self._local_sizes[:self.target_space.comm.rank])
        end = start + self.source_size[0]
        contiguous_indices = numpy.arange(start, end, dtype=IntType)
        perm = numpy.zeros(self.nleaves, dtype=IntType)  # result stored in here
        self.sf.bcastBegin(MPI.INT, contiguous_indices, perm, MPI.REPLACE)
        self.sf.bcastEnd(MPI.INT, contiguous_indices, perm, MPI.REPLACE)
        rows = numpy.arange(self.target_size[0] + 1, dtype=IntType)
        # Vector and Tensor valued functions are stored in a flattened array, so
        # we need to space out the column indices according to the block size
        cols = (self.target_space.block_size * perm[:, None] + numpy.arange(self.target_space.block_size, dtype=IntType)[None, :]).reshape(-1)
        mat.setValuesCSR(rows, cols, numpy.ones_like(cols, dtype=IntType))
        mat.assemble()
        if self.forward_reduce and not self.is_adjoint:
            # The mat we have constructed thus far takes us from the input-ordering VOM to the
            # immersed VOM. If we're going the other way, then we need to transpose it,
            # unless we're doing the adjoint interpolation, since source_mesh and target_mesh
            # are defined assuming we're doing forward interpolation.
            mat.transpose()
        return mat

    def _wrap_python_mat(self) -> PETSc.Mat:
        """Wrap this object as a PETSc Mat. Used for matfree interpolation.

        Returns
        -------
        PETSc.Mat
            A PETSc Mat of type python with this object as its context.
        """
        mat = PETSc.Mat().create(comm=self.target_space.comm)
        if self.forward_reduce:
            mat_size = (self.source_size, self.target_size)
        else:
            mat_size = (self.target_size, self.source_size)
        mat.setSizes(mat_size)
        mat.setType(mat.Type.PYTHON)
        mat.setPythonContext(self)
        mat.setUp()
        return mat

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
            VomOntoVomMat wrapped as a PETSc Mat of type python.
        """
        return self._wrap_python_mat()


class MixedInterpolator(Interpolator):
    """Interpolator between MixedFunctionSpaces."""
    def __init__(self, expr: Interpolate):
        """Initialise MixedInterpolator. Should not be called directly; use `get_interpolator`.

        Parameters
        ----------
        expr : Interpolate
            Symbolic Interpolate expression.
        """
        super().__init__(expr)

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

        # We need a Coargument in order to split the Interpolate
        needs_action = not any(isinstance(a, Coargument) for a in self.interpolate_args)
        if needs_action:
            # Split the dual argument
            dual_split = dict(split_form(self.dual_arg))
            # Create the Jacobian to be split into blocks
            self.ufl_interpolate = self.ufl_interpolate._ufl_expr_reconstruct_(self.operand, self.target_space)

        # Get sub-interpolators and sub-bcs for each block
        Isub: dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]] = {}
        for indices, form in split_form(self.ufl_interpolate):
            if isinstance(form, ZeroBaseForm):
                # Ensure block sparsity
                continue
            sub_bcs = []
            for space, index in zip(spaces, indices):
                subspace = space.sub(index)
                sub_bcs.extend(bc for bc in bcs if space_equals(bc.function_space(), subspace))
            if needs_action:
                # Take the action of each sub-cofunction against each block
                form = action(form, dual_split[indices[-1:]])
            Isub[indices] = (get_interpolator(form), sub_bcs)

        return Isub

    def _build_matnest(
            self,
            Isub: dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]],
            sub_mat_type: Literal["aij", "baij"],
    ) -> PETSc.Mat:
        """Return a PETSc nested matrix built from sub-interpolator matrices."""
        shape = tuple(len(a.function_space()) for a in self.interpolate_args)
        blocks = numpy.full(shape, PETSc.Mat(), dtype=object)
        for indices, (interp, sub_bcs) in Isub.items():
            blocks[indices] = interp._get_callable(bcs=sub_bcs, mat_type=sub_mat_type)()
        return PETSc.Mat().createNest(blocks)

    def _build_aij(
            self,
            Isub: dict[tuple[int] | tuple[int, int], tuple[Interpolator, list[DirichletBC]]],
    ) -> PETSc.Mat:
        """Return a PETSc AIJ matrix built from sub-interpolator matrices by converting a
        nested matrix."""
        matnest = self._build_matnest(Isub, sub_mat_type="aij")
        return matnest.convert("aij")

    def _get_callable(self, tensor=None, bcs=None, mat_type=None, sub_mat_type=None):
        mat_type = mat_type or "aij"
        sub_mat_type = sub_mat_type or "aij"
        Isub = self._get_sub_interpolators(bcs=bcs)
        V_dest = self.ufl_interpolate.function_space() or self.target_space
        f = tensor or Function(V_dest)
        if self.rank == 2:
            if mat_type == "nest":
                callable = partial(self._build_matnest, Isub, sub_mat_type)
            else:
                assert mat_type == "aij"
                callable = partial(self._build_aij, Isub)
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
