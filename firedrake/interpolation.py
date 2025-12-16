from __future__ import annotations

import enum
from threading import local
import numpy
import os
import tempfile
import abc

from functools import partial, singledispatch
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

import pyop3 as op3
from pyop3.cache import memory_and_disk_cache
from pyop3.dtypes import get_mpi_dtype

from FIAT.reference_element import Point

from finat.element_factory import create_element, as_fiat_cell
from finat.ufl import TensorElement, VectorElement, MixedElement
from finat.fiat_elements import ScalarFiatElement
from finat.quadrature import QuadratureRule
from finat.quadrature_element import QuadratureElement
from finat.point_set import UnknownPointSet
from finat.tensorfiniteelement import TensorFiniteElement

from gem.gem import Variable

from tsfc.driver import compile_expression_dual_evaluation
from tsfc.ufl_utils import extract_firedrake_constants, hash_expr

import gem
import finat

from firedrake import tsfc_interface, utils, functionspaceimpl
from firedrake.parloops import pack_tensor, pack_pyop3_tensor, transform_packed_cell_closure_dat, transform_packed_cell_closure_mat
from firedrake.ufl_expr import Argument, Coargument, action, adjoint as expr_adjoint
from firedrake.mesh import MissingPointsBehaviour, VertexOnlyMeshMissingPointsError, VertexOnlyMeshTopology, get_iteration_spec
from firedrake.petsc import PETSc
from firedrake.cofunction import Cofunction
from firedrake.utils import IntType, ScalarType, tuplify
from firedrake.tsfc_interface import extract_numbered_coefficients, _cachedir
from firedrake.ufl_expr import Argument, Coargument, action
from firedrake.mesh import MissingPointsBehaviour, VertexOnlyMeshMissingPointsError, VertexOnlyMeshTopology, MeshGeometry, MeshTopology, VertexOnlyMesh
from firedrake.petsc import PETSc
from firedrake.functionspaceimpl import WithGeometry
from firedrake.matrix import MatrixBase, AssembledMatrix
from firedrake.bcs import DirichletBC
from firedrake.formmanipulation import split_form
from firedrake.functionspace import VectorFunctionSpace, TensorFunctionSpace, FunctionSpace
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.cofunction import Cofunction

from mpi4py import MPI

from pyadjoint.tape import stop_annotating, no_annotations

__all__ = (
    "interpolate",
    "Interpolate",
    "get_interpolator",
    "DofNotDefinedError",
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
    matfree : bool
        If ``False``, then construct the permutation matrix for interpolating
        between a VOM and its input ordering. Defaults to ``True`` which uses SF broadcast
        and reduce operations. Only applies when interpolating between a :func:`.VertexOnlyMesh`
        and its associated input ordering; is ignored in all other cases.
    """
    subset: op2.Subset | None = None
    access: Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC] | None = None
    allow_missing_dofs: bool = False
    default_missing_val: float | None = None
    matfree: bool = True


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
        self.source_mesh = extract_unique_domain(operand) or self.target_mesh
        """The domain we are interpolating from."""

        # Interpolation options
        self.subset = expr.options.subset
        self.allow_missing_dofs = expr.options.allow_missing_dofs
        self.default_missing_val = expr.options.default_missing_val
        self.matfree = expr.options.matfree
        self.access = expr.options.access

    @abc.abstractmethod
    def _get_callable(
        self,
        tensor: Function | Cofunction | MatrixBase | None = None,
        bcs: Iterable[DirichletBC] | None = None
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
        """
        pass

    def assemble(
        self,
        tensor: Function | Cofunction | MatrixBase | None = None,
        bcs: Iterable[DirichletBC] | None = None
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
        Returns
        -------
        Function | Cofunction | MatrixBase | numbers.Number
            The function, cofunction, matrix, or scalar resulting from the
            interpolation.
        """
        result = self._get_callable(tensor=tensor, bcs=bcs)()
        if self.rank == 2:
            # Assembling the operator
            assert isinstance(tensor, MatrixBase | None)
            assert isinstance(result, PETSc.Mat)
            if tensor:
                result.copy(tensor.petscmat)
                return tensor
            return AssembledMatrix(self.interpolate_args, bcs, result)
        else:
            assert isinstance(tensor, Function | Cofunction | None)
            return tensor.assign(result) if tensor else result


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
    arguments = expr.arguments()
    has_mixed_arguments = any(len(arg.function_space()) > 1 for arg in arguments)
    if len(arguments) == 2 and has_mixed_arguments:
        return MixedInterpolator(expr)

    operand, = expr.ufl_operands
    target_mesh = expr.target_space.mesh()

    try:
        source_mesh = extract_unique_domain(operand) or target_mesh
    except ValueError:
        raise NotImplementedError(
            "Interpolating an expression with no arguments defined on multiple meshes is not implemented yet."
        )

    try:
        target_mesh = target_mesh.unique()
        source_mesh = source_mesh.unique()
    except RuntimeError:
        return MixedInterpolator(expr)

    submesh_interp_implemented = (
        all(isinstance(m.topology, MeshTopology) for m in [target_mesh, source_mesh])
        and target_mesh.submesh_ancesters[-1] is source_mesh.submesh_ancesters[-1]
        and target_mesh.topological_dimension == source_mesh.topological_dimension
    )
    if target_mesh is source_mesh or submesh_interp_implemented:
        return SameMeshInterpolator(expr)

    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        if isinstance(source_mesh.topology, VertexOnlyMeshTopology):
            return VomOntoVomInterpolator(expr)
        if target_mesh.geometric_dimension != source_mesh.geometric_dimension:
            raise ValueError("Cannot interpolate onto a VertexOnlyMesh of a different geometric dimension.")
        return SameMeshInterpolator(expr)

    if has_mixed_arguments or len(expr.target_space) > 1:
        return MixedInterpolator(expr)

    return CrossMeshInterpolator(expr)


class DofNotDefinedError(Exception):
    r"""Raised when attempting to interpolate across function spaces where the
    target function space contains degrees of freedom (i.e. nodes) which cannot
    be defined in the source function space. This typically occurs when the
    target mesh covers a larger domain than the source mesh.

    Attributes
    ----------
    src_mesh : :func:`.Mesh`
        The source mesh.
    dest_mesh : :func:`.Mesh`
        The destination mesh.

    """

    def __init__(self, src_mesh, dest_mesh):
        self.src_mesh = src_mesh
        self.dest_mesh = dest_mesh

    def __str__(self):
        return (
            f"The given target function space on domain {repr(self.dest_mesh)} "
            "contains degrees of freedom which cannot cannot be defined in the "
            f"source function space on domain {repr(self.src_mesh)}. "
            "This may be because the target mesh covers a larger domain than the "
            "source mesh. To disable this error, set allow_missing_dofs=True."
        )


class CrossMeshInterpolator(Interpolator):
    """
    Interpolate a function from one mesh and function space to another.

    For arguments, see :class:`.Interpolator`.
    """

    @no_annotations
    def __init__(self, expr: Interpolate):
        super().__init__(expr)
        self.target_mesh = self.target_mesh.unique()
        if self.access and self.access != op3.WRITE:
            raise NotImplementedError(
                "Access other than op2.WRITE not implemented for cross-mesh interpolation."
            )
        else:
            self.access = op3.WRITE

        if self.target_space.ufl_element().mapping() != "identity":
            # Identity mapping between reference cell and physical coordinates
            # implies point evaluation nodes.
            raise NotImplementedError(
                "Can only cross-mesh interpolate into spaces with point evaluation nodes."
            )

        if self.allow_missing_dofs:
            self.missing_points_behaviour = MissingPointsBehaviour.IGNORE
        else:
            self.missing_points_behaviour = MissingPointsBehaviour.ERROR

        if self.source_mesh.geometric_dimension != self.target_mesh.geometric_dimension:
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
        """Return the symbolic ``Interpolate`` expressions for cross-mesh interpolation.

        Raises
        ------
        DofNotDefinedError
            If some DoFs in the target function space cannot be defined
            in the source function space.
        """
        from firedrake.assemble import assemble
        # Immerse coordinates of target space point evaluation dofs in src_mesh
        target_space_vec = VectorFunctionSpace(self.target_mesh, self.dest_element)
        f_dest_node_coords = assemble(interpolate(self.target_mesh.coordinates, target_space_vec))
        dest_node_coords = f_dest_node_coords.dat.data_ro.reshape(-1, self.target_mesh.geometric_dimension)
        try:
            vom = VertexOnlyMesh(
                self.source_mesh,
                dest_node_coords,
                redundant=False,
                missing_points_behaviour=self.missing_points_behaviour,
            )
        except VertexOnlyMeshMissingPointsError:
            raise DofNotDefinedError(self.source_mesh, self.target_mesh)

        # Get the correct type of function space
        shape = self.target_space.ufl_function_space().value_shape
        if len(shape) == 0:
            fs_type = FunctionSpace
        elif len(shape) == 1:
            fs_type = partial(VectorFunctionSpace, dim=shape[0])
        else:
            fs_type = partial(TensorFunctionSpace, shape=shape)

        # Get expression for point evaluation at the dest_node_coords
        P0DG_vom = fs_type(vom, "DG", 0)
        point_eval = interpolate(self.operand, P0DG_vom)

        # If assembling the operator, we need the concrete permutation matrix
        matfree = False if self.rank == 2 else self.matfree

        # Interpolate into the input-ordering VOM
        P0DG_vom_input_ordering = fs_type(vom.input_ordering, "DG", 0)

        arg = Argument(P0DG_vom, 0 if self.ufl_interpolate.is_adjoint else 1)
        point_eval_input_ordering = interpolate(arg, P0DG_vom_input_ordering, matfree=matfree)
        return point_eval, point_eval_input_ordering

    def _get_callable(self, tensor=None, bcs=None):
        from firedrake.assemble import assemble
        if bcs:
            raise NotImplementedError("bcs not implemented for cross-mesh interpolation.")
        # self.ufl_interpolate.function_space() is None in the 0-form case
        V_dest = self.ufl_interpolate.function_space() or self.target_space
        f = tensor or Function(V_dest)

        point_eval, point_eval_input_ordering = self._get_symbolic_expressions()
        P0DG_vom_input_ordering = point_eval_input_ordering.argument_slots()[0].function_space().dual()

        if self.rank == 2:
            # The cross-mesh interpolation matrix is the product of the
            # `self.point_eval_interpolate` and the permutation
            # given by `self.to_input_ordering_interpolate`.
            if self.ufl_interpolate.is_adjoint:
                symbolic = action(point_eval, point_eval_input_ordering)
            else:
                symbolic = action(point_eval_input_ordering, point_eval)

            def callable() -> PETSc.Mat:
                return assemble(symbolic).petscmat
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


class SameMeshInterpolator(Interpolator):
    """
    An interpolator for interpolation within the same mesh or onto a validly-
    defined :func:`.VertexOnlyMesh`.

    For arguments, see :class:`.Interpolator`.
    """

    @no_annotations
    def __init__(self, expr):
        super().__init__(expr)
        self.target_mesh = self.target_mesh.unique()
        subset = self.subset
        if subset is None:
            target = self.target_mesh.topology
            source = self.source_mesh.topology
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
                    raise NotImplementedError
                    subset = op2.Subset(target.cell_set, numpy.where(indices_active))
                else:
                    # Do not need subset as target <= source.
                    pass
        self.subset = subset

        if not isinstance(self.dual_arg, Coargument):
            # Matrix-free assembly of 0-form or 1-form requires INC access
            if self.access and self.access != op3.INC:
                raise ValueError("Matfree adjoint interpolation requires INC access")
            self.access = op3.INC
        elif self.access is None:
            # Default access for forward 1-form or 2-form (forward and adjoint)
            self.access = op3.WRITE

    def _get_tensor(self) -> op3.Mat | Function | Cofunction:
        """Return a suitable tensor to interpolate into.

        Returns
        -------
        op2.Mat | Function | Cofunction
            The tensor to interpolate into.
        """
        if self.rank == 0:
            R = FunctionSpace(self.target_mesh, "Real", 0)
            f = Function(R, dtype=ScalarType)
        elif self.rank == 1:
            f = Function(self.ufl_interpolate.function_space())
            if self.access in {op3.MIN_WRITE, op3.MAX_WRITE}:
                finfo = numpy.finfo(f.dat.dtype)
                if self.access == op3.MIN_WRITE:
                    val = Constant(finfo.max)
                else:
                    val = Constant(finfo.min)
                f.assign(val)
        elif self.rank == 2:
            Vrow = self.interpolate_args[0].function_space()
            Vcol = self.interpolate_args[1].function_space()
            if len(Vrow) > 1 or len(Vcol) > 1:
                raise NotImplementedError("Interpolation matrix with MixedFunctionSpace requires MixedInterpolator")

            # Pretend that we are assembling the operator to populate the sparsity.
            sparsity = op3.Mat.sparsity(Vrow.axes, Vcol.axes)
            iter_spec = get_iteration_spec(self.target_mesh, "cell")
            op3.loop(
                c := iter_spec.loop_index,
                sparsity[Vrow.entity_node_map(iter_spec), Vcol.entity_node_map(iter_spec)].assign(666),
                eager=True,
            )
            f = op3.Mat.from_sparsity(sparsity)
        else:
            raise ValueError(f"Cannot interpolate an expression with {self.rank} arguments")
        return f

    def _get_callable(self, tensor=None, bcs=None):
        if (isinstance(tensor, Cofunction) and isinstance(self.dual_arg, Cofunction)) and set(tensor.dat).intersection(set(self.dual_arg.dat)):
            # adjoint one-form case: we need an empty tensor, so if it shares dats with
            # the dual_arg we cannot use it directly, so we store it
            f = self._get_tensor()
            copyout = (lambda: tensor.dat.assign(f.dat, eager=True),)
        else:
            f = tensor or self._get_tensor()
            copyout = ()

        op2_tensor = f if isinstance(f, op3.Mat) else f.dat
        loops = []
        if self.access is op3.INC:
            loops.append(lambda: op2_tensor.zero(eager=True))

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
            indices = tuple(idx if idx is not None else Ellipsis for idx in indices)
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


class VomOntoVomInterpolator(SameMeshInterpolator):

    def __init__(self, expr: Interpolate):
        super().__init__(expr)

    def _get_callable(self, tensor=None, bcs=None):
        if bcs:
            raise NotImplementedError("bcs not implemented for vom-to-vom interpolation.")
        self.mat = VomOntoVomMat(self)
        if self.rank == 1:
            f = tensor or self._get_tensor()
            # NOTE: get_mpi_type ensures we get the correct MPI type for the
            # data, including the correct data size and dimensional information
            # (so for vector function spaces in 2 dimensions we might need a
            # concatenation of 2 MPI.DOUBLE types when we are in real mode)
            self.mat.mpi_type, _ = get_mpi_dtype(f.dat.dtype, f.function_space().block_size)
            if self.ufl_interpolate.is_adjoint:
                assert isinstance(self.dual_arg, Cofunction)
                assert isinstance(f, Cofunction)

                def callable() -> Cofunction:
                    with self.dual_arg.vec_ro as source_vec:
                        coeff = self.mat.expr_as_coeff(source_vec)
                        with coeff.vec_ro as coeff_vec, f.vec_wo as target_vec:
                            self.mat.handle.multHermitian(coeff_vec, target_vec)
                    return f
            else:
                assert isinstance(f, Function)

                def callable() -> Function:
                    coeff = self.mat.expr_as_coeff()
                    with coeff.vec_ro as coeff_vec, f.vec_wo as target_vec:
                        self.mat.handle.mult(coeff_vec, target_vec)
                    return f
        elif self.rank == 2:
            # Create a temporary function to get the correct MPI type
            temp_source_func = Function(self.interpolate_args[1].function_space())
            self.mat.mpi_type, _ = get_mpi_dtype(temp_source_func.dat.dtype, temp_source_func.function_space().block_size)

            def callable() -> PETSc.Mat:
                return self.mat.handle

        return callable


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
        if access is op3.INC:
            return ()
        elif access is op3.WRITE:
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

    try:
        to_element = create_element(V.ufl_element())
    except KeyError:
        # FInAT only elements
        raise NotImplementedError(f"Don't know how to create FIAT element for {V.ufl_element()}")

    if access is op3.READ:
        raise ValueError("Can't have READ access for output function")

    # NOTE: The par_loop is always over the target mesh cells.
    target_mesh = V.mesh()
    source_mesh = extract_unique_domain(operand) or target_mesh
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        # For interpolation onto a VOM, we use a FInAT QuadratureElement as the
        # target element with runtime point set expressions as their
        # quadrature rule point set.
        rt_var_name = 'rt_X'
        try:
            cell = operand.ufl_element().ufl_cell()
        except AttributeError:
            # expression must be pure function of spatial coordinates so
            # domain has correct ufl cell
            cell = source_mesh.ufl_cell()
        to_element = rebuild(to_element, cell, rt_var_name)

    iter_spec = get_iteration_spec(target_mesh, "cell")

    if not (subset is None or subset is Ellipsis):
        raise NotImplementedError
        assert subset.superset == cell_set
        cell_set = subset

    parameters = {}
    parameters['scalar_type'] = ScalarType

    copyin = ()
    copyout = ()

    # For the matfree adjoint 1-form and the 0-form, the cellwise kernel will add multiple
    # contributions from the facet DOFs of the dual argument.
    # The incoming Cofunction needs to be weighted by the reciprocal of the DOF multiplicity.
    if isinstance(dual_arg, Cofunction) and not to_element.is_dg():
        # Create a buffer for the weighted Cofunction
        W = dual_arg.function_space()
        v = Function(W)
        expr = expr._ufl_expr_reconstruct_(operand, v=v)
        copyin += (lambda: v.dat.assign(dual_arg.dat, eager=True),)

        weight = Function(W)
        op3.loop(
            c := iter_spec.loop_index,
            weight.dat[target_mesh.closure(c)].iassign(1),
            eager=True,
        )
        with weight.vec_rw as w:
            w.reciprocal()

        # Create a callable to apply the weight
        with weight.vec_ro as w, v.vec_wo as y:
            copyin += (lambda: y.pointwiseMult(y, w),)

    # We need to pass both the ufl element and the finat element
    # because the finat elements might not have the right mapping
    # (e.g. L2 Piola, or tensor element with symmetries)
    # FIXME: for the runtime unknown point set (for cross-mesh
    # interpolation) we have to pass the finat element we construct
    # here. Ideally we would only pass the UFL element through.
    kernel = compile_expression(target_mesh.comm, expr, to_element, V.ufl_element(),
                                domain=source_mesh, parameters=parameters)

    local_kernel_args = []

    coefficients = extract_numbered_coefficients(expr, kernel.coefficient_numbers)
    if kernel.needs_external_coords:
        coefficients = [source_mesh.coordinates] + coefficients

    if any(c.dat == tensor for c in coefficients):
        output = tensor
        tensor = op3.Dat.empty_like(tensor)
        if access is not op3.WRITE:
            copyin += (lambda: tensor.assign(output, eager=True),)
        copyout += (lambda: output.assign(tensor, eager=True),)

    arguments = expr.arguments()
    if not arguments:
        V_dest = FunctionSpace(target_mesh, "Real", 0)
        packed_tensor = pack_pyop3_tensor(tensor, V_dest, iter_spec)
        local_kernel_args.append(packed_tensor)
    elif len(arguments) < 2:
        V_dest = utils.just_one(arguments).function_space()
        packed_tensor = pack_pyop3_tensor(tensor, V_dest, iter_spec)
        local_kernel_args.append(packed_tensor)
    else:
        assert access == op3.WRITE  # Other access descriptors not done for Matrices.
        Vrow = arguments[0].function_space()
        Vcol = arguments[1].function_space()
        assert tensor.handle.getSize() == (Vrow.dim(), Vcol.dim())

        lgmaps = None
        if bcs:
            # NOTE: Probably shouldn't overwrite Vrow and Vcol here...
            if is_dual(Vrow):
                Vrow = Vrow.dual()
            if is_dual(Vcol):
                Vcol = Vcol.dual()
            bc_rows = [bc for bc in bcs if bc.function_space() == Vrow]
            bc_cols = [bc for bc in bcs if bc.function_space() == Vcol]
            lgmaps = [(functionspaceimpl.mask_lgmap(tensor.buffer.mat_spec.row_spec.lgmap, bc_rows), functionspaceimpl.mask_lgmap(tensor.buffer.mat_spec.column_spec.lgmap, bc_cols))]

        packed_tensor = pack_pyop3_tensor(tensor, Vrow, Vcol, iter_spec)
        local_kernel_args.append(packed_tensor)

    if kernel.oriented:
        local_kernel_args.append(pack_tensor(target_mesh.cell_orientations(), iter_spec))

    if kernel.needs_cell_sizes:
        local_kernel_args.append(pack_tensor(source_mesh.cell_sizes, iter_spec))

    for coefficient in coefficients:
        local_kernel_args.append(pack_tensor(coefficient, iter_spec))

    for const in extract_firedrake_constants(expr):
        local_kernel_args.append(const.dat)

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
            if rt_var_name in [arg.name for arg in kernel.ast[kernel.name].args]:
                # Add the coordinates of the target mesh quadrature points in the
                # source mesh's reference cell as an extra argument for the inner
                # loop. (With a vertex only mesh this is a single point for each
                # vertex cell.)
                coefficients.append(target_mesh.reference_coordinates)

    if any(c.dat.buffer == tensor.buffer for c in coefficients):
        output = tensor
        tensor = op3.Dat.empty_like(tensor)
        if access is not op3.WRITE:
            copyin += (lambda: tensor.assign(output, eager=True),)
        copyout += (lambda: output.assign(tensor, eager=True),)


    expression_kernel = op3.Function(kernel.ast, [access] + [op3.READ for _ in local_kernel_args[1:]])
    parloop = op3.loop(iter_spec.loop_index, expression_kernel(*local_kernel_args))
    def parloop_callable():
        parloop(compiler_parameters={"optimize": True})

    if isinstance(tensor, op3.Mat):
        return parloop_callable, tensor.assemble
    else:
        return copyin + (parloop_callable, ) + copyout


try:
    _expr_cachedir = os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"]
except KeyError:
    _expr_cachedir = os.path.join(tempfile.gettempdir(),
                                  f"firedrake-tsfc-expression-kernel-cache-uid{os.getuid()}")


def _compile_expression_key(comm, expr, to_element, ufl_element, domain, parameters) -> tuple[Hashable, ...]:
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


@singledispatch
def rebuild(element, expr_cell, rt_var_name):
    """Construct a FInAT QuadratureElement for interpolation onto a
    VertexOnlyMesh. The quadrature point is an UnknownPointSet of shape
    (1, tdim) where tdim is the topological dimension of expr_cell. The
    weight is [1.0], since the single local dof in the VertexOnlyMesh function
    space corresponds to a point evaluation at the vertex.

    Parameters
    ----------
    element : finat.FiniteElementBase
        The FInAT element to construct a QuadratureElement for.
    expr_cell : ufl.Cell
        The UFL cell of the expression being interpolated.
    rt_var_name : str
        String beginning with 'rt_' which is used as the name of the
        gem.Variable used to represent the UnknownPointSet. The `rt_` prefix
        forces TSFC to do runtime tabulation.

    Raises
    ------
    NotImplementedError
        If the element type is not implemented yet.
    """
    raise NotImplementedError(f"Point evaluation not implemented for a {element} element.")


@rebuild.register(ScalarFiatElement)
def rebuild_dg(element, expr_cell, rt_var_name):
    # QuadratureElements have a dual basis which is point evaluation at the
    # quadrature points. By using an UnknownPointSet with one point, TSFC
    # will generate a kernel with an argument to which we can pass the reference
    # coordinates of a point and evaluate the expression at that point at runtime.
    if element.degree != 0 or not isinstance(element.cell, Point):
        raise NotImplementedError("Interpolation onto a VOM only implemented for P0DG on vertex cells.")

    # gem.Variable name starting with rt_ forces TSFC runtime tabulation
    assert rt_var_name.startswith("rt_")
    runtime_points_expr = Variable(rt_var_name, (1, expr_cell.topological_dimension))
    rule_pointset = UnknownPointSet(runtime_points_expr)
    # What we use for the weight doesn't matter since we are not integrating
    rule = QuadratureRule(rule_pointset, weights=[0.0])
    return QuadratureElement(as_fiat_cell(expr_cell), rule)


@rebuild.register(TensorFiniteElement)
def rebuild_te(element, expr_cell, rt_var_name):
    return TensorFiniteElement(rebuild(element.base_element, expr_cell, rt_var_name),
                               element._shape,
                               transpose=element._transpose)


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
    as a PETSc Mat, or a concrete PETSc seqaij Mat, depending on whether
    matfree interpolation is requested.
    """
    def __init__(self, interpolator: VomOntoVomInterpolator):
        """Initialise the VomOntoVomMat.

        Parameters
        ----------
        interpolator : VomOntoVomInterpolator
            A :class:`VomOntoVomInterpolator` object.

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

        if interpolator.matfree:
            # If matfree, we use the SF wrapped as a PETSc Mat
            # to perform the permutation. This is the default.
            self.handle = self._wrap_python_mat()
        else:
            # If matfree=False, then we build the concrete permutation
            # matrix as a PETSc seqaij Mat. This is used to build the
            # cross-mesh interpolation matrix.
            self.handle = self._create_permutation_mat()

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
        with coeff.vec_ro as coeff_vec:
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

    def _create_permutation_mat(self) -> PETSc.Mat:
        """Create the PETSc matrix that represents the interpolation operator from a vertex-only mesh to
        its input ordering vertex-only mesh.

        Returns
        -------
        PETSc.Mat
            PETSc seqaij matrix
        """
        # To create the permutation matrix we broadcast an array of indices which are contiguous
        # across all ranks and then use these indices to set the values of the matrix directly.
        mat = PETSc.Mat().createAIJ((self.target_size, self.source_size), nnz=1, comm=self.target_space.comm)
        mat.setUp()
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
                subspace = space.sub(index) if index is not None else space
                sub_bcs.extend(bc for bc in bcs if space_equals(bc.function_space(), subspace))
            if needs_action:
                # Take the action of each sub-cofunction against each block
                form = action(form, dual_split[indices[-1:]])
            Isub[indices] = (get_interpolator(form), sub_bcs)

        return Isub

    def _get_callable(self, tensor=None, bcs=None):
        Isub = self._get_sub_interpolators(bcs=bcs)
        V_dest = self.ufl_interpolate.function_space() or self.target_space
        f = tensor or Function(V_dest)
        if self.rank == 2:
            def callable() -> PETSc.Mat:
                shape = tuple(len(a.function_space()) for a in self.interpolate_args)
                blocks = numpy.full(shape, PETSc.Mat(), dtype=object)
                for indices, (interp, sub_bcs) in Isub.items():
                    blocks[indices] = interp._get_callable(bcs=sub_bcs)()
                return PETSc.Mat().createNest(blocks)
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
