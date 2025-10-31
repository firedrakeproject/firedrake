import numpy
import os
import tempfile
import abc

from functools import partial, singledispatch
from typing import Hashable, Literal, Callable, Iterable
from dataclasses import asdict, dataclass
from numbers import Number

import FIAT
import ufl
import finat.ufl
from ufl.algorithms import extract_arguments
from ufl.domain import extract_unique_domain
from ufl.classes import Expr
from ufl.duals import is_dual

from pyop2 import op2
from pyop2.caching import memory_and_disk_cache

from finat.element_factory import create_element, as_fiat_cell
from tsfc import compile_expression_dual_evaluation
from tsfc.ufl_utils import extract_firedrake_constants, hash_expr

import gem
import finat

import firedrake
from firedrake import tsfc_interface, utils
from firedrake.ufl_expr import Argument, Coargument, action
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.mesh import MissingPointsBehaviour, VertexOnlyMeshMissingPointsError, VertexOnlyMeshTopology, MeshGeometry
from firedrake.petsc import PETSc
from firedrake.halo import _get_mtype as get_dat_mpi_type
from firedrake.functionspaceimpl import WithGeometry
from firedrake.matrix import MatrixBase
from firedrake.bcs import DirichletBC
from mpi4py import MPI

from pyadjoint import stop_annotating, no_annotations

__all__ = (
    "interpolate",
    "Interpolate",
    "get_interpolator",
    "DofNotDefinedError",
    "InterpolateOptions",
    "Interpolator"
)


@dataclass
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
        and reduce operations.
    """
    subset: op2.Subset | None = None
    access: Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC] | None = None
    allow_missing_dofs: bool = False
    default_missing_val: float | None = None
    matfree: bool = True


class Interpolate(ufl.Interpolate):

    def __init__(self, expr: Expr, V: WithGeometry | ufl.BaseForm, **kwargs):
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
        expr = ufl.as_ufl(expr)
        expr_arg_numbers = {arg.number() for arg in extract_arguments(expr) if not is_dual(arg)}
        self.is_adjoint = expr_arg_numbers == {0}
        if isinstance(V, WithGeometry):
            # Need to create a Firedrake Coargument so it has a .function_space() method
            V = Argument(V.dual(), 1 if self.is_adjoint else 0)

        self.target_space = V.arguments()[0].function_space()
        if expr.ufl_shape != self.target_space.value_shape:
            raise ValueError(f"Shape mismatch: Expression shape {expr.ufl_shape}, FunctionSpace shape {self.target_space.value_shape}.")

        super().__init__(expr, V)

        self._options = InterpolateOptions(**kwargs)

    function_space = ufl.Interpolate.ufl_function_space

    def _ufl_expr_reconstruct_(
            self, expr: Expr, v: WithGeometry | ufl.BaseForm | None = None, **interp_data
    ):
        interp_data = interp_data or asdict(self.options)
        return ufl.Interpolate._ufl_expr_reconstruct_(self, expr, v=v, **interp_data)

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
def interpolate(expr: Expr, V: WithGeometry | ufl.BaseForm, **kwargs) -> Interpolate:
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
        self.expr = expr
        """The symbolic UFL Interpolate expression."""
        self.expr_args = expr.arguments()
        """Arguments of the Interpolate expression."""
        self.rank = len(self.expr_args)
        """Number of arguments in the Interpolate expression."""
        self.operand = operand
        """The primal argument slot of the Interpolate expression."""
        self.dual_arg = dual_arg
        """The dual argument slot of the Interpolate expression."""
        self.target_space = dual_arg.function_space().dual()
        """The primal space we are interpolating into."""
        self.target_mesh = self.target_space.mesh()
        """The domain we are interpolating into."""
        self.source_mesh = extract_unique_domain(operand) or self.target_mesh
        """The domain we are interpolating from."""
        self.callable = None
        """The function which performs the interpolation."""

        # Interpolation options
        self.subset = expr.options.subset
        self.allow_missing_dofs = expr.options.allow_missing_dofs
        self.default_missing_val = expr.options.default_missing_val
        self.matfree = expr.options.matfree
        self.callable = None
        self.access = expr.options.access

    @abc.abstractmethod
    def _build_callable(self, tensor: Function | Cofunction | MatrixBase | None = None, bcs: Iterable[DirichletBC] | None = None) -> None:
        """Builds callable to perform interpolation. Stored in ``self.callable``.

        If ``self.rank == 2``, then ``self.callable()`` must return an object with a ``handle``
        attribute that stores a PETSc matrix. If ``self.rank == 1``, then `self.callable()` must
        return a ``Function`` or ``Cofunction`` (in the forward and adjoint cases respectively).
        If ``self.rank == 0``, then ``self.callable()`` must return a number.

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

        * rank-2: assemble the operator and return a matrix
        * rank-1: assemble the action and return a function or cofunction
        * rank-0: assemble the action and return a scalar by applying the dual argument

        Parameters
        ----------
        tensor
            Optional tensor to store the interpolated result. For rank-2
            expressions this is expected to be a subclass of
            :class:`~firedrake.matrix.MatrixBase`. For lower-rank expressions
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
        self._build_callable(tensor=tensor, bcs=bcs)
        result = self.callable()
        if self.rank == 2:
            # Assembling the operator
            assert isinstance(tensor, MatrixBase | None)
            # Get the interpolation matrix
            petsc_mat = result.handle
            if tensor:
                petsc_mat.copy(tensor.petscmat)
                return tensor
            return firedrake.AssembledMatrix(self.expr_args, bcs, petsc_mat)
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
    source_mesh = extract_unique_domain(operand) or target_mesh
    submesh_interp_implemented = (
        all(isinstance(m.topology, firedrake.mesh.MeshTopology) for m in [target_mesh, source_mesh])
        and target_mesh.submesh_ancesters[-1] is source_mesh.submesh_ancesters[-1]
        and target_mesh.topological_dimension == source_mesh.topological_dimension
    )
    if target_mesh is source_mesh or submesh_interp_implemented:
        return SameMeshInterpolator(expr)

    target_topology = target_mesh.topology
    source_topology = source_mesh.topology

    if isinstance(target_topology, VertexOnlyMeshTopology):
        if isinstance(source_topology, VertexOnlyMeshTopology):
            return VomOntoVomInterpolator(expr)
        if target_mesh.geometric_dimension != source_mesh.geometric_dimension:
            raise ValueError("Cannot interpolate onto a mesh of a different geometric dimension")
        if not hasattr(target_mesh, "_parent_mesh") or target_mesh._parent_mesh is not source_mesh:
            raise ValueError("Can only interpolate across meshes where the source mesh is the parent of the target")
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
        if self.access and self.access != op2.WRITE:
            raise NotImplementedError(
                "Access other than op2.WRITE not implemented for cross-mesh interpolation."
            )
        else:
            self.access = op2.WRITE
        if self.target_space.ufl_element().mapping() != "identity":
            # Identity mapping between reference cell and physical coordinates
            # implies point evaluation nodes. A more general version would
            # require finding the global coordinates of all quadrature points
            # of the target function space in the source mesh.
            raise NotImplementedError(
                "Can only interpolate into spaces with point evaluation nodes."
            )
        if self.allow_missing_dofs:
            self.missing_points_behaviour = MissingPointsBehaviour.IGNORE
        else:
            self.missing_points_behaviour = MissingPointsBehaviour.ERROR

        if self.source_mesh.geometric_dimension != self.target_mesh.geometric_dimension:
            raise ValueError("Geometric dimensions of source and destination meshes must match.")

        dest_element = self.target_space.ufl_element()
        if isinstance(dest_element, finat.ufl.MixedElement):
            if isinstance(dest_element, finat.ufl.VectorElement | finat.ufl.TensorElement):
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

        self._build_symbolic_expressions()

    def _build_symbolic_expressions(self) -> None:
        """Constructs the symbolic ``Interpolate`` expressions for cross-mesh interpolation.

        Raises
        ------
        DofNotDefinedError
            If some DoFs in the target function space cannot be defined
            in the source function space.
        """
        from firedrake.assemble import assemble
        # Immerse coordinates of target space point evaluation dofs in src_mesh
        target_space_vec = firedrake.VectorFunctionSpace(self.target_mesh, self.dest_element)
        f_dest_node_coords = assemble(interpolate(self.target_mesh.coordinates, target_space_vec))
        dest_node_coords = f_dest_node_coords.dat.data_ro.reshape(-1, self.target_mesh.geometric_dimension)
        try:
            self.vom = firedrake.VertexOnlyMesh(
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
            fs_type = firedrake.FunctionSpace
        elif len(shape) == 1:
            fs_type = partial(firedrake.VectorFunctionSpace, dim=shape[0])
        else:
            fs_type = partial(firedrake.TensorFunctionSpace, shape=shape)

        # Get expression for point evaluation at the dest_node_coords
        self.P0DG_vom = fs_type(self.vom, "DG", 0)
        self.point_eval = interpolate(self.operand, self.P0DG_vom)

        # If assembling the operator, we need the concrete permutation matrix
        matfree = False if self.rank == 2 else self.matfree

        # Interpolate into the input-ordering VOM
        self.P0DG_vom_input_ordering = fs_type(self.vom.input_ordering, "DG", 0)

        arg = Argument(self.P0DG_vom, 0 if self.expr.is_adjoint else 1)
        self.point_eval_input_ordering = interpolate(arg, self.P0DG_vom_input_ordering, matfree=matfree)

    def _build_callable(self, tensor=None, bcs=None):
        from firedrake.assemble import assemble
        if bcs:
            raise NotImplementedError("bcs not implemented for cross-mesh interpolation.")
        # self.expr.function_space() is None in the 0-form case
        V_dest = self.expr.function_space() or self.target_space
        f = tensor or Function(V_dest)

        if self.rank == 2:
            # The cross-mesh interpolation matrix is the product of the
            # `self.point_eval_interpolate` and the permutation
            # given by `self.to_input_ordering_interpolate`.
            if self.expr.is_adjoint:
                symbolic = action(self.point_eval, self.point_eval_input_ordering)
            else:
                symbolic = action(self.point_eval_input_ordering, self.point_eval)
            self.handle = assemble(symbolic).petscmat

            def callable() -> CrossMeshInterpolator:
                return self
        elif self.expr.is_adjoint:
            assert self.rank == 1
            # f_src is a cofunction on V_dest.dual
            cofunc = self.dual_arg
            assert isinstance(cofunc, Cofunction)

            # Our first adjoint operation is to assign the dat values to a
            # P0DG cofunction on our input ordering VOM.
            f_input_ordering = Cofunction(self.P0DG_vom_input_ordering.dual())
            f_input_ordering.dat.data_wo[:] = cofunc.dat.data_ro[:]

            # The rest of the adjoint interpolation is the composition
            # of the adjoint interpolators in the reverse direction.
            # We don't worry about skipping over missing points here
            # because we're going from the input ordering VOM to the original VOM
            # and all points from the input ordering VOM are in the original.
            def callable() -> Cofunction:
                f_src_at_src_node_coords = assemble(action(self.point_eval_input_ordering, f_input_ordering))
                assemble(action(self.point_eval, f_src_at_src_node_coords), tensor=f)
                return f
        else:
            assert self.rank in {0, 1}
            # We evaluate the operand at the node coordinates of the destination space
            f_point_eval = assemble(self.point_eval)

            # We create the input-ordering Function before interpolating so we can
            # set default missing values if required.
            f_point_eval_input_ordering = Function(self.P0DG_vom_input_ordering)
            if self.default_missing_val is not None:
                f_point_eval_input_ordering.assign(self.default_missing_val)
            elif self.allow_missing_dofs:
                # If we allow missing points there may be points in the target
                # mesh that are not in the source mesh. If we don't specify a
                # default missing value we set these to NaN so we can identify
                # them later.
                f_point_eval_input_ordering.dat.data_wo[:] = numpy.nan

            def callable() -> Function | Number:
                assemble(action(self.point_eval_input_ordering, f_point_eval), tensor=f_point_eval_input_ordering)
                # We assign these values to the output function
                if self.allow_missing_dofs and self.default_missing_val is None:
                    indices = numpy.where(~numpy.isnan(f_point_eval_input_ordering.dat.data_ro))[0]
                    f.dat.data_wo[indices] = f_point_eval_input_ordering.dat.data_ro[indices]
                else:
                    f.dat.data_wo[:] = f_point_eval_input_ordering.dat.data_ro[:]

                if self.rank == 0:
                    # We take the action of the dual_arg on the interpolated function
                    assert not isinstance(self.dual_arg, ufl.Coargument)
                    return assemble(action(self.dual_arg, f))
                else:
                    return f
        self.callable = callable


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
            target = self.target_mesh.topology
            source = self.source_mesh.topology
            if all(isinstance(m, firedrake.mesh.MeshTopology) for m in [target, source]) and target is not source:
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

        if not isinstance(self.dual_arg, ufl.Coargument):
            # Matrix-free assembly of 0-form or 1-form requires INC access
            if self.access and self.access != op2.INC:
                raise ValueError("Matfree adjoint interpolation requires INC access")
            self.access = op2.INC
        elif self.access is None:
            # Default access for forward 1-form or 2-form (forward and adjoint)
            self.access = op2.WRITE

    def _get_tensor(self) -> op2.Mat | Function | Cofunction:
        """Return the tensor to interpolate into.

        Returns
        -------
        op2.Mat | Function | Cofunction
            The tensor to interpolate into.
        """
        if self.rank == 0:
            R = firedrake.FunctionSpace(self.target_mesh, "Real", 0)
            f = Function(R, dtype=utils.ScalarType)
        elif self.rank == 1:
            f = Function(self.expr.function_space())
            if self.access in {firedrake.MIN, firedrake.MAX}:
                finfo = numpy.finfo(f.dat.dtype)
                if self.access == firedrake.MIN:
                    val = firedrake.Constant(finfo.max)
                else:
                    val = firedrake.Constant(finfo.min)
                f.assign(val)
        elif self.rank == 2:
            Vrow = self.expr_args[0].function_space()
            Vcol = self.expr_args[1].function_space()
            if len(Vrow) > 1 or len(Vcol) > 1:
                raise NotImplementedError("Interpolation matrix with MixedFunctionSpace requires MixedInterpolator")
            Vrow_map = get_interp_node_map(self.source_mesh, self.target_mesh, Vrow)
            Vcol_map = get_interp_node_map(self.source_mesh, self.target_mesh, Vcol)
            sparsity = op2.Sparsity((Vrow.dof_dset, Vcol.dof_dset),
                                    [(Vrow_map, Vcol_map, None)],  # non-mixed
                                    name=f"{Vrow.name}_{Vcol.name}_sparsity",
                                    nest=False,
                                    block_sparse=True)
            f = op2.Mat(sparsity)
        else:
            raise ValueError(f"Cannot interpolate an expression with {self.rank} arguments")
        return f

    def _build_callable(self, tensor=None, bcs=None):
        f = tensor or self._get_tensor()
        op2_tensor = f if isinstance(f, op2.Mat) else f.dat

        loops = []

        # Arguments in the operand are allowed to be from a MixedFunctionSpace
        # We need to split the target space V and generate separate kernels
        if self.rank == 2:
            expressions = {(0,): self.expr}
        elif isinstance(self.dual_arg, Coargument):
            # Split in the coargument
            expressions = dict(firedrake.formmanipulation.split_form(self.expr))
        else:
            assert isinstance(self.dual_arg, Cofunction)
            # Split in the cofunction: split_form can only split in the coargument
            # Replace the cofunction with a coargument to construct the Jacobian
            interp = self.expr._ufl_expr_reconstruct_(self.operand, self.target_space)
            # Split the Jacobian into blocks
            interp_split = dict(firedrake.formmanipulation.split_form(interp))
            # Split the cofunction
            dual_split = dict(firedrake.formmanipulation.split_form(self.dual_arg))
            # Combine the splits by taking their action
            expressions = {i: action(interp_split[i], dual_split[i[-1:]]) for i in interp_split}

        # Interpolate each sub expression into each function space
        for indices, sub_expr in expressions.items():
            sub_op2_tensor = op2_tensor[indices[0]] if self.rank == 1 else op2_tensor
            loops.extend(_build_interpolation_callables(sub_expr, sub_op2_tensor, self.access, self.subset, bcs))

        if bcs and self.rank == 1:
            loops.extend(partial(bc.apply, f) for bc in bcs)

        def callable(loops, f):
            for l in loops:
                l()
            return f.dat.data.item() if self.rank == 0 else f

        self.callable = partial(callable, loops, f)


class VomOntoVomInterpolator(SameMeshInterpolator):

    def __init__(self, expr: Interpolate):
        super().__init__(expr)

    def _build_callable(self, tensor=None, bcs=None):
        if bcs:
            raise NotImplementedError("bcs not implemented for vom-to-vom interpolation.")
        self.mat = VomOntoVomMat(self)
        if self.rank == 1:
            f = tensor or self._get_tensor()
            # NOTE: get_dat_mpi_type ensures we get the correct MPI type for the
            # data, including the correct data size and dimensional information
            # (so for vector function spaces in 2 dimensions we might need a
            # concatenation of 2 MPI.DOUBLE types when we are in real mode)
            self.mat.mpi_type = get_dat_mpi_type(f.dat)[0]
            if self.expr.is_adjoint:
                assert isinstance(self.dual_arg, ufl.Cofunction)
                assert isinstance(f, Cofunction)

                def callable() -> Cofunction:
                    with self.dual_arg.dat.vec_ro as source_vec, f.dat.vec_wo as target_vec:
                        self.mat.handle.multHermitian(source_vec, target_vec)
                    return f
            else:
                assert isinstance(f, Function)

                def callable() -> Function:
                    coeff = self.mat.expr_as_coeff()
                    with coeff.dat.vec_ro as coeff_vec, f.dat.vec_wo as target_vec:
                        self.mat.handle.mult(coeff_vec, target_vec)
                    return f
        elif self.rank == 2:
            # we know we will be outputting either a function or a cofunction,
            # both of which will use a dat as a data carrier. At present, the
            # data type does not depend on function space dimension, so we can
            # safely use the argument function space. NOTE: If this changes
            # after cofunctions are fully implemented, this will need to be
            # reconsidered.
            temp_source_func = Function(self.expr_args[1].function_space())
            self.mat.mpi_type = get_dat_mpi_type(temp_source_func.dat)[0]
            # Leave mat inside a callable so we can access the handle
            # property. If matfree is True, then the handle is a PETSc SF
            # pretending to be a PETSc Mat. If matfree is False, then this
            # will be a PETSc Mat representing the equivalent permutation matrix

            def callable() -> VomOntoVomMat:
                return self.mat

        self.callable = callable


@utils.known_pyop2_safe
def _build_interpolation_callables(
    expr: ufl.Interpolate | ufl.ZeroBaseForm,
    tensor: op2.Dat | op2.Mat | op2.Global,
    access: Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC],
    subset: op2.Subset | None = None,
    bcs: Iterable[DirichletBC] | None = None
) -> tuple[Callable, ...]:
    """Returns tuple of callables which calculate the interpolation.

    Parameters
    ----------
    expr : ufl.Interpolate | ufl.ZeroBaseForm
        The symbolic interpolation expression, or a zero form. Zero forms
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
    if isinstance(expr, ufl.ZeroBaseForm):
        # Zero simplification, avoid code-generation
        if access is op2.INC:
            return ()
        elif access is op2.WRITE:
            return (partial(tensor.zero, subset=subset),)
        # Unclear how to avoid codegen for MIN and MAX
        # Reconstruct the expression as an Interpolate
        V = expr.arguments()[-1].function_space().dual()
        expr = interpolate(ufl.zero(V.value_shape), V)
    if not isinstance(expr, ufl.Interpolate):
        raise ValueError("Expecting to interpolate a ufl.Interpolate")
    dual_arg, operand = expr.argument_slots()
    V = dual_arg.function_space().dual()
    try:
        to_element = create_element(V.ufl_element())
    except KeyError:
        # FInAT only elements
        raise NotImplementedError(f"Don't know how to create FIAT element for {V.ufl_element()}")

    if access is op2.READ:
        raise ValueError("Can't have READ access for output function")

    # NOTE: The par_loop is always over the target mesh cells.
    target_mesh = V.mesh()
    source_mesh = extract_unique_domain(operand) or target_mesh
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        # For trans-mesh interpolation we use a FInAT QuadratureElement as the
        # (base) target element with runtime point set expressions as their
        # quadrature rule point set and weights from their dual basis.
        # NOTE: This setup is useful for thinking about future design - in the
        # future this `rebuild` function can be absorbed into FInAT as a
        # transformer that eats an element and gives you an equivalent (which
        # may or may not be a QuadratureElement) that lets you do run time
        # tabulation. Alternatively (and this all depends on future design
        # decision about FInAT how dual evaluation should work) the
        # to_element's dual basis (which look rather like quadrature rules) can
        # have their pointset(s) directly replaced with run-time tabulated
        # equivalent(s) (i.e. finat.point_set.UnknownPointSet(s))
        rt_var_name = 'rt_X'
        try:
            cell = operand.ufl_element().ufl_cell()
        except AttributeError:
            # expression must be pure function of spatial coordinates so
            # domain has correct ufl cell
            cell = source_mesh.ufl_cell()
        to_element = rebuild(to_element, cell, rt_var_name)

    cell_set = target_mesh.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset

    parameters = {}
    parameters['scalar_type'] = utils.ScalarType

    copyin = ()
    copyout = ()

    # For the matfree adjoint 1-form and the 0-form, the cellwise kernel will add multiple
    # contributions from the facet DOFs of the dual argument.
    # The incoming Cofunction needs to be weighted by the reciprocal of the DOF multiplicity.
    needs_weight = isinstance(dual_arg, ufl.Cofunction) and not to_element.is_dg()
    if needs_weight:
        # Create a buffer for the weighted Cofunction
        W = dual_arg.function_space()
        v = firedrake.Function(W)
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

    # We need to pass both the ufl element and the finat element
    # because the finat elements might not have the right mapping
    # (e.g. L2 Piola, or tensor element with symmetries)
    # FIXME: for the runtime unknown point set (for cross-mesh
    # interpolation) we have to pass the finat element we construct
    # here. Ideally we would only pass the UFL element through.
    kernel = compile_expression(cell_set.comm, expr, to_element, V.ufl_element(),
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

    coefficients = tsfc_interface.extract_numbered_coefficients(expr, coefficient_numbers)
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
            if ufl.duals.is_dual(Vrow):
                Vrow = Vrow.dual()
            if ufl.duals.is_dual(Vcol):
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
        if access == op2.INC:
            copyin += (tensor.zero,)
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


def _compile_expression_key(comm, expr, to_element, ufl_element, domain, parameters) -> tuple[Hashable, ...]:
    """Generate a cache key suitable for :func:`tsfc.compile_expression_dual_evaluation`."""
    dual_arg, operand = expr.argument_slots()
    return (hash_expr(operand), type(dual_arg), hash(ufl_element), utils.tuplify(parameters))


@memory_and_disk_cache(
    hashkey=_compile_expression_key,
    cachedir=tsfc_interface._cachedir
)
@PETSc.Log.EventDecorator()
def compile_expression(comm, *args, **kwargs):
    return compile_expression_dual_evaluation(*args, **kwargs)


@singledispatch
def rebuild(element, expr_cell, rt_var_name):
    raise NotImplementedError(f"Cross mesh interpolation not implemented for a {element} element.")


@rebuild.register(finat.fiat_elements.ScalarFiatElement)
def rebuild_dg(element, expr_cell, rt_var_name):
    # To tabulate on the given element (which is on a different mesh to the
    # expression) we must do so at runtime. We therefore create a quadrature
    # element with runtime points to evaluate for each point in the element's
    # dual basis. This exists on the same reference cell as the input element
    # and we can interpolate onto it before mapping the result back onto the
    # target space.
    expr_tdim = expr_cell.topological_dimension
    # Need point evaluations and matching weights from dual basis.
    # This could use FIAT's dual basis as below:
    # num_points = sum(len(dual.get_point_dict()) for dual in element.fiat_equivalent.dual_basis())
    # weights = []
    # for dual in element.fiat_equivalent.dual_basis():
    #     pts = dual.get_point_dict().keys()
    #     for p in pts:
    #         for w, _ in dual.get_point_dict()[p]:
    #             weights.append(w)
    # assert len(weights) == num_points
    # but for now we just fix the values to what we know works:
    if element.degree != 0 or not isinstance(element.cell, FIAT.reference_element.Point):
        raise NotImplementedError("Cross mesh interpolation only implemented for P0DG on vertex cells.")
    num_points = 1
    weights = [1.]*num_points
    # gem.Variable name starting with rt_ forces TSFC runtime tabulation
    assert rt_var_name.startswith("rt_")
    runtime_points_expr = gem.Variable(rt_var_name, (num_points, expr_tdim))
    rule_pointset = finat.point_set.UnknownPointSet(runtime_points_expr)
    rule = finat.quadrature.QuadratureRule(rule_pointset, weights=weights)
    return finat.QuadratureElement(as_fiat_cell(expr_cell), rule)


@rebuild.register(finat.TensorFiniteElement)
def rebuild_te(element, expr_cell, rt_var_name):
    return finat.TensorFiniteElement(rebuild(element.base_element,
                                             expr_cell, rt_var_name),
                                     element._shape,
                                     transpose=element._transpose)


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


class GlobalWrapper(object):
    """Wrapper object that fakes a Global to behave like a Function."""
    def __init__(self, glob):
        self.dat = glob
        self.cell_node_map = lambda *arguments: None
        self.ufl_domain = lambda: None


class VomOntoVomMat:
    """Object that facilitates interpolation between two vertex-only meshes."""
    def __init__(self, interpolator: VomOntoVomInterpolator):
        """Initialises the VomOntoVomMat.

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
            self.original_vom = interpolator.source_mesh
        elif interpolator.target_mesh.input_ordering is interpolator.source_mesh:
            self.forward_reduce = False
            self.original_vom = interpolator.target_mesh
        else:
            raise ValueError(
                "The target vom and source vom must be linked by input ordering!"
            )
        self.sf = self.original_vom.input_ordering_without_halos_sf
        self.V = interpolator.target_space
        self.source_vom = interpolator.source_mesh
        self.expr = interpolator.operand
        self.arguments = extract_arguments(self.expr)
        self.is_adjoint = interpolator.expr.is_adjoint

        # Calculate correct local and global sizes for the matrix
        nroots, leaves, _ = self.sf.getGraph()
        self.nleaves = len(leaves)
        self._local_sizes = self.V.comm.allgather(nroots)
        self.source_size = (self.V.block_size * nroots, self.V.block_size * sum(self._local_sizes))
        self.target_size = (
            self.V.block_size * self.nleaves,
            self.V.block_size * self.V.comm.allreduce(self.nleaves, op=MPI.SUM),
        )

        if interpolator.matfree:
            # If matfree, we use the SF to perform the interpolation
            self.handle = self._wrap_python_mat()
        else:
            # Otherwise we create the permutation matrix
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
        """Return a coefficient that corresponds to the expression used at
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
            element = self.V.ufl_element()  # Could be vector/tensor valued
            P0DG = firedrake.FunctionSpace(self.source_vom, element)
            # if we have any arguments in the expression we need to replace
            # them with equivalent coefficients now
            coeff_expr = self.expr
            if len(self.arguments):
                if len(self.arguments) > 1:
                    raise NotImplementedError(
                        "Can only interpolate expressions with one argument!"
                    )
                if source_vec is None:
                    raise ValueError("Need to provide a source dat for the argument!")
                arg = self.arguments[0]
                arg_coeff = firedrake.Function(arg.function_space())
                arg_coeff.dat.data_wo[:] = source_vec.getArray(readonly=True).reshape(
                    arg_coeff.dat.data_wo.shape
                )
                coeff_expr = ufl.replace(self.expr, {arg: arg_coeff})
            coeff = firedrake.Function(P0DG).interpolate(coeff_expr)
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
        """Broadcast data in source_vec using the PETSc SF.

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
        """Applies the interpolation operator.

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
        """Applies the adjoint of the interpolation operator.
        Since ``VomOntoVomMat`` represents a permutation, it is
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
        """Applies the tranpose of the interpolation operator. Called by `self.multHermitian`.

        Parameters
        ----------
        mat : PETSc.Mat
            Required by petsc4py but unused.
        source_vec : PETSc.Vec
            The vector to transpose interpolate.
        target_vec : PETSc.Vec
            The vector to store the result in.

        """
        # can only do adjoint if our expression exclusively contains a
        # single argument, making the application of the adjoint operator
        # straightforward (haven't worked out how to do this otherwise!)
        if not len(self.arguments) == 1:
            raise NotImplementedError(
                "Can only apply adjoint to expressions with one argument!"
            )
        if self.arguments[0] is not self.expr:
            raise NotImplementedError(
                "Can only apply adjoint to expressions consisting of a single argument at the moment."
            )
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
        """Creates the PETSc matrix that represents the interpolation operator from a vertex-only mesh to
        its input ordering vertex-only mesh.

        Returns
        -------
        PETSc.Mat
            PETSc seqaij matrix
        """
        # To create the permutation matrix we broadcast an array of indices which are contiguous
        # across all ranks and then use these indices to set the values of the matrix directly.
        mat = PETSc.Mat().createAIJ((self.target_size, self.source_size), nnz=1, comm=self.V.comm)
        mat.setUp()
        start = sum(self._local_sizes[:self.V.comm.rank])
        end = start + self.source_size[0]
        contiguous_indices = numpy.arange(start, end, dtype=utils.IntType)
        perm = numpy.zeros(self.nleaves, dtype=utils.IntType)
        self.sf.bcastBegin(MPI.INT, contiguous_indices, perm, MPI.REPLACE)
        self.sf.bcastEnd(MPI.INT, contiguous_indices, perm, MPI.REPLACE)
        rows = numpy.arange(self.target_size[0] + 1, dtype=utils.IntType)
        cols = (self.V.block_size * perm[:, None] + numpy.arange(self.V.block_size, dtype=utils.IntType)[None, :]).reshape(-1)
        mat.setValuesCSR(rows, cols, numpy.ones_like(cols, dtype=utils.IntType))
        mat.assemble()
        if self.forward_reduce and not self.is_adjoint:
            mat.transpose()
        return mat

    def _wrap_python_mat(self) -> PETSc.Mat:
        """Wraps this object as a PETSc Mat. Used for matfree interpolation.

        Returns
        -------
        PETSc.Mat
            A PETSc Mat of type python with this object as its context.
        """
        mat = PETSc.Mat().create(comm=self.V.comm)
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
        """Duplicates the matrix. Needed to wrap as a PETSc Python Mat.

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

    def _get_sub_interpolators(self, bcs: Iterable[DirichletBC] | None = None) -> dict[tuple[int, int], tuple[Interpolator, list[DirichletBC]]]:
        """Gets `Interpolator`s for each sub-Interpolate in the mixed expression.

        Returns
        -------
        dict[tuple[int, int], tuple[Interpolator, list[DirichletBC]]]
            A map from block index tuples to `Interpolator`s and bcs.
        """
        # Get the primal spaces
        spaces = tuple(
            a.function_space().dual() if isinstance(a, Coargument) else a.function_space() for a in self.expr_args
        )
        # TODO consider a stricter equality test for indexed MixedFunctionSpace
        # See https://github.com/firedrakeproject/firedrake/issues/4668
        space_equals = lambda V1, V2: V1 == V2 and V1.parent == V2.parent and V1.index == V2.index

        # We need a Coargument in order to split the Interpolate
        needs_action = not any(isinstance(a, Coargument) for a in self.expr_args)
        if needs_action:
            # Split the dual argument
            dual_split = dict(firedrake.formmanipulation.split_form(self.dual_arg))
            # Create the Jacobian to be split into blocks
            self.expr = self.expr._ufl_expr_reconstruct_(self.operand, self.target_space)

        # Get sub-interpolators for each block
        Isub: dict[tuple[int, int], tuple[Interpolator, list[DirichletBC]]] = {}
        for indices, form in firedrake.formmanipulation.split_form(self.expr):
            if isinstance(form, ufl.ZeroBaseForm):
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

    def _build_callable(self, tensor=None, bcs=None):
        Isub = self._get_sub_interpolators(bcs=bcs)
        V_dest = self.expr.function_space() or self.target_space
        f = tensor or Function(V_dest)
        if self.rank == 2:
            shape = tuple(len(a.function_space()) for a in self.expr_args)
            blocks = numpy.full(shape, PETSc.Mat(), dtype=object)
            for indices, (interp, sub_bcs) in Isub.items():
                interp._build_callable(bcs=sub_bcs)
                blocks[indices] = interp.callable().handle
            self.handle = PETSc.Mat().createNest(blocks)

            def callable() -> MixedInterpolator:
                return self
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
        self.callable = callable
