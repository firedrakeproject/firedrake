import enum
import functools
import numbers
import types
import warnings
from functools import cached_property
from typing import Any, Callable

import numpy as np
from mpi4py import MPI

from pyadjoint.tape import annotate_tape
import pyop3 as op3
import ufl.classes
from pyop3.dtypes import get_mpi_dtype
from pyop3.mpi import temp_internal_comm
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.corealg.dag_traverser import DAGTraverser

import firedrake.ufl_expr
from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.cython import dmcommon
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.petsc import PETSc
from firedrake.utils import IntType


def _ancestor_node_sf(
    space: firedrake.functionspaceimpl.WithGeometryBase,
    ancestor_space: firedrake.functionspaceimpl.AbstractFunctionSpace,
    comm: MPI.Comm,
) -> PETSc.SF:
    """Map each owned node to the process that owns its node on the ancestor."""
    ancestors = space.mesh().submesh_ancestors
    point_map = np.arange(space.mesh().num_points, dtype=IntType)
    for child in ancestors[:ancestors.index(ancestor_space.mesh())]:
        point_map = child._submesh_to_parent_plex_index_map[point_map]

    # Convert points to nodes after composition, so both spaces use the same
    # ancestor numbering even when they have different constrained nodes.
    node_map = dmcommon.make_node_to_node_map(
        point_map, space.local_section, ancestor_space.local_section, space.block_size
    )
    # The owned nodes come before the ghost nodes.
    ancestor_indices = node_map[:space.nodes.sf.num_owned]

    ancestor_nodes = ancestor_space.nodes
    locations = np.empty((ancestor_nodes.local_size, 2), dtype=IntType)
    locations[:, 0] = comm.rank
    locations[:, 1] = np.arange(ancestor_nodes.local_size, dtype=IntType)
    halo = ancestor_nodes.sf
    locations[halo.ileaf] = halo.iremote
    sf = PETSc.SF().create(comm)
    sf.setGraph(ancestor_nodes.local_size, None, locations[ancestor_indices])
    return sf


def _submesh_node_sf(
    source: firedrake.functionspaceimpl.WithGeometryBase,
    target: firedrake.functionspaceimpl.WithGeometryBase,
) -> PETSc.SF:
    """Return the star forest from owned source nodes to the matching owned target nodes."""
    ancestor = target.mesh().submesh_youngest_common_ancestor(source.mesh())
    if ancestor is None:
        raise ValueError("The source and target must belong to the same submesh family")

    ancestor_space = FunctionSpace(ancestor, target.ufl_element())
    with temp_internal_comm(target.comm) as comm:
        source_to_ancestor = _ancestor_node_sf(source, ancestor_space, comm)
        target_to_ancestor = _ancestor_node_sf(target, ancestor_space, comm)
        # Each ancestor node has at most one owned source node, so the inverse
        # exists. The composition drops the target nodes without a source node.
        return source_to_ancestor.createInverse().compose(target_to_ancestor)


class AssignExprBuilder(DAGTraverser):
    """Traverser used for converting an expression into a pyop3 expression."""

    def __init__(self, function_space: firedrake.functionspaceimpl.WithGeometryBase):
        self.function_space = function_space
        self.loop_index = None
        self.array_assign_safe = True
        self.operands = {}
        self.transfers = []
        super().__init__()

    def prepare_cross_mesh(
        self, expression: ufl.core.expr.Expr | ufl.form.BaseForm, subset: op3.Slice | types.EllipsisType
    ) -> op3.LoopIndex:
        """Build transfers and return the loop over target nodes with all source matches."""
        nodes = self.function_space.nodes
        selected = np.zeros(nodes.local_size, dtype=bool)
        selected[nodes[subset].buffer_slice(include_ghosts=True)] = True
        # A subset can select a ghost node before it selects its owner.
        halo = nodes.sf
        halo.reduce_begin(selected, MPI.LOR)
        halo.reduce_end(selected, MPI.LOR)
        selected[halo.num_owned:] = False

        for func in extract_coefficients(expression):
            dat, nodal = self._coefficient(func, broadcast=True)
            if nodal and func.function_space().mesh() != self.function_space.mesh():
                sf = _submesh_node_sf(func.function_space(), self.function_space)
                _, matched, _ = sf.getGraph()
                selected &= np.isin(np.arange(nodes.local_size), matched)
                dat = self.function_space.make_dat(valuetype=dat.dtype)
                self.transfers.append((sf, func.dat, dat))
            self.operands[func] = dat, nodal

        indices = np.flatnonzero(selected).astype(IntType, copy=False)
        self.loop_index = nodes[parse_subset(indices)].iter()
        return self.loop_index

    def transfer_inputs(self) -> None:
        """Read the source values at the owners of the matched target nodes."""
        block_size = self.function_space.block_size
        for sf, source, target in self.transfers:
            # The loop reads only the target nodes that are leaves of the star forest.
            source_values = source.buffer.data_ro
            target_values = target.buffer.data_wo
            unit, _ = get_mpi_dtype(source.dtype, cdim=block_size)
            sf.bcastBegin(unit, source_values, target_values, MPI.REPLACE)
            sf.bcastEnd(unit, source_values, target_values, MPI.REPLACE)

    def _coefficient(
        self, func: Function | Cofunction, *, broadcast: bool = False
    ) -> tuple[op3.Dat | op3.Scalar, bool]:
        """Return the data of the coefficient and whether it is indexed by node."""
        if func.ufl_element().family() == "Real":
            data = func.dat.global_data
            shape = func.function_space().shape
            if not shape:
                return op3.Scalar(data[0]), False
            if broadcast:
                data = np.broadcast_to(data.reshape(shape), self.function_space.shape).copy()
                shape = data.shape
            axes = op3.AxisTree.from_iterable(op3.Axis(size, f"dim{i}") for i, size in enumerate(shape))
            return op3.Dat(axes, data=data), False

        if func.ufl_element() != self.function_space.ufl_element():
            raise ValueError(
                "All functions in the expression must have the same element as the assignee"
            )
        return func.dat, True

    @functools.singledispatchmethod
    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)

    @process.register
    def _(self, func: Function | Cofunction, /) -> tuple[op3.Dat | op3.Scalar, bool]:
        """Return the coefficient as a pyop3 expression and whether it is nodal."""
        if self.loop_index is not None:
            dat, nodal = self.operands[func]
            return (dat[self.loop_index] if nodal else dat), nodal

        dat, nodal = self._coefficient(func)
        if nodal and any(
            func_fs.boundary_set != self_fs.boundary_set
            for func_fs, self_fs in zip(func.function_space(), self.function_space, strict=True)
        ):
            self.array_assign_safe = False
        return dat, nodal

    @process.register(Constant)
    def _(self, const) -> tuple[op3.Dat, bool, bool]:
        return const.dat, False

    @process.register(ufl.classes.ScalarValue)
    def _(self, num) -> numbers.Number:
        return num.value(), False

    @process.register(ufl.classes.Zero)
    def _(self, zero) -> numbers.Number:
        return 0, False

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Product, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b

        if a_is_vector and b_is_vector:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")

        return a_expr * b_expr, a_is_vector or b_is_vector

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b

        if b_is_vector:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

        return a_expr / b_expr, a_is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Sum, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b

        return a_expr + b_expr, a_is_vector or b_is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Power, a, b):
        a_expr, a_is_vector = a
        b_expr, b_is_vector = b
        assert not a_is_vector and not b_is_vector, "Only valid if a and b are scalars"
        return a_expr ** b_expr, False

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Abs, a):
        a_expr, is_vector = a
        return abs(a_expr), is_vector

    @process.register
    def _(self, _: ufl.classes.MultiIndex):
        # never used by parent types
        pass

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Indexed, a, ii):
        return a

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.ComponentTensor, a, ii):
        return a


class AssignmentMode(enum.Enum):
    STANDARD = enum.auto()
    IADD = enum.auto()
    ISUB = enum.auto()
    IMUL = enum.auto()
    IDIV = enum.auto()


class Assigner:
    """Class performing pointwise assignment of an expression to a function or a cofunction.

    Parameters
    ----------
    assignee
        Function or cofunction being assigned to.
    expression
        Expression to be assigned.
    subset
        Subset to apply the assignment over.
    mode
        The assignment mode (standard, iadd, isub, imul or idiv).
    allow_missing_dofs
        Permit assignment between objects with mismatching nodes. If `True` then
        assignee nodes with no matching assigner nodes are ignored.

    """
    def __init__(
        self,
        assignee: Function | Cofunction,
        expression: ufl.core.expr.Expr | ufl.form.BaseForm,
        subset: op3.Slice | types.EllipsisType = Ellipsis,
        *,
        mode: AssignmentMode = AssignmentMode.STANDARD,
        allow_missing_dofs: bool = False,
    ):
        self.assignee = assignee
        self.expression = as_ufl(expression)
        self.subset = parse_subset(subset)
        self.mode = mode
        # NOTE: We now don't do anything with this option
        self.allow_missing_dofs = allow_missing_dofs

    @PETSc.Log.EventDecorator()
    def assign(self, allow_missing_dofs: bool | None = None):
        """Perform the assignment.

        Parameters
        ----------
        allow_missing_dofs
            Deprecated option. See the `Assigner` constructor instead.

        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )
        if allow_missing_dofs is not None:
            warnings.warn(
                "The 'allow_missing_dofs' option should be passed to the "
                "Assigner constructor. It is ignored here."
            )
        self._assign_op()

    @cached_property
    def _cross_mesh(self) -> bool:
        """Return whether the expression and assignee are on different (sub)meshes."""
        expr_meshes = firedrake.ufl_expr.extract_domains(self.expression)
        assignee_meshes = self.assignee.function_space().mesh()
        return any(mesh not in assignee_meshes for mesh in expr_meshes)

    @cached_property
    def _assign_op(self) -> Callable[[], None]:
        """Return the operation that updates the assignee."""
        expr_builder = AssignExprBuilder(self.assignee.function_space())
        if self._cross_mesh:
            if self.mode != AssignmentMode.STANDARD:
                raise NotImplementedError("Augmented assignment across meshes is not supported")
            loop_index = expr_builder.prepare_cross_mesh(self.expression, self.subset)
            op3_assignee = self.assignee.dat[loop_index]
        else:
            op3_assignee = self.assignee.dat[self.subset]

        op3_expr, _ = expr_builder(self.expression)

        match self.mode:
            case AssignmentMode.STANDARD:
                pass
            case AssignmentMode.IADD:
                op3_expr = op3_assignee + op3_expr
            case AssignmentMode.ISUB:
                op3_expr = op3_assignee - op3_expr
            case AssignmentMode.IMUL:
                op3_expr = op3_assignee * op3_expr
            case AssignmentMode.IDIV:
                op3_expr = op3_assignee / op3_expr
            case _:
                raise NotImplementedError

        if self._cross_mesh:
            loop = op3.loop(loop_index, op3_assignee.assign(op3_expr, _weakref=False))

            def assign_op() -> None:
                expr_builder.transfer_inputs()
                self.assignee.dat.buffer.sync_roots()
                loop()

        # If possible try to do the assignment by operating on numpy arrays
        elif self.subset is Ellipsis and expr_builder.array_assign_safe:
            # TODO: This is technically less efficient than the compile strategy
            # for repeated use. This should be exposed to the user.
            def assign_op() -> None:
                op3_assignee.assign(op3_expr, eager=True, eager_strategy="array")

        else:
            assign_op = op3_assignee.assign(op3_expr, _weakref=False)

        return assign_op


@functools.singledispatch
def parse_subset(obj: Any) -> op3.Slice | types.EllipsisType:
    raise TypeError


@parse_subset.register
def _(slice_: op3.Slice | op3.IndexTree) -> op3.Slice:
    return slice_


@parse_subset.register
def _(ellipsis: types.EllipsisType) -> types.EllipsisType:
    return ellipsis


@parse_subset.register
def _(none: None) -> types.EllipsisType:
    return Ellipsis


@parse_subset.register
def _(subset: op3.SubsetSliceComponent) -> op3.Slice:
    return op3.Slice("nodes", [subset])


@parse_subset.register(list)
@parse_subset.register(tuple)
@parse_subset.register(np.ndarray)
def _(subset: list | tuple | np.ndarray) -> op3.Slice:
    subset_dat = op3.Dat.from_sequence(subset, dtype=IntType)
    subset = op3.SubsetSliceComponent(None, subset_dat)
    return parse_subset(subset)
