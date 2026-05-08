import enum
import functools
import numbers
import operator
from functools import cached_property
from types import EllipsisType
from typing import Any

import numpy as np
from mpi4py import MPI


from pyadjoint.tape import annotate_tape
import finat.ufl
import numpy as np
import pyop3 as op3
import pytools
import ufl.classes
from pyadjoint.tape import annotate_tape
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.corealg.dag_traverser import DAGTraverser

from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.utils import IntType, ScalarType, split_by


class AssignExprBuilder(DAGTraverser):
    """Multifunction used for converting an expression into a weighted sum of coefficients.

    Calling ``map_expr_dag(CoefficientCollector(), expr)`` will return a tuple whose entries
    are of the form ``(coefficient, weight)``. Expressions that cannot be expressed as a
    weighted sum will raise an exception.

    Note: As well as being simple weighted sums (e.g. ``u.assign(2*v1 + 3*v2)``), one can
    also assign constant expressions of the appropriate shape (e.g. ``u.assign(1.0)`` or
    ``u.assign(2*v + 3)``). Therefore the returned tuple must be split since ``coefficient``
    may be either a :class:`firedrake.constant.Constant` or :class:`firedrake.function.Function`.
    """

    def __init__(self, function_space):
        self.function_space = function_space
        self.array_assign_allowed = True
        super().__init__()

    @functools.singledispatchmethod
    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)

    @process.register(Function)
    @process.register(Cofunction)
    def _(self, func) -> op3.Dat:
        if (
            func.ufl_element().family() != "Real"
            and func.ufl_element() != self.function_space.ufl_element()
        ):
            raise ValueError("All functions in the expression must have the same "
                             "element as the assignee")

        # NOTE: Is it really valid to consider Real a scalar type here? It means that we are
        is_scalar = func.ufl_element().family() == "Real"
        is_vector = not is_scalar

        func_mesh = func.function_space().mesh()
        if func_mesh != self.function_space.mesh():
            if not self.function_space.mesh().submesh_youngest_common_ancestor(func_mesh):
                raise ValueError(
                    "All functions in the expression must be defined on a single domain "
                    "that is in the same submesh family as domain of the assignee"
                )

            raise NotImplementedError("TODO")

        if func.function_space() != self.function_space:
            # If we have a restricted function space we have different data
            # layouts so naive array assignment will fail
            self.array_assign_allowed = False

        return func.dat, is_scalar, is_vector

    @process.register(Constant)
    def _(self, const) -> tuple[op3.Dat, bool, bool]:
        # TODO: Might want to restrict the allowed shapes here to only scalar and
        # self.function_space.shape
        return const.dat, True, False

    @process.register(ufl.classes.ScalarValue)
    def _(self, num) -> numbers.Number:
        return num.value(), True, False

    @process.register(ufl.classes.Zero)
    def _(self, zero) -> numbers.Number:
        return 0, True, False

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Product, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        if a_is_vector and b_is_vector:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector or b_is_vector
        return a_expr * b_expr, is_scalar, is_vector

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        if b_is_vector:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector
        return a_expr / b_expr, is_scalar, is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Sum, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector or b_is_vector
        return a_expr + b_expr, is_scalar, is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, o: ufl.classes.Power, a, b):
        breakpoint()
        # Only valid if a and b are scalars
        return ((Constant(self._as_scalar(a) ** self._as_scalar(b)), 1),)

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Abs, a):
        breakpoint()
        # Only valid if a is a scalar
        return ((Constant(abs(self._as_scalar(a))), 1),)

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
    assignee : firedrake.function.Function or firedrake.cofunction.Cofunction
        Function or Cofunction being assigned to.
    expression : ufl.core.expr.Expr or ufl.form.BaseForm
        Expression to be assigned.
    subset : pyop2.types.set.Set or pyop2.types.set.Subset or pyop2.types.set.MixedSet
        Subset to apply the assignment over.

    """
    # symbol = "="

    # _coefficient_collector = AssignExprBuilder()

    def __init__(self, assignee, expression, subset=Ellipsis, *, mode: AssignmentMode = AssignmentMode.STANDARD):
        expression = as_ufl(expression)

        self._assignee = assignee
        self._expression = expression
        self._subset = parse_subset(subset)
        self._mode = mode

        expr_builder = AssignExprBuilder(assignee.function_space())
        self._assign_expr, self._expr_is_scalar, self._expr_is_vector = expr_builder(expression)
        self._array_assign_allowed = expr_builder.array_assign_allowed

    @PETSc.Log.EventDecorator()
    def assign(self, allow_missing_dofs=False):
        """Perform the assignment.

        Parameters
        ----------
        allow_missing_dofs : bool
            Permit assignment between objects with mismatching nodes. If `True` then
            assignee nodes with no matching assigner nodes are ignored.

        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )

        array_assign_allowed = self._array_assign_allowed and self._subset is Ellipsis

        match self._mode:
            case AssignmentMode.STANDARD:
                expr = self._assign_expr
            case AssignmentMode.IADD:
                expr = self._assignee.dat + self._assign_expr
            case AssignmentMode.ISUB:
                expr = self._assignee.dat - self._assign_expr
            case AssignmentMode.IMUL:
                assert self._expr_is_scalar
                expr = self._assignee.dat * self._assign_expr
            case AssignmentMode.IDIV:
                assert self._expr_is_scalar
                expr = self._assignee.dat / self._assign_expr
            case _:
                raise NotImplementedError

        assignee = self._assignee.dat[self._subset]
        if array_assign_allowed:
            # TODO: This is technically less efficient than the compile strategy
            # for repeated use. This should be exposed to the user.
            assignee.assign(expr, eager=True, eager_strategy="array")
        else:
            # TODO: cache the expression for faster reuse of the assembler
            assignee.assign(expr, eager=True, eager_strategy="compile")

    # def _assign_single_mesh(self, lhs_func, subset, funcs, operator):
    #     data_ro = operator.attrgetter("data_ro")
    #     # subset_indices = Ellipsis if subset is None else indices(subset)
    #
    #     # def source_indices(f):
    #     #     target_space = lhs_func.function_space()
    #     #     target_map = target_space.cell_node_map()
    #     #     source_map = f.function_space().cell_node_map()
    #     #     if source_map is target_map:
    #     #         # Source and target spaces have the same DoF ordering.
    #     #         return subset_indices
    #     #     else:
    #     #         # Permute source indices into the target ordering.
    #     #         size = target_space.dof_dset.total_size
    #     #         perm = np.empty((size,), dtype=source_map.values.dtype)
    #     #         np.put(perm, values(target_map), values(source_map))
    #     #         perm = perm[:target_space.axes.owned.local_size]
    #     #         return perm[subset_indices]
    #
    #     func_data = np.array([data_ro(f.dat[subset]) for f in funcs])
    #     rvalue = self._compute_rvalue(func_data)
    #     self._assign_single_dat(lhs_func.dat, subset, rvalue)
    #
    # def _assign_multi_mesh(self, lhs_func, subset, funcs, operator, allow_missing_dofs):
    #     target_mesh = extract_unique_domain(lhs_func)
    #     target_V = lhs_func.function_space()
    #     source_V, = set(f.function_space() for f in funcs)
    #     raise NotImplementedError("entity node map is the wrong choice")
    #     composed_map = source_V.topological.entity_node_map(target_mesh.topology, "cell", "everywhere", None)
    #     indices_active = composed_map.indices_active_with_halo
    #     indices_active_all = indices_active.all()
    #     indices_active_all = target_mesh.comm.allreduce(indices_active_all, op=MPI.LAND)
    #     if subset is None:
    #         if not indices_active_all and not allow_missing_dofs:
    #             raise ValueError("Found assignee nodes with no matching assigner nodes: run with `allow_missing_dofs=True`")
    #         subset_indices_target = target_V.cell_node_map().values_with_halo[indices_active, :].flatten()
    #         subset_indices_source = composed_map.values_with_halo[indices_active, :].flatten()
    #     else:
    #         subset_indices_target, perm, _ = np.intersect1d(
    #             target_V.cell_node_map().values_with_halo[indices_active, :].flatten(),
    #             subset.indices,
    #             return_indices=True,
    #         )
    #         if len(subset.indices) > len(subset_indices_target) and not allow_missing_dofs:
    #             raise ValueError("Found assignee nodes with no matching assigner nodes: run with `allow_missing_dofs=True`")
    #         subset_indices_source = composed_map.values_with_halo[indices_active, :].flatten()[perm]
    #     # Use buffer array to make sure that owned DoFs are updated upon assigning.
    #     # The following example illustrates the issue that a naive assignment would cause.
    #     #
    #     # Consider the following target/source meshes distributed over 2 processes
    #     # with no partition overlap:
    #     #
    #     #                0----0----0----1----1
    #     #                |         |         |
    #     # target         0    0    0    1    1
    #     # (parent mesh)  |         |         |
    #     #                0----0----0----1----1  (owning ranks are shown)
    #     #
    #     #                          1----1----1
    #     #                          |         |
    #     # source                   1    1    1
    #     # (submesh)                |         |
    #     #                          1----1----1  (owning ranks are shown)
    #     #
    #     # Consider CG1 functions f (on parent) and fsub (on submesh). By a naive
    #     # f.assign(fsub, subset=...), the DoFs shared by rank 0 and rank 1 would
    #     # only be updated on rank 1, which sees those DoFs as ghost, and those
    #     # updated values on rank 1 would be overridden by the old values on rank 0
    #     # upon a halo exchange.
    #     #
    #     # TODO: Use work array for buffer?
    #     buffer = type(lhs_func)(target_V)
    #     finfo = np.finfo(lhs_func.dat.dtype)
    #     buffer.dat._data[:] = finfo.max
    #     func_data = np.array([f.dat.data_ro_with_halos[subset_indices_source] for f in funcs])
    #     rvalue = self._compute_rvalue(func_data)
    #     self._assign_single_dat(buffer.dat, subset_indices_target, rvalue, True)
    #     # Make all owned DoFs up-to-date; ghost DoFs may or may not be up-to-date after this.
    #     buffer.dat.local_to_global_begin(op2.MIN)
    #     buffer.dat.local_to_global_end(op2.MIN)
    #     indices = np.where(buffer.dat.data_ro_with_halos < finfo.max * 0.999999999999)
    #     lhs_func.dat.data_wo_with_halos[indices] = buffer.dat.data_ro_with_halos[indices]

@functools.singledispatch
def parse_subset(obj: Any) -> op3.Slice | EllipsisType:
    raise TypeError


@parse_subset.register
def _(slice_: op3.Slice) -> op3.Slice:
    return slice_


@parse_subset.register
def _(ellipsis: EllipsisType) -> EllipsisType:
    return ellipsis


@parse_subset.register
def _(none: None) -> EllipsisType:
    return Ellipsis


@parse_subset.register
def _(subset: op3.Subset) -> op3.Slice:
    return op3.Slice("nodes", [subset])


@parse_subset.register(list)
@parse_subset.register(tuple)
def _(subset: list | tuple) -> op3.Slice:
    subset_dat = op3.Dat.from_sequence(subset, dtype=IntType)
    subset = op3.Subset(None, subset_dat)
    return parse_subset(subset)
