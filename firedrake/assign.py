import enum
import functools
import numbers
import operator
import types
import warnings
from functools import cached_property
from typing import Any, Literal, Callable

import numpy as np
from immutabledict import immutabledict as idict
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

import firedrake.ufl_expr
from firedrake import utils
from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.petsc import PETSc
from firedrake.utils import IntType, ScalarType, split_by


class AssignExprBuilder(DAGTraverser):
    """Traverser used for converting an expression into a pyop3 expression."""

    def __init__(
        self,
        function_space: firedrake.functionspaceimpl.WithGeometry,
        loop_index: op3.LoopIndex | None,
    ) -> None:
        self.function_space = function_space
        self.loop_index = loop_index
        self.array_assign_safe = True
        super().__init__()

    @functools.singledispatchmethod
    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)

    @process.register
    def _(self, func: Function | Cofunction, /) -> op3.Dat:
        if (
            func.ufl_element().family() != "Real"
            and func.ufl_element() != self.function_space.ufl_element()
        ):
            raise ValueError(
                "All functions in the expression must have the same element as the assignee"
            )

        if self.loop_index is not None:
            assignee_mesh = self.function_space.mesh()
            func_mesh = func.function_space().mesh()
            if func_mesh != assignee_mesh:
                common_ancestor = assignee_mesh.submesh_youngest_common_ancestor(func_mesh)
                if not common_ancestor:
                    raise ValueError(
                        "All functions in the expression must be defined on a single domain "
                        "that is in the same submesh family as domain of the assignee"
                    )

                loop_index = self.loop_index
                # We are looping over the points of the target mesh, so we need to go
                # target points -> common ancestor -> func points
                aa = assignee_mesh.submesh_ancestors
                for a in reversed(aa[:aa.index(common_ancestor)]):
                    # This seems very inefficient
                    afs = self.function_space.reconstruct(mesh=a)
                    loop_index = afs.submesh_child_to_parent_map(loop_index)

                bb = func_mesh.submesh_ancestors
                for b in bb[:bb.index(common_ancestor)]:
                    bfs = self.function_space.reconstruct(mesh=b)
                    loop_index = bfs.submesh_parent_to_child_map(loop_index)

                op3_expr = func.dat[loop_index]

            else:
                op3_expr = func.dat[self.loop_index]

        else:
            op3_expr = func.dat

            # If we have a restricted function space we have different data
            # layouts so naive array assignment will fail and we have to fall
            # back to generating code.
            if func.function_space().boundary_set != self.function_space.boundary_set:
                self.array_assign_safe = False

        # NOTE: Is it really valid to consider Real a scalar type here?
        is_scalar = func.ufl_element().family() == "Real"
        is_vector = not is_scalar

        return op3_expr, is_scalar, is_vector

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
    def _(self, _: ufl.classes.Power, a, b):
        a_expr, a_is_scalar, a_is_vector = a
        b_expr, b_is_scalar, b_is_vector = b

        is_scalar = a_is_scalar and b_is_scalar
        is_vector = a_is_vector or b_is_vector

        # Only valid if a and b are scalars
        assert is_scalar

        return a_expr ** b_expr, is_scalar, is_vector

    @process.register
    @DAGTraverser.postorder
    def _(self, _: ufl.classes.Abs, a):
        a_expr, is_scalar, is_vector = a
        assert is_scalar
        return abs(a_expr), is_scalar, is_vector

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
        if (
            not expr_meshes
            or len(expr_meshes) == 1 and expr_meshes[0] == self.assignee.function_space().mesh()
        ):
            return False
        else:
            return True

    @cached_property
    def _assign_op(self) -> Callable[[], None]:
        if self._cross_mesh:
            # If we are assigning between submeshes then we have to generate a
            # full parloop in order to be able to include maps. For example:
            #
            #     for i
            #       dat1[i] <- dat2[g(f(i))]
            loop_index = self.assignee.function_space().nodes[self.subset].iter()
            op3_assignee = self.assignee.dat[self.subset][loop_index]
        else:
            loop_index = None
            op3_assignee = self.assignee.dat[self.subset]

        expr_builder = AssignExprBuilder(self.assignee.function_space(), loop_index)
        op3_expr, is_scalar, _ = expr_builder(self.expression)

        match self.mode:
            case AssignmentMode.STANDARD:
                pass
            case AssignmentMode.IADD:
                op3_expr = op3_assignee + op3_expr
            case AssignmentMode.ISUB:
                op3_expr = op3_assignee - op3_expr
            case AssignmentMode.IMUL:
                assert is_scalar
                op3_expr = op3_assignee * op3_expr
            case AssignmentMode.IDIV:
                assert is_scalar
                op3_expr = op3_assignee / op3_expr
            case _:
                raise NotImplementedError

        if self._cross_mesh:
            # We use an intermediate buffer array to make sure that owned DoFs are
            # updated upon assigning. This is because the point ownership is not
            # guaranteed to be the same between target and source meshes (see
            # 'sanitizeSubMesh' kwarg for DMPlex.filter).
            #
            # The following example illustrates the issue that a naive assignment would cause.
            #
            # Consider the following target/source meshes distributed over 2 processes
            # with no partition overlap:
            #
            #                0----0----0----1----1
            #                |         |         |
            # target         0    0    0    1    1
            # (parent mesh)  |         |         |
            #                0----0----0----1----1  (owning ranks are shown)
            #
            #                          1----1----1
            #                          |         |
            # source                   1    1    1
            # (submesh)                |         |
            #                          1----1----1  (owning ranks are shown)
            #
            # Consider CG1 functions f (on parent) and fsub (on submesh). By a naive
            # f.assign(fsub, subset=...), the DoFs shared by rank 0 and rank 1 would
            # only be updated on rank 1, which sees those DoFs as ghost, and those
            # updated values on rank 1 would be overridden by the old values on rank 0
            # upon a halo exchange.
            #
            # NOTE: I believe that this limitation only exists because if we don't
            # have enough overlap then we can get orphaned subpoints (i.e. edges or
            # vertices) on one rank with no available cell. I think a much more
            # sensible thing to do here is make sure that we have an appropriate
            # overlap from the outset.

            # create a persistent loop object so we can reuse it
            loop = op3.loop(loop_index, op3_assignee.assign(op3_expr))

            def assign_op() -> None:
                # swap out another buffer for the computation
                assignee_buffer = self.assignee.dat.buffer
                orig_assignee_data = assignee_buffer._host_data.copy()

                fmin = np.finfo(assignee_buffer.dtype).min
                assignee_buffer._host_data[...] = fmin

                # FIXME: These parameters shouldn't be needed if we correctly
                # mask things (by intersecting meshes)
                loop(compiler_parameters={"propagate_negatives": True, "mask_array_accesses": True})

                # now write back only the values that were touched
                assignee_buffer._reduce_leaves_to_roots(MPI.MAX)
                unchanged_idxs = np.flatnonzero(np.isclose(assignee_buffer._host_data, fmin))
                assignee_buffer._host_data[unchanged_idxs] = orig_assignee_data[unchanged_idxs]

        # If possible try to do the assignment by operating on numpy arrays
        elif self.subset is Ellipsis and expr_builder.array_assign_safe:
            # TODO: This is technically less efficient than the compile strategy
            # for repeated use. This should be exposed to the user.
            def assign_op() -> None:
                op3_assignee.assign(op3_expr, eager=True, eager_strategy="array")

        else:
            assign_op = op3_assignee.assign(op3_expr)

        return assign_op


@functools.singledispatch
def parse_subset(obj: Any) -> op3.Slice | types.EllipsisType:
    raise TypeError


@parse_subset.register
def _(slice_: op3.Slice) -> op3.Slice:
    return slice_


@parse_subset.register
def _(ellipsis: types.EllipsisType) -> types.EllipsisType:
    return ellipsis


@parse_subset.register
def _(none: None) -> types.EllipsisType:
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
