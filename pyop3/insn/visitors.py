from __future__ import annotations

import abc
from ctypes import Array
import collections
>>>>>>> connorjward/pyop3
import functools
import itertools
from itertools import zip_longest
import numbers
from collections.abc import Iterable, Mapping
from os import access
from re import I
from typing import Any

import numpy as np
from petsc4py import PETSc
from immutabledict import immutabledict as idict

import pyop3.expr.base as expr_types
from pyop3.expr.buffer import MatArrayBufferExpression
import pyop3.expr.visitors as expr_visitors
from pyop3 import utils

from pyop3.node import NodeTransformer, NodeVisitor, NodeCollector
from pyop3.expr.tensor.base import TensorTransform, InPlaceTensorTransform, OutOfPlaceTensorTransform
from pyop3.expr import Scalar, Dat, Tensor, Mat, LinearDatBufferExpression, BufferExpression, MatPetscMatBufferExpression
from pyop3.tree.axis_tree import AxisTree, AxisForest
from pyop3.tree.axis_tree.tree import merge_axis_trees
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, NullBuffer, ArrayBuffer, BufferRef

from pyop3.tree.index_tree.tree import LoopIndex
from pyop3.tree.index_tree.parse import _as_context_free_indices
import pyop3.insn as insn_types
# TODO: remove all these in favour of op3_insn
from pyop3.insn.base import (
    INC,
    READ,
    RW,
    WRITE,
    AssignmentType,
    DummyKernelArgument,
    ArrayAccessType,
    enlist,
    maybe_enlist,
    non_null,
    filter_null,
)
from pyop3.utils import OrderedFrozenSet

import pyop3.extras.debug


class InstructionTransformer(NodeTransformer):

    @functools.singledispatchmethod
    def process(self, insn: insn_types.Instruction, /, **kwargs) -> insn_types.Instruction:
        return super().process(insn, **kwargs)

    # Instruction lists have a common pattern
    @process.register(insn_types.InstructionList)
    @NodeTransformer.postorder
    def _(self, insn_list: insn_types.InstructionList, /, *insns, **kwargs) -> insn_types.Instruction:
        breakpoint()  # likely wrong
        return maybe_enlist(insns)


class LoopContextExpander(InstructionTransformer):

    @functools.singledispatchmethod
    def process(self, insn: insn_types.Instruction, /, **kwargs) -> insn_types.Instruction:
        return super().process(insn, **kwargs)

    @process.register(insn_types.Loop)
    def _(self, loop: insn_types.Loop, /, *, loop_context) -> insn_types.Loop | insn_types.InstructionList:
        expanded_loops = []
        iterset = loop.index.iterset
        for leaf_path in iterset.leaf_paths:
            # collect the possible targets per leaf
            # leaf_target_paths = tuple(
            #     leaf_target_paths_per_target[leaf_path]
            #     for leaf_target_paths_per_target in iterset.leaf_target_paths
            # )
            # loop_context = {loop.index.id: leaf_target_paths}
            loop_context_ = {loop.index.id: leaf_path}

            restricted_loop_index = utils.just_one(_as_context_free_indices(loop.index, loop_context))

            # skip empty loops
            if restricted_loop_index.iterset.size == 0:
                continue

            loop_context_acc_ = loop_context | loop_context_
            expanded_loop = type(loop)(
                restricted_loop_index,
                [
                    self(stmt, loop_context=loop_context_acc_)
                    for stmt in loop.statements
                ]
            )
            expanded_loops.append(expanded_loop)
        return maybe_enlist(expanded_loops)


    @process.register(insn_types.CalledFunction)
    def _(self, func: insn_types.CalledFunction, /, *, loop_context) -> insn_types.CalledFunction:
        new_arguments = tuple(arg.with_context(loop_context) for arg in func.arguments)
        return func.__record_init__(_arguments=new_arguments)

    @process.register(insn_types.ArrayAssignment)
    def _(self, assignment: insn_types.ArrayAssignment, /, *, loop_context) -> insn_types.ArrayAssignment:
        assignee = expr_visitors.restrict_to_context(assignment.assignee, loop_context)
        expression = expr_visitors.restrict_to_context(assignment.expression, loop_context)
        return assignment.__record_init__(_assignee=assignee, _expression=expression)

    @process.register(insn_types.Exscan)  # for now assume we are fine
    def _(self, insn: insn_types.Instruction, /, **kwargs) -> insn_types.Instruction:
        return self.reuse_if_untouched(insn)

@_expand_loop_contexts_rec.register(ArrayAssignment)
def _(assignment: ArrayAssignment, /, *, loop_context_acc) -> ArrayAssignment:
    assignee = restrict_expression_to_context(assignment.assignee, loop_context_acc)
    expression = restrict_expression_to_context(assignment.expression, loop_context_acc)
    return assignment.__record_init__(_assignee=assignee, _expression=expression)


# NOTE: This is a bad name for this transformation. 'expand_multi_component_loops'?
def expand_loop_contexts(insn: insn_types.Instruction, /) -> insn_types.Instruction:
    return LoopContextExpander()(insn, loop_context=idict())


class ImplicitPackUnpackExpander(NodeTransformer):
    def __init__(self):
        self._name_generator = utils.UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    @_apply.register(insn_types.NullInstruction)
    @_apply.register(insn_types.Exscan)  # assume we are fine
    def _(self, insn, /):
        return insn

    # TODO Can I provide a generic "operands" thing? Put in the parent class?
    @_apply.register(insn_types.Loop)
    def _(self, loop: insn_types.Loop) -> insn_types.Loop:
        new_statements = [s for stmt in loop.statements for s in enlist(self._apply(stmt))]
        return loop.__record_init__(statements=new_statements)

    @_apply.register
    def _(self, insn_list: insn_types.InstructionList):
        return type(insn_list)([insn_ for insn in insn_list for insn_ in enlist(self._apply(insn))])

    # # TODO: Should be the same as Assignment
    # @_apply.register
    # def _(self, assignment: PetscMatInstruction):
    #     # FIXME: Probably will not work for things like mat[x, y].assign(dat[z])
    #     # where the expression is indexed.
    #     return (assignment,)

    @_apply.register
    def _(self, assignment: insn_types.ArrayAssignment):
        # I think this is fine...
        return assignment

    @_apply.register
    def _(self, terminal: insn_types.CalledFunction):
        gathers = []
        # NOTE: scatters are executed in LIFO order
        scatters = []
        arguments = []
        for (arg, intent), shape in zip(
            terminal.function_arguments, terminal.argument_shapes, strict=True
        ):
            # bare_arg, arg_pack_insns, arg_unpack_insns = _expand_reshapes(arg, intent)
            # gathers.extend(arg_pack_insns)
            # scatters.extend(arg_unpack_insns)
            #
            #     if intent == READ:
            #         gathers.extend(ArrayAssignment(temporary, arg, "write"))
            #     elif intent == WRITE:
            #         # This is currently necessary because some local kernels
            #         # (interpolation) actually increment values instead of setting
            #         # them directly. This should ideally be addressed.
            #         gathers.append(ArrayAssignment(temporary, 0, "write"))
            #         scatters.insert(0, ArrayAssignment(arg, temporary, "write"))
            #     elif intent == RW:
            #         gathers.append(ArrayAssignment(temporary, arg, "write"))
            #         scatters.insert(0, ArrayAssignment(arg, temporary, "write"))
            #     else:
            #         assert intent == INC

            # emit pack/unpack instructions
            if _requires_pack_unpack(arg):
                # TODO: Make generic across Array types
                if isinstance(arg, Dat):
                    temporary = Dat.null(arg.axes.materialize().localize(), dtype=arg.dtype, prefix="t")
                else:
                    assert isinstance(arg, Mat)
                    temporary = Mat.null(arg.row_axes.materialize().localize(), arg.caxes.materialize().localize(), dtype=arg.dtype, prefix="t")

                if intent == READ:
                    gathers.append(insn_types.ArrayAssignment(temporary, arg, "write"))
                elif intent == WRITE:
                    scatters.insert(0, insn_types.ArrayAssignment(arg, temporary, "write"))
                elif intent == RW:
                    gathers.append(insn_types.ArrayAssignment(temporary, arg, "write"))
                    scatters.insert(0, insn_types.ArrayAssignment(arg, temporary, "write"))
                else:
                    assert intent == INC
                    gathers.append(insn_types.ArrayAssignment(temporary, 0, "write"))
                    scatters.insert(0, insn_types.ArrayAssignment(arg, temporary, "inc"))

                function_arg = LinearDatBufferExpression(temporary.buffer, 0)
            else:
                if arg.buffer.is_nested:
                    raise NotImplementedError("Assume cannot have nest indices here")
                function_arg = LinearDatBufferExpression(arg.buffer, 0)
            arguments.append(function_arg)

        return maybe_enlist((*gathers, insn_types.StandaloneCalledFunction(terminal.function, arguments), *scatters))


# TODO check this docstring renders correctly
def expand_implicit_pack_unpack(expr: insn_types.Instruction):
    """Expand implicit pack and unpack operations.

    An implicit pack/unpack is something of the form

    .. code::
        kernel(dat[f(p)])

    In order for this to work the ``dat[f(p)]`` needs to be packed
    into a temporary. Assuming that its intent in ``kernel`` is
    `pyop3.WRITE`, we would expand this function into

    .. code::
        tmp <- [0, 0, ...]
        kernel(tmp)
        dat[f(p)] <- tmp

    Notes
    -----
    For this routine to work, any context-sensitive loops must have
    been expanded already (with `expand_loop_contexts`). This is
    because context-sensitive arrays may be packed into temporaries
    in some contexts but not others.

    """
    return ImplicitPackUnpackExpander().apply(expr)


@functools.singledispatch
def _requires_pack_unpack(arg: insn_types.FunctionArgument) -> bool:
    raise TypeError


@_requires_pack_unpack.register(Scalar)
def _(scalar: Scalar) -> bool:
    return False


@_requires_pack_unpack.register(Dat)
def _(dat: Dat) -> bool:
    # This is overly restrictive since we could pass something contiguous like
    # dat[i0, :] directly to a local kernel
    return not (isinstance(dat.buffer, ConcreteBuffer) and _layouts_match(dat.axes) and not has_materialized_temporaries(dat))


@_requires_pack_unpack.register(Mat)
def _(mat: Mat) -> bool:
    return not (not isinstance(mat.buffer, PetscMatBuffer) and _layouts_match(mat.row_axes) and _layouts_match(mat.caxes) and not has_materialized_temporaries(mat))


def _layouts_match(axes: AxisTreeT) -> bool:
    if isinstance(axes, AxisForest):
        return utils.strictly_all(map(_layouts_match, axes.trees))
    else:
        return axes.leaf_subst_layouts == axes.unindexed.leaf_subst_layouts


@functools.singledispatch
def expand_assignments(obj: Any, /) -> insn_types.InstructionList:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@expand_assignments.register(insn_types.InstructionList)
def _(insn_list: insn_types.InstructionList, /) -> insn_types.InstructionList:
    return maybe_enlist((expand_assignments(insn) for insn in insn_list))


@expand_assignments.register(insn_types.Loop)
def _(loop: insn_types.Loop, /) -> insn_types.Loop:
    return insn_types.Loop(
        loop.index,
        [
            stmt_ for stmt in loop.statements for stmt_ in enlist(expand_assignments(stmt))
        ],
    )


@expand_assignments.register(insn_types.StandaloneCalledFunction)
# @expand_assignments.register(PetscMatAssignment)
@expand_assignments.register(insn_types.NullInstruction)
@expand_assignments.register(insn_types.Exscan)  # assume we are fine
def _(func: insn_types.StandaloneCalledFunction, /) -> insn_types.StandaloneCalledFunction:
    return func


def _intent_as_access_type(intent):
    if intent == READ:
        return ArrayAccessType.READ
    if intent == WRITE:
        return ArrayAccessType.WRITE
    else:
        assert intent == INC
        return ArrayAccessType.INC



@expand_assignments.register(CalledFunction)
def _(called_func: CalledFunction, /) -> InstructionList:
    bare_func_args = []
    pack_insns = []
    unpack_insns = []

    for func_arg, intent in zip(
        called_func.arguments, called_func.function._access_descrs, strict=True
    ):
        arg_pack_insns = []
        arg_unpack_insns = []

        # function calls need materialised arrays
        # FIXME: INC'd globals with transforms (ie parents) have to be materialised
        if _requires_pack_unpack(func_arg):
            local_tensor = func_arg.materialize()

            if intent == READ:
                arg_pack_insns.append(local_tensor.assign(func_arg))
            elif intent == WRITE:
                # This is currently necessary because some local kernels
                # (interpolation) actually increment values instead of setting
                # them directly. This should ideally be addressed.
                arg_pack_insns.append(local_tensor.assign(0))
                arg_unpack_insns.insert(0, func_arg.assign(local_tensor))
            elif intent == RW:
                arg_pack_insns.append(local_tensor.assign(func_arg))
                arg_unpack_insns.insert(0, func_arg.assign(local_tensor))
            else:
                assert intent == INC
                arg_pack_insns.append(local_tensor.assign(0))
                arg_unpack_insns.insert(0, func_arg.iassign(local_tensor))

            materialized_arg = LinearDatBufferExpression(local_tensor.buffer, 0)
        else:
            materialized_arg = LinearDatBufferExpression(func_arg.buffer, 0)

        bare_func_args.append(materialized_arg)
        pack_insns.extend(arg_pack_insns)
        unpack_insns.extend(arg_unpack_insns)

    bare_called_func = StandaloneCalledFunction(called_func.function, bare_func_args)
    return maybe_enlist((*pack_insns, bare_called_func, *unpack_insns))


@expand_assignments.register(insn_types.ArrayAssignment)
def _(assignment: insn_types.ArrayAssignment, /) -> insn_types.InstructionList:
    # This function is complete magic and deserves some serious exposition:
    #
    # To begin with, consider the assignment:
    #
    #     x <- y
    #
    # where 'y' is a dat with a parent. Having a parent means that 'y' is
    # the result of a transformation applied to another dat. When we generate
    # code for this assignment we therefore need to traverse the hierarchy of
    # transformations and emit something like:
    #
    #     t <- Y
    #     f(t)       -- in-place transform
    #     u <- g(t)  -- out-of-place transform
    #     x <- u
    #
    # where 'Y' is the global data structure at the top of the parent hierarchy.
    #
    # To make this happen, in this function we 'expand' the expression 'y',
    # giving us back 'u' and the sequence of transformation instructions.
    #
    # Now let's imagine what happens for 'x <- y' where the assignee ('x') is
    # the transformed object. We thus want to generate code like:
    #
    #     t <- y
    #     f(t)       -- in-place transform
    #     u <- g(t)  -- out-of-place transform
    #     X <- u
    #
    # where 'X' is the global data at the top of the parent hierarchy for 'x'.
    # Expanding the assignee will return 't' and the subsequent transformations.
    #
    # Lastly, if we consider incrementing, instead of assigning (i.e. 'x += y'),
    # then some changes are needed. We need to generate code like:
    #
    #     t <- y
    #     f(t)       -- in-place transform
    #     u <- g(t)  -- out-of-place transform
    #     X += u
    #
    # Note that the final instruction is where the increment takes place.
    #     t1 <- t0
    #     mat[f(p), f(p)] <- t1
    # if assignment.is_mat_access:
    #     raise NotImplementedError("think")
    #     return op3_insn.InstructionList([assignment])

    bare_expression, extra_input_insns = _expand_reshapes(
        assignment.expression, ArrayAccessType.READ
    )

    if assignment.assignment_type == AssignmentType.WRITE:
        access_type = ArrayAccessType.WRITE
    else:
        assert assignment.assignment_type == AssignmentType.INC
        access_type = ArrayAccessType.INC

    bare_assignee, assignee_transform_insns = _expand_reshapes(
        assignment.assignee, access_type
    )
    bare_assignment = assignment.__record_init__(_assignee=bare_assignee, _expression=bare_expression)

    if bare_assignee == assignment.assignee:
        # no extra assignments
        bare_assignment = assignment.__record_init__(_assignee=bare_assignee, _expression=bare_expression)
    else:
        bare_assignment = assignment.__record_init__(_assignee=bare_assignee, _expression=bare_expression, _assignment_type=AssignmentType.WRITE)

    return maybe_enlist((*expression_transform_insns, bare_assignment, *assignee_transform_insns))


@functools.singledispatch
def _expand_reshapes(expr: Any, /, *args, **kwargs):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@_expand_reshapes.register
def _(op: expr_types.UnaryOperator, /, access_type):
    bare_a, unpack_insns = _expand_reshapes(op.a, access_type)
    return (type(op)(bare_a), unpack_insns)


@_expand_reshapes.register
def _(op: expr_types.BinaryOperator, /, access_type):
    bare_a, a_unpack_insns = _expand_reshapes(op.a, access_type)
    bare_b, b_unpack_insns = _expand_reshapes(op.b, access_type)
    return (type(op)(bare_a, bare_b), a_unpack_insns+b_unpack_insns)


@_expand_reshapes.register
def _(op: expr_types.TernaryOperator, /, access_type):
    bare_operands = []
    unpack_insns = []
    for operand in op.operands:
        bare_operand, operand_unpack_insns = _expand_reshapes(operand, access_type)
        bare_operands.append(bare_operand)
        unpack_insns.extend(operand_unpack_insns)
    return (type(op)(*bare_operands), tuple(unpack_insns))


@_expand_reshapes.register(numbers.Number)
@_expand_reshapes.register(expr_types.AxisVar)
@_expand_reshapes.register(expr_types.LoopIndexVar)
@_expand_reshapes.register(BufferExpression)
@_expand_reshapes.register(expr_types.NaN)
def _(var, /, access_type):
    return (var, ())


# TODO: Add intermediate type here to assert that there is no longer a parent attr
@_expand_reshapes.register(Tensor)
def _(array: Tensor, /, access_type):
    if access_type == ArrayAccessType.READ:
        return _expand_transforms_in(array)
    else:
        # assert access_type == ArrayAccessType.WRITE
        # return _expand_transforms_out(array, access_type)
        return _expand_transforms_out(array, access_type)


def _expand_transforms_in(tensor: Tensor) -> tuple[Tensor, tuple[Instruction, ...]]:
    """
    I.e.

    * given: 'T'
    * want:

        f(U)         # in-place transform
        T <- g(U)    # out-of-place transform
        ...
        kernel(T)

      and 'U'

    """
    current_tensor = tensor
    pack_insns = ()
    while current_tensor.parent:
        parent_tensor = current_tensor.parent.untransformed

        bare_current_tensor = current_tensor.__record_init__(_parent=None)
        bare_parent_tensor = parent_tensor.__record_init__(_parent=None)

        if isinstance(current_tensor.parent, InPlaceTensorTransform):
            if isinstance(bare_current_tensor, Dat):
                bare_current_tensor_reshaped = bare_current_tensor.with_axes(bare_parent_tensor.axes.materialize())
            elif isinstance(bare_current_tensor, Mat):
                bare_current_tensor_reshaped = bare_current_tensor.with_axes(bare_parent_tensor.raxes.materialize(), bare_parent_tensor.caxes.materialize())
            else:
                raise NotImplementedError
            current_pack_insns = (
                bare_current_tensor_reshaped.assign(bare_parent_tensor),
                *current_tensor.parent.transform_in(bare_current_tensor),
                )
        else:
            assert isinstance(current_tensor.parent, OutOfPlaceTensorTransform)
            current_pack_insns = current_tensor.parent.transform_in(bare_parent_tensor, bare_current_tensor)

        pack_insns = (*pack_insns, *current_pack_insns)
        current_tensor = parent_tensor

    # for inputs return the first tensor (it's the local one)
    return tensor.__record_init__(_parent=None), pack_insns

def _expand_transforms_out(tensor: Tensor, access_type) -> tuple[Tensor, tuple[Instruction, ...]]:
    """
    I.e.

    * given: 'T'
    * want:

        kernel(T, ...)
        ...
        f'(T)        # in-place transform
        U <- g'(T)   # out-of-place transform

      and 'U'

    """
    current_tensor = tensor
    unpack_insns = ()
    while current_tensor.parent:
        parent_tensor = current_tensor.parent.untransformed

        bare_current_tensor = current_tensor.__record_init__(_parent=None)
        bare_parent_tensor = parent_tensor.__record_init__(_parent=None)

        # Since transformations are atomic we have to make sure that they
        # are handled as distinct instructions. This means that we have to
        # materialise them every time. As an example consider an in-place
        # permutation followed by a reshape:
        #
        #   for i < 6
        #     t1[i] = t0[perm[i]]
        #   for j < 3
        #     dat[f(j)] = t1[j]
        #   for k < 3
        #     dat[g(k)] = t1[k+3]
        #
        # This cannot be represented as:
        #
        #   for j < 3
        #     dat[f(j)] += t0[perm[j]]
        #   for k < 3
        #     dat[g(k)] += t0[perm[k+3]]
        #
        # because the system is not that clever: the shapes of t0 and dat[???]
        # do not match.
        #
        # TODO: This materialisation is not actually needed if we don't index
        # the tensor.
        if isinstance(current_tensor.parent, InPlaceTensorTransform):
            if isinstance(bare_current_tensor, Dat):
                # bare_parent_tensor_reshaped = bare_parent_tensor.with_axes(bare_current_tensor.axes.materialize())
                bare_current_tensor_reshaped = bare_current_tensor.with_axes(bare_parent_tensor.axes.materialize())
            elif isinstance(bare_current_tensor, Mat):
                # bare_parent_tensor_reshaped = bare_parent_tensor.with_axes(bare_current_tensor.raxes.materialize(), bare_current_tensor.caxes.materialize())
                bare_current_tensor_reshaped = bare_current_tensor.with_axes(bare_parent_tensor.row_axes.materialize(), bare_parent_tensor.column_axes.materialize())
            else:
                raise NotImplementedError
            # NOTE: It seems a bit weird to have an assignment given that this is 'inplace'
            current_unpack_insns = [
                *current_tensor.parent.transform_out(bare_current_tensor),
            ]

            # at the end of the traversal, maybe emit an INC
            if not parent_tensor.parent and access_type == ArrayAccessType.INC:
                current_unpack_insns.append(bare_parent_tensor.iassign(bare_current_tensor_reshaped))
            else:
                current_unpack_insns.append(bare_parent_tensor.assign(bare_current_tensor_reshaped))
        else:
            assert isinstance(current_tensor.parent, OutOfPlaceTensorTransform)
            current_unpack_insns = current_tensor.parent.transform_out(bare_current_tensor, bare_parent_tensor)

        unpack_insns = (*current_unpack_insns, *unpack_insns)
        current_tensor = parent_tensor

    # for inputs return the last tensor (it's the global one)
    # return current_tensor, unpack_insns
    # no, don't
    return tensor.__record_init__(_parent=None), unpack_insns


def has_materialized_temporaries(tensor: Tensor) -> bool:
    while tensor.parent:
        if isinstance(tensor.parent, OutOfPlaceTensorTransform):
            return True
        else:
            tensor = tensor.parent.untransformed
    return False


@functools.singledispatch
def concretize_layouts(obj: Any, /) -> insn_types.Instruction:
    """Lock in the layout expressions that data arguments are accessed with.

    For example this converts Dats to DatArrayBufferExpressions that cannot
    be indexed further.

    This function also trims expressions to remove any zero-sized bits.

    """
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@concretize_layouts.register(insn_types.NullInstruction)
@concretize_layouts.register(insn_types.Exscan)  # assume we are fine
def _(null: insn_types.NullInstruction, /) -> insn_types.NullInstruction:
    return null


@concretize_layouts.register(insn_types.InstructionList)
def _(insn_list: insn_types.InstructionList, /) -> insn_types.Instruction:
    return maybe_enlist(
        filter(non_null, (map(concretize_layouts, insn_list)))
    )


@concretize_layouts.register(insn_types.Loop)
def _(loop: insn_types.Loop, /) -> insn_types.Loop | insn_types.NullInstruction:
    statements = tuple(filter_null(map(concretize_layouts, loop.statements)))
    return loop.__record_init__(statements=statements) if statements else insn_types.NullInstruction()


@concretize_layouts.register(insn_types.StandaloneCalledFunction)
def _(func: insn_types.StandaloneCalledFunction, /) -> insn_types.StandaloneCalledFunction:
    return func


@concretize_layouts.register(insn_types.ArrayAssignment)
def _(assignment: insn_types.ArrayAssignment, /) -> insn_types.NonEmptyArrayAssignment | insn_types.NullInstruction:
    assignee = expr_visitors.concretize_layouts(assignment.assignee, assignment.shape)
    expression = expr_visitors.concretize_layouts(assignment.expression, assignment.shape)

    return insn_types.NonEmptyArrayAssignment(assignee, expression, assignment.shape, assignment.assignment_type, comm=assignment.comm)


MAX_COST_CONSIDERATION_FACTOR = 5
"""Maximum factor an expression cost can exceed the minimum and still be considered."""


@PETSc.Log.EventDecorator()
def materialize_indirections(insn: insn_types.Instruction, *, compress: bool = False) -> insn_types.Instruction:
    # try setting a 'global' cache here
    # TODO: formalise this.
    mycache = {}

    expr_candidates = collect_candidate_indirections(insn, compress=compress)

    if not expr_candidates:
        # For things like null instructions there are no expression candidates
        # to think about so we can stop early
        return insn

    # Combine the best per-arg candidates into the initial overall best candidate
    best_candidate = {}
    max_cost = 0
    for arg_id, arg_candidates in expr_candidates.items():
        expr, expr_cost = min(arg_candidates, key=lambda item: item[1])
        best_candidate[arg_id] = (expr, expr_cost)
        max_cost += expr_cost

    # Optimise by dropping any immediately bad candidates
    trimmed_expr_candidates = {}
    for arg_id, arg_candidates in expr_candidates.items():
        trimmed_arg_candidates = []
        min_arg_cost = min((cost for _, cost in arg_candidates))
        for arg_candidate, cost in arg_candidates:
            if cost <= max_cost and cost <= min_arg_cost * MAX_COST_CONSIDERATION_FACTOR:
                trimmed_arg_candidates.append((arg_candidate, cost))
        trimmed_expr_candidates[arg_id] = tuple(trimmed_arg_candidates)
    expr_candidates = trimmed_expr_candidates

    # Now select the combination with the lowest combined cost. We can make savings here
    # by sharing indirection maps between different arguments. For example, if we have
    #
    #     dat1[mapA[mapB[mapC[i]]]]
    #     dat2[mapB[mapC[i]]]
    #
    # then we can (sometimes) minimise the data cost by having
    #     dat1[mapA[mapBC[i]]]
    #     dat2[mapBC[i]]
    #
    # instead of
    #
    #     dat1[mapABC[i]]
    #     dat2[mapBC[i]]
    min_cost = max_cost
    for shared_candidate in utils.expand_collection_of_iterables(expr_candidates):
        cost = 0
        seen_exprs = set()
        for expr, expr_cost in shared_candidate.values():
            if expr not in seen_exprs:
                cost += expr_cost
                seen_exprs.add(expr)

        if cost < min_cost:
            best_candidate = shared_candidate
            min_cost = cost

    # Drop cost information from 'best_candidate'
    best_candidate = {key: expr for key, (expr, _) in best_candidate.items()}

    # Materialise any symbolic (composite) dats
    composite_dats = frozenset.union(*map(expr_visitors.collect_composite_dats, best_candidate.values()))
    replace_map = {
        comp_dat: expr_visitors.materialize_composite_dat(comp_dat)
        for comp_dat in composite_dats
    }
    best_candidate = {
        key: expr_visitors.replace(expr, replace_map)
        for key, expr in best_candidate.items()
    }

    # Lastly propagate the materialised indirections back through the instruction tree
    return concretize_materialized_indirections(insn, best_candidate)



def collect_candidate_indirections(insn: insn_types.Instruction, /, *, compress: bool) -> idict:
    return _collect_candidate_indirections(insn, compress=compress, loop_indices=())


@functools.singledispatch
def _collect_candidate_indirections(obj: Any, /, **kwargs) -> idict:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_collect_candidate_indirections.register(insn_types.NullInstruction)
@_collect_candidate_indirections.register(insn_types.Exscan)  # assume we are fine
def _(null: insn_types.InstructionList, /, **kwargs) -> idict:
    return idict()


@_collect_candidate_indirections.register(insn_types.InstructionList)
def _(insn_list: insn_types.InstructionList, /, **kwargs) -> idict:
    return utils.merge_dicts(
        (_collect_candidate_indirections(insn, **kwargs) for insn in insn_list),
    )


@_collect_candidate_indirections.register(insn_types.Loop)
def _(loop: insn_types.Loop, /, *, compress: bool, loop_indices: tuple[LoopIndex, ...]) -> idict:
    loop_indices_ = loop_indices + (loop.index,)
    return utils.merge_dicts(
        (
            _collect_candidate_indirections(stmt, compress=compress, loop_indices=loop_indices_)
            for stmt in loop.statements
        ),
    )


@_collect_candidate_indirections.register(insn_types.NonEmptyTerminal)
def _(terminal: insn_types.NonEmptyTerminal, /, *, loop_indices: tuple[LoopIndex, ...], compress: bool) -> idict:
    candidates = {}
    for i, arg in enumerate(terminal.arguments):
        per_arg_candidates = expr_visitors.collect_tensor_candidate_indirections(
            arg, axis_trees=terminal.axis_trees, loop_indices=loop_indices, compress=compress
        )
        for arg_key, value in per_arg_candidates.items():
            candidates[(terminal, i, arg_key)] = value
    return idict(candidates)


@functools.singledispatch
def concretize_materialized_indirections(obj, layouts) -> insn_types.Instruction:
    raise TypeError


@concretize_materialized_indirections.register(insn_types.InstructionList)
def _(insn_list: insn_types.InstructionList, /, layouts: Mapping[Any, Any]) -> insn_types.InstructionList:
    return maybe_enlist(concretize_materialized_indirections(insn, layouts) for insn in insn_list)


@concretize_materialized_indirections.register(insn_types.Loop)
def _(loop: insn_types.Loop, /, layouts: Mapping[Any, Any]) -> insn_types.Loop:
    return loop.__record_init__(statements=tuple(concretize_materialized_indirections(stmt, layouts) for stmt in loop.statements))


@concretize_materialized_indirections.register(insn_types.StandaloneCalledFunction)
def _(func: insn_types.StandaloneCalledFunction, /, layouts: Mapping[Any, Any]) -> insn_types.StandaloneCalledFunction:
    return func


@concretize_materialized_indirections.register(insn_types.NonEmptyArrayAssignment)
def _(assignment: insn_types.NonEmptyArrayAssignment, /, layouts: Mapping[Any, Any]) -> insn_types.ConcretizedNonEmptyArrayAssignment:
    assignee, expression = (
        expr_visitors.concretize_materialized_tensor_indirections(arg, layouts, (assignment, i))
        for i, arg in enumerate(assignment.arguments)
    )
    return insn_types.ConcretizedNonEmptyArrayAssignment(
        assignee, expression, assignment.assignment_type, assignment.axis_trees, comm=assignment.comm
    )


# does this live here?
class Renamer:
    def __init__(self):
        self._store = {}
        self._counter_by_type = collections.defaultdict(itertools.count)

    def __getitem__(self, key):
        return self._store[key]

    def add(self, obj: Any):
        try:
            return self._store[obj]
        except KeyError:
            index = next(self._counter_by_type[type(obj)])
            label = f"{type(obj).__name__}_{index}"
            return self._store.setdefault(obj, label)


class DiskCacheKeyGetter(NodeVisitor):

    def __init__(self):
        self._renamer = Renamer()
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: insn_types.Instruction) -> Hashable:
        return super().process(obj)


    @process.register(insn_types.InstructionList)
    @process.register(insn_types.NullInstruction)
    @NodeVisitor.postorder
    def _(self, insn: insn_types.Instruction, *visited: Hashable) -> Hashable:
        return (type(insn), *visited)

    @process.register(insn_types.Loop)
    def _(self, loop: insn_types.Loop) -> Hashable:
        from pyop3.tree.axis_tree.visitors import (
            get_disk_cache_key as get_axis_tree_disk_cache_key
        )

        self._renamer.add(loop.index)
        return (
            type(loop),
            get_axis_tree_disk_cache_key(loop.index.iterset, self._renamer),
            *(self(stmt) for stmt in loop.statements),
        )

    @process.register(insn_types.StandaloneCalledFunction)
    def _(self, func: insn_types.StandaloneCalledFunction) -> Hashable:
        from pyop3.expr.visitors import get_disk_cache_key as get_expr_disk_cache_key

        return (
            type(func),
            func.function,
            *(get_expr_disk_cache_key(arg, self._renamer) for arg in func.arguments),
        )

    # TODO: Could have a nice visiter that checks fields (except where hash=False)
    @process.register(insn_types.ConcretizedNonEmptyArrayAssignment)
    def _(self, assignment: insn_types.ConcretizedNonEmptyArrayAssignment, /) -> Hashable:
        from pyop3.tree.axis_tree.visitors import get_disk_cache_key as get_axis_tree_disk_cache_key
        from pyop3.expr.visitors import get_disk_cache_key as get_expr_disk_cache_key

        return (
            type(assignment),
            get_expr_disk_cache_key(assignment.assignee, self._renamer),
            get_expr_disk_cache_key(assignment.expression, self._renamer),
            assignment.assignment_type,
            tuple(get_axis_tree_disk_cache_key(tree, self._renamer) for tree in assignment.axis_trees),
        )


def get_disk_cache_key(insn: insn_types.Instruction) -> Hashable:
    return DiskCacheKeyGetter()(insn)


class BufferCollector(NodeCollector):

    def __init__(self):
        from pyop3.expr.visitors import BufferCollector as ExprBufferCollector
        from pyop3.tree.axis_tree.visitors import BufferCollector as TreeBufferCollector

        expr_collector = ExprBufferCollector()
        tree_collector = TreeBufferCollector()
        expr_collector.tree_collector = tree_collector
        tree_collector.expr_collector = expr_collector

        self._expr_collector = expr_collector
        self._tree_collector = tree_collector
        super().__init__()

    @functools.singledispatchmethod
    def process(self, obj: Any) -> OrderedFrozenSet:
        return super().process(obj)

    @process.register(insn_types.InstructionList)
    @NodeCollector.postorder
    def _(self, insn_list: insn_types.InstructionList, visited, /) -> OrderedFrozenSet:
        breakpoint()
        return utils.reduce("|", visited.values(), OrderedFrozenSet())

    @process.register(insn_types.NullInstruction)
    def _(self, insn: insn_types.NullInstruction, /) -> OrderedFrozenSet:
        return OrderedFrozenSet()

    @process.register(insn_types.Loop)
    @NodeCollector.postorder
    def _(self, insn: insn_types.Loop, visited, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(
            self._tree_collector(insn.index.iterset),
            *visited["statements"],
        )

    @process.register(insn_types.StandaloneCalledFunction)
    def _(self, func: insn_types.StandaloneCalledFunction, /) -> OrderedFrozenSet:
        return OrderedFrozenSet().union(
            *(self._expr_collector(arg) for arg in func.arguments)
        )

    @process.register(insn_types.ConcretizedNonEmptyArrayAssignment)
    def _(self, assignment: insn_types.ConcretizedNonEmptyArrayAssignment, /) -> Hashable:
        return (
            self._expr_collector(assignment.assignee)
            | self._expr_collector(assignment.expression)
            | utils.reduce("|", map(self._tree_collector, assignment.axis_trees))
        )


def collect_buffers(insn: insn_types.Instruction) -> OrderedFrozenSet:
    return BufferCollector()(insn)


class LiteralInserter(NodeTransformer):

    @functools.singledispatchmethod
    def process(self, obj: Any) -> insn_types.Instruction:
        return super().process(obj)

    @process.register(insn_types.Loop)
    @process.register(insn_types.StandaloneCalledFunction)
    def _(self, insn: insn_types.Instruction) -> insn_types.Instruction:
        return self.reuse_if_untouched(insn)

    @process.register(insn_types.NonEmptyArrayAssignment)
    def _(self, assignment: insn_types.NonEmptyArrayAssignment, /) -> insn_types.NonEmptyArrayAssignment:
        # NOTE: This is not robust to if we have expressions that are not just ints, or
        # if the mat is on the rhs
        if (
            isinstance(assignment.assignee, MatPetscMatBufferExpression)
            and isinstance(assignment.expression, numbers.Number)
        ):
            # If we have an expression like
            #
            #     mat[f(p), f(p)] <- 666
            #
            # then we have to convert `666` into an appropriately sized temporary
            # for Mat{Get,Set}Values to work.
            row_axis_tree, column_axis_tree = assignment.axis_trees
            nrows = row_axis_tree.local_max_size
            ncols = column_axis_tree.local_max_size
            expr_data = np.full((nrows, ncols), assignment.expression, dtype=assignment.assignee.buffer.buffer.dtype)

            new_buffer = BufferRef(ArrayBuffer(expr_data, constant=True))
            new_expression = MatArrayBufferExpression(new_buffer, idict(), idict())
            return assignment.__record_init__(_expression=new_expression)
        else:
            return assignment


def insert_literals(insn: insn_types.Instruction) -> insn_types.Instruction:
    return LiteralInserter()(insn)
