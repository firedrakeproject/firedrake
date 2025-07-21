from __future__ import annotations

import abc
import collections
import functools
from itertools import zip_longest
import numbers
import operator
from collections.abc import Iterable, Mapping
from os import access
from typing import Any, Union

import numpy as np
from petsc4py import PETSc
from pyop3.tensor.dat import BufferExpression
from pyop3.sf import local_sf
from pyrsistent import pmap, PMap
from immutabledict import immutabledict

from pyop3 import utils
from pyop3.tensor import Scalar, Dat, Tensor, Mat, NonlinearDatBufferExpression, LinearDatBufferExpression, NonlinearMatBufferExpression, LinearMatBufferExpression
from pyop3.axtree import Axis, AxisTree, ContextFree, ContextSensitive, ContextMismatchException, ContextAware
from pyop3.axtree.tree import UnaryOperator, BinaryOperator, AxisVar, IndexedAxisTree, TernaryOperator, merge_axis_trees2, prune_zero_sized_branches, NaN
from pyop3.buffer import AbstractBuffer, BufferRef, PetscMatBuffer, ArrayBuffer, NullBuffer, AllocatedPetscMatBuffer
from pyop3.dtypes import IntType
from pyop3.itree import Map, TabulatedMapComponent, collect_loop_contexts
from pyop3.itree.tree import LoopIndex, LoopIndexVar, Slice, AffineSliceComponent, IndexTree
from pyop3.itree.parse import _as_context_free_indices
from pyop3.expr_visitors import (
    # collect_tensor_shape,
    get_shape,
    replace as replace_expression,
    replace_terminals,
    collect_composite_dats,
    CompositeDat,
    materialize_composite_dat,
    collect_candidate_indirections,
    concretize_layouts as concretize_expression_layouts,
    extract_axes,
    restrict_to_context as restrict_expression_to_context,
    collect_candidate_indirections as collect_expression_candidate_indirections,
    collect_tensor_candidate_indirections,
    concretize_materialized_tensor_indirections,
)
from pyop3.lang import (
    INC,
    READ,
    RW,
    WRITE,
    AbstractAssignment,
    ArrayAssignment,
    ConcretizedNonEmptyArrayAssignment,
    NonEmptyTerminal,
    StandaloneCalledFunction,
    FunctionArgument,
    # NonEmptyPetscMatAssignment,
    NullInstruction,
    ArrayAssignment,
    AssignmentType,
    CalledFunction,
    NonEmptyArrayAssignment,
    DummyKernelArgument,
    Instruction,
    Loop,
    InstructionList,
    # PetscMatAssignment,
    ArrayAccessType,
    Terminal,
    enlist,
    maybe_enlist,
    non_null,
    filter_null,
)
from pyop3.utils import UniqueNameGenerator, just_one, single_valued, OrderedSet, merge_dicts, expand_collection_of_iterables, strictly_all

import pyop3.extras.debug


# NOTE: A sensible pattern is to have a public and private (rec) implementations of a
# transformation. Then the outer one can also drop extra instruction lists.


# GET RID OF THIS
# TODO Is this generic for other parsers/transformers? Esp. lower.py
class Transformer(abc.ABC):
    @abc.abstractmethod
    def apply(self, expr):
        pass


# NOTE: This is a bad name for this transformation. 'expand_multi_component_loops'?
def expand_loop_contexts(insn: Instruction, /) -> InstructionList:
    """
    This function also drops zero-sized loops.
    """
    return _expand_loop_contexts_rec(insn, loop_context_acc=pmap())


@functools.singledispatch
def _expand_loop_contexts_rec(obj: Any, /, *, loop_context_acc) -> InstructionList:
    raise TypeError


@_expand_loop_contexts_rec.register(InstructionList)
def _(insn_list: InstructionList, /, **kwargs) -> Instruction:
    return maybe_enlist([_expand_loop_contexts_rec(insn, **kwargs) for insn in insn_list])


@_expand_loop_contexts_rec.register(Loop)
def _(loop: Loop, /, *, loop_context_acc) -> Loop | InstructionList:
    expanded_loops = []
    for leaf_path in loop.index.iterset.leaf_paths:
        paths = tuple(
            target_acc[leaf_path][0]
            for target_acc in loop.index.iterset.targets_acc
        )
        loop_context = {loop.index.id: paths}

        restricted_loop_index = just_one(_as_context_free_indices(loop.index, loop_context))

        # skip empty loops
        if restricted_loop_index.iterset.size == 0:
            continue

        loop_context_acc_ = loop_context_acc | loop_context
        expanded_loop = type(loop)(
            restricted_loop_index,
            [
                _expand_loop_contexts_rec(stmt, loop_context_acc=loop_context_acc_)
                for stmt in loop.statements
            ]
        )
        expanded_loops.append(expanded_loop)
    return maybe_enlist(expanded_loops)


@_expand_loop_contexts_rec.register(CalledFunction)
def _(func: CalledFunction, /, *, loop_context_acc) -> CalledFunction:
    return CalledFunction(
        func.function,
        [arg.with_context(loop_context_acc) for arg in func.arguments],
    )


@_expand_loop_contexts_rec.register(ArrayAssignment)
def _(assignment: ArrayAssignment, /, *, loop_context_acc) -> ArrayAssignment:
    assignee = restrict_expression_to_context(assignment.assignee, loop_context_acc)
    expression = restrict_expression_to_context(assignment.expression, loop_context_acc)
    return ArrayAssignment(assignee, expression, assignment.assignment_type)


class ImplicitPackUnpackExpander(Transformer):
    def __init__(self):
        self._name_generator = UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    @_apply.register(NullInstruction)
    def _(self, insn, /):
        return insn

    # TODO Can I provide a generic "operands" thing? Put in the parent class?
    @_apply.register(Loop)
    def _(self, loop: Loop) -> Loop:
        new_statements = [s for stmt in loop.statements for s in enlist(self._apply(stmt))]
        return loop.__record_init__(statements=new_statements)

    @_apply.register
    def _(self, insn_list: InstructionList):
        return type(insn_list)([insn_ for insn in insn_list for insn_ in enlist(self._apply(insn))])

    # # TODO: Should be the same as Assignment
    # @_apply.register
    # def _(self, assignment: PetscMatInstruction):
    #     # FIXME: Probably will not work for things like mat[x, y].assign(dat[z])
    #     # where the expression is indexed.
    #     return (assignment,)

    @_apply.register
    def _(self, assignment: ArrayAssignment):
        # I think this is fine...
        return assignment

        # # same as for CalledFunction
        # gathers = []
        # # NOTE: scatters are executed in LIFO order
        # scatters = []
        # arguments = []
        #
        # # lazy coding, tidy up
        # if isinstance(assignment, ReplaceAssignment):
        #     access = WRITE
        # else:
        #     assert isinstance(assignment, AddAssignment)
        #     access = INC
        # for arg, intent in [
        #     (assignment.assignee, access),
        #     (assignment.expression, READ),
        # ]:
        #     if isinstance(arg, numbers.Number):
        #         arguments.append(arg)
        #         continue
        #
        #     # emit function calls for PetscMat
        #     if isinstance(arg, Mat):
        #         axes = AxisTree(arg.axes.node_map)
        #         new_arg = Dat(
        #             axes,
        #             data=NullBuffer(arg.dtype),  # does this need a size?
        #             prefix="t",
        #         )
        #
        #         if intent == READ:
        #             gathers.append(PetscMatLoad(arg, new_arg))
        #         elif intent == WRITE:
        #             scatters.insert(0, PetscMatStore(arg, new_arg))
        #         elif intent == RW:
        #             gathers.append(PetscMatLoad(arg, new_arg))
        #             scatters.insert(0, PetscMatStore(arg, new_arg))
        #         else:
        #             assert intent == INC
        #             scatters.insert(0, PetscMatAdd(arg, new_arg))
        #
        #         arguments.append(new_arg)
        #     else:
        #         arguments.append(arg)
        #
        # return maybe_enlist((*gathers, assignment.with_arguments(arguments), *scatters))

    @_apply.register
    def _(self, terminal: CalledFunction):
        gathers = []
        # NOTE: scatters are executed in LIFO order
        scatters = []
        arguments = []
        for (arg, intent), shape in zip(
            terminal.function_arguments, terminal.argument_shapes, strict=True
        ):
            if isinstance(arg, DummyKernelArgument):
                arguments.append(arg)
                continue

            # emit pack/unpack instructions
            if _requires_pack_unpack(arg):
                # TODO: Make generic across Array types
                if isinstance(arg, Dat):
                    temporary = Dat.null(arg.axes.materialize(), dtype=arg.dtype, prefix="t")
                else:
                    assert isinstance(arg, Mat)
                    temporary = Mat.null(arg.raxes.materialize(), arg.caxes.materialize(), dtype=arg.dtype, prefix="t")

                if intent == READ:
                    gathers.append(ArrayAssignment(temporary, arg, "write"))
                elif intent == WRITE:
                    # This is currently necessary because some local kernels
                    # (interpolation) actually increment values instead of setting
                    # them directly. This should ideally be addressed.
                    gathers.append(ArrayAssignment(temporary, 0, "write"))
                    scatters.insert(0, ArrayAssignment(arg, temporary, "write"))
                elif intent == RW:
                    gathers.append(ArrayAssignment(temporary, arg, "write"))
                    scatters.insert(0, ArrayAssignment(arg, temporary, "write"))
                else:
                    assert intent == INC
                    gathers.append(ArrayAssignment(temporary, 0, "write"))
                    scatters.insert(0, ArrayAssignment(arg, temporary, "inc"))

                function_arg = LinearDatBufferExpression(BufferRef(temporary.buffer), 0, temporary.shape, temporary.loop_axes)
            else:
                if arg.buffer.is_nested:
                    raise NotImplementedError("Assume cannot have nest indices here")
                function_arg = LinearDatBufferExpression(BufferRef(arg.buffer), 0, arg.shape, arg.loop_axes)
            arguments.append(function_arg)

        return maybe_enlist((*gathers, StandaloneCalledFunction(terminal.function, arguments), *scatters))


# class ExprMarker


# TODO check this docstring renders correctly
def expand_implicit_pack_unpack(expr: Instruction):
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
def _requires_pack_unpack(arg: FunctionArgument) -> bool:
    raise TypeError


@_requires_pack_unpack.register(Scalar)
def _(scalar: Scalar) -> bool:
    return False


@_requires_pack_unpack.register(Dat)
def _(dat: Dat) -> bool:
    # This is overly restrictive since we could pass something contiguous like
    # dat[i0, :] directly to a local kernel
    return not (isinstance(dat.buffer, AbstractBuffer) and _layouts_match(dat.axes))


@_requires_pack_unpack.register(Mat)
def _(mat: Mat) -> bool:
    return not (not isinstance(mat.buffer, PetscMatBuffer) and _layouts_match(mat.raxes) and _layouts_match(mat.caxes))


def _layouts_match(axis_tree) -> bool:
    return axis_tree.leaf_subst_layouts == axis_tree.unindexed.leaf_subst_layouts


@functools.singledispatch
def expand_assignments(obj: Any, /) -> InstructionList:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@expand_assignments.register(InstructionList)
def _(insn_list: InstructionList, /) -> InstructionList:
    return maybe_enlist((expand_assignments(insn) for insn in insn_list))


@expand_assignments.register(Loop)
def _(loop: Loop, /) -> Loop:
    return Loop(
        loop.index,
        [
            stmt_ for stmt in loop.statements for stmt_ in enlist(expand_assignments(stmt))
        ],
    )


@expand_assignments.register(StandaloneCalledFunction)
# @expand_assignments.register(PetscMatAssignment)
@expand_assignments.register(NullInstruction)
def _(func: StandaloneCalledFunction, /) -> StandaloneCalledFunction:
    return func


@expand_assignments.register(ArrayAssignment)
def _(assignment: ArrayAssignment, /) -> InstructionList:
    # NOTE: This is incorrect, we only include this because if we have a 'basic' matrix assignment
    # like
    #
    #     mat[f(p), f(p)] <- t0
    #
    # we don't want to expand it into
    #
    #     t1 <- t0
    #     mat[f(p), f(p)] <- t1
    # if assignment.is_mat_access:
    #     raise NotImplementedError("think")
    #     return InstructionList([assignment])

    bare_expression, extra_input_insns = _expand_reshapes(
        assignment.expression, ArrayAccessType.READ
    )

    if assignment.assignment_type == AssignmentType.WRITE:
        assignee_access_type = ArrayAccessType.WRITE
    else:
        assert assignment.assignment_type == AssignmentType.INC
        assignee_access_type = ArrayAccessType.INC

    bare_assignee, extra_output_insns = _expand_reshapes(
        assignment.assignee, assignee_access_type
    )

    if bare_assignee == assignment.assignee:
        bare_assignment = ArrayAssignment(bare_assignee, bare_expression, assignment.assignment_type)
    else:
        bare_assignment = ArrayAssignment(bare_assignee, bare_expression, "write")

    return maybe_enlist((*extra_input_insns, bare_assignment, *extra_output_insns))


# TODO: better word than "mode"? And use an enum.
@functools.singledispatch
def _expand_reshapes(expr: Any, /, mode):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@_expand_reshapes.register
def _(op: UnaryOperator, /, access_type):
    bare_a, a_insns = _expand_reshapes(op.a, access_type)
    return (type(op)(bare_a), a_insns)


@_expand_reshapes.register
def _(op: BinaryOperator, /, access_type):
    bare_a, a_insns = _expand_reshapes(op.a, access_type)
    bare_b, b_insns = _expand_reshapes(op.b, access_type)
    return (type(op)(bare_a, bare_b), a_insns + b_insns)


@_expand_reshapes.register
def _(op: TernaryOperator, /, access_type):
    bare_a, a_insns = _expand_reshapes(op.a, access_type)
    bare_b, b_insns = _expand_reshapes(op.b, access_type)
    bare_c, c_insns = _expand_reshapes(op.c, access_type)
    return (type(op)(bare_a, bare_b, bare_c), a_insns + b_insns + c_insns)


@_expand_reshapes.register(numbers.Number)
@_expand_reshapes.register(AxisVar)
@_expand_reshapes.register(LoopIndexVar)
@_expand_reshapes.register(BufferExpression)
@_expand_reshapes.register(NaN)
def _(var, /, access_type):
    return (var, ())


# TODO: Add intermediate type here to assert that there is no longer a parent attr
@_expand_reshapes.register(Tensor)
def _(array: Tensor, /, access_type):
    if array.parent:
        # .materialize?
        if isinstance(array, Dat):
            temp_initial = Dat.null(
                AxisTree(array.parent.axes.node_map),
                dtype=array.dtype,
                prefix="t"
            )
            temp_reshaped = temp_initial.with_axes(array.axes)
        else:
            assert isinstance(array, Mat)
            raxes = AxisTree(array.parent.raxes.node_map)
            caxes = AxisTree(array.parent.caxes.node_map)
            temp_initial = Mat.null(raxes, caxes, dtype=array.dtype, prefix="t")
            temp_reshaped = temp_initial.with_axes(array.raxes, array.caxes)

        transformed_dat, extra_insns = _expand_reshapes(array.parent, access_type)

        if extra_insns:
            raise NotImplementedError("Pretty sure this doesn't work as is")

        if access_type == ArrayAccessType.READ:
            assignment = ArrayAssignment(temp_initial, transformed_dat, "write")
        elif access_type == ArrayAccessType.WRITE:
            assignment = ArrayAssignment(transformed_dat, temp_initial, "write")
        else:
            assert access_type == ArrayAccessType.INC
            assignment = ArrayAssignment(transformed_dat, temp_initial, "inc")

        return (temp_reshaped, extra_insns + (assignment,))
    else:
        return (array, ())


@functools.singledispatch
def concretize_layouts(obj: Any, /) -> Instruction:
    """Lock in the layout expressions that data arguments are accessed with.

    For example this converts Dats to DatArrayBufferExpressions that cannot
    be indexed further.

    This function also trims expressions to remove any zero-sized bits.

    """
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@concretize_layouts.register(NullInstruction)
def _(null: NullInstruction, /) -> NullInstruction:
    return null


@concretize_layouts.register(InstructionList)
def _(insn_list: InstructionList, /) -> Instruction:
    return maybe_enlist(
        filter(non_null, (map(concretize_layouts, insn_list)))
    )


@concretize_layouts.register(Loop)
def _(loop: Loop, /) -> Loop | NullInstruction:
    statements = tuple(filter_null(map(concretize_layouts, loop.statements)))
    return loop.__record_init__(statements=statements) if statements else NullInstruction()


@concretize_layouts.register(StandaloneCalledFunction)
def _(func: StandaloneCalledFunction, /) -> StandaloneCalledFunction:
    return func


@concretize_layouts.register(ArrayAssignment)
def _(assignment: ArrayAssignment, /) -> NonEmptyArrayAssignment | NullInstruction:
    # Determine the overall shape of the assignment by merging the shapes of the
    # arguments. This allows for assignments with mismatching indices like:
    #
    #     dat1[i, j] = dat2[j]
    axis_trees = []
    axis_trees_per_arg = tuple(
        trees
        for trees in map(get_shape, assignment.arguments)
        if trees is not None
    )

    # We can get a mismatch here if we are assigning a scalar (single tree
    # shape) to a matrix (double tree shape). We should probably be stricter
    # here (e.g. by asserting it has to be a scalar).
    for arg_axis_trees in zip_longest(*axis_trees_per_arg):
        merged_axis_tree = merge_axis_trees2(arg_axis_trees)

        # drop zero-sized bits
        pruned_axis_tree = merged_axis_tree.prune()

        if not pruned_axis_tree:
            # the assignment is zero-sized
            return NullInstruction()

        axis_trees.append(pruned_axis_tree)
    axis_trees = tuple(axis_trees)

    assignee = concretize_expression_layouts(assignment.assignee, axis_trees)
    expression = concretize_expression_layouts(assignment.expression, axis_trees)

    return NonEmptyArrayAssignment(assignee, expression, axis_trees, assignment.assignment_type)


@PETSc.Log.EventDecorator()
def materialize_indirections(insn: Instruction, *, compress: bool = False) -> Instruction:
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

    # Optimise by dropping any immediately bad candidates. We do this by dropping
    # any candidates whose cost (per-arg) is greater than the current best candidate.
    expr_candidates = {
        arg_id: tuple(
            (arg_candidate, cost)
            for arg_candidate, cost in arg_candidates
            if cost <= max_cost
        )
        for arg_id, arg_candidates in expr_candidates.items()
    }

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
    for shared_candidate in expand_collection_of_iterables(expr_candidates):
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
    composite_dats = frozenset.union(*map(collect_composite_dats, best_candidate.values()))
    replace_map = {
        comp_dat: materialize_composite_dat(comp_dat)
        for comp_dat in composite_dats
    }
    best_candidate = {
        key: replace_expression(expr, replace_map)
        for key, expr in best_candidate.items()
    }

    # Lastly propagate the materialised indirections back through the instruction tree
    return concretize_materialized_indirections(insn, best_candidate)



def collect_candidate_indirections(insn: Instruction, /, *, compress: bool) -> immutabledict:
    return _collect_candidate_indirections(insn, compress=compress, loop_indices=())


@functools.singledispatch
def _collect_candidate_indirections(obj: Any, /, **kwargs) -> immutabledict:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@_collect_candidate_indirections.register(NullInstruction)
def _(null: InstructionList, /, **kwargs) -> immutabledict:
    return immutabledict()


@_collect_candidate_indirections.register(InstructionList)
def _(insn_list: InstructionList, /, **kwargs) -> immutabledict:
    return merge_dicts(
        (_collect_candidate_indirections(insn, **kwargs) for insn in insn_list),
    )


@_collect_candidate_indirections.register(Loop)
def _(loop: Loop, /, *, compress: bool, loop_indices: tuple[LoopIndex, ...]) -> immutabledict:
    loop_indices_ = loop_indices + (loop.index,)
    return merge_dicts(
        (
            _collect_candidate_indirections(stmt, compress=compress, loop_indices=loop_indices_)
            for stmt in loop.statements
        ),
    )


@_collect_candidate_indirections.register(NonEmptyTerminal)
def _(terminal: NonEmptyTerminal, /, *, loop_indices: tuple[LoopIndex, ...], compress: bool) -> immutabledict:
    candidates = {}
    for i, arg in enumerate(terminal.arguments):
        per_arg_candidates = collect_tensor_candidate_indirections(
            arg, axis_trees=terminal.axis_trees, loop_indices=loop_indices, compress=compress
        )
        for arg_key, value in per_arg_candidates.items():
            candidates[(terminal, i, arg_key)] = value
    return immutabledict(candidates)


@functools.singledispatch
def concretize_materialized_indirections(obj, layouts) -> Instruction:
    raise TypeError


@concretize_materialized_indirections.register(InstructionList)
def _(insn_list: InstructionList, /, layouts: Mapping[Any, Any]) -> InstructionList:
    return maybe_enlist(concretize_materialized_indirections(insn, layouts) for insn in insn_list)


@concretize_materialized_indirections.register(Loop)
def _(loop: Loop, /, layouts: Mapping[Any, Any]) -> Loop:
    return loop.__record_init__(statements=tuple(concretize_materialized_indirections(stmt, layouts) for stmt in loop.statements))


@concretize_materialized_indirections.register(StandaloneCalledFunction)
def _(func: StandaloneCalledFunction, /, layouts: Mapping[Any, Any]) -> StandaloneCalledFunction:
    return func


@concretize_materialized_indirections.register(NonEmptyArrayAssignment)
def _(assignment: NonEmptyArrayAssignment, /, layouts: Mapping[Any, Any]) -> ConcretizedNonEmptyArrayAssignment:
    assignee, expression = (
        concretize_materialized_tensor_indirections(arg, layouts, (assignment, i))
        for i, arg in enumerate(assignment.arguments)
    )
    return ConcretizedNonEmptyArrayAssignment(
        assignee, expression, assignment.assignment_type, assignment.axis_trees
    )
