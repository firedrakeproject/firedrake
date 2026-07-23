from __future__ import annotations

import abc
import collections
import functools
import itertools
import numbers
from collections.abc import Iterable, Mapping
from os import access
from typing import Any, Hashable

import numpy as np
from petsc4py import PETSc
from immutabledict import immutabledict as idict

from pyop3.cache import memory_cache
import pyop3.compile
import pyop3.axis_tree
import pyop3.expr
import pyop3.expr.visitors
from pyop3.expr.buffer import MatArrayBufferExpression, ScalarBufferExpression
from pyop3.expr.tensor import mat
from pyop3.expr.tensor.dat import AggregateDat
from pyop3.expr.tensor.mat import AggregateMat
from pyop3 import utils

from pyop3.constants import INC, READ, RW, WRITE
from pyop3.node import NodeTransformer, NodeVisitor, NodeCollector, postorder
from pyop3.expr.tensor.base import OutOfPlaceCallableTensorTransform, ReshapeTensorTransform, TensorTransform
from pyop3.expr import Scalar, Dat, Tensor, Mat, LinearDatBufferExpression, BufferExpression, MatPetscMatBufferExpression
from pyop3.axis_tree import AxisTree, AxisForest
from pyop3.axis_tree.tree import UNIT_AXIS_TREE, merge_axis_trees
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, NullBuffer, ArrayBuffer

from pyop3.index_tree.tree import LoopIndex
from pyop3.index_tree.parse import _as_context_free_indices
import pyop3.insn
from pyop3.insn.base import (
    AssignmentType,
    ArrayAccessType,
    enlist,
    maybe_enlist,
    non_null,
    filter_null,
)
from pyop3.collections import OrderedFrozenSet


class InstructionTransformer(NodeTransformer):

    @functools.singledispatchmethod
    def process(self, insn: pyop3.insn.Instruction, /, **kwargs) -> pyop3.insn.Instruction:
        return super().process(insn, **kwargs)

    # Instruction lists have a common pattern
    @process.register(pyop3.insn.InstructionList)
    @postorder
    def _(self, insn_list: pyop3.insn.InstructionList, /, *insns, **kwargs) -> pyop3.insn.Instruction:
        raise NotImplementedError
        return maybe_enlist(insns)


class LoopContextExpander(InstructionTransformer):

    @functools.singledispatchmethod
    def process(self, insn: pyop3.insn.Instruction, /, **kwargs) -> pyop3.insn.Instruction:
        return super().process(insn, **kwargs)

    @process.register(pyop3.insn.Loop)
    def _(self, loop: pyop3.insn.Loop, /, *, loop_context) -> pyop3.insn.Loop | pyop3.insn.InstructionList:
        expanded_loops = []
        iterset = loop.index.iterset
        for leaf_path in iterset.leaf_paths:
            loop_context_ = {loop.index.id: leaf_path}

            restricted_loop_index = utils.just_one(_as_context_free_indices(loop.index, loop_context_))

            # (not always safe to do)
            # # skip empty loops
            # if restricted_loop_index.iterset.size == 0:
            #     continue

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


    @process.register(pyop3.insn.CalledFunction)
    def _(self, func: pyop3.insn.CalledFunction, /, *, loop_context) -> pyop3.insn.CalledFunction:
        new_arguments = tuple(arg.with_context(loop_context) for arg in func.arguments)
        return func.record_new(_arguments=new_arguments)

    @process.register(pyop3.insn.Assignment)
    def _(self, assignment: pyop3.insn.Assignment, /, *, loop_context) -> pyop3.insn.Assignment:
        assignee = pyop3.expr.visitors.restrict_to_context(assignment.assignee, loop_context)
        expression = pyop3.expr.visitors.restrict_to_context(assignment.expression, loop_context)
        return assignment.record_new(_assignee=assignee, _expression=expression)

    @process.register(pyop3.insn.Exscan)  # for now assume we are fine
    def _(self, insn: pyop3.insn.Instruction, /, **kwargs) -> pyop3.insn.Instruction:
        return self.reuse_if_untouched(insn)


# NOTE: This is a bad name for this transformation. 'expand_multi_component_loops'?
def expand_loop_contexts(insn: pyop3.insn.Instruction, /) -> pyop3.insn.Instruction:
    return LoopContextExpander()(insn, loop_context=idict())


class ImplicitPackUnpackExpander(NodeTransformer):
    def __init__(self):
        self._name_generator = utils.UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    @_apply.register(pyop3.insn.NullInstruction)
    @_apply.register(pyop3.insn.Exscan)  # assume we are fine
    def _(self, insn, /):
        return insn

    # TODO Can I provide a generic "operands" thing? Put in the parent class?
    @_apply.register(pyop3.insn.Loop)
    def _(self, loop: pyop3.insn.Loop) -> pyop3.insn.Loop:
        new_statements = [s for stmt in loop.statements for s in enlist(self._apply(stmt))]
        return loop.record_new(statements=new_statements)

    @_apply.register
    def _(self, insn_list: pyop3.insn.InstructionList):
        return type(insn_list)([insn_ for insn in insn_list for insn_ in enlist(self._apply(insn))])

    # # TODO: Should be the same as Assignment
    # @_apply.register
    # def _(self, assignment: PetscMatInstruction):
    #     # FIXME: Probably will not work for things like mat[x, y].assign(dat[z])
    #     # where the expression is indexed.
    #     return (assignment,)

    @_apply.register
    def _(self, assignment: pyop3.insn.Assignment):
        # I think this is fine...
        return assignment

    @_apply.register
    def _(self, terminal: pyop3.insn.CalledFunction):
        gathers = []
        # NOTE: scatters are executed in LIFO order
        scatters = []
        arguments = []
        for (arg, intent), shape in zip(
            terminal.function_arguments, terminal.argument_shapes, strict=True
        ):
            # emit pack/unpack instructions
            if _requires_pack_unpack(arg):
                # TODO: Make generic across Array types
                if isinstance(arg, Dat):
                    temporary = Dat.null(arg.axes.materialize().regionless(), dtype=arg.dtype, prefix="t")
                else:
                    assert isinstance(arg, Mat)
                    temporary = Mat.null(arg.row_axes.materialize().regionless(), arg.column_axes.materialize().regionless(), dtype=arg.dtype, prefix="t")

                if intent == READ:
                    gathers.append(pyop3.insn.Assignment(temporary, arg, "write"))
                elif intent == WRITE:
                    scatters.insert(0, pyop3.insn.Assignment(arg, temporary, "write"))
                elif intent == RW:
                    gathers.append(pyop3.insn.Assignment(temporary, arg, "write"))
                    scatters.insert(0, pyop3.insn.Assignment(arg, temporary, "write"))
                else:
                    assert intent == INC
                    gathers.append(pyop3.insn.Assignment(temporary, 0, "write"))
                    scatters.insert(0, pyop3.insn.Assignment(arg, temporary, "inc"))

                function_arg = LinearDatBufferExpression(temporary.buffer, 0)
            else:
                if arg.buffer.is_nested:
                    raise NotImplementedError("Assume cannot have nest indices here")
                function_arg = LinearDatBufferExpression(arg.buffer, 0)
            arguments.append(function_arg)

        return maybe_enlist((*gathers, pyop3.insn.StandaloneCalledFunction(terminal.function, arguments), *scatters))


# TODO check this docstring renders correctly
def expand_implicit_pack_unpack(expr: pyop3.insn.Instruction):
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
def _requires_pack_unpack(arg: pyop3.insn.FunctionArgument) -> bool:
    utils.raise_missing_dispatch_handler(arg)


@_requires_pack_unpack.register(Scalar)
@_requires_pack_unpack.register(pyop3.expr.OpaqueTerminal)
def _(scalar: Scalar) -> bool:
    return False


@_requires_pack_unpack.register(Dat)
def _(dat: Dat) -> bool:
    # This is overly restrictive since we could pass something contiguous like
    # dat[i0, :] directly to a local kernel
    return not (isinstance(dat.buffer, ConcreteBuffer) and _layouts_match(dat.axes) and not has_materialized_temporaries(dat))


@_requires_pack_unpack.register(Mat)
def _(mat: Mat) -> bool:
    return not (not isinstance(mat.buffer, PetscMatBuffer) and _layouts_match(mat.row_axes) and _layouts_match(mat.column_axes) and not has_materialized_temporaries(mat))


@_requires_pack_unpack.register(AggregateDat)
@_requires_pack_unpack.register(AggregateMat)
def _(amat) -> bool:
    return True


def _layouts_match(axes: AxisTreeT) -> bool:
    if isinstance(axes, AxisForest):
        return utils.strictly_all(map(_layouts_match, axes.trees))
    else:
        return axes.leaf_subst_layouts == axes.unindexed.leaf_subst_layouts


@functools.singledispatch
def expand_transforms(obj: Any, /) -> pyop3.insn.InstructionList:
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@expand_transforms.register(pyop3.insn.InstructionList)
def _(insn_list: pyop3.insn.InstructionList, /) -> pyop3.insn.InstructionList:
    return maybe_enlist((expand_transforms(insn) for insn in insn_list))


@expand_transforms.register(pyop3.insn.Loop)
def _(loop: pyop3.insn.Loop, /) -> pyop3.insn.Loop:
    return pyop3.insn.Loop(
        loop.index,
        [
            stmt_ for stmt in loop.statements for stmt_ in enlist(expand_transforms(stmt))
        ],
    )


@expand_transforms.register(pyop3.insn.StandaloneCalledFunction)
# @expand_assignments.register(PetscMatAssignment)
@expand_transforms.register(pyop3.insn.NullInstruction)
@expand_transforms.register(pyop3.insn.Exscan)  # assume we are fine
def _(func: pyop3.insn.StandaloneCalledFunction, /) -> pyop3.insn.StandaloneCalledFunction:
    return func


def _intent_as_access_type(intent):
    if intent == READ:
        return ArrayAccessType.READ
    if intent == WRITE:
        return ArrayAccessType.WRITE
    else:
        assert intent == INC
        return ArrayAccessType.INC



@expand_transforms.register(pyop3.insn.CalledFunction)
def _(called_func: pyop3.insn.CalledFunction, /) -> pyop3.insn.InstructionList:
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
                arg_unpack_insns.insert(0, func_arg.assign(local_tensor))
            elif intent == RW:
                arg_pack_insns.append(local_tensor.assign(func_arg))
                arg_unpack_insns.insert(0, func_arg.assign(local_tensor))
            else:
                assert intent == INC
                arg_pack_insns.append(local_tensor.assign(0))
                arg_unpack_insns.insert(0, func_arg.iassign(local_tensor))

            materialized_arg = LinearDatBufferExpression(local_tensor.buffer, 0)
        elif isinstance(func_arg, pyop3.expr.OpaqueTerminal):
            materialized_arg = func_arg
        else:
            materialized_arg = LinearDatBufferExpression(func_arg.buffer, 0)

        bare_func_args.append(materialized_arg)
        pack_insns.extend(arg_pack_insns)
        unpack_insns.extend(arg_unpack_insns)

    bare_called_func = pyop3.insn.StandaloneCalledFunction(called_func.function, bare_func_args)
    return maybe_enlist((*pack_insns, bare_called_func, *unpack_insns))


@expand_transforms.register(pyop3.insn.Assignment)
def _(assignment: pyop3.insn.Assignment, /) -> pyop3.insn.InstructionList:
    # This function is complete magic and deserves some serious exposition:
    #
    # To begin with, consider the assignment:
    #
    #     x <- y
    #
    # where 'y' is a transformed dat. To generate code for this assignment we
    # need to traverse the hierarchy of transformations and emit something like:
    #
    #     t <- Y
    #     f(t)       -- in-place transform
    #     u <- g(t)  -- out-of-place transform
    #     x <- u     -- original assignment
    #
    # where 'Y' is the global data structure at the top of the transform hierarchy.
    #
    # To make this happen, in this function we 'expand' the expression 'y',
    # giving us back 'u' and the sequence of transformation instructions. Note
    # that here we are expanding the assignment *expression* (as opposed to the
    # assignee 'x') and so the transformation instructions are emitted in order
    # from global to local data structures.
    #
    # Now let's imagine what happens for 'x <- y' where the assignee ('x') is
    # the transformed object. We thus want to generate code like:
    #
    #     t <- y
    #     f(t)       -- in-place transform
    #     u <- g(t)  -- out-of-place transform
    #     X <- u
    #
    # where 'X' is the global data at the top of the transform hierarchy for 'x'.
    # Expanding the assignee will return 't' and the subsequent transformations.
    # Since the transformation here is applied to the assignee the transformation
    # instructions go from local data structures to global ones.
    #
    # Lastly, if we consider incrementing instead of assigning (i.e. 'x += y'),
    # then some changes are needed. We need to generate code like:
    #
    #     t <- y
    #     f(t)       -- in-place transform
    #     u <- g(t)  -- out-of-place transform
    #     X += u
    #
    # To make this work we extract the increment by materialising 'u'.
    bare_expression, expression_insns = pyop3.expr.visitors.expand_transforms(
        assignment.expression, ArrayAccessType.READ
    )

    if assignment.assignment_type == AssignmentType.WRITE:
        access_type = ArrayAccessType.WRITE
    else:
        assert assignment.assignment_type == AssignmentType.INC
        access_type = ArrayAccessType.INC
    bare_assignee, assignee_insns = pyop3.expr.visitors.expand_transforms(
        assignment.assignee, access_type
    )

    assignment_type = assignment.assignment_type
    if assignment_type == AssignmentType.INC and assignee_insns:
        # If we are emitting assignee transformation instruction for an
        # increment assignment then the final instruction must be the
        # increment into the global data structure. This means that we
        # should only write here, not increment.
        assert assignee_insns[-1].assignment_type == AssignmentType.INC
        assignment_type = AssignmentType.WRITE

    # PETSc matrix assignment requires the expression to be a materialised
    # temporary. Note that we expand literals at a later point, which is silly.
    # We should do this together.
    if (
        isinstance(bare_assignee.buffer, PetscMatBuffer)
        and isinstance(bare_expression, Mat)
        and not all(
            isinstance(t, pyop3.axis_tree.AbstractUnindexedAxisTree)
            for t in bare_expression.axis_trees
        )
    ):
        assert not any(
            isinstance(t, pyop3.axis_tree.AxisForest)
            for t in bare_expression.axis_trees
        )
        expression_temp = bare_expression.materialize()
        expression_insns += (expression_temp.assign(bare_expression),)
        bare_expression = expression_temp

    bare_assignment = assignment.record_new(
        _assignee=bare_assignee,
        _expression=bare_expression,
        _assignment_type=assignment_type,
    )
    return maybe_enlist((*expression_insns, bare_assignment, *assignee_insns))


def has_materialized_temporaries(tensor: Tensor) -> bool:
    while tensor.transform:
        if isinstance(tensor.transform, OutOfPlaceTensorTransform):
            return True
        else:
            tensor = tensor.transform.prev
    return False


@functools.singledispatch
def concretize_layouts(obj: Any, /) -> pyop3.insn.Instruction:
    """Lock in the layout expressions that data arguments are accessed with.

    For example this converts Dats to DatArrayBufferExpressions that cannot
    be indexed further.

    This function also trims expressions to remove any zero-sized bits.

    """
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@concretize_layouts.register(pyop3.insn.NullInstruction)
@concretize_layouts.register(pyop3.insn.Exscan)  # assume we are fine
def _(null: pyop3.insn.NullInstruction, /) -> pyop3.insn.NullInstruction:
    return null


@concretize_layouts.register(pyop3.insn.InstructionList)
def _(insn_list: pyop3.insn.InstructionList, /) -> pyop3.insn.Instruction:
    return maybe_enlist(
        filter(non_null, (map(concretize_layouts, insn_list)))
    )


@concretize_layouts.register(pyop3.insn.Loop)
def _(loop: pyop3.insn.Loop, /) -> pyop3.insn.Loop | pyop3.insn.NullInstruction:
    index = loop.index.record_new(iterset=loop.index.iterset.materialize())
    statements = tuple(filter_null(map(concretize_layouts, loop.statements)))
    return loop.record_new(index=index, statements=statements) if statements else pyop3.insn.NullInstruction()


@concretize_layouts.register
def _(func: pyop3.insn.StandaloneCalledFunction, /) -> pyop3.insn.StandaloneCalledFunction:
    return func


@concretize_layouts.register
def _(assignment: pyop3.insn.Assignment, /) -> pyop3.insn.NonEmptyArrayAssignment | pyop3.insn.NullInstruction:
    # The assignee may have an axis forest as its shape, but we can only
    # emit loops for one of them. Try all candidates and hopefully one will match.
    # For matrices there are two shape axes and so we need to try the product
    # of all candidates.
    for axis_trees in itertools.product(*(tree.trees for tree in assignment.shape)):
        try:
            assignee = pyop3.expr.visitors.concretize_layouts(assignment.assignee, axis_trees)
            expression = pyop3.expr.visitors.concretize_layouts(assignment.expression, axis_trees)
        except pyop3.exceptions.IncompatibleAxisTargetException:
            continue
        else:
            shape = tuple(tree.materialize() for tree in axis_trees)
            break
    else:
        raise pyop3.exceptions.IncompatibleAxisTargetException

    return pyop3.insn.NonEmptyArrayAssignment(assignee, expression, shape, assignment.assignment_type, comm=assignment.comm)


# TODO: move into compress_indirections.py
class MaterializedIndirectionsConcretizer(NodeVisitor):

    @functools.singledispatchmethod
    def process(self, obj: ExpressionT, /, *args, **kwargs) -> tuple[tuple[Any, int, int], ...]:
        return super().process(obj, *args, **kwargs)

    @process.register(pyop3.insn.InstructionList)
    def _(self, insn_list: pyop3.insn.InstructionList, /, layouts: Mapping[Any, Any]) -> pyop3.insn.InstructionList:
        return maybe_enlist(self._call(insn, layouts=layouts) for insn in insn_list)


    @process.register(pyop3.insn.Loop)
    def _(self, loop: pyop3.insn.Loop, /, layouts: Mapping[Any, Any]) -> pyop3.insn.Loop:
        return loop.record_new(statements=tuple(self._call(stmt, layouts=layouts) for stmt in loop.statements))


    @process.register(pyop3.insn.StandaloneCalledFunction)
    @process.register(pyop3.insn.Exscan)
    @process.register(pyop3.insn.NullInstruction)
    def _(self, func: pyop3.insn.StandaloneCalledFunction, /, layouts: Mapping[Any, Any]) -> pyop3.insn.StandaloneCalledFunction:
        return func


    @process.register(pyop3.insn.NonEmptyArrayAssignment)
    def _(self, assignment: pyop3.insn.NonEmptyArrayAssignment, /, layouts: Mapping[Any, Any]) -> pyop3.insn.ConcretizedNonEmptyArrayAssignment:
        assignee, expression = (
            pyop3.expr.visitors.concretize_materialized_tensor_indirections(arg, layouts, (self.index, i))
            for i, arg in enumerate(assignment.arguments)
        )
        return pyop3.insn.ConcretizedNonEmptyArrayAssignment(
            assignee, expression, assignment.assignment_type, assignment.axis_trees, comm=assignment.comm
        )


def concretize_materialized_indirections(obj, layouts) -> pyop3.insn.Instruction:
    return MaterializedIndirectionsConcretizer()(obj, layouts=layouts)



class InstructionCacheKeyGetter(NodeVisitor):
    @functools.singledispatchmethod
    def process(self, obj: pyop3.insn.Instruction) -> Hashable:
        return super().process(obj)

    @process.register(pyop3.insn.InstructionList)
    @process.register(pyop3.insn.NullInstruction)
    @postorder
    def _(self, insn: pyop3.insn.Instruction, *visited: Hashable) -> Hashable:
        return (type(insn), *visited)



class LiteralInserter(NodeTransformer):

    @functools.singledispatchmethod
    def process(self, obj: Any) -> pyop3.insn.Instruction:
        return super().process(obj)

    @process.register(pyop3.insn.InstructionList)
    @process.register(pyop3.insn.Loop)
    @process.register(pyop3.insn.Exscan)
    @process.register(pyop3.insn.StandaloneCalledFunction)
    @process.register(pyop3.insn.NullInstruction)
    def _(self, insn: pyop3.insn.Instruction) -> pyop3.insn.Instruction:
        return self.reuse_if_untouched(insn)

    @process.register(pyop3.insn.NonEmptyArrayAssignment)
    def _(self, assignment: pyop3.insn.NonEmptyArrayAssignment, /) -> pyop3.insn.NonEmptyArrayAssignment:
        # NOTE: This is not robust to if we have expressions that are not just ints, or
        # if the mat is on the rhs
        if (
            isinstance(assignment.assignee, MatPetscMatBufferExpression)
            and isinstance(assignment.assignee.buffer.handle, PETSc.Mat)
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
            expr_data = np.full((nrows, ncols), assignment.expression, dtype=assignment.assignee.buffer.dtype)

            new_buffer = ArrayBuffer(expr_data, constant=True)
            new_expression = MatArrayBufferExpression(new_buffer, idict(), idict())
            return assignment.record_new(_expression=new_expression)
        else:
            return assignment


def insert_literals(insn: pyop3.insn.Instruction) -> pyop3.insn.Instruction:
    return LiteralInserter()(insn)


class CompilerOptionsCollector(NodeVisitor):

    @functools.singledispatchmethod
    def process(self, obj: Any, /, *args, **kwargs) -> NoReturn:
        raise TypeError(f"No handler defined for {utils.pretty_type(obj)}")

    @process.register
    @postorder
    def _(self, insn: pyop3.insn.InstructionList, /, visited) -> pyop3.compile.CompilerOptions:
        return sum(visited["instructions"], pyop3.compile.CompilerOptions())

    @process.register
    def _(self, insn: pyop3.insn.TerminalInstruction) -> pyop3.compile.CompilerOptions:
        return insn.compiler_options

    @process.register
    @postorder
    def _(self, insn: pyop3.insn.Loop, /, visited) -> pyop3.compile.CompilerOptions:
        return sum(visited["statements"], pyop3.compile.CompilerOptions())


def collect_compiler_options(insn: pyop3.insn.Instruction) -> pyop3.compile.CompilerOptions:
    return CompilerOptionsCollector()(insn)
