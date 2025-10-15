from __future__ import annotations

import abc
from ctypes import Array
import functools
from itertools import zip_longest
import numbers
from collections.abc import Iterable, Mapping
from re import I
from typing import Any

from petsc4py import PETSc
from immutabledict import immutabledict

import pyop3.expr.base as expr_types
from pyop3 import utils
from pyop3.expr import Scalar, Dat, Tensor, Mat, LinearDatBufferExpression, BufferExpression
from pyop3.expr.tensor.base import InPlaceTensorTransform, OutOfPlaceTensorTransform
from pyop3.tree.axis_tree import AxisTree
from pyop3.tree.axis_tree.tree import merge_axis_trees
from pyop3.buffer import AbstractBuffer, PetscMatBuffer
from pyop3.tree.index_tree.tree import LoopIndex
from pyop3.tree.index_tree.parse import _as_context_free_indices
from pyop3.expr.visitors import (
    get_shape,
    replace as replace_expression,
    collect_composite_dats,
    materialize_composite_dat,
    collect_candidate_indirections,
    concretize_layouts as concretize_expression_layouts,
    restrict_to_context as restrict_expression_to_context,
    collect_candidate_indirections as collect_expression_candidate_indirections,
    collect_tensor_candidate_indirections,
    concretize_materialized_tensor_indirections,
)
from pyop3.insn.base import (
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
    Exscan,
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
    return _expand_loop_contexts_rec(insn, loop_context_acc=immutabledict())


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
    return assignment.__record_init__(_assignee=assignee, _expression=expression)


# for now assume we are fine
@_expand_loop_contexts_rec.register(Exscan)
def _(exscan: Exscan, /, *, loop_context_acc) -> ArrayAssignment:
    return exscan


class ImplicitPackUnpackExpander(Transformer):
    def __init__(self):
        self._name_generator = UniqueNameGenerator()

    def apply(self, expr):
        return self._apply(expr)

    @functools.singledispatchmethod
    def _apply(self, expr: Any):
        raise NotImplementedError(f"No handler provided for {type(expr).__name__}")

    @_apply.register(NullInstruction)
    @_apply.register(Exscan)  # assume we are fine
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
        assert False, "old code"
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
                    temporary = Dat.null(arg.axes.materialize().regionless, dtype=arg.dtype, prefix="t")
                else:
                    assert isinstance(arg, Mat)
                    temporary = Mat.null(arg.raxes.materialize().regionless, arg.caxes.materialize().regionless, dtype=arg.dtype, prefix="t")

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

                function_arg = LinearDatBufferExpression(temporary.buffer, 0)
            else:
                if arg.buffer.is_nested:
                    raise NotImplementedError("Assume cannot have nest indices here")
                function_arg = LinearDatBufferExpression(arg.buffer, 0)
            arguments.append(function_arg)

        breakpoint()  # TODO: reverse the scatters
        return maybe_enlist((*gathers, StandaloneCalledFunction(terminal.function, arguments), *scatters))


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
    return not (isinstance(dat.buffer, AbstractBuffer) and _layouts_match(dat.axes) and not has_materialized_temporaries(dat))


@_requires_pack_unpack.register(Mat)
def _(mat: Mat) -> bool:
    return not (not isinstance(mat.buffer, PetscMatBuffer) and _layouts_match(mat.raxes) and _layouts_match(mat.caxes) and not has_materialized_temporaries(mat))


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
@expand_assignments.register(Exscan)  # assume we are fine
def _(func: StandaloneCalledFunction, /) -> StandaloneCalledFunction:
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
        access_type = _intent_as_access_type(intent)

        bare_func_arg, arg_pack_insns, arg_unpack_insns = _expand_reshapes(func_arg, access_type)
        arg_pack_insns = list(arg_pack_insns)
        arg_unpack_insns = list(arg_unpack_insns)

        # function calls need materialised arrays
        if _requires_pack_unpack(bare_func_arg):
            local_tensor = bare_func_arg.materialize()

            if intent == READ:
                arg_pack_insns.append(local_tensor.assign(bare_func_arg))
            elif intent == WRITE:
                # This is currently necessary because some local kernels
                # (interpolation) actually increment values instead of setting
                # them directly. This should ideally be addressed.
                arg_pack_insns.append(local_tensor.assign(0))
                arg_unpack_insns.insert(0, bare_func_arg.assign(local_tensor))
            elif intent == RW:
                arg_pack_insns.append(local_tensor.assign(bare_func_arg))
                arg_unpack_insns.insert(0, bare_func_arg.assign(local_tensor))
            else:
                assert intent == INC
                arg_pack_insns.append(local_tensor.assign(0))
                arg_unpack_insns.insert(0, bare_func_arg.iassign(local_tensor))

            materialized_arg = LinearDatBufferExpression(local_tensor.buffer, 0)
        else:
            materialized_arg = LinearDatBufferExpression(bare_func_arg.buffer, 0)

        bare_func_args.append(materialized_arg)
        pack_insns.extend(arg_pack_insns)
        unpack_insns.extend(arg_unpack_insns)

    bare_called_func = StandaloneCalledFunction(called_func.function, bare_func_args)
    return maybe_enlist((*pack_insns, bare_called_func, *unpack_insns))


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

    bare_expression, extra_input_insns, _ = _expand_reshapes(
        assignment.expression, ArrayAccessType.READ
    )

    # NOTE: This might have broken things, be careful (30/09/25)
    # if assignment.assignment_type == AssignmentType.WRITE:
    #     assignee_access_type = ArrayAccessType.WRITE
    # else:
    #     assert assignment.assignment_type == AssignmentType.INC
    #     assignee_access_type = ArrayAccessType.INC

    bare_assignee, _, extra_output_insns = _expand_reshapes(
        assignment.assignee, ArrayAccessType.WRITE
    )

    if bare_assignee == assignment.assignee:
        # no extra assignments
        bare_assignment = assignment.__record_init__(_assignee=bare_assignee, _expression=bare_expression)
    else:
        bare_assignment = assignment.__record_init__(_assignee=bare_assignee, _expression=bare_expression, _assignment_type="write")

    return maybe_enlist((*extra_input_insns, bare_assignment, *reversed(extra_output_insns)))


# TODO: better word than "mode"? And use an enum.
@functools.singledispatch
def _expand_reshapes(expr: Any, /, mode):
    raise TypeError(f"No handler provided for {type(expr).__name__}")


@_expand_reshapes.register
def _(op: expr_types.UnaryOperator, /, access_type):
    bare_a, pack_insns, unpack_insns = _expand_reshapes(op.a, access_type)
    return (type(op)(bare_a), pack_insns, unpack_insns)


@_expand_reshapes.register
def _(op: expr_types.BinaryOperator, /, access_type):
    bare_a, a_pack_insns, a_unpack_insns = _expand_reshapes(op.a, access_type)
    bare_b, b_pack_insns, b_unpack_insns = _expand_reshapes(op.b, access_type)
    return (type(op)(bare_a, bare_b), a_pack_insns+b_pack_insns, a_unpack_insns+b_unpack_insns)


@_expand_reshapes.register
def _(op: expr_types.TernaryOperator, /, access_type):
    bare_operands = []
    pack_insns = []
    unpack_insns = []
    for operand in op.operands:
        bare_operand, operand_pack_insns, operand_unpack_insns = _expand_reshapes(operand, access_type)
        bare_operands.append(bare_operand)
        pack_insns.extend(operand_pack_insns)
        unpack_insns.extend(operand_unpack_insns)
    return (type(op)(*bare_operands), tuple(pack_insns), tuple(unpack_insns))


@_expand_reshapes.register(numbers.Number)
@_expand_reshapes.register(expr_types.AxisVar)
@_expand_reshapes.register(expr_types.LoopIndexVar)
@_expand_reshapes.register(BufferExpression)
@_expand_reshapes.register(expr_types.NaN)
def _(var, /, access_type):
    return (var, (), ())


# TODO: Add intermediate type here to assert that there is no longer a parent attr
@_expand_reshapes.register(Tensor)
def _(array: Tensor, /, access_type):
    """
    Example:

    Consider:

        kernel(dat[?])  # INC

    into

        t0 <- 0
        kernel(t0)
        f(t0)  # in-place
        t1 <- g(t0)  # out-of-place
        dat[?] += t1
    """
    if not array.parent:
        return array, (), ()

    pack_insns = []
    unpack_insns = []

    # Mumble, INC accesses are inherently incompatible with transforms because
    # transforms imply R/W accesses.
    #
    # e.g. consider 'kernel' with INC access and a global. The following won't work
    # because the transformation won't only be over the single contribution.
    #
    #   kernel(glob)
    #   f(glob)  # in-place transform
    #
    # vs
    #
    #   kernel(t0)
    #   f(t0)
    #   glob += t0
    #
    # which is safe to do.
    #
    # As a consequence it means that INC accesses with transforms must always be
    # expanded to have temporaries.
    #
    # N.B. I am fairly confident that this is right but I haven't quite got it
    # straight in my head exactly why.
    if access_type == ArrayAccessType.INC:
        transformed_temporary, local_output_tensor, global_tensor = _materialize_untransformed_tensor(array)

        bare_temporary = transformed_temporary.__record_init__(_parent=None)
        pack_insns.insert(0, bare_temporary.zero())

        unpack_insns.append(global_tensor.iassign(local_output_tensor))

        array = transformed_temporary
        access_type = ArrayAccessType.WRITE

    if access_type == ArrayAccessType.READ:
        insns = _expand_transforms_in(array)
        pack_insns.extend(insns)
    else:
        assert access_type == ArrayAccessType.WRITE
        insns = _expand_transforms_out(array)
        unpack_insns = [*insns, *unpack_insns]

    bare_array = array.__record_init__(_parent=None)
    return bare_array, tuple(pack_insns), tuple(unpack_insns)


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
            bare_current_tensor_reshaped = bare_current_tensor.with_axes(bare_parent_tensor.axes.materialize())
            current_pack_insns = (
                bare_current_tensor_reshaped.assign(bare_parent_tensor),
                *current_tensor.parent.transform_in(bare_current_tensor),
            )
        else:
            assert isinstance(current_tensor.parent, OutOfPlaceTensorTransform)
            current_pack_insns = current_tensor.parent.transform_in(bare_parent_tensor, bare_current_tensor)

        pack_insns = (*pack_insns, *current_pack_insns)
        current_tensor = parent_tensor
    return pack_insns


def _expand_transforms_out(tensor: Tensor) -> tuple[Tensor, tuple[Instruction, ...]]:
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
            bare_current_tensor_reshaped = bare_current_tensor.with_axes(bare_parent_tensor.axes.materialize())
            current_unpack_insns = (
                *current_tensor.parent.transform_out(bare_current_tensor),
                bare_parent_tensor.assign(bare_current_tensor_reshaped),
            )
        else:
            assert isinstance(current_tensor.parent, OutOfPlaceTensorTransform)
            current_unpack_insns = current_tensor.parent.transform_out(bare_current_tensor, bare_parent_tensor)

        unpack_insns = (*current_unpack_insns, *unpack_insns)
        current_tensor = parent_tensor
    return unpack_insns


def has_materialized_temporaries(tensor: Tensor) -> bool:
    while tensor.parent:
        if isinstance(tensor.parent, OutOfPlaceTensorTransform):
            return True
        else:
            tensor = tensor.parent.untransformed
    return False


def _materialize_untransformed_tensor(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """
    I.e.

    * given 'T' implying:

        kernel(T, ...)
        ...
        f'(T)        # in-place transform
        U <- g'(T)   # out-of-place transform
        dat += U

    * want to materialise U and return U and dat

    This effectively means we have to look at all the parents and return the top-most, we also
    need to swap out 'parent'

    """
    if tensor.parent:
        new_parent_tensor, root_temp, root = _materialize_untransformed_tensor(tensor.parent.untransformed)
        new_parent = tensor.parent.__record_init__(untransformed=new_parent_tensor)
        return tensor.__record_init__(_parent=new_parent), root_temp, root
    else:
        U = tensor.materialize()
        return U, U, tensor



@functools.singledispatch
def concretize_layouts(obj: Any, /) -> Instruction:
    """Lock in the layout expressions that data arguments are accessed with.

    For example this converts Dats to DatArrayBufferExpressions that cannot
    be indexed further.

    This function also trims expressions to remove any zero-sized bits.

    """
    raise TypeError(f"No handler provided for {type(obj).__name__}")


@concretize_layouts.register(NullInstruction)
@concretize_layouts.register(Exscan)  # assume we are fine
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
    # FIXME: I think actually the assignee should just prescribe this.
    axis_trees_per_arg = (axis_trees_per_arg[0],)
    for arg_axis_trees in zip_longest(*axis_trees_per_arg):
        merged_axis_tree = merge_axis_trees(arg_axis_trees)

        # drop zero-sized bits
        pruned_axis_tree = merged_axis_tree.prune()

        if not pruned_axis_tree:
            # the assignment is zero-sized
            return NullInstruction()

        axis_trees.append(pruned_axis_tree)
    axis_trees = tuple(axis_trees)

    assignee = concretize_expression_layouts(assignment.assignee, axis_trees)
    expression = concretize_expression_layouts(assignment.expression, axis_trees)

    return NonEmptyArrayAssignment(assignee, expression, axis_trees, assignment.assignment_type, comm=assignment.internal_comm)


MAX_COST_CONSIDERATION_FACTOR = 5
"""Maximum factor an expression cost can exceed the minimum and still be considered."""


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
@_collect_candidate_indirections.register(Exscan)  # assume we are fine
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
        assignee, expression, assignment.assignment_type, assignment.axis_trees, comm=assignment.user_comm
    )
