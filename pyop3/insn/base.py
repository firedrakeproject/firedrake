from __future__ import annotations

import abc
import collections
from collections.abc import Hashable, Mapping
import dataclasses
import enum
import functools
import itertools
import numbers
from os import stat
import textwrap
import typing
from functools import cached_property
from typing import Any, ClassVar, Iterable, Tuple

from immutabledict import immutabledict as idict
import loopy as lp
import numpy as np
from pyop3.buffer import BufferRef, PetscMatBufferSubMat
from pyop3.expr.buffer import LinearDatBufferExpression, ScalarBufferExpression
import pytools
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.record
from pyop3 import utils
from pyop3.cache import with_heavy_caches, with_self_heavy_cache, memory_cache, cached_method
from pyop3.collections import OrderedFrozenSet, OrderedSet, is_ordered_mapping
from pyop3.node import Node, Terminal
from pyop3.tree.axis_tree import AxisTree
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE, AxisForest, ContextFree, ContextSensitive, axis_tree_is_valid_subset, matching_axis_tree
from pyop3.expr import BufferExpression, Tensor, Scalar, Dat, Mat
from pyop3.sf import DistributedObject
from pyop3.dtypes import dtype_limits
from pyop3.exceptions import Pyop3Exception
from pyop3.utils import (
    auto,
)

if typing.TYPE_CHECKING:
    from .exec import InstructionExecutionContext


# TODO I don't think that this belongs in this file, it belongs to the function?
# create a function.py file?
class Intent(enum.Enum):
    # developer note, MIN_RW and MIN_WRITE are distinct (unlike PyOP2) to avoid
    # passing "requires_zeroed_output_arguments" around, yuck

    READ = "read"
    WRITE = "write"
    RW = "rw"
    INC = "inc"
    MIN_WRITE = "min_write"
    MIN_RW = "min_rw"
    MAX_WRITE = "max_write"
    MAX_RW = "max_rw"


READ = Intent.READ
WRITE = Intent.WRITE
RW = Intent.RW
INC = Intent.INC
MIN_RW = Intent.MIN_RW
MIN_WRITE = Intent.MIN_WRITE
MAX_RW = Intent.MAX_RW
MAX_WRITE = Intent.MAX_WRITE
# TODO: This exception is not actually ever raised. We should check the
# intents of the kernel arguments and complain if something illegal is
# happening.
class IntentMismatchError(Exception):
    pass


# FIXME: This is not a thing any more
class KernelArgument(abc.ABC):
    """Abstract class for types that may be passed as arguments to kernels.

    Note that some types that can be passed to *functions* are not in fact
    kernel arguments. This is because they either wrap actual kernel arguments
    (e.g. `Dat`), or because no argument is actually passed
    (e.g. a temporary).

    """

    # needed? the motivation is that one can consider arrays as having 2 dtypes. E.g.
    # 'double*' or 'double' (the whole thing or the entries)
    # @property
    # @abc.abstractmethod
    # def kernel_dtype(self):
    #     pass


class UnprocessedExpressionException(Pyop3Exception):
    """Exception raised when pyop3 expected a preprocessed expression."""


class Instruction(Node, DistributedObject, abc.ABC):

    def __init__(self) -> None:
        object.__setattr__(self, "_hit_executor_cache", True)

    # FIXME: This is very similar to PreprocessedOperation.buffers but *not the same*
    #  Here we only permit the 'shallow' buffers (i.e. not the layouts) whereas there
    # it is everything that gets passed in
    @property
    @abc.abstractmethod
    def buffer_arguments(self) -> OrderedFrozenSet[AbstractBufferExpression]:
        """Mapping from name to tensor that is passed in as an argument."""

    @property
    @abc.abstractmethod
    def comm(self) -> MPI.Comm:
        pass

    @with_self_heavy_cache
    def __call__(self, *, compiler_parameters=None, **kwargs) -> None:
        self._get_execution_context(compiler_parameters)(**kwargs)

    @cached_method()
    def _get_execution_context(self, compiler_parameters) -> InstructionExecutionContext:
        from .exec import InstructionExecutionContext

        return InstructionExecutionContext(self, compiler_parameters)



_DEFAULT_LOOP_NAME = "pyop3_loop"


@pyop3.record.frozenrecord()
class Loop(Instruction):

    # {{{ instance attrs

    index: LoopIndex
    statements: tuple[Instruction, ...]

    def __init__(
        self,
        index: LoopIndex,
        statements: Iterable[Instruction] | Instruction,
    ) -> None:
        statements = utils.as_tuple(statements)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "statements", statements)
        super().__init__()

    # }}}

    # {{{ interface impls

    child_attrs = ("statements",)

    @cached_property
    def buffer_arguments(self) -> OrderedFrozenSet[Tensor]:
        return OrderedFrozenSet().union(*(stmt.buffer_arguments for stmt in self.statements))

    @property
    def comm(self) -> MPI.Comm:
        # TODO: check iterset
        return utils.common_comm(self.statements, "comm")

    # }}}

    def __str__(self) -> str:
        stmt_strs = [textwrap.indent(str(stmt), "    ") for stmt in self.statements]
        return f"""loop(
  {self.index},
  [
{'\n'.join(stmt_strs)}
  ]
)"""


@pyop3.record.frozenrecord()
class InstructionList(Instruction):
    """A list of instructions."""

    # {{{ instance attrs

    instructions: tuple[Instruction]

    def __init__(self, instructions: Iterable[Instruction]) -> None:
        instructions = tuple(instructions)
        object.__setattr__(self, "instructions", instructions)

    # }}}

    # {{{ interface impls

    child_attrs = ("instructions",)

    @property
    def comm(self) -> MPI.Comm:
        return utils.common_comm(self.instructions, "comm")

    @property
    def buffer_arguments(self) -> OrderedFrozenSet[Tensor]:
        return OrderedFrozenSet().union(*(insn.buffer_arguments for insn in self.instructions))

    # }}}

    def __iter__(self):
        return iter(self.instructions)

    def __str__(self) -> str:
        return "\n".join(map(str, self.instructions))

    @cached_property
    def datamap(self):
        return utils.merge_dicts(insn.datamap for insn in self.instructions)


def enlist(insn: Instruction) -> InstructionList:
    if isinstance(insn, InstructionList):
        return insn
    elif isinstance(insn, NullInstruction):
        return InstructionList(())
    else:
        return InstructionList([insn])


def maybe_enlist(instructions) -> Instruction:
    flattened_insns = []
    for insn in filter_null(instructions):
        if isinstance(insn, InstructionList):
            flattened_insns.extend(insn.instructions)
        else:
            flattened_insns.append(insn)

    if not flattened_insns:
        return NullInstruction()
    elif len(flattened_insns) > 1:
        return InstructionList(flattened_insns)
    else:
        return utils.just_one(flattened_insns)


def non_null(instruction: Instruction) -> bool:
    return not isinstance(instruction, NullInstruction)


def filter_null(iterable: Iterable[Instruction]):
    return filter(non_null, iterable)


class TerminalInstruction(Instruction, Terminal, abc.ABC):

    @property
    @abc.abstractmethod
    def arguments(self) -> tuple[Any, ...]:
        pass

    @property
    def buffer_arguments(self) -> OrderedFrozenSet[BufferExpression, ...]:
        from pyop3.expr.visitors import collect_arguments

        return OrderedFrozenSet().union(
            *(collect_arguments(arg) for arg in self.arguments)
        )


class NonEmptyTerminal(TerminalInstruction, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def axis_trees(self) -> AxisTree:
        pass


@dataclasses.dataclass(frozen=True)
class ArgumentSpec:
    intent: Intent
    dtype: np.dtype
    space: Tuple[int]  # TODO: definitely am not using this...


class FunctionArgument(abc.ABC):
    """Abstract class for types that may be passed to functions."""


@pyop3.record.frozenrecord()
class Function:
    """A callable function."""

    code: Any
    _access_descrs: tuple[Intent, ...]

    def __init__(self, loopy_kernel, access_descrs):
        lpy_args = loopy_kernel.default_entrypoint.args
        if len(lpy_args) != len(access_descrs):
            raise ValueError("Wrong number of access descriptors given")
        for lpy_arg, access in zip(lpy_args, access_descrs):
            if access in {MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE} and lpy_arg.shape != (
                1,
            ):
                raise ValueError("Reduction operations are only valid for scalars")

        loopy_kernel = fix_intents(loopy_kernel, access_descrs)
        access_descrs = tuple(access_descrs)

        object.__setattr__(self, "code", loopy_kernel)
        object.__setattr__(self, "_access_descrs", access_descrs)

    # unfortunately needed because loopy translation units aren't immediately hashable
    def __hash__(self) -> int:
        if not hasattr(self, "_saved_hash"):
            kb = lp.tools.LoopyKeyBuilder()
            hash_ = hash((
                type(self),
                kb(self.code),
                self._access_descrs,
            ))
            object.__setattr__(self, "_saved_hash", hash_)
        return self._saved_hash

    # unfortunately needed because loopy translation units aren't immediately hashable
    def __eq__(self, other, /) -> bool:
        return type(other) is type(self) and other.code == self.code and other._access_descrs == self._access_descrs

    def __call__(self, *args):
        # if not all(isinstance(a, FunctionArgument) for a in args):
        #     raise TypeError("invalid kernel argument type")
        if len(args) != len(self.argspec):
            raise ValueError(
                f"Wrong number of arguments provided, expected {len(self.argspec)} "
                f"but received {len(args)}"
            )
        # if any(
        #     spec.dtype.numpy_dtype != arg.kernel_dtype
        #     for spec, arg in checked_zip(self.argspec, args)
        #     if arg.kernel_dtype is not auto
        # ):
        #     raise ValueError("Arguments to the kernel have the wrong dtype")
        return CalledFunction(self, args)

    @property
    def argspec(self):
        spec = []
        for access, arg in zip(
            self._access_descrs, self.code.default_entrypoint.args, strict=True
        ):
            shape = arg.shape if not isinstance(arg, lp.ValueArg) else ()
            spec.append(ArgumentSpec(access, arg.dtype, shape))
        return tuple(spec)

    @property
    def name(self):
        return self.code.default_entrypoint.name

    @property
    def num_flops(self) -> int:
        import pyop3.extras.debug
        pyop3.extras.debug.warn_todo("Function.num_flops isn't implemented, returning 666 for now")
        return 666


class AbstractCalledFunction(NonEmptyTerminal, metaclass=abc.ABCMeta):

    # def __init__(
    #     self, function: Function, arguments: Iterable[FunctionArgument], **kwargs
    # ) -> None:
    #     object.__setattr__(self, "function", function)
    #     super().__init__(arguments, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(arg.name for arg in self.arguments)})"

    @property
    @abc.abstractmethod
    def function(self) -> Function:
        pass

    @property
    def axis_trees(self) -> tuple[AxisTree, ...]:
        return (UNIT_AXIS_TREE,)

    @property
    def name(self):
        return self.function.name

    @property
    def argspec(self):
        return self.function.argspec

    @cached_property
    def function_arguments(self):
        return tuple((arg, spec.intent) for arg, spec in zip(self.arguments, self.argspec, strict=True))

    @property
    def argument_shapes(self):
        return tuple(
            arg.shape if not isinstance(arg, lp.ValueArg) else ()
            for arg in self.function.code.default_entrypoint.args
        )

    @property
    def comm(self) -> MPI.Comm:
        return utils.common_comm(self.arguments, "comm", allow_undefined=True) or MPI.COMM_SELF


@pyop3.record.frozenrecord()
class CalledFunction(AbstractCalledFunction):

    # {{{ instance attrs

    _function: Function
    _arguments: tuple[Any]

    def __init__(self, function: Function, arguments: Iterable):
        arguments = tuple(arguments)

        function = self._fixup_function_argument_shapes(function, arguments)

        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_arguments", arguments)

    # }}}

    # {{{ interface impls

    function: ClassVar[property] = pyop3.record.attr("_function")
    arguments: ClassVar[property] = pyop3.record.attr("_arguments")

    # }}}

    @classmethod
    def _fixup_function_argument_shapes(cls, function, arguments):
        loopy_kernel = function.code.default_entrypoint
        if all(a.shape is not None for a in loopy_kernel.args):
            return function

        new_loopy_args = []
        for loopy_arg, arg in zip(loopy_kernel.args, arguments, strict=True):
            if loopy_arg.shape is None:
                loopy_arg = loopy_arg.copy(shape=(arg.size,))
            new_loopy_args.append(loopy_arg)
        new_loopy_args = tuple(new_loopy_args)
        return function.__record_init__(
            code=function.code.with_kernel(loopy_kernel.copy(args=new_loopy_args))
        )



@pyop3.record.frozenrecord()
class StandaloneCalledFunction(AbstractCalledFunction):
    """A called function whose arguments do not need packing/unpacking."""

    _function: Function
    _arguments: Iterable[FunctionArgument]

    function: ClassVar[property] = property(lambda self: self._function)
    arguments: ClassVar[property] = property(lambda self: self._arguments)

    def __init__(self, function: Function, arguments: Iterable):
        arguments = tuple(arguments)

        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_arguments", arguments)


# TODO: Make this a singleton like UNIT_AXIS_TREE
class NullInstruction(TerminalInstruction):
    """An instruction that does nothing."""

    arguments = ()

    # COMM_DYNAMIC?
    comm = MPI.COMM_SELF


# TODO: With Python 3.11 can be made a StrEnum
class AssignmentType(enum.Enum):
    WRITE = "write"
    INC = "inc"


def assignment_type_as_intent(assignment_type: AssignmentType) -> Intent:
    match assignment_type:
        case AssignmentType.WRITE:
            return Intent.WRITE
        case AssignmentType.INC:
            return Intent.INC
        case _:
            raise AssertionError(f"{assignment_type} not recognised")


class AbstractAssignment(TerminalInstruction, metaclass=abc.ABCMeta):

    # {{{ Abstract methods

    @property
    @abc.abstractmethod
    def assignee(self) -> Any:
        pass

    @property
    @abc.abstractmethod
    def expression(self) -> Any:
        pass

    @property
    @abc.abstractmethod
    def assignment_type(self) -> AssignmentType:
        pass

    # }}}

    # {{{ Interface impls

    @property
    def arguments(self) -> tuple[Any, Any]:
        return (self.assignee, self.expression)

    # }}}


    # {{{ Dunders

    # def __init__(self, assignee, expression, assignment_type, **kwargs):
    #     arguments = (assignee, expression)
    #     assignment_type = AssignmentType(assignment_type)
    #
    #     object.__setattr__(self, "assignment_type", assignment_type)
    #     super().__init__(arguments, **kwargs)

    def __str__(self) -> str:
        if self.assignment_type == AssignmentType.WRITE:
            operator = "="
        else:
            assert self.assignment_type == AssignmentType.INC
            operator = "+="

        # 'assignee' and 'expression' might be multi-component and thus have
        # multi-line representations. We want to line these up.
        # NOTE: This might not be the ideal solution, eagerly break the Assignment up?

        assignee_strs = str(self.assignee).split("\n")
        expression_strs = str(self.expression).split("\n")

        if len(assignee_strs) > 1:
            if len(expression_strs) > 1:
                return "\n".join((
                    f"{assignee} {operator} {expression}"
                    for assignee, expression in zip(assignee_strs, expression_strs, strict=True)
                ))
            else:
                return "\n".join((
                    f"{assignee} {operator} {utils.just_one(expression_strs)}"
                    for assignee in assignee_strs
                ))
        else:
            if len(expression_strs) > 1:
                return "\n".join((
                    f"{utils.just_one(assignee_strs)} {operator} {expr}"
                    for expr in expression_strs
                ))
            else:
                return f"{utils.just_one(assignee_strs)} {operator} {utils.just_one(expression_strs)}"

    # }}}

    @property
    def assignee(self):
        return self.arguments[0]

    @property
    def expression(self):
        return self.arguments[1]


# TODO: not sure need to specify 'array' here
@pyop3.record.frozenrecord()
class ArrayAssignment(AbstractAssignment):

    # {{{ instance attrs

    _assignee: Any
    _expression: Any
    _assignment_type: AssignmentType

    def __init__(self, assignee: Any, expression: Any, assignment_type: AssignmentType | str) -> None:
        assignment_type = AssignmentType(assignment_type)

        object.__setattr__(self, "_assignee", assignee)
        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "_assignment_type", assignment_type)
        super().__init__()
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ interface impls

    assignee: ClassVar[property] = pyop3.record.attr("_assignee")
    expression: ClassVar[property] = pyop3.record.attr("_expression")
    assignment_type: ClassVar[property] = pyop3.record.attr("_assignment_type")

    @property
    def comm(self) -> MPI.Comm:
        return utils.common_comm([self.assignee, self.expression], "comm", allow_undefined=True) or MPI.COMM_SELF

    # NOTE: Wrong type here...
    @property
    def shape(self) -> tuple[AxisTree, ...]:
        from pyop3.expr.visitors import get_shape

        assignee_shapes = get_shape(self.assignee)
        expr_shapes = get_shape(self.expression)
        if expr_shapes == (UNIT_AXIS_TREE,):
            expr_shapes = itertools.repeat(UNIT_AXIS_TREE, len(assignee_shapes))

        # The shape of the assignment is simply the shape of the assignee, nothing else
        # makes sense. For more complex things loops should be used.
        # FIXME: This logic is dreadful
        axis_trees = []
        for assignee_shape, expr_shape in zip(assignee_shapes, expr_shapes, strict=True):
            if isinstance(assignee_shape, AxisForest):
                if isinstance(expr_shape, AxisForest):
                    # take the first match
                    assignee_shape = [
                            shape
                            for shape in assignee_shape.trees
                            if any(axis_tree_is_valid_subset(es, shape) for es in expr_shape.trees)
                        ][0]
                else:
                    # take the first match
                    assignee_shape = [
                            shape
                            for shape in assignee_shape.trees
                            if axis_tree_is_valid_subset(expr_shape, shape)
                        ][0]  
            axis_trees.append(assignee_shape)
        return tuple(axis_trees)

    # }}}



# FIXME: inconsistent argument ordering vs Concretized
@pyop3.record.frozenrecord()
class NonEmptyArrayAssignment(AbstractAssignment, NonEmptyTerminal):

    # {{{ instance attrs

    _assignee: Any
    _expression: Any
    _axis_trees: tuple[AxisTree, ...]
    _assignment_type: AssignmentType
    # is this still needed?
    _comm: MPI.Comm = dataclasses.field(hash=False)

    def __init__(self, assignee: Any, expression: Any, axis_trees, assignment_type: AssignmentType | str, *, comm: MPI.Comm) -> None:
        assignment_type = AssignmentType(assignment_type)

        object.__setattr__(self, "_assignee", assignee)
        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "_axis_trees", axis_trees)
        object.__setattr__(self, "_assignment_type", assignment_type)
        object.__setattr__(self, "_comm", comm)
        self.__post_init__()

    def __post_init__(self):
        pass

    # }}}

    # {{{ interface impls

    assignee = pyop3.record.attr("_assignee")
    expression = pyop3.record.attr("_expression")
    axis_trees = pyop3.record.attr("_axis_trees")
    assignment_type = pyop3.record.attr("_assignment_type")
    comm = pyop3.record.attr("_comm")

    # }}}


@pyop3.record.frozenrecord()
class ConcretizedNonEmptyArrayAssignment(AbstractAssignment):

    # {{{ Instance attrs

    _assignee: Any
    _expression: Any
    _assignment_type: AssignmentType
    _axis_trees: tuple[AxisTree, ...]
    _comm: MPI.Comm = dataclasses.field(hash=False)

    def __init__(self, assignee: Any, expression: Any, assignment_type: AssignmentType | str, axis_trees, *, comm: MPI.Comm) -> None:
        assignment_type = AssignmentType(assignment_type)

        object.__setattr__(self, "_assignee", assignee)
        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "_assignment_type", assignment_type)
        object.__setattr__(self, "_axis_trees", axis_trees)
        object.__setattr__(self, "_comm", comm)
        self.__post_init__()

    def __post_init__(self):
        pass

    # }}}

    # {{{ Interface impls

    assignee: ClassVar = pyop3.record.attr("_assignee")
    expression: ClassVar = pyop3.record.attr("_expression")
    assignment_type: ClassVar = pyop3.record.attr("_assignment_type")
    axis_trees: ClassVar = pyop3.record.attr("_axis_trees")
    comm: ClassVar = pyop3.record.attr("_comm")

    # }}}


@pyop3.record.frozenrecord()
class Exscan(TerminalInstruction):

    # {{{ instance attrs

    assignee: Any
    expression: Any
    scan_type: Any
    scan_axis: Axis
    _comm: MPI.Comm = dataclasses.field(hash=False)

    # }}}

    # {{{ interface impls

    @property
    def arguments(self) -> tuple[Any, Any]:
        return (self.assignee, self.expression)

    @property
    def comm(self) -> MPI.Comm:
        return self._comm

    @cached_property
    def extent(self):
        return self.scan_axis.component.size - 1

    # }}}


def exscan(*args, eager: bool = False, **kwargs):
    expr = Exscan(*args, **kwargs)
    return expr() if eager else expr



# TODO: With Python 3.11 can be made a StrEnum
# The idea is basically RW isn't allowed here
class ArrayAccessType(enum.Enum):
    READ = "read"
    WRITE = "write"
    INC = "inc"


class OpaqueKernelArgument(KernelArgument, ContextFree):
    def __init__(self, dtype=auto):
        self._dtype = dtype

    @property
    def kernel_dtype(self):
        return self._dtype


class DummyKernelArgument(OpaqueKernelArgument):
    """Placeholder kernel argument.

    This class is useful when one simply wants to generate code from a loop
    expression and not execute it.

    ### dtypes not required here as sniffed from local kernel/context?

    """


def loop_(*args, eager: bool = False, **kwargs) -> Loop | None:
    """
    Notes
    -----
    This function has a trailing underscore to avoid clashing with any variables
    called ``loop``. It is exported as ``op3.loop``.

    """
    if eager:
        compiler_parameters = kwargs.pop("compiler_parameters", None)

    loop_expr = Loop(*args, **kwargs)
    return loop_expr(compiler_parameters=compiler_parameters) if eager else loop_expr


# TODO: better to pass eager kwarg
def do_loop(index, statements, *, compiler_parameters: Mapping | None = None):
    loop_(index, statements)(compiler_parameters=compiler_parameters)


def fix_intents(tunit, accesses):
    """

    The local kernel has underspecified accessors (is_input, is_output).
    Here coerce them to match the access descriptors provided.

    This should arguably be done properly in TSFC.

    Note that even if this isn't done in TSFC we need to guard against this properly
    as the default error is very unclear.

    """
    kernel = tunit.default_entrypoint
    new_args = []
    for arg, access in zip(kernel.args, accesses, strict=True):
        assert isinstance(access, Intent)
        is_input = access in {READ, RW, INC, MIN_RW, MAX_RW}
        is_output = access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_WRITE, MAX_RW}
        new_args.append(arg.copy(is_input=is_input, is_output=is_output))
    return tunit.with_kernel(kernel.copy(args=new_args))
