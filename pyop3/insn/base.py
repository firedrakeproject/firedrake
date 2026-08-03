from __future__ import annotations

import abc
import collections
from collections.abc import Hashable, Mapping, Iterable
import dataclasses
import enum
import functools
import itertools
import numbers
from os import stat
import textwrap
import typing
from functools import cached_property
from typing import Any, ClassVar, Tuple

from immutabledict import immutabledict as idict
import loopy as lp
import loopy.tools
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import pyop3.compile
import pyop3.expr
import pyop3.record
import pyop3.visitors
from pyop3 import utils
from pyop3.cache import with_heavy_caches, with_self_heavy_cache, memory_cache, cached_method
from pyop3.collections import OrderedFrozenSet, OrderedSet, is_ordered_mapping
from pyop3.constants import Intent, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE
from pyop3.node import Node, Terminal, Operator
from pyop3.axis_tree import AxisTree
from pyop3.axis_tree.tree import UNIT_AXIS_TREE, AxisForest, ContextFree, ContextSensitive, axis_tree_is_valid_subset, matching_axis_tree
from pyop3.expr import BufferExpression, Tensor, Scalar, Dat, Mat
from pyop3.dtypes import dtype_limits
from pyop3.exceptions import Pyop3Exception
from pyop3.utils import (
    auto,
)

if typing.TYPE_CHECKING:
    from .exec import InstructionExecutionContext


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


class Instruction(Node, abc.ABC):

    def __init__(self) -> None:
        object.__setattr__(self, "_hit_executor_cache", True)

    # FIXME: This is very similar to PreprocessedOperation.buffers but *not the same*
    #  Here we only permit the 'shallow' buffers (i.e. not the layouts) whereas there
    # it is everything that gets passed in
    # TODO: Call 'named_terminals'? because that's the type that we have...
    # exec_arguments?
    @property
    @abc.abstractmethod
    def global_arguments(self) -> OrderedFrozenSet[AbstractBufferExpression]:
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


class NonTerminalInstruction(Instruction, Operator):
    pass


class TerminalInstruction(Instruction, Terminal, abc.ABC):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def arguments(self) -> tuple[Any, ...]:
        pass

    @property
    @abc.abstractmethod
    def compiler_options(self) -> pyop3.compile.CompilerOptions:
        """Extra options needed to compile this terminal."""

    # }}}

    @property
    def global_arguments(self) -> OrderedFrozenSet[BufferExpression, ...]:
        from pyop3.expr.visitors import collect_arguments

        return OrderedFrozenSet().union(
            *(collect_arguments(arg) for arg in self.arguments)
        )




# TODO not a useful thing to have any more
_DEFAULT_LOOP_NAME = "pyop3_loop"


@pyop3.record.frozenrecord()
class Loop(NonTerminalInstruction):

    # {{{ instance attrs

    index: LoopIndex
    statements: tuple[Instruction, ...]

    def collect_buffers(self, visitor):
        return visitor(self.index).union(*(map(visitor, self.statements)))

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self.index), tuple(map(visitor, self.statements)))

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self.index, inside=True),
            tuple(map(visitor, self.statements)),
        )

    @cached_property
    def comm(self) -> MPI.Comm:
        return pyop3.visitors.common_comm(self.index, *self.statements)

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
    def global_arguments(self) -> OrderedFrozenSet[Tensor]:
        return OrderedFrozenSet().union(*(stmt.global_arguments for stmt in self.statements))

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
class InstructionList(NonTerminalInstruction):
    """A list of instructions."""

    # {{{ instance attrs

    instructions: tuple[Instruction]

    def get_disk_cache_key(self, visitor):
        return (type(self), tuple(visitor(insn) for insn in self.instructions))

    get_instruction_executor_cache_key = get_disk_cache_key

    def collect_buffers(self, visitor):
        return OrderedFrozenSet().union(*(map(visitor, self.instructions)))

    @cached_property
    def comm(self) -> MPI.Comm:
        return pyop3.visitors.common_comm(*self.instructions)

    def __init__(self, instructions: Iterable[Instruction]) -> None:
        instructions = tuple(instructions)
        object.__setattr__(self, "instructions", instructions)

    # }}}

    # {{{ interface impls

    child_attrs = ("instructions",)

    @property
    def global_arguments(self) -> OrderedFrozenSet[Tensor]:
        return OrderedFrozenSet().union(*(insn.global_arguments for insn in self.instructions))

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
class Function(pyop3.obj.Pyop3Object):
    """A callable function."""

    # {{{ instance attrs

    code: Any
    _access_descrs: tuple[Intent, ...]
    _compiler_options: pyop3.compile.CompilerOptions

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            loopy.tools.LoopyKeyBuilder()(self.code),
            self._access_descrs,
        )

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(
        self,
        loopy_kernel,
        access_descrs,
        *,
        include_dirs: Iterable[str] = (),
        lib_dirs: Iterable[str] = (),
        libs: Iterable[str] = (),
    ):
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

        compiler_options = pyop3.compile.CompilerOptions(
            include_dirs=tuple(include_dirs),
            lib_dirs=tuple(lib_dirs),
            libs=tuple(libs),
        )

        object.__setattr__(self, "code", loopy_kernel)
        object.__setattr__(self, "_access_descrs", access_descrs)
        object.__setattr__(self, "_compiler_options", compiler_options)

    # }}}

    # {{{ interface impls

    compiler_options = pyop3.record.attr("_compiler_options")

    # }}}

    # {{{ factory methods

    @classmethod
    def from_c_string(
        cls,
        /,
        name: str,
        c_code: str,
        args: Iterable[tuple[str, DTypeT, Intent]],
        *,
        preambles=(),
        **kwargs,
    ) -> Function:
        from pyop3 import LOOPY_TARGET, LOOPY_LANG_VERSION

        loopy_insn = lp.CInstruction(
            (),
            c_code,
            frozenset((name_ for name_, _, _ in args)),
            tuple(name_ for name_, _, intent in args if intent != Intent.READ),
        )
        loopy_args = []
        for name_, dtype, intent in args:
            match intent:
                case Intent.READ:
                    is_input = True
                    is_output = False
                case Intent.WRITE:
                    is_input = True  # is this needed?
                    is_output = True
                case Intent.INC:
                    is_input = True
                    is_output = True
                case _:
                    raise NotImplementedError

            if isinstance(dtype, lp.types.OpaqueType):
                # no packing, passthrough arg
                loopy_arg = lp.ValueArg(name_, dtype, is_input=is_input, is_output=is_output)
            else:
                loopy_arg = lp.GlobalArg(name_, dtype, is_input=is_input, is_output=is_output)
            loopy_args.append(loopy_arg)
        loopy_kernel = lp.make_kernel(
            [],  # no extra loops
            [loopy_insn],
            loopy_args,
            name=name,
            preambles=[
                ("20_petsc", "#include <petsc.h>"),
                *preambles,
            ],
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        )

        intents = [intent for _, _, intent in args]
        return cls(loopy_kernel, intents, **kwargs)

    # }}}

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
        import pyop3.debug
        pyop3.debug.warn_todo("Function.num_flops isn't implemented, returning 666 for now")
        return 666


class AbstractCalledFunction(NonEmptyTerminal, metaclass=abc.ABCMeta):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def function(self) -> Function:
        pass

    # }}}

    # {{{ interface impls

    # TODO: look at the arguments too
    @cached_property
    def compiler_options(self) -> pyop3.compile.CompilerOptions:
        return self.function.compiler_options

    # }}}

    def __str__(self) -> str:
        return f"{self.name}({', '.join(arg.name for arg in self.arguments)})"

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


@pyop3.record.frozenrecord()
class CalledFunction(AbstractCalledFunction):

    # {{{ instance attrs

    _function: Function
    _arguments: tuple[Any]

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self._function),
            tuple(map(visitor, self._arguments)),
        )

    @cached_property
    def comm(self) -> MPI.Comm:
        return pyop3.visitors.common_comm(*self._arguments)

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
        if all(
            a.shape is not None for a in loopy_kernel.args
            if isinstance(a, lp.ArrayArg)
        ):
            return function

        new_loopy_args = []
        for loopy_arg, arg in zip(loopy_kernel.args, arguments, strict=True):
            if isinstance(loopy_arg, lp.ArrayArg):
                loopy_arg = loopy_arg.copy(shape=(arg.size,), dim_tags=None)
            new_loopy_args.append(loopy_arg)
        new_loopy_args = tuple(new_loopy_args)
        return function.__record_init__(
            code=function.code.with_kernel(loopy_kernel.copy(args=new_loopy_args))
        )



@pyop3.record.frozenrecord()
class StandaloneCalledFunction(AbstractCalledFunction):
    """A called function whose arguments do not need packing/unpacking."""

    # {{{ instance attrs

    _function: Function
    _arguments: Iterable[FunctionArgument]

    def get_disk_cache_key(self, visitor):
        return (
            type(self),
            visitor(self._function),
            tuple(map(visitor, self._arguments)),
        )

    def collect_buffers(self, visitor):
        return OrderedFrozenSet().union(*(map(visitor, self._arguments)))

    @cached_property
    def comm(self) -> MPI.Comm:
        return pyop3.visitors.common_comm(*self._arguments)

    def __init__(self, function: Function, arguments: Iterable):
        arguments = tuple(arguments)

        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_arguments", arguments)

    # }}}

    function: ClassVar[property] = property(lambda self: self._function)
    arguments: ClassVar[property] = property(lambda self: self._arguments)


# TODO: Make this a singleton like UNIT_AXIS_TREE
class NullInstruction(TerminalInstruction):
    """An instruction that does nothing."""

    # {{{ instance attrs (there aren't any)

    def collect_buffers(self, visitor):
        return OrderedFrozenSet()

    # }}}

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

    # {{{ abstract methods

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

    # {{{ interface impls

    @property
    def arguments(self) -> tuple[Any, Any]:
        return (self.assignee, self.expression)

    # TODO: can do things like add #include <petscmat.h> here...
    @cached_property
    def compiler_options(self) -> pyop3.compile.CompilerOptions:
        return pyop3.compile.CompilerOptions()

    # }}}

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

    @property
    def assignee(self):
        return self.arguments[0]

    @property
    def expression(self):
        return self.arguments[1]


@pyop3.record.frozenrecord()
class Assignment(AbstractAssignment):

    # {{{ instance attrs

    _assignee: Any
    _expression: Any
    _assignment_type: AssignmentType

    def get_instruction_executor_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self._assignee),
            visitor(self._expression),
            self._assignment_type,
        )

    @cached_property
    def comm(self) -> MPI.Comm:
        return pyop3.visitors.common_comm(self._assignee, self._expression)

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

    # NOTE: Wrong type here...
    @cached_property
    def shape(self) -> tuple[AxisTree, ...]:
        return pyop3.expr.visitors.get_shape(self.assignee)

        # the below doesn't really work, need shapes to match exactly
        # assignee_shapes = pyop3.expr.visitors.get_shape(self.assignee)
        # expr_shapes = pyop3.expr.visitors.get_shape(self.expression)
        #
        # # sometimes the expression may not be matrix-valued
        # if len(assignee_shapes) != len(expr_shapes):
        #     assert len(assignee_shapes) == 2 and len(expr_shapes) == 1
        #     expr_shapes = expr_shapes * 2
        #
        # # Set 'only_unit' here because we are happy for 'expr_shapes' to be
        # # different to 'assignee_shape' up to unit axes. For example, we
        # # want to allow the operation
        # #
        # #     loop(p, dat1[p].assign(dat2[f(p)])
        # #
        # # even though dat2[f(p)] will have an extra axis introduced by
        # # the map. Provided f(p) only has size 1 the LHS and RHS are
        # # still the same shape.
        # return tuple(
        #     pyop3.axis_tree.merge_axis_trees([assignee_shape, expr_shape], only_unit=True)
        #     for assignee_shape, expr_shape in zip(assignee_shapes, expr_shapes, strict=True)
        # )

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

    def collect_buffers(self, visitor) -> OrderedFrozenSet[ConcreteBuffer]:
        return OrderedFrozenSet().union(
            visitor(self._assignee),
            visitor(self._expression),
            *(visitor(tree) for tree in self._axis_trees),
        )

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self._assignee),
            visitor(self._expression),
            *(map(visitor, self._axis_trees)),
            self._assignment_type,
        )

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

    def collect_buffers(self, visitor):
        return OrderedFrozenSet().union(
            visitor(self.assignee),
            visitor(self.expression),
            visitor(self.scan_axis),
        )

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self.assignee),
            visitor(self.expression),
            self.scan_type,
            visitor(self.scan_axis),
        )

    def get_instruction_executor_cache_key (self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self.assignee),
            visitor(self.expression),
            self.scan_type,
            visitor(self.scan_axis, inside=True),
        )

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

    @cached_property
    def compiler_options(self) -> pyop3.compile.CompilerOptions:
        return pyop3.compile.CompilerOptions()

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
        is_input = access in {Intent.READ, Intent.RW, Intent.INC, Intent.MIN_RW, Intent.MAX_RW}
        is_output = access in {Intent.WRITE, Intent.RW, Intent.INC, Intent.MIN_RW, Intent.MIN_WRITE, Intent.MAX_WRITE, Intent.MAX_RW}
        new_args.append(arg.copy(is_input=is_input, is_output=is_output))
    return tunit.with_kernel(kernel.copy(args=new_args))
