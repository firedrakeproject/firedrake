from __future__ import annotations

import abc
import collections
from collections.abc import Hashable, Mapping
import dataclasses
import enum
import functools
import numbers
from os import stat
import textwrap
from functools import cached_property
from typing import Any, ClassVar, Iterable, Tuple

import immutabledict
import loopy as lp
import numpy as np
import pytools
from mpi4py import MPI
from petsc4py import PETSc

from pyop3 import utils
from pyop3.node import Node, Terminal
from pyop3.tree.axis_tree import AxisTree
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE, AxisForest, ContextFree, ContextSensitive
from pyop3.expr import BufferExpression
from pyop3.sf import DistributedObject
from pyop3.dtypes import dtype_limits
from pyop3.exceptions import Pyop3Exception
from pyop3.utils import (
    deprecated,
    OrderedSet,
    as_tuple,
    auto,
    just_one,
    merge_dicts,
    single_valued,
    is_ordered_mapping,
)


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompilerParameters:

    # {{{ optimisation options

    compress_indirection_maps: bool = False
    interleave_comp_comm: bool = False

    # }}}

    # {{{ profiling options

    add_likwid_markers: bool = False
    add_petsc_event: bool = False

    # }}}

    # {{{ debugging options

    attach_debugger: bool = False

    # }}}


DEFAULT_COMPILER_PARAMETERS = CompilerParameters()


META_COMPILER_PARAMETERS = immutabledict.immutabledict({
    # TODO: when implemented should also set interleave_comp_comm to True
    "optimize": {"compress_indirection_maps": True}
})
"""'Meta' compiler parameters that set multiple options at once."""
# NOTE: These must be boolean options


class ParsedCompilerParameters(CompilerParameters):
    pass


def parse_compiler_parameters(compiler_parameters) -> ParsedCompilerParameters:
    """
    The process of parsing ``compiler_parameters`` is as follows:

        1. Begin with the default options (`DEFAULT_COMPILER_PARAMETERS`).
        2. In the order specified in ``compiler_parameters``, parse any
           'macro' options and tweak the parameters as appropriate.
        3. Lastly, any non-macro options are added.

    By setting macro options before individual options the user can make
    more specific overrides.

    """
    if isinstance(compiler_parameters, ParsedCompilerParameters):
        return compiler_parameters

    if compiler_parameters is None:
        compiler_parameters = {}
    else:
        # TODO: nice error message
        assert is_ordered_mapping(compiler_parameters)
        compiler_parameters = dict(compiler_parameters)

    parsed_parameters = dataclasses.asdict(DEFAULT_COMPILER_PARAMETERS)
    for macro_param, specific_params in META_COMPILER_PARAMETERS.items():
        # Do not rely on the truthiness of variables here. We want to make
        # sure that the user has provided a boolean value.
        if compiler_parameters.pop(macro_param, False) == True:
            for key, value in specific_params.items():
                parsed_parameters[key] = value

    for key, value in compiler_parameters.items():
        # TODO: If a KeyError then invalid params provided, should raise a helpful error
        assert key in parsed_parameters
        parsed_parameters[key] = value

    return ParsedCompilerParameters(**parsed_parameters)


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

    def __call__(self, replacement_buffers: Mapping[Hashable, ConcreteBuffer] | None = None, *, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)

        executable = self.compile(compiler_parameters)
        executable(replacement_buffers)

    def preprocess(self, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)
        return self._preprocess(compiler_parameters)

    @utils.cached_method()
    def _preprocess(self, compiler_parameters: ParsedCompilerParameters) -> Instruction:
        from .visitors import (
            expand_implicit_pack_unpack,
            expand_loop_contexts,
            expand_assignments,
            materialize_indirections,
            concretize_layouts,
            insert_literals,
        )

        insn = self
        insn = expand_loop_contexts(insn)

        # bad name, this expands all transformations and pack/unpacks for called functions
        # 'flatten?'
        # Since the expansion can add new nodes requiring parsing we do a fixed point iteration
        old_insn = insn
        insn = expand_assignments(insn)
        while insn != old_insn:
            old_insn = insn
            insn = expand_assignments(insn)

        # if "form" in str(self):
        #     breakpoint()

        # TODO: remove zero-sized bits here!
        insn = concretize_layouts(insn)
        insn = insert_literals(insn)
        insn = materialize_indirections(insn, compress=compiler_parameters.compress_indirection_maps)

        return PreprocessedOperation(insn)

    # TODO: only really an attr of lowered ones...
    def compile(self, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)
        return self._compile(compiler_parameters)

    @utils.cached_method()
    def _compile(self, compiler_parameters: ParsedCompilerParameters):
        from pyop3.ir.lower import compile

        preprocessed = self.preprocess(compiler_parameters)
        return compile(preprocessed, compiler_parameters=compiler_parameters)


class ContextAwareInstruction(Instruction):
    @property
    @abc.abstractmethod
    def datamap(self):
        """Map from names to arrays."""

    # @property
    # @abc.abstractmethod
    # def kernel_arguments(self):
    #     pass


_DEFAULT_LOOP_NAME = "pyop3_loop"


@utils.frozenrecord()
class PreprocessedOperation:
    root_insn: Instruction

    @property
    def comm(self) -> MPI.Comm:
        return self.root_insn.comm

    @cached_property
    def buffers(self) -> OrderedFrozenSet:
        """The buffers (global data) that are present in the operation."""
        from pyop3.insn.visitors import collect_buffers

        return collect_buffers(self.root_insn)

    @cached_property
    def disk_cache_key(self) -> Hashable:
        """Key used to write the operation to disk.

        The returned key should be consistent across ranks and not include
        overly specific information such as buffer names or array values.

        """
        from pyop3.insn.visitors import get_disk_cache_key

        return get_disk_cache_key(self.root_insn)


@utils.frozenrecord()
class Loop(Instruction):

    # {{{ instance attrs

    index: LoopIndex
    statements: tuple[Instruction, ...]

    def __init__(
        self,
        index: LoopIndex,
        statements: Iterable[Instruction] | Instruction,
    ) -> None:
        statements = as_tuple(statements)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "statements", statements)

    # }}}

    # {{{ interface impls

    child_attrs = ("statements",)

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

    def __call__(self, replacement_buffers: Mapping | None = None, *, compiler_parameters=None):
        # TODO just parse into ContextAwareLoop and call that
        from pyop3.ir.lower import compile
        from pyop3.tree.index_tree.tree import partition_iterset
        from pyop3.buffer import ArrayBuffer

        compiler_parameters = parse_compiler_parameters(compiler_parameters)

        code = self.compile(compiler_parameters)

        # TODO: Move to executor class
        if compiler_parameters.interleave_comp_comm:
            raise NotImplementedError
            new_index, (icore, iroot, ileaf) = partition_iterset(
                self.index, [a for a, _ in self.function_arguments]
            )
            #buffer_intents
            # assert self.index.id == new_index.id
            #
            # # substitute subsets into loopexpr, should maybe be done in partition_iterset
            # parallel_loop = self.copy(index=new_index)

            for init in initializers:
                init()

            # replace the parallel axis subset with one for the specific indices here
            extent = just_one(icore.axes.root.components).count
            core_kwargs = merge_dicts(
                [kwargs, {icore.name: icore, extent.name: extent}]
            )

            with PETSc.Log.Event(f"compute_{self.name}_core"):
                code(**core_kwargs)

            # await reductions
            for red in reductions:
                red()

            # roots
            # replace the parallel axis subset with one for the specific indices here
            root_extent = just_one(iroot.axes.root.components).count
            root_kwargs = merge_dicts(
                [kwargs, {icore.name: iroot, extent.name: root_extent}]
            )
            with PETSc.Log.Event(f"compute_{self.name}_root"):
                code(**root_kwargs)

            # await broadcasts
            for broadcast in broadcasts:
                broadcast()

            # leaves
            leaf_extent = just_one(ileaf.axes.root.components).count
            leaf_kwargs = merge_dicts(
                [kwargs, {icore.name: ileaf, extent.name: leaf_extent}]
            )
            with PETSc.Log.Event(f"compute_{self.name}_leaf"):
                code(**leaf_kwargs)

            # also may need to eagerly assemble Mats, or be clever and spike the accessors?
        else:
            # TODO: reenable logging (what is 'self.name')?
            # with PETSc.Log.Event(f"apply_{self.name}"):
            code(replacement_buffers)

    @cached_property
    def function_arguments(self) -> tuple:
        args = {}  # ordered
        for stmt in self.statements:
            for arg, intent in stmt.function_arguments:
                args[arg] = intent
        return tuple((arg, intent) for arg, intent in args.items())

    @cached_property
    def kernel_arguments(self):
        args = OrderedSet()
        for stmt in self.statements:
            for arg in stmt.kernel_arguments:
                args.add(arg)
        return tuple(args)


@utils.frozenrecord()
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

    # }}}

    def __iter__(self):
        return iter(self.instructions)

    def __str__(self) -> str:
        return "\n".join(map(str, self.instructions))

    @cached_property
    def datamap(self):
        return merge_dicts(insn.datamap for insn in self.instructions)


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
        return just_one(flattened_insns)


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
    def buffer_arguments(self) -> tuple[BufferExpression, ...]:
        return tuple(utils.filter_type(BufferExpression, self.arguments))


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


@utils.frozenrecord()
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
    @abc.abstractmethod
    def arguments(self) -> tuple[Any]:
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


@utils.frozenrecord()
class CalledFunction(AbstractCalledFunction):

    # {{{ instance attrs

    _function: Function
    _arguments: tuple[Any]

    def __init__(self, function: Function, arguments: Iterable):
        arguments = tuple(arguments)

        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_arguments", arguments)

    # }}}

    # {{{ interface impls

    function: ClassVar[property] = utils.attr("_function")
    arguments: ClassVar[property] = utils.attr("_arguments")

    # }}}



@utils.frozenrecord()
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
                    f"{assignee} {operator} {just_one(expression_strs)}"
                    for assignee in assignee_strs
                ))
        else:
            if len(expression_strs) > 1:
                return "\n".join((
                    f"{just_one(assignee_strs)} {operator} {expr}"
                    for expr in expression_strs
                ))
            else:
                return f"{just_one(assignee_strs)} {operator} {just_one(expression_strs)}"

    # }}}

    @property
    def assignee(self):
        return self.arguments[0]

    @property
    def expression(self):
        return self.arguments[1]


# @utils.frozenrecord()
# class AssignmentShape:
#     """
#     This class is necessary to encapsulate more complex assignments. Examples
#     include:
#
#         dat1[i] = dat2[j]
#         mat[i, j] = dat1[i] + dat2[j]
#
#     Ah, this latter case demonstrates that this isn't quite right... the expression
#     takes bits of the first axis tree and bits of the second.
#
#     Also, it would be extremely confusing to have
#
#         dat1[i] = dat1[j]
#
#     with 2 axis trees. Maybe we need loops in most cases...
#
#     """
#     axis_trees: tuple[AxisTree, ...]
#     assignee_axis_trees: tuple[AxisTree | None, ...]
#     expression_axis_trees: tuple[AxisTree | None, ...]
#
#     def __post_init__(self) -> None:
#         assert utils.single_valued(
#             map(len, [self.axis_trees, self.assignee_axis_trees, self.expression_axis_trees])
#         )


# TODO: not sure need to specify 'array' here
@utils.frozenrecord()
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
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ interface impls

    assignee: ClassVar[property] = utils.attr("_assignee")
    expression: ClassVar[property] = utils.attr("_expression")
    assignment_type: ClassVar[property] = utils.attr("_assignment_type")

    @property
    def comm(self) -> MPI.Comm:
        return utils.common_comm([self.assignee, self.expression], "comm", allow_undefined=True) or MPI.COMM_SELF

    # NOTE: Wrong type here...
    @property
    def shape(self) -> tuple[AxisTree, ...]:
        from pyop3.expr.visitors import get_shape

        # The shape of the assignment is simply the shape of the assignee, nothing else
        # makes sense. For more complex things loops should be used.
        axis_trees = []
        for axis_obj in get_shape(self.assignee):
            if isinstance(axis_obj, AxisForest):
                # just take the first
                axis_obj = axis_obj.trees[0]
            axis_trees.append(axis_obj)
        return tuple(axis_trees)

    # }}}



# FIXME: inconsistent argument ordering vs Concretized
@utils.frozenrecord()
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

    assignee = utils.attr("_assignee")
    expression = utils.attr("_expression")
    axis_trees = utils.attr("_axis_trees")
    assignment_type = utils.attr("_assignment_type")
    comm = utils.attr("_comm")

    # }}}


@utils.frozenrecord()
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

    assignee: ClassVar = utils.attr("_assignee")
    expression: ClassVar = utils.attr("_expression")
    assignment_type: ClassVar = utils.attr("_assignment_type")
    axis_trees: ClassVar = utils.attr("_axis_trees")
    comm: ClassVar = utils.attr("_comm")

    # }}}


@utils.frozenrecord()
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
