# TODO Rename this file insn.py - the pyop3 language is everything, not just this

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
from cachetools import cachedmethod
from petsc4py import PETSc
from pyrsistent import PMap, pmap

from pyop3 import utils
from pyop3.tree.axis_tree import AxisTree
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE, ContextFree, ContextSensitive
from pyop3.config import config
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

# NOTE: I dont think that these are needed any more. Just RW access?
MIN_RW = Intent.MIN_RW
MIN_WRITE = Intent.MIN_WRITE
MAX_RW = Intent.MAX_RW
MAX_WRITE = Intent.MAX_WRITE


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompilerParameters:
    # Optimisation options

    compress_indirection_maps: bool = False
    interleave_comp_comm: bool = False

    # Profiling options

    add_likwid_markers: bool = False
    add_petsc_event: bool = False

    # Debugging options

    attach_debugger: bool = False

    def __post_init__(self):
        if self.attach_debugger and not config["debug"]:
            raise RuntimeError("Will only work in debug mode (PYOP3_DEBUG=1)")


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


@dataclasses.dataclass(frozen=True)
class PreprocessedExpression:
    """Wrapper for an expression indicating that it has been prepared for code generation."""
    expression: Instruction


class Instruction(abc.ABC):

    @property
    def _cache(self) -> collections.defaultdict[dict]:
        if not hasattr(self, "_lazy_cache"):
            object.__setattr__(self, "_lazy_cache", collections.defaultdict(dict))
        return self._lazy_cache

    def __call__(self, replacement_buffers: Mapping[Hashable, ConcreteBuffer] | None = None, *, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)

        executable = self.compile(compiler_parameters)
        executable(replacement_buffers)

    def preprocess(self, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)
        return self._preprocess(compiler_parameters)

    @cachedmethod(lambda self: self._cache["Instruction._preprocess"])
    def _preprocess(self, compiler_parameters: ParsedCompilerParameters):
        from .visitors import (
            expand_implicit_pack_unpack,
            expand_loop_contexts,
            expand_assignments,
            materialize_indirections,
            concretize_layouts,
        )

        insn = self

        if "mat" in str(self) and "form_cell" in str(self):
            import pyop3
            pyop3.extras.debug.enable_conditional_breakpoints()
        insn = expand_loop_contexts(insn)
        insn = expand_implicit_pack_unpack(insn)

        insn = expand_assignments(insn)  # specifically reshape bits

        # TODO: remove zero-sized bits here!
        insn = concretize_layouts(insn)

        insn = materialize_indirections(insn, compress=compiler_parameters.compress_indirection_maps)

        return PreprocessedExpression(insn)

    def compile(self, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)
        return self._compile(compiler_parameters)

    @cachedmethod(lambda self: self._cache["Instruction._compile"])
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
class Loop(Instruction):

    # {{{ Instance attrs

    index: LoopIndex
    statements: tuple[Instruction]

    # }}}

    def __init__(
        self,
        index: LoopIndex,
        statements: Iterable[Instruction] | Instruction,
    ):
        statements = as_tuple(statements)

        object.__setattr__(self, "index", index)
        object.__setattr__(self, "statements", statements)

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

        initializers = []
        reductions = []
        broadcasts = []
        if self.comm.size > 1:
            for data_arg in code.data_arguments:
                if not isinstance(data_arg, ArrayBuffer):
                    continue

                inits, reds, bcasts = Loop._buffer_exchanges(
                    data_arg, code.global_buffer_intents[data_arg.name]
                )
                initializers.extend(inits)
                reductions.extend(reds)
                broadcasts.extend(bcasts)

        # TODO: handle interleaving as a compiler_parameter somehow
        if compiler_parameters.interleave_comp_comm:
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
            # Unoptimised case: perform all transfers eagerly
            for init in initializers:
                init()
            for red in reductions:
                red()
            for bcast in broadcasts:
                bcast()

            # TODO: reenable logging (what is 'self.name')?
            # with PETSc.Log.Event(f"apply_{self.name}"):
            code(replacement_buffers)

    @property
    def comm(self):
        # maybe collect the comm by looking at everything?
        return self.index.iterset.comm

    @cached_property
    def datamap(self) -> PMap:
        return self.index.datamap | merge_dicts(s.datamap for s in self.statements)

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

    def _array_updates(self):
        """Collect appropriate callables for updating shared values in the right order.

        Returns
        -------
        (initializers, (finalizers0, finalizers1))
            Collections of callables to be executed at the right times.

        """
        from pyop3 import Dat, Mat
        from pyop3.buffer import ArrayBuffer

        initializers = []
        reductions = []
        broadcasts = []
        for arg, intent in self.function_arguments:
            if isinstance(arg, Dat):
                buffer = arg.buffer
                if isinstance(buffer, ArrayBuffer) and buffer.is_distributed:
                    # for now assume the most conservative case
                    touches_ghost_points = True

                    inits, reds, bcasts = self._buffer_exchanges(
                        buffer, intent, touches_ghost_points=touches_ghost_points
                    )
                    initializers.extend(inits)
                    reductions.extend(reds)
                    broadcasts.extend(bcasts)
            elif isinstance(arg, ContextSensitiveDat):
                # assumed to not be distributed
                pass
            else:
                assert isinstance(arg, Mat)
                # just in case
                broadcasts.append(arg.assemble)

        return initializers, reductions, broadcasts

    # I hate staticmethods now, refactor
    @staticmethod
    def _buffer_exchanges(buffer, intent):
        initializers, reductions, broadcasts = [], [], []

        # Possibly instead of touches_ghost_points we could produce custom SFs for each loop
        # (we have filter_star_forest())
        # For now we just disregard the optimisation
        touches_ghost_points = True

        if intent in {READ, RW}:
            if touches_ghost_points:
                if not buffer._roots_valid:
                    initializers.append(buffer._reduce_leaves_to_roots_begin)
                    reductions.extend([
                        buffer._reduce_leaves_to_roots_end,
                        buffer._broadcast_roots_to_leaves_begin,
                    ])
                    broadcasts.append(buffer._broadcast_roots_to_leaves_end)
                else:
                    initializers.append(buffer._broadcast_roots_to_leaves_begin)
                    broadcasts.append(buffer._broadcast_roots_to_leaves_end)
            else:
                if not buffer._roots_valid:
                    initializers.append(buffer._reduce_leaves_to_roots_begin)
                    reductions.append(buffer._reduce_leaves_to_roots_end)

        elif intent == WRITE:
            # Assumes that all points are written to (i.e. not a subset). If
            # this is not the case then a manual reduction is needed.
            buffer._leaves_valid = False
            buffer._pending_reduction = None

        else:
            # reductions
            assert intent in {INC, MIN_WRITE, MIN_RW, MAX_WRITE, MAX_RW}
            # We don't need to update roots if performing the same reduction
            # again. For example we can increment into a buffer as many times
            # as we want. The reduction only needs to be done when the
            # data is read.
            if buffer._roots_valid or intent == buffer._pending_reduction:
                pass
            else:
                # We assume that all points are visited, and therefore that
                # WRITE accesses do not need to update roots. If only a subset
                # of entities are written to then a manual reduction is required.
                # This is the same assumption that we make for data_wo.
                if intent in {INC, MIN_RW, MAX_RW}:
                    assert buffer._pending_reduction is not None
                    initializers.append(buffer._reduce_leaves_to_roots_begin)
                    reductions.append(buffer._reduce_leaves_to_roots_end)

                # set leaves to appropriate nil value
                if intent == INC:
                    nil = 0
                elif intent in {MIN_WRITE, MIN_RW}:
                    nil = dtype_limits(buffer.dtype).max
                else:
                    assert intent in {MAX_WRITE, MAX_RW}
                    nil = dtype_limits(buffer.dtype).min

                def _init_nil():
                    buffer._data[buffer.sf.ileaf] = nil

                reductions.append(_init_nil)

            # We are modifying owned values so the leaves must now be wrong
            buffer._leaves_valid = False

            # If ghost points are not modified then no future reduction is required
            if not touches_ghost_points:
                buffer._pending_reduction = None
            else:
                buffer._pending_reduction = intent

        return tuple(initializers), tuple(reductions), tuple(broadcasts)

    @cached_property
    def datamap(self):
        return self.index.datamap | merge_dicts(stmt.datamap for stmt in self.statements)


@utils.frozenrecord()
class InstructionList(Instruction):
    """A list of instructions."""

    # {{{ Instance attrs

    instructions: tuple[Instruction]

    # }}}

    def __init__(self, instructions: Iterable[Instruction]) -> None:
        instructions = tuple(instructions)
        object.__setattr__(self, "instructions", instructions)

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


class Terminal(Instruction, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def arguments(self) -> tuple[Any, ...]:
        pass

    @property
    def buffer_arguments(self) -> tuple[BufferExpression, ...]:
        from pyop3.expr.tensor.dat import BufferExpression

        return tuple(utils.filter_type(BufferExpression, self.arguments))


class NonEmptyTerminal(Terminal, metaclass=abc.ABCMeta):

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


class Function:
    """A callable function."""

    def __init__(self, loopy_kernel, access_descrs):
        lpy_args = loopy_kernel.default_entrypoint.args
        if len(lpy_args) != len(access_descrs):
            raise ValueError("Wrong number of access descriptors given")
        for lpy_arg, access in zip(lpy_args, access_descrs):
            if access in {MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE} and lpy_arg.shape != (
                1,
            ):
                raise ValueError("Reduction operations are only valid for scalars")

        self.code = fix_intents(loopy_kernel, access_descrs)
        self._access_descrs = access_descrs

    def __call__(self, *args):
        if not all(isinstance(a, FunctionArgument) for a in args):
            raise TypeError("invalid kernel argument type")
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


@utils.frozenrecord()
class CalledFunction(AbstractCalledFunction):

    _function: Function
    _arguments: tuple[Any]

    function: ClassVar[property] = property(lambda self: self._function)
    arguments: ClassVar[property] = property(lambda self: self._arguments)

    def __init__(self, function: Function, arguments: Iterable):
        arguments = tuple(arguments)

        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_arguments", arguments)


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
class NullInstruction(Terminal):
    """An instruction that does nothing."""

    arguments = ()


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


class AbstractAssignment(Terminal, metaclass=abc.ABCMeta):

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


# TODO: not sure need to specify 'array' here
@utils.frozenrecord()
class ArrayAssignment(AbstractAssignment):

    # {{{ instance attrs

    _assignee: Any
    _expression: Any
    _assignment_type: AssignmentType

    # }}}

    # {{{ interface impls

    assignee: ClassVar[property] = property(lambda self: self._assignee)
    expression: ClassVar[property] = property(lambda self: self._expression)
    assignment_type: ClassVar[property] = property(lambda self: self._assignment_type)

    # }}}

    def __init__(self, assignee: Any, expression: Any, assignment_type: AssignmentType | str) -> None:
        assignment_type = AssignmentType(assignment_type)

        object.__setattr__(self, "_assignee", assignee)
        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "_assignment_type", assignment_type)


@utils.frozenrecord()
class NonEmptyArrayAssignment(AbstractAssignment, NonEmptyTerminal):

    # {{{ instance attrs

    _assignee: Any
    _expression: Any
    _axis_trees: tuple[AxisTree, ...]
    _assignment_type: AssignmentType

    # }}}

    # {{{ interface impls

    assignee: ClassVar[property] = property(lambda self: self._assignee)
    expression: ClassVar[property] = property(lambda self: self._expression)
    axis_trees: ClassVar[property] = property(lambda self: self._axis_trees)
    assignment_type: ClassVar[property] = property(lambda self: self._assignment_type)

    # }}}


@utils.frozenrecord()
class ConcretizedNonEmptyArrayAssignment(AbstractAssignment):

    # {{{ Instance attrs

    _assignee: Any
    _expression: Any
    _assignment_type: AssignmentType
    _axis_trees: tuple[AxisTree, ...]

    # }}}

    # {{{ Interface impls

    assignee: ClassVar[property] = property(lambda self: self._assignee)
    expression: ClassVar[property] = property(lambda self: self._expression)
    assignment_type: ClassVar[property] = property(lambda self: self._assignment_type)
    axis_trees: ClassVar[property] = property(lambda self: self._axis_trees)

    # }}}


# TODO: With Python 3.11 can be made a StrEnum
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


def _loop(*args, **kwargs):
    """
    Notes
    -----
    This function has a leading underscore to avoid clashing with any variables
    called ``loop``. It is exported as ``op3.loop``.

    """
    return Loop(*args, **kwargs)


# alias
loop_ = _loop


# TODO: better to pass eager kwarg
def do_loop(index, statements, *, compiler_parameters: Mapping | None = None):
    _loop(index, statements)(compiler_parameters=compiler_parameters)


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
