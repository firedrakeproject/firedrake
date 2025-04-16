# TODO Rename this file insn.py - the pyop3 language is everything, not just this

from __future__ import annotations

import abc
import collections
import dataclasses
import enum
import functools
import numbers
import textwrap
from functools import cached_property
from typing import Iterable, Tuple

import immutabledict
import loopy as lp
import numpy as np
import pytools
from cachetools import cachedmethod
from petsc4py import PETSc
from pyrsistent import PMap, pmap

from pyop3.axtree import AxisTree
from pyop3.axtree.tree import ContextFree, ContextSensitive
from pyop3.config import config
from pyop3.dtypes import dtype_limits
from pyop3.exceptions import Pyop3Exception
from pyop3.utils import (
    UniqueRecord,
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


META_COMPILER_PARAMETERS = immutabledict.ImmutableOrderedDict({
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


class Instruction(UniqueRecord, abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache = collections.defaultdict(dict)

    def __call__(self, *, compiler_parameters=None, **kwargs):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)

        executable = self.compile(compiler_parameters)
        executable(**kwargs)

    def preprocess(self, compiler_parameters=None):
        compiler_parameters = parse_compiler_parameters(compiler_parameters)
        return self._preprocess(compiler_parameters)

    @cachedmethod(lambda self: self._cache["Instruction._preprocess"])
    def _preprocess(self, compiler_parameters: ParsedCompilerParameters):
        from pyop3.insn_visitors import (
            expand_implicit_pack_unpack,
            expand_loop_contexts,
            expand_assignments,
            prepare_petsc_calls,
            compress_indirection_maps,
            concretize_arrays,
            drop_zero_sized_paths,
        )

        insn = self

        if isinstance(insn, NullInstruction):
            raise NotImplementedError("crash gracefully, nothing to do")

        insn = expand_loop_contexts(insn)
        insn = expand_implicit_pack_unpack(insn)

        insn = expand_assignments(insn)  # specifically reshape bits

        # do this as early as possible because we don't like dealing with mats
        insn = prepare_petsc_calls(insn)

        insn = drop_zero_sized_paths(insn)

        if compiler_parameters.compress_indirection_maps:
            insn = compress_indirection_maps(insn)

        insn = concretize_arrays(insn)

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


class Loop(Instruction):
    fields = Instruction.fields | {"index", "statements", "name"}

    # doubt that I need an ID here
    id_generator = pytools.UniqueNameGenerator()

    def __init__(
        self,
        index: LoopIndex,
        statements: Iterable[Instruction],
        *,
        name: str = _DEFAULT_LOOP_NAME,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index = index
        self.statements = as_tuple(statements)
        self.name = name

    def __str__(self) -> str:
        stmt_strs = [textwrap.indent(str(stmt), "    ") for stmt in self.statements]
        return f"""loop(
  {self.index},
  [
{'\n'.join(stmt_strs)}
  ]
)"""

    def __call__(self, *, compiler_parameters=None, **kwargs):
        # TODO just parse into ContextAwareLoop and call that
        from pyop3.ir.lower import compile
        from pyop3.itree.tree import partition_iterset
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
            #
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
            code(**kwargs)

    @property
    def comm(self):
        # maybe collect the comm by looking at everything?
        return self.index.iterset.comm

    @cached_property
    def datamap(self) -> PMap:
        return self.index.datamap | merge_dicts(s.datamap for s in self.statements)

    @cached_property
    def is_parallel(self):
        from pyop3.buffer import ArrayBuffer

        for arg in self.kernel_arguments:
            if isinstance(arg, ArrayBuffer):
                # if arg.is_distributed:
                if arg.comm.size > 1:
                    return True
            else:
                assert isinstance(arg, PETSc.Mat)
                for local_size, global_size in arg.getSizes():
                    if local_size != global_size:
                        return True
        return False
        return len(self._distarray_args) > 0

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
        from pyop3.array import Dat, Mat
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


# get rid of this
class ContextAwareLoop(ContextAwareInstruction):
    fields = Instruction.fields | {"index", "statements"}

    def __init__(self, index, statements, **kwargs):
        assert False, "dead code"
        super().__init__(**kwargs)
        self.index = index
        self.statements = statements

    @cached_property
    def datamap(self):
        return self.index.datamap | merge_dicts(
            stmt.datamap for stmts in self.statements.values() for stmt in stmts
        )


class InstructionList(Instruction):
    """
    A list of instructions.
    """
    fields = Instruction.fields | {"instructions"}

    def __init__(self, instructions, *, name=_DEFAULT_LOOP_NAME, **kwargs):
        super().__init__(**kwargs)
        self.instructions = tuple(instructions)
        self.name = name

    def __iter__(self):
        return iter(self.instructions)

    def __str__(self) -> str:
        return "\n".join(map(str, self.instructions))

    @cached_property
    def datamap(self):
        return merge_dicts(insn.datamap for insn in self.instructions)

    @property
    @deprecated("instructions")
    def loops(self):
        return self.instructions


def enlist(insn: Instruction) -> InstructionList:
    if isinstance(insn, InstructionList):
        return insn
    elif isinstance(insn, NullInstruction):
        return InstructionList(())
    else:
        return InstructionList([insn])


def maybe_enlist(instructions) -> Instruction:
    flattened_insns = []
    for insn in instructions:
        if isinstance(insn, InstructionList):
            flattened_insns.extend(insn.instructions)
        else:
            flattened_insns.append(insn)

    if len(flattened_insns) > 1:
        return InstructionList(flattened_insns)
    else:
        return just_one(flattened_insns)


def non_null(instruction: Instruction) -> bool:
    return not isinstance(instruction, NullInstruction)


# TODO singledispatch
# TODO perhaps this is simply "has non unit stride"?
def _has_nontrivial_stencil(array):
    """

    This is a proxy for 'this array touches halo points'.

    """
    # FIXME This is WRONG, there are cases (e.g. support(extfacet)) where
    # the halo might be touched but the size (i.e. map arity) is 1. I need
    # to look at index_exprs probably.
    from pyop3.array import Dat

    if isinstance(array, Dat):
        return _has_nontrivial_stencil(array)
    else:
        raise TypeError


class Terminal(Instruction, abc.ABC):
    # @cached_property
    # def datamap(self):
    #     return merge_dicts(a.datamap for a, _ in self.function_arguments)
    #
    # @property
    # @abc.abstractmethod
    # def argument_shapes(self):
    #     pass
    #
    # @abc.abstractmethod
    # def with_arguments(self, arguments: Iterable[KernelArgument]):
    #     pass
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


class AbstractCalledFunction(Terminal, metaclass=abc.ABCMeta):
    def __init__(
        self, function: Function, arguments: Iterable[FunctionArgument], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.function = function
        self.arguments = arguments

    def __str__(self) -> str:
        return f"{self.name}({', '.join(arg.name for arg in self.arguments)})"

    @property
    def name(self):
        return self.function.name

    fields = Terminal.fields | {"function", "arguments"}

    @property
    def argspec(self):
        return self.function.argspec

    @cached_property
    def function_arguments(self):
        return tuple((arg, spec.intent) for arg, spec in zip(self.arguments, self.argspec, strict=True))

    @cached_property
    def kernel_arguments(self):
        kargs = OrderedSet()
        for func_arg in self.arguments:
            for karg in _collect_kernel_arguments(func_arg):
                kargs.add(karg)
        return tuple(kargs)

    @property
    def argument_shapes(self):
        return tuple(
            arg.shape if not isinstance(arg, lp.ValueArg) else ()
            for arg in self.function.code.default_entrypoint.args
        )




class CalledFunction(AbstractCalledFunction):
    def with_arguments(self, arguments):
        return self.copy(arguments=arguments)


class ExplicitCalledFunction(AbstractCalledFunction):
    """A `CalledFunction` whose arguments do not need packing/unpacking."""


# TODO: Make this a singleton like UNIT_AXIS_TREE
class NullInstruction(Terminal):
    """An instruction that does nothing."""


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


class AbstractAssignment(Terminal):
    def __init__(self, assignee, expression, assignment_type, **kwargs):
        assignment_type = AssignmentType(assignment_type)

        super().__init__(**kwargs)
        self.assignee = assignee
        self.expression = expression
        self.assignment_type = assignment_type

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


class AbstractBufferAssignment(AbstractAssignment, metaclass=abc.ABCMeta):
    pass


class BufferAssignment(AbstractBufferAssignment):
    fields = Terminal.fields | {"assignee", "expression", "assignment_type"}

    # not really important any more
    name = "pyop3_assignment"

    # def __init__(self, assignee, *args, **kwargs):
    #     if assignee.name == "t_5": # deebug
    #         breakpoint()
    #     super().__init__(assignee, *args, **kwargs)

    @property
    def arguments(self):
        # FIXME Not sure this is right for complicated expressions
        return (self.assignee, self.expression)

    @property
    def arrays(self):
        from pyop3.array import Dat

        arrays_ = [self.assignee]
        if isinstance(self.expression, Dat):
            arrays_.append(self.expression)
        else:
            if not isinstance(self.expression, numbers.Number):
                raise NotImplementedError
        return tuple(arrays_)

    @property
    def argument_shapes(self):
        return (None,) * len(self.kernel_arguments)

    def with_arguments(self, arguments):
        if len(arguments) != 2:
            raise ValueError("Must provide 2 arguments")

        assignee, expression = arguments
        return self.copy(assignee=assignee, expression=expression)

    @property
    def _expression_kernel_arguments(self):
        from pyop3.array import Dat

        if isinstance(self.expression, Dat):
            return ((self.expression, READ),)
        elif isinstance(self.expression, numbers.Number):
            return ()
        else:
            raise NotImplementedError("Complicated rvalues not yet supported")

    @property
    def kernel_arguments(self):
        from pyop3.array import Dat, Mat

        args = OrderedSet()
        for array, _ in self.function_arguments:
            if isinstance(array, Dat):
                args.add(array.buffer)
            elif isinstance(array, Mat):
                args.add(array.mat)
        return tuple(args)

    def with_axes(self, axes: AxisTree) -> NonEmptyAssignmentMixin:
        return NonEmptyBufferAssignment(self.assignee, self.expression, self.assignment_type, axes)


class AbstractPetscMatAssignment(AbstractAssignment, metaclass=abc.ABCMeta):
    def __init__(self, mat, values, access_type):
        if access_type == ArrayAccessType.READ:
            assignment_type = AssignmentType.WRITE
            assignee = values
            expression = mat
        elif access_type == ArrayAccessType.WRITE:
            assignee = mat
            expression = values
            assignment_type = AssignmentType.WRITE
        else:
            assert access_type == ArrayAccessType.INC
            assignee = mat
            expression = values
            assignment_type = AssignmentType.INC

        super().__init__(assignee, expression, assignment_type)
        self.mat = mat
        self.values = values
        self.access_type = access_type


class PetscMatAssignment(AbstractPetscMatAssignment):
    def with_axes(self, row_axis_tree, col_axis_tree):
        return NonEmptyPetscMatAssignment(self.mat, self.values, self.access_type, row_axis_tree, col_axis_tree)


# NOTE: These are internal classes, can be moved elsewhere
class NonEmptyAssignmentMixin(abc.ABC):
    @property
    @abc.abstractmethod
    def axis_trees(self) -> AxisTree:
        pass


class NonEmptyBufferAssignment(AbstractBufferAssignment, NonEmptyAssignmentMixin):
    fields = Terminal.fields | {"assignee", "expression", "assignment_type", "axis_trees"}

    def __init__(self, assignee, expression, assignment_type, axis_trees, **kwargs):
        super().__init__(assignee, expression, assignment_type, **kwargs)
        self._axis_trees = tuple(axis_trees)

    @property
    def axis_trees(self) -> tuple[AxisTree]:
        return self._axis_trees


class NonEmptyPetscMatAssignment(AbstractPetscMatAssignment, NonEmptyAssignmentMixin):
    def __init__(self, mat, values, access_type, row_axis_tree, col_axis_tree, **kwargs):
        super().__init__(mat, values, access_type, **kwargs)
        # self._axis_trees = (row_axes, col_axes)
        self.row_axis_tree = row_axis_tree
        self.col_axis_tree = col_axis_tree

    @property
    def axis_trees(self) -> tuple[AxisTree, AxisTree]:
        return self._axis_trees


# TODO: With Python 3.11 can be made a StrEnum
class ArrayAccessType(enum.Enum):
    READ = "read"
    WRITE = "write"
    INC = "inc"


# class PetscMatAccess(AbstractAssignment):
#     fields = AbstractAssignment.fields | {"mat_arg", "array_arg", "access_type"}
#
#     def __init__(self, mat_arg, array_arg, access_type):
#         from pyop3.array import Dat
#
#         access_type = PetscMatAccessType(access_type)
#
#         if isinstance(array_arg, numbers.Number):
#             array_arg = Dat(
#                 mat_arg.axes,
#                 data=np.full(mat_arg.axes.size, array_arg, dtype=mat_arg.dtype),
#                 prefix="t",
#                 constant=True,
#             )
#
#
#         assert mat_arg.dtype == array_arg.dtype
#
#         self.mat_arg = mat_arg
#         self.array_arg = array_arg
#         self.access_type = access_type
#
#     @property
#     def kernel_arguments(self):
#         args = (self.mat_arg.mat,)
#         if isinstance(self.array_arg, ContextSensitive):
#             args += tuple(dat.buffer for dat in self.array_arg.context_map.values())
#         else:
#             args += (self.array_arg.buffer,)
#         return args


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


@functools.singledispatch
def _collect_kernel_arguments(func_arg: FunctionArgument) -> tuple:
    from pyop3.array import Dat, Mat
    from pyop3.buffer import ArrayBuffer, NullBuffer

    if isinstance(func_arg, Dat):
        return _collect_kernel_arguments(func_arg.buffer)
    elif isinstance(func_arg, Mat):
        return _collect_kernel_arguments(func_arg.mat)
    elif isinstance(func_arg, ArrayBuffer):
        return (func_arg,)
    elif isinstance(func_arg, NullBuffer):
        return ()
    else:
        raise TypeError(f"No handler defined for {type(func_arg).__name__}")


@_collect_kernel_arguments.register
def _(mat: PETSc.Mat) -> tuple:
    return (mat,)
