"""Coordinate the execution of instructions."""
from __future__ import annotations

import ctypes
import dataclasses
import functools
import os
import re
from collections.abc import Callable, Hashable, Mapping
from functools import cached_property
from typing import Any

import cachetools
import loopy as lp
import loopy.tools
import numpy as np
import petsctools
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.buffer
import pyop3.cache
import pyop3.cc
import pyop3.collections
import pyop3.config
import pyop3.expr
import pyop3.insn.base
from pyop3 import utils
from pyop3.cache import cached_method, memory_cache
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE


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

    # {{{ compilation options

    # TODO: handle these - need to build CompilerOptions

    backend: str = "loopy"
    """ Option to select 'loopy' or 'mlir' as code generation backends. """
    # extra_cflags: tuple[str, ...] = ()
    # extra_ldflags: tuple[str, ...] = ()

    # }}}

    # {{{ other options

    propagate_negatives: bool = False
    """Whether to propagate negative values in indirections."""

    mask_array_accesses: bool = False
    """Whether to check for and skip expressions like 'dat[-1]'."""

    # }}}



DEFAULT_COMPILER_PARAMETERS = CompilerParameters()


META_COMPILER_PARAMETERS = idict({
    # TODO: when implemented should also set interleave_comp_comm to True
    "optimize": {"compress_indirection_maps": True}
})
"""'Meta' compiler parameters that set multiple options at once."""
# NOTE: These must be boolean options


class ParsedCompilerParameters(CompilerParameters):
    pass


CompilerParametersT = CompilerParameters | Mapping[str, Hashable]


def parse_compiler_parameters(compiler_parameters: CompilerParametersT) -> ParsedCompilerParameters:
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
        assert pyop3.collections.is_ordered_mapping(compiler_parameters)
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


class InstructionExecutionContext:
    """Class that coordinates the compilation and execution of an instruction."""

    def __init__(self, root_insn: Instruction, compiler_parameters) -> None:
        compiler_parameters = parse_compiler_parameters(compiler_parameters)

        self.root_insn = root_insn
        self.compiler_parameters = compiler_parameters

        # Flag for detecting whether or not we hit cache
        # TODO: rename to 'preprocess_called'?
        self._has_called_compile = False
        self._preprocessed = None

    @property
    def comm(self) -> MPI.Comm:
        return self.root_insn.comm

    def __call__(self, **kwargs) -> None:
        executable = self.compile()

        # unpack instruction arguments into buffers, as these are what are
        # actually passed to the compiled code
        new_buffers = {}
        for arg_name, new_arg in kwargs.items():
            orig_arg_buffers = self._argument_name_to_buffer_map[arg_name]
            new_arg_buffers = self._extract_buffers(new_arg)
            for orig_buffer, new_buffer in zip(orig_arg_buffers, new_arg_buffers, strict=True):
                new_buffers[orig_buffer] = new_buffer

        # We shouldn't be calling preprocess() if we are hitting cache, this is
        # an important performance check. Perform the check at the last second
        # to make sure we're not calling it anywhere.
        if not self._has_called_compile:
            assert self._preprocessed is None

        executable(new_buffers)

    @cached_method()
    def compile(self) -> Callable[[int, ...], None]:
        executor, orig_arguments = self._compile()

        if (
            not self._has_called_compile
            and orig_arguments != self.root_insn.global_arguments
        ):
            # If the returned executor is cached from a previous invocation then we
            # have to duplicate it with new buffers. For example consider the expressions:
            #
            #     dat1.assign(2*dat2)
            #     dat3.assign(2*dat4)
            #
            # Assuming that all the dats have the same axis trees then this will hit
            # the code executor cache but we will have to replace the buffers
            # dat1 -> dat3 and dat2 -> dat4.

            # Build the mapping between buffers from arguments. Note that this
            # isn't an exhaustive list of buffers: buffers from axis trees etc
            # are unchanged and will not be replaced.
            arg_buffer_map = {}
            for arg_index, orig_arg in enumerate(orig_arguments):
                new_arg = self.root_insn.global_arguments[arg_index]
                for orig_buf, new_buf in zip(
                    self._extract_buffers(orig_arg),
                    self._extract_buffers(new_arg),
                    strict=True,
                ):
                    arg_buffer_map[orig_buf] = new_buf

            new_kernel_name_to_buffer_info = {}
            for name, buf_view in executor.kernel_name_to_buffer_info.items():
                new_buffer_view = pyop3.buffer.IndexedBuffer(
                    arg_buffer_map.get(buf_view.buffer, buf_view.buffer),
                    buf_view.nest_indices,
                )
                new_kernel_name_to_buffer_info[name] = new_buffer_view
            new_kernel_name_to_buffer_info = idict(new_kernel_name_to_buffer_info)
            new_buffer_intents = idict({
                arg_buffer_map.get(buf, buf): intent
                for buf, intent in executor.buffer_intents.items()
            })

            executor = CompiledCodeExecutor(
                executor.executable,
                new_kernel_name_to_buffer_info,
                new_buffer_intents,
                executor.comm,
            )

        return executor

    # We use an LRU cache here because long running simulations with a
    # timestepping loop can easily miss cache (if assembling into a
    # different tensor each time).
    # @memory_cache(
    #     hashkey=lambda self: self._executor_cache_key,
    #     get_comm=lambda self: self.comm,
    #     heavy=True,
    #     make_cache=lambda: cachetools.LRUCache(100),
    # )
    def _compile(self) -> CompiledCodeExecutor:
        from pyop3.insn.visitors import collect_compiler_options
        from pyop3.compile.core import _compile_static

        # Preprocess the instruction. This is an expensive operation so we
        # want to avoid doing it if at all possible.
        self.preprocess()
        assert not self._has_called_compile
        self._has_called_compile = True

        # A very common and insidious caching bug happens when we incorrectly hit
        # the compile_static cache and then try to load buffers using their index
        # when the number of buffers does not match the initial time we hit cache.
        # To catch this as early as possible we look for the number of unique
        # buffer keys that appear in the disk cache key and compare to the buffers
        # that we actually have.
        # TODO: make this check conditional
        # if pyop3.config.config.debug_checks:
        #     ...
        num_buffers = 0
        cache_key_str = str(self.disk_cache_key)
        array_pattern = \
            r"\(<class 'pyop3.buffer.ArrayBuffer'>, dtype\('\S+'\), 'ArrayBuffer_\d+', \w+, \w+, \w+\)"
        petscmat_pattern = r"\(<class 'pyop3.buffer.PetscMatBuffer'>, 'PetscMatBuffer_\d+', \w+\)"
        for pattern in [array_pattern, petscmat_pattern]:
            num_buffers += len(utils.unique(re.findall(pattern, cache_key_str)))
        assert num_buffers == len(self.preprocessed_buffers)

        compiler_parameters = parse_compiler_parameters(self.compiler_parameters)
        loopy_code, kernel_name_to_buffer_info, buffer_intents = \
            _compile_static(self, compiler_parameters)

        extra_compiler_options = collect_compiler_options(self._preprocessed)

        if compiler_parameters.add_petsc_event:
            petsc_events = (loopy_code.default_entrypoint.name,)
        else:
            petsc_events = ()

        executable = Executable(
            loopy_code,
            self.comm,
            extra_compiler_options=extra_compiler_options,
            petsc_events=petsc_events,
        )

        # replace buffer indices with the real things
        kernel_name_to_buffer_views = {}
        for name, (buf_index, nest_idxs) in kernel_name_to_buffer_info.items():
            kernel_name_to_buffer_views[name] = pyop3.buffer.IndexedBuffer(
                self.preprocessed_buffers[buf_index], nest_idxs
            )
        kernel_name_to_buffer_views = idict(kernel_name_to_buffer_views)
        buffer_intents = idict({
            self.preprocessed_buffers[buffer_index]: intent
            for buffer_index, intent in buffer_intents.items()
        })
        executor = CompiledCodeExecutor(executable, kernel_name_to_buffer_views, buffer_intents, self.comm)

        return executor, self.root_insn.global_arguments

    def preprocess(self) -> Instruction:
        import pyop3.visitors

        from .visitors import (
            expand_loop_contexts,
            expand_transforms,
            insert_literals,
        )

        if self._preprocessed is None:
            insn = self.root_insn
            insn = expand_loop_contexts(insn)

            # bad name, this expands all transformations and pack/unpacks for called functions
            # 'flatten?'
            # Since the expansion can add new nodes requiring parsing we do a fixed point iteration
            old_insn = insn
            insn = expand_transforms(insn)
            while insn != old_insn:
                old_insn = insn
                insn = expand_transforms(insn)

            insn = pyop3.visitors.concretize(insn)
            insn = insert_literals(insn)
            insn = pyop3.visitors.materialize_indirections(insn, compress=self.compiler_parameters.compress_indirection_maps)

            self._preprocessed = insn

        return self._preprocessed

    @cached_property
    def preprocessed_buffers(self) -> OrderedFrozenSet:
        """Data structures that are arguments to the compiled code."""
        from pyop3.visitors import collect_buffers

        assert self._preprocessed is not None
        return collect_buffers(self._preprocessed)

    @cached_property
    def disk_cache_key(self) -> Hashable:
        """Key used to write the operation to disk.

        The returned key should be consistent across ranks and not include
        overly specific information such as buffer names or array values.

        """
        from pyop3.visitors import get_disk_cache_key

        assert self._preprocessed is not None
        return get_disk_cache_key(self._preprocessed)

    @cached_property
    def _argument_name_to_buffer_map(self) -> dict[str, tuple[AbstractBuffer, ...]]:
        # This attribute is only used for argument replacement.
        # We don't want to get conflicts if we pass in two tensors with the same name
        name_to_buffer_map = {}
        names_to_skip = set()
        for arg in self.root_insn.global_arguments:
            if arg.name in names_to_skip:
                continue

            arg_buffers = self._extract_buffers(arg)
            if arg.name in name_to_buffer_map:
                if name_to_buffer_map[arg.name] != arg_buffers:
                    # found a duplicate arg name but with different buffers,
                    # replacement by name isn't valid
                    del name_to_buffer_map[arg.name]
                    names_to_skip.add(arg.name)
            else:
                name_to_buffer_map[arg.name] = arg_buffers
        return name_to_buffer_map

    @cached_property
    def _executor_cache_key(self) -> Hashable:
        from pyop3.visitors import get_instruction_executor_cache_key

        return get_instruction_executor_cache_key(self.root_insn)

    @functools.singledispatchmethod
    def _extract_buffers(self, arg: Any, /) -> tuple[pyop3.buffer.AbstractBuffer, ...]:
        utils.raise_missing_dispatch_handler(arg)

    @_extract_buffers.register(pyop3.expr.OpaqueTerminal)
    def _(self, expr: Any, /) -> tuple[pyop3.buffer.AbstractBuffer, ...]:
        return (expr.buffer,)

    @_extract_buffers.register(pyop3.expr.ScalarBufferExpression)
    @_extract_buffers.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, expr: Any, /) -> tuple[pyop3.buffer.AbstractBuffer, ...]:
        if expr.buffer_view.nest_indices != ():
            raise NotImplementedError(
                "Extracting nested buffers that aren't PETSc MATNESTS not yet supported"
            )
        return (expr.buffer_view.buffer,)

    @_extract_buffers.register(pyop3.expr.Scalar)
    @_extract_buffers.register(pyop3.expr.Dat)
    @_extract_buffers.register(pyop3.expr.Mat)
    def _(self, expr: Any, /) -> tuple[pyop3.buffer.AbstractBuffer, ...]:
        return (expr.buffer,)

    @_extract_buffers.register
    def _(self, agg_dat: pyop3.expr.AggregateDat, /) -> tuple[pyop3.buffer.AbstractBuffer, ...]:
        return tuple(buf for subdat in agg_dat.subdats for buf in self._extract_buffers(subdat))

    @_extract_buffers.register
    def _(self, agg_mat: pyop3.expr.AggregateMat, /) -> tuple[pyop3.buffer.AbstractBuffer, ...]:
        return tuple(buf for submat in agg_mat.submats.flatten() for buf in self._extract_buffers(submat))


@dataclasses.dataclass(frozen=True)
class Executable:
    """A callable function.

    Parameters
    ----------
    code:
        The computation to be performed.
    comm
        The communicator.

    Notes
    -----
    This class is intentionally distinct from `CompiledCodeExecutor` because
    the executable may be reused by multiple executors (for instance if the
    buffers are changed) and we want to reuse the work needed to generate
    the function pointer.

    """
    code: lp.TranslationUnit
    comm: MPI.Comm
    extra_compiler_options: pyop3.cc.CompilerOptions = dataclasses.field(
        default=pyop3.cc.CompilerOptions(), kw_only=True
    )
    petsc_events: tuple[str, ...] = dataclasses.field(default=(), kw_only=True)

    def __call__(self, *args: int) -> None:
        self._callable(*args)

    @cached_property
    def _callable(self) -> collections.abc.Callable[[int, ...], None]:
        """Compile the code and return a function pointer."""
        # ideally move this logic somewhere else
        cppargs = (
            *petsctools.get_petsc_dirs(prefix="-I", subdir="include"),
            *(f"-I{incdir}" for incdir in self.extra_compiler_options.include_dirs),
        )
        ldargs = (
            *petsctools.get_petsc_dirs(prefix="-L", subdir="lib"),
            *petsctools.get_petsc_dirs(prefix="-Wl,-rpath,", subdir="lib"),
            "-lpetsc",
            "-lm",
            *(f"-L{libdir}" for libdir in self.extra_compiler_options.lib_dirs),
            *(f"-l{lib}" for lib in self.extra_compiler_options.libs),
        )

        # NOTE: no - instead of this inspect the compiler parameters!!!
        # TODO: Make some sort of function in config.py
        if "LIKWID_MODE" in os.environ:
            cppargs += ("-DLIKWID_PERFMON",)
            ldargs += ("-llikwid",)

        dll = pyop3.cc.load(self._device_code, "c", cppargs, ldargs, comm=self.comm)

        for event in self.petsc_events:
            # Create the event in python and then set in the shared library to avoid
            # allocating memory over and over again in the C kernel.
            ctypes.c_int.in_dll(dll, f"id_{event}").value = PETSc.Log.Event(event).id

        func = getattr(dll, self.code.default_entrypoint.name)
        func.argtypes = [
            cast_loopy_arg_to_ctypes_type(arg) for arg in self.code.default_entrypoint.args
        ]
        func.restype = None
        return func

    @cached_property
    def _device_code(self):
        return _loopy_to_c_string(self.code, self.comm)


class CompiledCodeExecutor:
    """Class that executes compiled code.

    Parameters
    ----------
    executable
        The compiled operation.
    kernel_name_to_buffer_info
        Mapping from local kernel argument names to the buffer and nest indices.
    buffer_intents
        Mapping from buffers to their intents.

    Notes
    -----
    This class has a large number of cached properties to reduce overhead when it
    is called.

    This class is basically executable+buffers. It is useful because we want to cache the executable globally
    but we don't want to cache this globally because the buffers are likely to change.

    """

    def __init__(
        self,
        executable: Executable,
        kernel_name_to_buffer_info,
        buffer_intents,
        comm: MPI.Comm,
    ):
        self.executable = executable
        self.kernel_name_to_buffer_info = kernel_name_to_buffer_info
        self.buffer_intents = buffer_intents
        self.comm = comm

    # @cached_property
    # def _buffer_global_id_to_name_in_kernel_map(self):
    #     return {buffer: name_in_kernel for name_in_kernel, (buffer, _) in self.buffer_map.items()}
    #
    @cached_property
    def _default_buffers(self):
        return tuple(self.buffer_intents.keys())

    def __call__(self, new_buffers: Mapping[ConcreteBuffer, ConcreteBuffer]) -> None:
        """
        Notes
        -----
        This code is performance critical.

        """
        # if "MatSetValues" in str(self):
        # #     breakpoint()
        #     import pyop3.debug
        #     pyop3.debug.maybe_breakpoint()

        if not new_buffers:  # shortcut for the most common case
            buffer_intents = self.buffer_intents
            exec_arguments = self._default_exec_arguments
        else:
            buffer_intents = {
                new_buffers.get(orig_buffer, orig_buffer): intent
                for orig_buffer, intent in self.buffer_intents.items()
            }
            exec_arguments = [
                self._handle_to_pointer(
                    buf_view.record_new(
                        buffer=new_buffers.get(buf_view.buffer, buf_view.buffer)
                    ).handle
                )
                for buf_view in self.kernel_name_to_buffer_info.values()
            ]

        utils.debug_assert(
            lambda: all(arg is not None for arg in exec_arguments),
            "Attempting to pass a null pointer to the executable",
        )

        # TODO
        # if self.compiler_parameters.interleave_comp_comm:
        if False:
            raise NotImplementedError
            # new_index, (icore, iroot, ileaf) = partition_iterset(
            #     self.index, [a for a, _ in self.function_arguments]
            # )
            # #buffer_intents
            # # assert self.index.id == new_index.id
            # #
            # # # substitute subsets into loopexpr, should maybe be done in partition_iterset
            # # parallel_loop = self.copy(index=new_index)
            #
            # for init in initializers:
            #     init()
            #
            # # replace the parallel axis subset with one for the specific indices here
            # extent = utils.just_one(icore.axes.root.components).count
            # core_kwargs = utils.merge_dicts(
            #     [kwargs, {icore.name: icore, extent.name: extent}]
            # )
            #
            # with PETSc.Log.Event(f"compute_{self.name}_core"):
            #     code(**core_kwargs)
            #
            # # await reductions
            # for red in reductions:
            #     red()
            #
            # # roots
            # # replace the parallel axis subset with one for the specific indices here
            # root_extent = utils.just_one(iroot.axes.root.components).count
            # root_kwargs = utils.merge_dicts(
            #     [kwargs, {icore.name: iroot, extent.name: root_extent}]
            # )
            # with PETSc.Log.Event(f"compute_{self.name}_root"):
            #     code(**root_kwargs)
            #
            # # await broadcasts
            # for broadcast in broadcasts:
            #     broadcast()
            #
            # # leaves
            # leaf_extent = utils.just_one(ileaf.axes.root.components).count
            # leaf_kwargs = utils.merge_dicts(
            #     [kwargs, {icore.name: ileaf, extent.name: leaf_extent}]
            # )
            # with PETSc.Log.Event(f"compute_{self.name}_leaf"):
            #     code(**leaf_kwargs)

            # This is a bit of a misnomer - the idea here is that for data to be ready to compute we
            # must first update all roots and then update all leaves from these roots.
            # Recall that points on a rank may be partitioned into 'core', 'root' and 'leaf' where a
            # 'leaf' is a point owned by another process, 'root' is a point that exists as a ghost on
            # another process, and 'core' are the rest.
            # * It is valid to compute on parts of the iteration set that only touch 'core' points
            # before any communication takes place
            # * it is valid to compute on parts that touch core and root once all roots have been
            # updated via reductions
            # * you can only compute using leaf values once these have been updated

        initializers = []
        reductions = []
        broadcasts = []
        finalizers = []
        for buffer, intent in buffer_intents.items():
            inits, reds, bcasts, fins = self._buffer_exchanges(buffer, intent)
            initializers.extend(inits)
            reductions.extend(reds)
            broadcasts.extend(bcasts)
            finalizers.extend(fins)

        # Unoptimised case: perform all transfers eagerly
        for init in initializers:
            init()
        for red in reductions:
            red()
        for bcast in broadcasts:
            bcast()

        # Now all the data is correct, compute!
        self.executable(*exec_arguments)

        # if "MatSetValues" in str(self) and "form" in str(self):
        #     buf = list(self.buffer_intents.keys())[0]
        #     breakpoint()

        for fin in finalizers:
            fin()

    def __str__(self) -> str:
        sep = "*" * 80
        str_ = []
        str_.append(sep)
        str_.append(self.executable._device_code)
        str_.append(sep)

        for arg in self.executable.code.default_entrypoint.args:
            size, buffer = self._buffer_str(self.kernel_name_to_buffer_info[arg.name].buffer)
            str_.append(f"{arg.name} {size} : {buffer}")

        str_.append(sep)
        return "\n".join(str_)

    @functools.singledispatchmethod
    def _buffer_str(self, buffer):
        utils.raise_missing_dispatch_handler(arg)

    @_buffer_str.register
    def _(self, buffer: pyop3.buffer.ArrayBuffer):
        return f"({buffer.size})", str(buffer._current_device_array)

    @_buffer_str.register
    def _(self, buffer: pyop3.buffer.PetscMatBuffer) -> str:
        return "", "<PetscMat>"

    @cached_property
    def _default_exec_arguments(self) -> tuple[int]:
        return tuple(
            self._handle_to_pointer(buffer_view.handle)
            for buffer_view in self.kernel_name_to_buffer_info.values()
        )

    @functools.singledispatchmethod
    def _handle_to_pointer(self, handle: Any, /) -> int:
        utils.raise_missing_dispatch_handler(handle)

    # not used because we pass the handle in already
    # @_as_exec_argument.register
    # def _(self, opaque: pyop3.expr.OpaqueTerminal):
    #     return opaque.handle

    @_handle_to_pointer.register
    def _(self, arr: np.ndarray, /) -> int:
        return arr.ctypes.data

    try:
        import cupy as cp

        @_handle_to_pointer.register(cp.ndarray)
        def _(self, arr: cp.ndarray, /) -> int:
            # NOTE: This gives a pointer to a GPU memory address.
            # Loopy cannot work with GPU so this will lead to a segfault. 
            raise MemoryError("Segfault will occur if you pass a CuPy GPU pointer to Loopy/C code")
    except ImportError:
        pass

    @_handle_to_pointer.register
    def _(self, mat: PETSc.Mat, /) -> int:
        # Sometime the matrix is in an invalid state and we cannot return a handle.
        # This happens for example when reusing a loop that initially used a
        # preallocator matrix. Once used the preallocator matrix is no longer in a
        # valid state. This is generally fine though because when we compute things
        # we replace this matrix with a fully allocated one. We therefore pass a
        # None here and check things later.
        if not mat:
            assert False, "old code"
        #     return None

        assert mat.type != PETSc.Mat.Type.PYTHON, \
            "Python-type mats should be unpacked by now"

        return mat.handle

    # NOTE: This is probably very slow to have to do every time - a lot of this can be cached
    # the rest (initial state) can be checked each time
    @functools.singledispatchmethod
    def _buffer_exchanges(self, buffer: pyop3.buffer.ConcreteBuffer, intent):
        utils.raise_missing_dispatch_handler(buffer)

    @_buffer_exchanges.register
    def _(self, buffer: pyop3.buffer.ArrayBuffer, intent):
        initializers, reductions, bcasts, finalizers = [], [], [], []

        # Possibly instead of touches_ghost_points we could produce custom SFs for each loop
        # (we have filter_star_forest())
        # For now we just disregard the optimisation
        touches_ghost_points = True

        if intent in {READ, RW}:
            if touches_ghost_points:
                if not buffer._roots_valid:
                    initializers.append(buffer.sync_roots_begin)
                    reductions.extend([
                        buffer.sync_roots_end,
                        buffer.sync_leaves_begin,
                    ])
                    bcasts.append(buffer.sync_leaves_end)
                elif not buffer._leaves_valid:
                    initializers.append(buffer.sync_leaves_begin)
                    bcasts.append(buffer.sync_leaves_end)
                else:
                    pass
            else:
                if not buffer._roots_valid:
                    initializers.append(buffer.sync_roots_begin)
                    reductions.append(buffer.sync_roots_end)

        elif intent == WRITE:
            # Assumes that all points are written to (i.e. not a subset). If
            # this is not the case then a manual reduction is needed.
            initializers.append(lambda: setattr(buffer, "_pending_reduction", None))
            finalizers.append(lambda: setattr(buffer, "_leaves_valid", False))

        else:
            # reductions
            assert intent in {INC, MIN_WRITE, MIN_RW, MAX_WRITE, MAX_RW}
            # We don't need to update roots if performing the same reduction
            # again. For example we can increment into a buffer as many times
            # as we want. The reduction only needs to be done when the
            # data is read.
            if buffer._pending_reduction == intent:
                pass
            else:
                # We assume that all points are visited, and therefore that
                # WRITE accesses do not need to update roots. If only a subset
                # of entities are written to then a manual reduction is required.
                # This is the same assumption that we make for data_wo.
                if not buffer._roots_valid and intent in {INC, MIN_RW, MAX_RW}:
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
                    # Not modifying owned values so don't want to update state via intent
                    np.ravel(buffer._current_device_array)[buffer.sf.ileaf] = nil

                reductions.append(_init_nil)

            # We are modifying owned values so the leaves must now be wrong
            finalizers.append(lambda: setattr(buffer, "_leaves_valid", False))

            # If ghost points are not modified then no future reduction is required
            if not touches_ghost_points:
                finalizers.append(lambda: setattr(buffer, "_pending_reduction", None))
            else:
                finalizers.append(lambda: setattr(buffer, "_pending_reduction", intent))

        if intent != READ:
            finalizers.append(lambda: buffer.inc_state())

        return initializers, reductions, bcasts, finalizers

    @_buffer_exchanges.register
    def _(self, buffer: pyop3.buffer.PetscMatBuffer, intent):
        begin_insns = []
        end_insns = []
        finalizers = []
        if intent == READ:
            begin_insns.append(lambda: buffer.assemble_begin(final=True))
            end_insns.append(lambda: buffer.assemble_end(final=True))
        else:
            if intent == WRITE:
                insert_mode = PETSc.InsertMode.INSERT_VALUES
            else:
                assert intent == INC
                insert_mode = PETSc.InsertMode.ADD_VALUES
            begin_insns.append(lambda: buffer.maybe_flush_assemble_begin(insert_mode))
            end_insns.append(lambda: buffer.maybe_flush_assemble_end(insert_mode))

        # Manually increment the state here instead of deferring to PETSc
        # because state tracking needs to happen collectively
        if intent != READ:
            finalizers.append(lambda: buffer.inc_state())

        # TODO: We need all communication to happen before we begin computing, but if
        # we have multiple matrices we can at least overlap their communication.
        return begin_insns+end_insns, (), (), finalizers


@functools.singledispatch
def cast_loopy_arg_to_ctypes_type(obj: Any) -> type:
    utils.raise_missing_dispatch_handler(obj)


@cast_loopy_arg_to_ctypes_type.register(lp.ArrayArg)
def _(arg: lp.ArrayArg) -> type:
    return ctypes.c_voidp


@cast_loopy_arg_to_ctypes_type.register(lp.ValueArg)
def _(arg: lp.ValueArg):
    if isinstance(arg.dtype, pyop3.dtypes.OpaqueType):
        return ctypes.c_voidp
    else:
        return np.ctypeslib.as_ctypes_type(arg.dtype)


# TODO: This should probably get folded into '_compile_static', otherwise we
# have to get the translation unit from cache, hash it, then get the thing
# we actually want from the cache.
@pyop3.cache.memory_cache(hashkey=lambda tu, _: utils._loopy_key_builder(tu))
@pyop3.cache.disk_only_cache(hashkey=lambda tu, _: utils._loopy_key_builder(tu), bcast=True)
def _loopy_to_c_string(tu: lp.TranslationUnit, comm: MPI.Comm) -> str:
    return lp.generate_code_v2(tu).device_code()
