"""Coordinate the execution of instructions."""
from __future__ import annotations

import ctypes
import dataclasses
import functools
import os
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Callable, Hashable

import loopy as lp
import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import petsctools

import pyop3.buffer
import pyop3.collections
import pyop3.compile
import pyop3.expr
import pyop3.insn.base
import pyop3.pyop2_utils
from pyop3.cache import cached_method, memory_cache
from pyop3.insn.visitors import canonicalize_labels


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

    # {{{ other options

    check_negatives: bool = False
    """Whether to propagate negative values in indirections."""

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
        root_insn = canonicalize_labels(root_insn)
        compiler_parameters = parse_compiler_parameters(compiler_parameters)

        self.root_insn = root_insn
        self.compiler_parameters = compiler_parameters

        # Flag for detecting whether or not we hit cache
        self._has_called_compile = False

    @property
    def comm(self) -> MPI.Comm:
        return self.root_insn.comm

    def __call__(self, **kwargs) -> None:
        executable = self.compile()

        new_buffers = {}
        if kwargs:
            for arg_name, new_arg in kwargs.items():
                buffer_name = self._argument_name_to_buffer_name_map[arg_name]
                buffer = self._extract_buffer(new_arg)
                new_buffers[buffer_name] = buffer

        executable(**new_buffers)

    @cached_property
    def _preprocess(self) -> Instruction:
        from .visitors import (
            expand_implicit_pack_unpack,
            expand_loop_contexts,
            expand_transforms,
            materialize_indirections,
            concretize_layouts,
            insert_literals,
        )

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

        insn = concretize_layouts(insn)
        insn = insert_literals(insn)
        insn = materialize_indirections(insn, compress=self.compiler_parameters.compress_indirection_maps)

        return insn

    @cached_method()
    def compile(self) -> Callable[[int, ...], None]:
        executor, argument_index_to_buffer_name_map = self._compile()

        # If the returned executor is cached from a previous invocation then we
        # have to duplicate it with new buffers. For example consider the expressions:
        #
        #     dat1.assign(2*dat2)
        #     dat3.assign(2*dat4)
        #
        # Assuming that all the dats have the same axis trees then this will hit
        # the code executor cache but we will have to replace the buffers 
        # dat1 -> dat3 and dat2 -> dat4.
        if not self._has_called_compile:
            new_buffer_map = dict(executor.buffer_map)
            for arg_index, buffer_name in argument_index_to_buffer_name_map.items():
                buffer_name_in_kernel = executor._buffer_global_name_to_name_in_kernel_map[buffer_name]
                # TODO: ick behaviour with buffer ref...
                _, intent = executor.buffer_map[buffer_name_in_kernel]
                arg = self.root_insn.buffer_arguments[arg_index]
                buffer = self._extract_buffer(arg)
                assert buffer is not None
                new_buffer_map[buffer_name_in_kernel] = (buffer, intent)
            new_buffer_map = idict(new_buffer_map)

            # can we do this check more eagerly?
            if new_buffer_map != executor.buffer_map:
                executor = CompiledCodeExecutor(executor.executable, new_buffer_map, executor.comm)

        return executor

    @memory_cache(
        hashkey=lambda self: self._executor_cache_key,
        get_comm=lambda self: self.comm,
        heavy=True,
    )
    def _compile(self) -> CompiledCodeExecutor:
        assert not self._has_called_compile
        self._has_called_compile = True
        executor = compile(self, compiler_parameters=self.compiler_parameters)

        return executor, self._argument_index_to_buffer_name_map

    @cached_property
    def buffers(self) -> OrderedFrozenSet:
        """The buffers (global data) that are present in the operation."""
        from pyop3.insn.visitors import collect_buffers

        return collect_buffers(self._preprocess)

    @cached_property
    def disk_cache_key(self) -> Hashable:
        """Key used to write the operation to disk.

        The returned key should be consistent across ranks and not include
        overly specific information such as buffer names or array values.

        """
        from pyop3.insn.visitors import get_disk_cache_key

        return get_disk_cache_key(self._preprocess)


    @cached_property
    def _argument_index_to_buffer_name_map(self) -> idict[int, str]:
        return idict({i: arg.buffer.name for i, arg in enumerate(self.root_insn.buffer_arguments)})

    @cached_property
    def _argument_name_to_buffer_name_map(self) -> idict:
        return idict({arg.name: arg.buffer.name for arg in self.root_insn.buffer_arguments})

    @cached_property
    def _executor_cache_key(self) -> Hashable:
        from pyop3.insn.visitors import get_instruction_executor_cache_key

        return get_instruction_executor_cache_key(self.root_insn)

    @functools.singledispatchmethod
    def _extract_buffer(self, arg):
        raise TypeError

    @_extract_buffer.register(pyop3.expr.Scalar)
    def _(self, scalar):
        return scalar.buffer

    @_extract_buffer.register(pyop3.expr.Dat)
    def _(self, dat):
        if not isinstance(dat.buffer.handle, np.ndarray):
            raise NotImplementedError
        return dat.buffer

    @_extract_buffer.register(pyop3.expr.ScalarBufferExpression)
    def _(self, scalar):
        return scalar.buffer

    @_extract_buffer.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, dat):
        return dat.buffer

    @_extract_buffer.register(pyop3.expr.Mat)
    def _(self, mat):
        if isinstance(mat.buffer.handle, PETSc.Mat):
            match mat.buffer.handle.type:
                case "nest":
                    return pyop3.buffer.PetscMatBufferSubMat(mat.buffer, mat.nest_indices)
                case "python":
                    return self._extract_buffer(mat.buffer.handle.getPythonContext().dat)
                case _:
                    return mat.buffer
        else:
            return mat.buffer


class Executable:
    def __init__(self, code: lp.TranslationUnit, compiler_parameters: Mapping, comm: Pyop3Comm):
        self.code = code
        self.compiler_parameters = compiler_parameters
        self.comm = comm

    @cached_property
    def callable(self):
        return compile_loopy(self.code, compiler_parameters=self.compiler_parameters, comm=self.comm)

    def __call__(self, *args) -> None:
        # if len(self.code.callables_table) > 1 and "form" in str(self.code):
        # #     # breakpoint()
        #     import pyop3.debug
        #     pyop3.debug.maybe_breakpoint()
        # if len(self.loopy_code.callables_table) > 1:
        # if len(self.buffer_map) == 5:

        if self.comm.size > 1:
            if self.compiler_parameters.interleave_comp_comm:
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

            initializers = []
            reductions = []
            broadcasts = []
            for buffer_ref, intent in self.buffer_map.values():
                if isinstance(buffer_ref, PetscMatBuffer):
                    continue
                else:
                    assert isinstance(buffer_ref, pyop3.buffer.ArrayBuffer)

                inits, reds, bcasts = self._buffer_exchanges(buffer_ref, intent)
                initializers.extend(inits)
                reductions.extend(reds)
                broadcasts.extend(bcasts)

            # Unoptimised case: perform all transfers eagerly
            for init in initializers:
                init()
            for red in reductions:
                red()
            for bcast in broadcasts:
                bcast()

            # Now all the data is correct, compute!
            self.callable(*args)
        else:
            self.callable(*args)


class CompiledCodeExecutor:
    """
    Notes
    -----
    This class has a large number of cached properties to reduce overhead when it
    is called.

    This class is basically executable+buffers. It is useful because we want to cache the executable globally
    but we don't want to cache this globally because the buffers are likely to change.

    """

    # TODO: decouple intents from the buffer map (put intents on the executable)
    def __init__(self, executable: Executable, buffer_map: WeakValueDictionary[str, ConcreteBuffer], comm: Pyop3Comm):
        self.executable = executable
        self.buffer_map = buffer_map
        self.comm = comm

    @cached_property
    def _buffer_refs(self) -> tuple[BufferRef]:
        return tuple(ref for ref, _ in self.buffer_map.values())

    @cached_property
    def _buffer_global_name_to_name_in_kernel_map(self):
        return {buffer_ref.name: name_in_kernel for name_in_kernel, (buffer_ref, _) in self.buffer_map.items()}

    @cached_property
    def _default_buffers(self) -> tuple[ConcreteBuffer]:
        return tuple(buffer_ref for buffer_ref in self._buffer_refs)

    def __call__(self, **kwargs) -> None:
        """
        Notes
        -----
        This code is performance critical.

        """
        if not kwargs:  # shortcut for the most common case
            buffers = self._default_buffers
            exec_arguments = self._default_exec_arguments
        else:
            buffers = list(self._default_buffers)
            exec_arguments = list(self._default_exec_arguments)

            # TODO:
            # if CONFIG.debug:
            if False:
                for buffer_name, replacement_buffer in kwargs.items():
                    self._check_buffer_is_valid(self.buffer_map[buffer_name], replacement_buffer)

            for buffer_key, replacement_buffer in kwargs.items():
                index = self._buffer_ref_indices[buffer_key]
                buffers[index] = replacement_buffer
                exec_arguments[index] = self._as_exec_argument(replacement_buffer.handle)

        for index in self._modified_buffer_indices:
            buffers[index].inc_state()

        self.executable(*exec_arguments)

    def __str__(self) -> str:
        sep = "*" * 80
        str_ = []
        str_.append(sep)
        str_.append(lp.generate_code_v2(self.executable.code).device_code())
        str_.append(sep)

        for arg in self.executable.code.default_entrypoint.args:
            size, buffer = self._buffer_str(self.buffer_map[arg.name][0])
            str_.append(f"{arg.name} {size} : {buffer}")

        str_.append(sep)
        return "\n".join(str_)

    @functools.singledispatchmethod
    def _buffer_str(self, buffer):
        raise TypeError

    @_buffer_str.register
    def _(self, buffer: pyop3.buffer.ArrayBuffer):
        return f"({buffer.size})", str(buffer._data)

    @_buffer_str.register
    def _(self, buffer: pyop3.buffer.PetscMatBuffer) -> str:
        return "", "<PetscMat>"

    @cached_property
    def _buffer_ref_indices(self) -> idict[str, int]:
        return idict({
            # (buffer_ref.buffer.name, buffer_ref.nest_indices): i for i, buffer_ref in enumerate(self._buffer_refs)
            buffer_ref.name: i for i, buffer_ref in enumerate(self._buffer_refs)
        })

    @cached_property
    def _modified_buffer_indices(self) -> tuple[int]:
        return tuple(
            i
            for i, (_, intent) in enumerate(self.buffer_map.values())
            if intent != pyop3.insn.base.READ
        )

    @cached_property
    def _default_exec_arguments(self) -> tuple[int]:
        return tuple(self._as_exec_argument(buffer_ref.handle) for buffer_ref in self._buffer_refs)

    @functools.singledispatchmethod
    def _as_exec_argument(self, handle: Any) -> int:
        raise TypeError

    @_as_exec_argument.register
    def _(self, handle: np.ndarray) -> int:
        return handle.ctypes.data

    @_as_exec_argument.register
    def _(self, handle: PETSc.Mat) -> int:
        assert handle.type not in {PETSc.Mat.Type.NEST, PETSc.Mat.Type.PYTHON}
        return handle.handle

    def _check_buffer_is_valid(self, orig_buffer: AbstractBuffer, new_buffer: AbstractBuffer, /) -> None:
        valid = (
            type(new_buffer) is type(orig_buffer)
            and new_buffer.size == orig_buffer.size
            and new_buffer.dtype == orig_buffer.dtype
        )
        if not valid:
            raise exc.BufferMismatchException()

    # NOTE: This is probably very slow to have to do every time - a lot of this can be cached
    # the rest (initial state) can be checked each time
    def _buffer_exchanges(self, buffer, intent):
        initializers, reductions, broadcasts = [], [], []

        # Possibly instead of touches_ghost_points we could produce custom SFs for each loop
        # (we have filter_star_forest())
        # For now we just disregard the optimisation
        touches_ghost_points = True

        if intent in {pyop3.insn.base.READ, pyop3.insn.base.RW}:
            if touches_ghost_points:
                if not buffer._roots_valid:
                    initializers.append(buffer.reduce_leaves_to_roots_begin)
                    reductions.extend([
                        buffer.reduce_leaves_to_roots_end,
                        buffer.broadcast_roots_to_leaves_begin,
                    ])
                    broadcasts.append(buffer.broadcast_roots_to_leaves_end)
                else:
                    initializers.append(buffer.broadcast_roots_to_leaves_begin)
                    broadcasts.append(buffer.broadcast_roots_to_leaves_end)
            else:
                if not buffer._roots_valid:
                    initializers.append(buffer.reduce_leaves_to_roots_begin)
                    reductions.append(buffer.reduce_leaves_to_roots_end)

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


# TODO: prefer generate_code?
def compile(op, compiler_parameters=None):
    from pyop3.ir.lower import _compile_static

    compiler_parameters = parse_compiler_parameters(compiler_parameters)

    loopy_code, buffer_index_map = _compile_static(op, compiler_parameters)
    executable = Executable(loopy_code, compiler_parameters, op.comm)

    # TODO: The handling of nest indices here is very confused
    sorted_buffers = {}
    for kernel_arg_name, buffer_info in buffer_index_map.items():
        buffer_index, nest_indices, intent = buffer_info
        global_buffer = op.buffers[buffer_index]
        sorted_buffers[kernel_arg_name] = (global_buffer, intent)

    return CompiledCodeExecutor(executable, sorted_buffers, op.comm)


# NOTE: A lot of this is more generic than just loopy, try to refactor
def compile_loopy(translation_unit, *, compiler_parameters, comm):
    """Build a shared library and return a function pointer from it.

    :arg jitmodule: The JIT Module which can generate the code to compile, or
        the string representing the source code.
    :arg extension: extension of the source file (c, cpp)
    :arg fn_name: The name of the function to return from the resulting library
    :arg cppargs: A tuple of arguments to the C compiler (optional)
    :arg ldargs: A tuple of arguments to the linker (optional)
    :arg argtypes: A list of ctypes argument types matching the arguments of
         the returned function (optional, pass ``None`` for ``void``). This is
         only used when string is passed in instead of JITModule.
    :arg restype: The return type of the function (optional, pass
         ``None`` for ``void``).
    :kwarg comm: Optional communicator to compile the code on (only
        rank 0 compiles code) (defaults to pyop2.mpi.COMM_WORLD).
    """
    code = lp.generate_code_v2(translation_unit).device_code()
    argtypes = [
        cast_loopy_arg_to_ctypes_type(arg) for arg in translation_unit.default_entrypoint.args
    ]
    restype = None

    # ideally move this logic somewhere else
    cppargs = (
        petsctools.get_petsc_dirs(prefix="-I", subdir="include")
        # + tuple("-I%s" % d for d in self.local_kernel.include_dirs)
        # + ("-I%s" % os.path.abspath(os.path.dirname(__file__)),)
    )
    ldargs = (
        petsctools.get_petsc_dirs(prefix="-L", subdir="lib")
        + petsctools.get_petsc_dirs(prefix="-Wl,-rpath,", subdir="lib")
        + ("-lpetsc", "-lm")
        # + tuple(self.local_kernel.ldargs)
    )

    # NOTE: no - instead of this inspect the compiler parameters!!!
    # TODO: Make some sort of function in config.py
    if "LIKWID_MODE" in os.environ:
        cppargs += ("-DLIKWID_PERFMON",)
        ldargs += ("-llikwid",)

    dll = pyop3.compile.load(code, "c", cppargs, ldargs, comm=comm)

    if compiler_parameters.add_petsc_event:
        # Create the event in python and then set in the shared library to avoid
        # allocating memory over and over again in the C kernel.
        event_name = translation_unit.default_entrypoint.name
        ctypes.c_int.in_dll(dll, f"id_{event_name}").value = PETSc.Log.Event(event_name).id

    func = getattr(dll, translation_unit.default_entrypoint.name)
    func.argtypes = argtypes
    func.restype = restype
    return func


@functools.singledispatch
def cast_loopy_arg_to_ctypes_type(obj: Any) -> type:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@cast_loopy_arg_to_ctypes_type.register(lp.ArrayArg)
def _(arg: lp.ArrayArg) -> type:
    return ctypes.c_voidp


@cast_loopy_arg_to_ctypes_type.register(lp.ValueArg)
def _(arg: lp.ValueArg):
    if isinstance(arg.dtype, OpaqueType):
        return ctypes.c_voidp
    else:
        return np.ctypeslib.as_ctypes_type(arg.dtype)
