"""Coordinate the execution of instructions."""
from __future__ import annotations

import dataclasses
import functools
from functools import cached_property
from typing import Any, Callable

import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.collections
import pyop3.expr
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


DEFAULT_COMPILER_PARAMETERS = CompilerParameters()


META_COMPILER_PARAMETERS = idict({
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
        from pyop3.ir.lower import CompiledCodeExecutor

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
        from pyop3.ir.lower import compile

        assert not self._has_called_compile
        self._has_called_compile = True
        preprocessed = self._preprocess
        # executor = compile(preprocessed, compiler_parameters=self.compiler_parameters)
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
                    return PetscMatBufferSubMat(mat.buffer, mat.nest_indices)
                case "python":
                    return self._extract_buffer(mat.buffer.handle.getPythonContext().dat)
                case _:
                    return mat.buffer

