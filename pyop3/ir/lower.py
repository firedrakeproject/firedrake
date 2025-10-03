from __future__ import annotations

import abc
import collections
import contextlib
import ctypes
import dataclasses
import enum
import functools
import os
import numbers
import textwrap
import warnings
import weakref
from collections.abc import Mapping
from functools import cached_property
from typing import Any
from weakref import WeakValueDictionary

from cachetools import cachedmethod
from petsc4py import PETSc

import loopy as lp
import numpy as np
import pymbolic as pym
from immutabledict import immutabledict as idict

import pyop2

from pyop3 import exceptions as exc, utils, expr as op3_expr
from pyop3.expr import NonlinearDatBufferExpression
from pyop3.expr.visitors import collect_axis_vars, replace
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE, IndexedAxisTree, AxisComponent, relabel_path
from pyop3.buffer import AbstractBuffer, BufferRef, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.config import config
from pyop3.dtypes import IntType
from pyop3.ir.transform import with_likwid_markers, with_petsc_event, with_attach_debugger
from pyop3.insn.base import (
    Intent,
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    AbstractAssignment,
    Exscan,
    NullInstruction,
    assignment_type_as_intent,
    parse_compiler_parameters,
    WRITE,
    AssignmentType,
    ConcretizedNonEmptyArrayAssignment,
    StandaloneCalledFunction,
    Loop,
    InstructionList,
)
from pyop3.utils import (
    StrictlyUniqueDict,
    UniqueNameGenerator,
    as_tuple,
    just_one,
    merge_dicts,
    strictly_all,
)

import pyop3.extras.debug

# FIXME this needs to be synchronised with TSFC, tricky
# shared base package? or both set by Firedrake - better solution
LOOPY_TARGET = lp.CWithGNULibcTarget()
LOOPY_LANG_VERSION = (2018, 2)


# Is this still needed? Loopy may have fixed this
class OpaqueType(lp.types.OpaqueType):
    def __repr__(self) -> str:
        return f"OpaqueType('{self.name}')"


class CodeGenerationException(exc.Pyop3Exception):
    pass


@dataclasses.dataclass(frozen=True)
class GlobalBufferArgSpec:
    buffer: ConcreteBuffer
    intent: Intent


class CodegenContext(abc.ABC):
    pass


class LoopyCodegenContext(CodegenContext):
    def __init__(self):
        self._domains = []
        self._instructions = []
        self._arguments = []
        self._subkernels = []

        self._within_inames = frozenset()
        self._last_insn_id = None

        self._name_generator = UniqueNameGenerator()

        # buffer name -> name in kernel
        self._kernel_names = {}

        # buffer name -> buffer
        self.global_buffers = {}
        self.global_buffer_intents = {}

        # initializer hash -> temporary name
        self._reusable_temporaries: dict[int, str] = {}

        # assignee name -> indirection expression
        self._assignees = {}

    @property
    def domains(self) -> tuple:
        return tuple(self._domains)

    @property
    def instructions(self) -> tuple:
        return tuple(self._instructions)

    @property
    def arguments(self) -> tuple:
        return tuple(sorted(self._arguments, key=lambda arg: arg.name))

    @property
    def subkernels(self) -> tuple:
        return tuple(self._subkernels)

    def add_domain(self, iname, *args):
        nargs = len(args)
        if nargs == 1:
            start, stop = 0, args[0]
        else:
            assert nargs == 2
            start, stop = args[0], args[1]
        domain_str = f"{{ [{iname}]: {start} <= {iname} < {stop} }}"
        self._domains.append(domain_str)

    def add_assignment(self, assignee, expression, prefix="insn"):
        insn = lp.Assignment(
            assignee,
            expression,
            id=self._name_generator(prefix),
            within_inames=frozenset(self._within_inames),
            depends_on=self._depends_on,
            depends_on_is_final=True,
        )
        self._add_instruction(insn)

    def add_cinstruction(self, insn_str, read_variables=frozenset()):
        cinsn = lp.CInstruction(
            (),
            insn_str,
            read_variables=frozenset(read_variables),
            id=self.unique_name("insn"),
            within_inames=self._within_inames,
            within_inames_is_final=True,
            depends_on=self._depends_on,
        )
        self._add_instruction(cinsn)

    def add_function_call(self, assignees, expression, prefix="insn"):
        insn = lp.CallInstruction(
            assignees,
            expression,
            id=self._name_generator(prefix),
            within_inames=self._within_inames,
            within_inames_is_final=True,
            depends_on=self._depends_on,
            depends_on_is_final=True,
        )
        self._add_instruction(insn)

    def add_buffer(self, buffer_ref: BufferRef, intent: Intent | None = None) -> str:
        # TODO: This should check to make that we do not encounter any
        # loop-carried dependencies. For that to work we need to track the intent and
        # the indirection expression. Something like:
        #
        #   for i
        #     dat1[i] = ???
        #     dat2[i] = dat1[map1[i]]
        #
        # is illegal, but
        #
        #   for i
        #     dat1[2*i] = ???
        #     dat2[i] = dat1[2*i]
        #
        # is not.

        buffer = buffer_ref.buffer
        buffer_key = (buffer.name, buffer_ref.nest_indices)
        if isinstance(buffer, NullBuffer):
            assert not buffer_ref.nest_indices
            # 'intent' is not important for temporaries
            if buffer_key in self._kernel_names:
                return self._kernel_names[buffer_key]
            shape = self._temporary_shapes.get(buffer_key, (buffer.size,))
            name_in_kernel = self.add_temporary("t", buffer.dtype, shape=shape)
        else:
            if intent is None:
                raise ValueError("Global data must declare intent")

            if buffer_key in self._kernel_names:
                if intent != self.global_buffer_intents[buffer_key]:
                    # We are accessing a buffer with different intents so have to
                    # pessimally claim RW access
                    self.global_buffer_intents[buffer_key] = RW
                return self._kernel_names[buffer_key]

            if isinstance(buffer_ref.handle, np.ndarray):
                # Inject constant buffer data into the generated code if sufficiently small
                # NOTE: We conflate 2 concepts for constant-ness here:
                # * The array cannot be modified
                # * The array is the same between ranks
                if (
                    buffer.constant
                    and isinstance(buffer.size, numbers.Integral)
                    and buffer.size < config["max_static_array_size"]
                ):
                    return self.add_temporary(
                        "t",
                        buffer.dtype,
                        initializer=buffer.data_ro,
                        shape=buffer.data_ro.shape,
                        read_only=True,
                    )

                if isinstance(buffer.dtype, np.dtypes.IntDType):
                    name_in_kernel = self.unique_name("idat")
                else:
                    name_in_kernel = self.unique_name("dat")

                # If the buffer is being passed straight through to a function then we
                # have to make sure that the shapes match
                shape = self._temporary_shapes.get(buffer_key, None)
                loopy_arg = lp.GlobalArg(name_in_kernel, dtype=buffer.dtype, shape=shape)
            else:
                assert isinstance(buffer_ref.handle, PETSc.Mat)

                name_in_kernel = self.unique_name("mat")
                loopy_arg = lp.ValueArg(name_in_kernel, dtype=OpaqueType("Mat"))

            self.global_buffers[buffer_key] = buffer_ref
            self.global_buffer_intents[buffer_key] = intent
            self._arguments.append(loopy_arg)

        self._kernel_names[buffer_key] = name_in_kernel
        return name_in_kernel

    def add_temporary(self, prefix="t", dtype=IntType, *, shape=(), initializer: np.ndarray = None, read_only: bool = False) -> str:
        # If multiple temporaries with the same initializer are used then they
        # can be shared.
        can_reuse = initializer is not None and read_only
        if can_reuse:
            key = initializer.data.tobytes()
            if key in self._reusable_temporaries:
                return self._reusable_temporaries[key]

        name_in_kernel = self.unique_name(prefix)
        arg = lp.TemporaryVariable(
            name_in_kernel,
            dtype=dtype,
            shape=shape,
            initializer=initializer,
            read_only=read_only,
            address_space=lp.AddressSpace.LOCAL,
        )
        self._arguments.append(arg)

        if can_reuse:
            self._reusable_temporaries[key] = name_in_kernel

        return name_in_kernel

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    def unique_name(self, prefix):
        return self._name_generator(prefix)

    @contextlib.contextmanager
    def within_inames(self, inames) -> None:
        orig_within_inames = self._within_inames
        self._within_inames |= inames
        yield
        self._within_inames = orig_within_inames

    # FIXME, bad API but it is context-dependent
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}

    def _add_instruction(self, insn):
        self._instructions.append(insn)
        self._last_insn_id = insn.id


class DummyModuleExecutor:
    def __call__(self, *args, **kwargs):
        pass


class CompiledCodeExecutor:
    """
    Notes
    -----
    This class has a large number of cached properties to reduce latency when it
    is called.

    """

    # TODO: intents and datamap etc maybe all go together. All relate to the same objects
    def __init__(self, loopy_code: lp.TranslationUnit, buffer_map: WeakValueDictionary[str, ConcretBuffer], compiler_parameters: Mapping, comm: Pyop3Comm):
        self.loopy_code = loopy_code
        self.buffer_map = buffer_map
        self.compiler_parameters = compiler_parameters
        self.comm = comm

    @property
    def loopy_kernel(self) -> lp.LoopKernel:
        return self.loopy_code.default_entrypoint

    @cached_property
    def _buffer_refs(self) -> tuple[BufferRef]:
        return tuple(ref for ref, _ in self.buffer_map.values())

    @cached_property
    def _default_buffers(self) -> tuple[ConcreteBuffer]:
        return tuple(buffer_ref.buffer for buffer_ref in self._buffer_refs)

    @cached_property
    def executable(self):
        return compile_loopy(self.loopy_code, pyop3_compiler_parameters=self.compiler_parameters)

    def __call__(self, replacement_buffers: Mapping[Hashable, ConcreteBuffer] | None = None) -> None:
        """
        Notes
        -----
        This code is performance critical.

        """
        if replacement_buffers is None:  # shortcut for the most common case
            buffers = self._default_buffers
            exec_arguments = self._default_exec_arguments
        else:
            buffers = list(self._default_buffers)
            exec_arguments = list(self._default_exec_arguments)

            # TODO:
            # if config.debug:
            if False:
                for buffer_name, replacement_buffer in kwargs.items():
                    self._check_buffer_is_valid(self.buffer_map[buffer_name], replacement_buffer)

            for buffer_key, replacement_buffer in replacement_buffers.items():
                index = self._buffer_ref_indices[buffer_key]
                buffers[index] = replacement_buffer
                replacement_handle = replacement_buffer.handle(
                    nest_indices=self._buffer_refs[index].nest_indices
                )
                exec_arguments[index] = self._as_exec_argument(replacement_handle)

        for index in self._modified_buffer_indices:
            buffers[index].inc_state()

        # if len(self.loopy_code.callables_table) > 1 and "form" in str(self):
        #     breakpoint()
        # if "MatSetValues" in str(self):
        #     breakpoint()
        # if "integral" in str(self):
        pyop3.extras.debug.maybe_breakpoint("closure")

        if self.comm.size > 1:
            if self.compiler_parameters.interleave_comp_comm:
                raise NotImplementedError

            initializers = []
            reductions = []
            broadcasts = []
            for buffer_ref, intent in self.buffer_map.values():
                if isinstance(buffer_ref.buffer, PetscMatBuffer):
                    continue
                else:
                    assert isinstance(buffer_ref.buffer, ArrayBuffer)

                inits, reds, bcasts = self._buffer_exchanges(buffer_ref.buffer, intent)
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
            self.executable(*exec_arguments)
        else:
            self.executable(*exec_arguments)

    def __str__(self) -> str:
        sep = "*" * 80
        str_ = []
        str_.append(sep)
        str_.append(lp.generate_code_v2(self.loopy_code).device_code())
        str_.append(sep)

        for arg in self.loopy_code.default_entrypoint.args:
            size, buffer = self._buffer_str(self.buffer_map[arg.name][0].buffer)
            str_.append(f"{arg.name} {size} : {buffer}")

        str_.append(sep)
        return "\n".join(str_)

    @functools.singledispatchmethod
    def _buffer_str(self, buffer):
        raise TypeError

    @_buffer_str.register
    def _(self, buffer: ArrayBuffer):
        return f"({buffer.size})", str(buffer._data)

    @_buffer_str.register
    def _(self, buffer: PetscMatBuffer) -> str:
        return "", "<PetscMat>"

    @cached_property
    def _buffer_ref_indices(self) -> idict[str, int]:
        return idict({
            (buffer_ref.buffer.name, buffer_ref.nest_indices): i for i, buffer_ref in enumerate(self._buffer_refs)
        })

    @cached_property
    def _modified_buffer_indices(self) -> tuple[int]:
        return tuple(
            i
            for i, (_, intent) in enumerate(self.buffer_map.values())
            if intent != READ
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

        if intent in {READ, RW}:
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


class BinarySearchCallable(lp.ScalarCallable):
    def __init__(self, name="bsearch", **kwargs):
        super().__init__(name, **kwargs)

    def with_types(self, arg_id_to_dtype, callables_table):
        new_arg_id_to_dtype = arg_id_to_dtype.copy()
        new_arg_id_to_dtype[-1] = int
        return (
            self.copy(name_in_target="bsearch", arg_id_to_dtype=new_arg_id_to_dtype),
            callables_table,
        )

    def with_descrs(self, arg_id_to_descr, callables_table):
        return self.copy(arg_id_to_descr=arg_id_to_descr), callables_table

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        assert False
        from pymbolic import var

        mat_descr = self.arg_id_to_descr[0]
        m, n = mat_descr.shape
        ecm = expression_to_code_mapper
        mat, vec = insn.expression.parameters
        (result,) = insn.assignees

        c_parameters = [
            var("CblasRowMajor"),
            var("CblasNoTrans"),
            m,
            n,
            1,
            ecm(mat).expr,
            1,
            ecm(vec).expr,
            1,
            ecm(result).expr,
            1,
        ]
        return (
            var(self.name_in_target)(*c_parameters),
            False,  # cblas_gemv does not return anything
        )

    def generate_preambles(self, target):
        assert isinstance(target, lp.CTarget)
        yield ("20_stdlib", "#include <stdlib.h>")
        return


# prefer generate_code?
def compile(expr, compiler_parameters=None):
    insn = expr

    compiler_parameters = parse_compiler_parameters(compiler_parameters)

    # function_name = insn.name
    function_name = "pyop3_loop"  # TODO: Provide as kwarg

    if isinstance(insn, InstructionList):
        cs_expr = insn.instructions
    else:
        cs_expr = (insn,)

    context = LoopyCodegenContext()
    # NOTE: so I think LoopCollection is a better abstraction here - don't want to be
    # explicitly dealing with contexts at this point. Can always sniff them out again.
    # for context, ex in cs_expr:
    for ex in cs_expr:
        # ex = expand_implicit_pack_unpack(ex)

        # add external loop indices as kernel arguments
        # FIXME: removed because cs_expr needs to sniff the context now
        loop_indices = {}
        # for index, (path, _) in context.items():
        #     if len(path) > 1:
        #         raise NotImplementedError("needs to be sorted")
        #
        #     # dummy = Dat(index.iterset, data=NullBuffer(IntType))
        #     dummy = Dat(Axis(1), dtype=IntType)
        #     # this is dreadful, pass an integer array instead
        #     ctx.add_argument(dummy)
        #     myname = ctx.actual_to_kernel_rename_map[dummy.name]
        #     replace_map = {
        #         axis: pym.subscript(pym.var(myname), (i,))
        #         for i, axis in enumerate(path.keys())
        #     }
        #     # FIXME currently assume that source and target exprs are the same, they are not!
        #     loop_indices[index] = (replace_map, replace_map)

        for e in as_tuple(ex): # TODO: get rid of this loop
            # context manager?
            context.set_temporary_shapes(_collect_temporary_shapes(e))
            _compile(e, loop_indices, context)

    if not context.global_buffers:
        warnings.warn(
            "The generated kernel does not modify any global data, this may indicate that something has gone wrong"
        )
        return DummyModuleExecutor()

    # add a no-op instruction touching all of the kernel arguments so they are
    # not silently dropped
    noop = lp.CInstruction(
        (),
        "",
        read_variables=frozenset({a.name for a in context.arguments}),
        within_inames=frozenset(),
        within_inames_is_final=True,
        depends_on=context._depends_on,
    )
    context._instructions.append(noop)

    preambles = [
        ("20_debug", "#include <stdio.h>"),  # dont always inject
        ("30_petsc", "#include <petsc.h>"),  # perhaps only if petsc callable used?
        (
            "30_bsearch",
            textwrap.dedent(
                """
                #include <stdlib.h>


                int32_t cmpfunc(const void * a, const void * b) {
                   return ( *(int32_t*)a - *(int32_t*)b );
                }
            """
            ),
        ),
    ]

    translation_unit = lp.make_kernel(
        context.domains,
        context.instructions,
        context.arguments,
        name=function_name,
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
        preambles=preambles,
    )
    translation_unit = lp.merge((translation_unit, *context.subkernels))

    entrypoint = translation_unit.default_entrypoint
    if compiler_parameters.add_likwid_markers:
        entrypoint = with_likwid_markers(entrypoint)
    if compiler_parameters.add_petsc_event:
        entrypoint = with_petsc_event(entrypoint)
    if compiler_parameters.attach_debugger:
        entrypoint = with_attach_debugger(entrypoint)
    translation_unit = translation_unit.with_kernel(entrypoint)

    # Sort the buffers by where they appear in the kernel signature
    kernel_to_buffer_names = utils.invert_mapping(context._kernel_names)
    sorted_buffers = {}
    for kernel_arg in entrypoint.args:
        buffer_key = kernel_to_buffer_names[kernel_arg.name]
        sorted_buffers[kernel_arg.name] = (context.global_buffers[buffer_key], context.global_buffer_intents[buffer_key])

    return CompiledCodeExecutor(translation_unit, sorted_buffers, compiler_parameters, expr.internal_comm)


# put into a class in transform.py?
@functools.singledispatch
def _collect_temporary_shapes(expr):
    raise TypeError(f"No handler defined for {type(expr).__name__}")


@_collect_temporary_shapes.register(InstructionList)
def _(insn_list: InstructionList, /) -> idict:
    return merge_dicts(_collect_temporary_shapes(insn) for insn in insn_list)


@_collect_temporary_shapes.register(Loop)
def _(loop: Loop, /):
    shapes = {}
    for stmt in loop.statements:
        for temp, shape in _collect_temporary_shapes(stmt).items():
            if shape is None:
                continue
            if temp in shapes:
                assert shapes[temp] == shape
            else:
                shapes[temp] = shape
    return shapes


@_collect_temporary_shapes.register(AbstractAssignment)
@_collect_temporary_shapes.register(NullInstruction)
@_collect_temporary_shapes.register(Exscan)  # assume we are fine
def _(assignment: AbstractAssignment, /) -> idict:
    return idict()


@_collect_temporary_shapes.register
def _(call: StandaloneCalledFunction):
    return idict(
        {
            (arg.buffer.buffer.name, arg.buffer.nest_indices): lp_arg.shape
            for lp_arg, arg in zip(
                call.function.code.default_entrypoint.args, call.arguments, strict=True
            )
            if isinstance(lp_arg, lp.ArrayArg)
        }
    )


@functools.singledispatch
def _compile(expr: Any, loop_indices, ctx: LoopyCodegenContext) -> None:
    raise TypeError(f"No handler defined for {type(expr).__name__}")


@_compile.register(NullInstruction)
def _(null: NullInstruction, *args, **kwargs):
    pass


@_compile.register(InstructionList)
def _(insn_list: InstructionList, /, loop_indices, ctx) -> None:
    for insn in insn_list:
        _compile(insn, loop_indices, ctx)


@_compile.register(Loop)
def _(
    loop,
    loop_indices,
    codegen_context: LoopyCodegenContext,
) -> None:
    parse_loop_properly_this_time(
        loop,
        loop.index.iterset,
        loop_indices,
        codegen_context,
    )


def parse_loop_properly_this_time(
    loop,
    axes,
    loop_indices,
    codegen_context,
    *,
    axis=None,
    path=None,
    iname_map=None,
) -> None:
    if axes is UNIT_AXIS_TREE:
        # NOTE: might need an expression here sometimes
        for statement in loop.statements:
            _compile(
                statement,
                # loop_indices | dict(loop_exprs),
                loop_indices,
                codegen_context,
            )
        return

    if strictly_all(x is None for x in {axis, path, iname_map}):
        axis = axes.root
        path = idict()
        iname_map = idict()

    for component in axis.components:
        path_ = path | {axis.label: component.label}

        if component.size != 1:
            iname = codegen_context.unique_name("i")
            domain_var = register_extent(
                component.size,
                iname_map,
                loop_indices,
                codegen_context,
            )
            codegen_context.add_domain(iname, domain_var)
            iname_replace_map_ = iname_map | {axis.label: pym.var(iname)}
            within_inames = frozenset({iname})
        else:
            iname_replace_map_ = iname_map | {axis.label: 0}
            within_inames = set()

        with codegen_context.within_inames(within_inames):
            # NOTE: The following bit is done for each axis, not sure if that's right or if
            # we should handle at the bottom
            loop_exprs = StrictlyUniqueDict()
            for index_exprs in axes.index_exprs:
                for axis_label, index_expr in index_exprs.get(path_, {}).items():
                    loop_exprs[(loop.index.id, axis_label)] = lower_expr(index_expr, [iname_replace_map_], loop_indices, codegen_context, paths=[path_])
            loop_exprs = idict(loop_exprs)

            if subaxis := axes.node_map[path_]:
                parse_loop_properly_this_time(
                    loop,
                    axes,
                    # I think we only want to do this at the end of the traversal
                    loop_indices | dict(loop_exprs),
                    # loop_indices,
                    codegen_context,
                    axis=subaxis,
                    path=path_,
                    iname_map=iname_replace_map_,
                )
            else:
                for statement in loop.statements:
                    _compile(
                        statement,
                        loop_indices | dict(loop_exprs),
                        codegen_context,
                    )


@_compile.register
def _(call: StandaloneCalledFunction, loop_indices, context: LoopyCodegenContext) -> None:
    subarrayrefs = {}
    loopy_args = call.function.code.default_entrypoint.args
    for loopy_arg, arg, spec in zip(loopy_args, call.arguments, call.argspec, strict=True):
        # this check fails because we currently assume that all arrays require packing
        # from pyop3.transform import _requires_pack_unpack
        # assert not _requires_pack_unpack(arg)
        name_in_kernel = context.add_buffer(arg.buffer, spec.intent)

        if not isinstance(loopy_arg, lp.ArrayArg):
            raise NotImplementedError

        if loopy_arg.shape is not None:
            shape = loopy_arg.shape
        else:
            shape = (np.prod(*(axis_tree.size for axis_tree in arg.shape), dtype=int),)

        # subarrayref nonsense/magic
        indices = []
        for s in shape:
            iname = context.unique_name("i")
            context.add_domain(iname, s)
            indices.append(pym.var(iname))
        indices = tuple(indices)

        subarrayrefs[arg] = lp.symbolic.SubArrayRef(
            indices, pym.var(name_in_kernel)[indices]
        )

    assignees = tuple(
        subarrayrefs[arg]
        for arg, spec in zip(call.arguments, call.argspec, strict=True)
        if spec.intent in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}
    )
    expression = pym.primitives.Call(
        pym.var(call.function.code.default_entrypoint.name),
        tuple(
            subarrayrefs[arg]
            for arg, spec in zip(call.arguments, call.argspec, strict=True)
            if spec.intent in {READ, RW, INC, MIN_RW, MAX_RW}
        ),
    )

    context.add_function_call(assignees, expression)
    subkernel = call.function.code.with_entrypoints(frozenset())
    context.add_subkernel(subkernel)


@_compile.register(ConcretizedNonEmptyArrayAssignment)
def parse_assignment(assignment: ConcretizedNonEmptyArrayAssignment, loop_indices, context: CodegenContext):
    if any(
        isinstance(arg.buffer.buffer, ConcreteBuffer)
        and isinstance(arg.buffer.handle, PETSc.Mat)
        for arg in assignment.buffer_arguments
    ):
        _compile_petsc_mat(assignment, loop_indices, context)
    else:
        compile_array_assignment(
            assignment,
            loop_indices,
            context,
            assignment.axis_trees,
        )


def _compile_petsc_mat(assignment: ConcretizedNonEmptyArrayAssignment, loop_indices, context) -> None:
    mat = assignment.assignee
    expr = assignment.expression

    if not isinstance(mat.buffer.buffer, PetscMatBuffer):
        raise NotImplementedError  # order must be different
    else:
        # We need to know whether the matrix is the assignee or not because we need
        # to know whether to put MatGetValues or MatSetValues
        setting_mat_values = True


    row_axis_tree, column_axis_tree = assignment.axis_trees

    if isinstance(expr, numbers.Number):
        # If we have an expression like
        #
        #     mat[f(p), f(p)] <- 666
        #
        # then we have to convert `666` into an appropriately sized temporary
        # for Mat{Get,Set}Values to work.
        # TODO: There must be a more elegant way of doing this
        nrows = row_axis_tree.max_size
        ncols = column_axis_tree.max_size
        expr_data = np.full((nrows, ncols), expr, dtype=mat.buffer.buffer.dtype)

        # if isinstance(nrows, numbers.Integral) and isinstance(ncols, numbers.Integral):
        # else:
        #     pyop3.extras.debug.warn_todo("Need expr.materialize() or similar to get the max size")
        #     max_size = 36  # assume that this is OK
        #     expr_data = np.full((max_size, max_size), expr, dtype=mat.buffer.buffer.dtype)
        array_buffer = BufferRef(ArrayBuffer(expr_data, constant=True))
    else:
        assert isinstance(expr, op3_expr.BufferExpression)
        array_buffer = expr.buffer

    # now emit the right line of code, this should properly be a lp.ScalarCallable
    # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
    mat_name = context.add_buffer(assignment.assignee.buffer, assignment_type_as_intent(assignment.assignment_type))

    # NOTE: Is this always correct? It is for now.
    array_name = context.add_buffer(array_buffer, READ)

    rsize = row_axis_tree.size
    csize = column_axis_tree.size

    # these sizes can be expressions that need evaluating
    rsize_var = register_extent(
        rsize,
        {},
        loop_indices,
        context,
    )

    csize_var = register_extent(
        csize,
        {},
        loop_indices,
        context,
    )

    # convert the generic expressions to 
    # for example:
    #
    #   map0[3*i0 + i1]
    #   map0[3*i0 + i2 + 3]
    #
    # to the shared top-level layout:
    #
    #   map0[3*i0]
    #
    # which is what Mat{Get,Set}Values() needs.
    layout_exprs = []
    for layout in [mat.row_layout, mat.column_layout]:
        assert isinstance(layout, NonlinearDatBufferExpression)
        subst_sublayout = layout.layouts[idict()]
        subst_layout = op3_expr.LinearDatBufferExpression(layout.buffer, subst_sublayout)
        layout_expr = lower_expr(subst_layout, ((),), loop_indices, context)
        layout_exprs.append(layout_expr)
    irow, icol = layout_exprs

    # FIXME:
    blocked = False

    # hacky
    myargs = [
        assignment, mat_name, array_name, rsize_var, csize_var, irow, icol, blocked
    ]
    if setting_mat_values:
        match assignment.assignment_type:
            case AssignmentType.WRITE:
                call_str = _petsc_mat_store(*myargs)
            case AssignmentType.INC:
                call_str = _petsc_mat_add(*myargs)
            case _:
                raise AssertionError
    else:
        raise NotImplementedError
        # call_str = _petsc_mat_load(*myargs)
        # but check cannot do INC here without extra step

    context.add_cinstruction(call_str)


def _petsc_mat_load(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatGetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"
    else:
        return f"MatGetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"


def _petsc_mat_store(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"
    else:
        return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"


def _petsc_mat_add(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"
    else:
        return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"

# TODO now I attach a lot of info to the context-free array, do I need to pass axes around?
def compile_array_assignment(
    assignment,
    loop_indices,
    codegen_context,
    axis_trees,
    *,
    iname_replace_maps=None,
    # TODO document these under "Other Parameters"
    axis_tree=None,
    paths=None,
):
    if paths is None:
        paths = []
    if iname_replace_maps is None:
        iname_replace_maps = []

    if axis_tree is None:
        axis_tree, *axis_trees = axis_trees

        paths += [idict()]
        iname_replace_maps += [idict()]

        if axis_tree.is_empty or axis_tree is UNIT_AXIS_TREE or isinstance(axis_tree, IndexedAxisTree):
            if axis_trees:
                raise NotImplementedError("need to refactor code here")

            add_leaf_assignment(
                assignment,
                paths,
                iname_replace_maps,
                codegen_context,
                loop_indices,
            )
            return

    axis = axis_tree.node_map[paths[-1]]

    for component in axis.components:
        if component.size != 1:
            iname = codegen_context.unique_name("i")

            extent_var = register_extent(
                component.size,
                iname_replace_maps[-1],
                loop_indices,
                codegen_context,
            )
            codegen_context.add_domain(iname, extent_var)
            new_iname_replace_maps = iname_replace_maps.copy()
            new_iname_replace_maps[-1] = iname_replace_maps[-1] | {axis.label: pym.var(iname)}
            within_inames = {iname}
        else:
            new_iname_replace_maps = iname_replace_maps.copy()
            new_iname_replace_maps[-1] = iname_replace_maps[-1] | {axis.label: 0}
            within_inames = set()

        new_paths = paths.copy()
        new_paths[-1] = paths[-1] | {axis.label: component.label}

        with codegen_context.within_inames(within_inames):
            if axis_tree.node_map[new_paths[-1]]:
                compile_array_assignment(
                    assignment,
                    loop_indices,
                    codegen_context,
                    axis_trees,
                    iname_replace_maps=new_iname_replace_maps,
                    axis_tree=axis_tree,
                    paths=new_paths,
                )
            elif axis_trees:
                compile_array_assignment(
                    assignment,
                    loop_indices,
                    codegen_context,
                    axis_trees,
                    iname_replace_maps=new_iname_replace_maps,
                    axis_tree=None,
                    paths=new_paths,
                )
            else:
                add_leaf_assignment(
                    assignment,
                    new_paths,
                    new_iname_replace_maps,
                    codegen_context,
                    loop_indices,
                )


def add_leaf_assignment(
    assignment,
    paths,
    iname_replace_maps,
    codegen_context,
    loop_indices,
):
    intent = assignment_type_as_intent(assignment.assignment_type)
    shape = tuple(axis_tree.size for axis_tree in assignment.axis_trees)
    lexpr = lower_expr(assignment.assignee, iname_replace_maps, loop_indices, codegen_context, intent=intent, paths=paths, shape=shape)
    rexpr = lower_expr(assignment.expression, iname_replace_maps, loop_indices, codegen_context, paths=paths, shape=shape)

    if assignment.assignment_type == AssignmentType.INC:
        rexpr = lexpr + rexpr

    codegen_context.add_assignment(lexpr, rexpr)


@_compile.register(Exscan)
def _(exscan: Exscan, loop_indices, context) -> None:
    if exscan.scan_type != "+":
        raise NotImplementedError
    component = exscan.scan_axis.component
    domain_var = register_extent(
        component.size-1,
        {},
        loop_indices,
        context,
    )
    iname = context.unique_name("i")
    context.add_domain(iname, domain_var)

    lexpr = lower_expr(exscan.assignee, [{exscan.scan_axis.label: pym.var(iname)+1}], loop_indices, context, intent=WRITE)
    lexpr2 = lower_expr(exscan.assignee, [{exscan.scan_axis.label: pym.var(iname)}], loop_indices, context)
    rexpr = lower_expr(exscan.expression, [{exscan.scan_axis.label: pym.var(iname)}], loop_indices, context)

    rexpr = lexpr2 + rexpr
    context.add_assignment(lexpr, rexpr)


def lower_expr(expr, iname_maps, loop_indices, ctx, *, intent=READ, paths=None, shape=None) -> pym.Expression:
    return _lower_expr(expr, iname_maps, loop_indices, ctx, intent=intent, paths=paths, shape=shape)


@functools.singledispatch
def _lower_expr(obj: Any, /, *args, **kwargs) -> pym.Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_lower_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
    return num


@_lower_expr.register(op3_expr.Add)
def _(add: op3_expr.Add, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(add.a, *args, **kwargs) + _lower_expr(add.b, *args, **kwargs)


@_lower_expr.register(op3_expr.Sub)
def _(sub: op3_expr.Sub, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(sub.a, *args, **kwargs) - _lower_expr(sub.b, *args, **kwargs)


@_lower_expr.register(op3_expr.Mul)
def _(mul: op3_expr.Mul, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mul.a, *args, **kwargs) * _lower_expr(mul.b, *args, **kwargs)


@_lower_expr.register(op3_expr.Modulo)
def _(mod: op3_expr.Mul, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mod.a, *args, **kwargs) % _lower_expr(mod.b, *args, **kwargs)


@_lower_expr.register(op3_expr.Or)
def _(or_: op3_expr.Or, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.LogicalOr((_lower_expr(or_.a, *args, **kwargs), _lower_expr(or_.b, *args, **kwargs)))


@_lower_expr.register(op3_expr.Neg)
def _(neg: op3_expr.Neg, /, *args, **kwargs) -> pym.Expression:
    return -_lower_expr(neg.a, *args, **kwargs)


@_lower_expr.register(op3_expr.FloorDiv)
def _(neg: op3_expr.Neg, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(neg.a, *args, **kwargs) // _lower_expr(neg.b, *args, **kwargs)


@_lower_expr.register(op3_expr.Comparison)
def _(cond, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.Comparison(
        _lower_expr(cond.a, *args, **kwargs),
        cond._symbol,
        _lower_expr(cond.b, *args, **kwargs),
    )


@_lower_expr.register(op3_expr.AxisVar)
def _(axis_var: op3_expr.AxisVar, /, iname_maps, *args, **kwargs) -> pym.Expression:
    try:
        return just_one(iname_maps)[axis_var.axis_label]
    except KeyError:
        breakpoint()  # debug


@_lower_expr.register(op3_expr.LoopIndexVar)
def _(loop_var: op3_expr.LoopIndexVar, /, iname_maps, loop_indices, *args, **kwargs) -> pym.Expression:
    return loop_indices[(loop_var.loop_id, loop_var.axis_label)]


@_lower_expr.register(op3_expr.Scalar)
def _(scalar: op3_expr.Scalar, /, iname_maps, loop_indices, context, *, intent, **kwargs) -> pym.Expression:
    # TODO: Need a ScalarBufferExpression or similar to encode nested-ness
    buffer_ref = BufferRef(scalar.buffer)
    name_in_kernel = context.add_buffer(buffer_ref, intent)
    return pym.subscript(pym.var(name_in_kernel), (0,))


@_lower_expr.register(op3_expr.ScalarBufferExpression)
def _(expr: op3_expr.ScalarBufferExpression, /, iname_maps, loop_indices, context, *, intent, **kwargs) -> pym.Expression:
    return lower_buffer_access(expr.buffer, [0], iname_maps, loop_indices, context, intent=intent)


@_lower_expr.register(op3_expr.LinearDatBufferExpression)
def _(expr: op3_expr.LinearDatBufferExpression, /, iname_maps, loop_indices, context, *, intent, **kwargs) -> pym.Expression:
    return lower_buffer_access(expr.buffer, [expr.layout], iname_maps, loop_indices, context, intent=intent)


@_lower_expr.register(op3_expr.NonlinearDatBufferExpression)
def _(expr: op3_expr.NonlinearDatBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, **kwargs) -> pym.Expression:
    path = just_one(paths)
    return lower_buffer_access(expr.buffer, [expr.layouts[path]], iname_maps, loop_indices, context, intent=intent)


# @_lower_expr.register(op3_expr.MatPetscMatBufferExpression)
# def _(mat_expr: op3_expr.MatPetscMatBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, shape) -> pym.Expression:
#     layouts = (mat_expr.row_layout, mat_expr.column_layout)
#     return lower_buffer_access(mat_expr.buffer, layouts, iname_maps, loop_indices, context, intent=intent, shape=shape)


@_lower_expr.register(op3_expr.MatArrayBufferExpression)
def _(expr: op3_expr.MatArrayBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, shape) -> pym.Expression:
    row_path, column_path = paths
    layouts = (expr.row_layouts[row_path], expr.column_layouts[column_path])
    return lower_buffer_access(expr.buffer, layouts, iname_maps, loop_indices, context, intent=intent, shape=shape)


def lower_buffer_access(buffer: AbstractBuffer, layouts, iname_maps, loop_indices, context, *, intent, shape=None) -> pym.Expression:
    name_in_kernel = context.add_buffer(buffer, intent)

    offset_expr = 0
    strides = utils.strides(shape) if shape else (1,)
    for stride, layout, iname_map in zip(strides, layouts, iname_maps, strict=True):
        stride = lower_expr(stride, [iname_map], loop_indices, context)
        offset_expr += stride * lower_expr(layout, [iname_map], loop_indices, context)

    indices = maybe_multiindex(buffer, offset_expr, context)
    return pym.subscript(pym.var(name_in_kernel), indices)


def maybe_multiindex(buffer_ref, offset_expr, context):
    # hack to handle the facbuffer.t that temporaries can have shape but we want to
    # linearly index it here
    buffer_key = (buffer_ref.buffer.name, buffer_ref.nest_indices)
    if buffer_key in context._temporary_shapes:
        shape = context._temporary_shapes[buffer_key]
        rank = len(shape)
        extra_indices = (0,) * (rank - 1)

        # also has to be a scalar, not an expression
        temp_offset_name = context.add_temporary("j")
        temp_offset_var = pym.var(temp_offset_name)
        context.add_assignment(temp_offset_var, offset_expr)
        indices = extra_indices + (temp_offset_var,)
    else:
        indices = (offset_expr,)

    return indices


@_lower_expr.register(op3_expr.Conditional)
def _(cond: op3_expr.Conditional, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.If(_lower_expr(cond.a, *args, **kwargs), _lower_expr(cond.b, *args, **kwargs), _lower_expr(cond.c, *args, **kwargs))


@functools.singledispatch
def register_extent(obj: Any, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@register_extent.register(numbers.Integral)
def _(num: numbers.Integral, *args, **kwargs):
    return num


@register_extent.register(op3_expr.Expression)
def _(expr: op3_expr.Expression, inames, loop_indices, context):
    pym_expr = lower_expr(expr, [inames], loop_indices, context)
    extent_name = context.add_temporary("p")
    context.add_assignment(pym.var(extent_name), pym_expr)
    return extent_name


# FIXME: We assume COMM_SELF here, this is maybe OK if we make sure to use a
# compilation comm at a higher point.
# NOTE: A lot of this is more generic than just loopy, try to refactor
def compile_loopy(translation_unit, *, pyop3_compiler_parameters):
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
    from pyop2.utils import get_petsc_dir
    from pyop2.compilation import load

    code = lp.generate_code_v2(translation_unit).device_code()
    argtypes = [
        cast_loopy_arg_to_ctypes_type(arg) for arg in translation_unit.default_entrypoint.args
    ]
    restype = None

    # ideally move this logic somewhere else
    cppargs = (
        tuple("-I%s/include" % d for d in get_petsc_dir())
        # + tuple("-I%s" % d for d in self.local_kernel.include_dirs)
        # + ("-I%s" % os.path.abspath(os.path.dirname(__file__)),)
    )
    ldargs = (
        tuple("-L%s/lib" % d for d in get_petsc_dir())
        + tuple("-Wl,-rpath,%s/lib" % d for d in get_petsc_dir())
        + ("-lpetsc", "-lm")
        # + tuple(self.local_kernel.ldargs)
    )

    # NOTE: no - instead of this inspect the compiler parameters!!!
    # TODO: Make some sort of function in config.py
    if "LIKWID_MODE" in os.environ:
        cppargs += ("-DLIKWID_PERFMON",)
        ldargs += ("-llikwid",)

    # TODO: needs a comm
    dll = load(code, "c", cppargs, ldargs, pyop2.mpi.COMM_SELF)

    if pyop3_compiler_parameters.add_petsc_event:
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
