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
from immutabledict import immutabledict
from pyop3.tensor.dat import ArrayBufferExpression, MatArrayBufferExpression, MatPetscMatBufferExpression, OpaqueBufferExpression
from pyop3.expr_visitors import collect_axis_vars

import pyop2

from pyop3 import exceptions as exc, utils
from pyop3.tensor import LinearDatArrayBufferExpression, NonlinearDatArrayBufferExpression, Scalar
from pyop3.axtree.tree import UNIT_AXIS_TREE, Add, AxisVar, IndexedAxisTree, Mul, AxisComponent, relabel_path
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.config import config
from pyop3.dtypes import IntType
from pyop3.ir.transform import with_likwid_markers, with_petsc_event, with_attach_debugger
from pyop3.itree.tree import AffineSliceComponent, LoopIndexVar, Slice, IndexTree
from pyop3.lang import (
    Intent,
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    AbstractAssignment,
    NullInstruction,
    assignment_type_as_intent,
    parse_compiler_parameters,
    WRITE,
    AssignmentType,
    ConcretizedNonEmptyArrayAssignment,
    StandaloneCalledFunction,
    # NonEmptyPetscMatAssignment,
    PreprocessedExpression,
    UnprocessedExpressionException,
    DummyKernelArgument,
    Loop,
    InstructionList,
    ArrayAccessType,
)
from pyop3.log import logger
from pyop3.utils import (
    PrettyTuple,
    StrictlyUniqueDict,
    UniqueNameGenerator,
    as_tuple,
    just_one,
    maybe_generate_name,
    merge_dicts,
    single_valued,
    strictly_all,
    Identified,
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
        self.global_buffers = weakref.WeakValueDictionary()
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

    def add_buffer(self, buffer: AbstractBuffer, intent: Intent | None = None) -> str:
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
        # TODO: make a singledispatchmethod
        if isinstance(buffer, NullBuffer):
            # 'intent' is not important for temporaries
            if buffer.name in self._kernel_names:
                return self._kernel_names[buffer.name]
            shape = self._temporary_shapes.get(buffer.name, (buffer.size,))
            name_in_kernel = self.add_temporary("t", buffer.dtype, shape=shape)
        else:
            if intent is None:
                raise ValueError("Global data must declare intent")

            if buffer.name in self._kernel_names:
                if intent != self.global_buffer_intents[buffer.name]:
                    # We are accessing a buffer with different intents so have to
                    # pessimally claim RW access
                    self.global_buffer_intents[buffer.name] = RW
                return self._kernel_names[buffer.name]

            if isinstance(buffer, ArrayBuffer):
                # Inject constant buffer data into the generated code if sufficiently small
                # TODO: Need to consider 'constant-ness'. Something may be immutable but still
                # not match across ranks.
                # Maybe sf check is enough?
                if buffer.constant and buffer.size < config["max_static_array_size"]:
                    assert not buffer.sf, "sufficient check?"
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
                shape = self._temporary_shapes.get(buffer.name, None)
                loopy_arg = lp.GlobalArg(name_in_kernel, dtype=buffer.dtype, shape=shape)
            else:
                assert isinstance(buffer, PetscMatBuffer)

                name_in_kernel = self.unique_name("mat")
                loopy_arg = lp.ValueArg(name_in_kernel, dtype=OpaqueType("Mat"))

            self.global_buffers[buffer.name] = buffer
            self.global_buffer_intents[buffer.name] = intent
            self._arguments.append(loopy_arg)

        self._kernel_names[buffer.name] = name_in_kernel
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
    def __call__(self, **kwargs):
        pass


class ModuleExecutor:
    """
    Notes
    -----
    This class has a large number of cached properties to reduce latency when it
    is called.

    """

    # TODO: intents and datamap etc maybe all go together. All relate to the same objects
    def __init__(self, loopy_code: lp.TranslationUnit, buffer_map: WeakValueDictionary[str, ConcretBuffer], buffer_intents: Mapping[str, Intent], compiler_parameters: Mapping):
        self.loopy_code = loopy_code
        self.buffer_map = buffer_map
        self.buffer_intents = buffer_intents
        self.compiler_parameters = compiler_parameters

    @property
    def loopy_kernel(self) -> lp.LoopKernel:
        return self.loopy_code.default_entrypoint

    @cached_property
    def _buffers(self) -> tuple[AbstractBuffer]:
        return tuple(self.buffer_map.values())

    @cached_property
    def executable(self):
        return compile_loopy(self.loopy_code, pyop3_compiler_parameters=self.compiler_parameters)

    def __call__(self, **kwargs):
        if not kwargs:  # shortcut for the most common case
            buffers = self._buffers
            exec_arguments = self._default_exec_arguments
        else:
            buffers = list(self._buffers)
            exec_arguments = list(self._default_exec_arguments)

            # TODO:
            # if config.debug:
            if False:
                for buffer_name, replacement_buffer in kwargs.items():
                    self._check_buffer_is_valid(self.buffer_map[buffer_name], replacement_buffer)

            for buffer_name, replacement_buffer in kwargs.items():
                index = self._buffer_indices[buffer_name]
                buffers[index] = replacement_buffer
                exec_arguments[index] = self._as_exec_argument(replacement_buffer)

        for index in self._modified_buffer_indices:
            buffers[index].inc_state()

        ccode = lambda: lp.generate_code_v2(self.loopy_code).device_code()

        # if len(self.loopy_code.callables_table) > 1:
        #     breakpoint()
        # pyop3.extras.debug.maybe_breakpoint()

        self.executable(*exec_arguments)

    @cached_property
    def _buffer_indices(self) -> immutabledict[str, int]:
        return immutabledict({
            buffer.name: i for i, buffer in enumerate(self._buffers)
        })

    @cached_property
    def _modified_buffer_indices(self) -> tuple[int]:
        return tuple(
            i
            for i, intent in enumerate(self.buffer_intents.values())
            if intent != READ
        )

    @cached_property
    def _default_exec_arguments(self) -> tuple[numbers.Number]:
        return tuple(self._as_exec_argument(buffer) for buffer in self._buffers)

    def _as_exec_argument(self, buffer: AbstractBuffer) -> numbers.Number:
        if isinstance(buffer, ArrayBuffer):
            # NOTE: Use the private accessor ._data here to avoid triggering
            # a halo exchange
            return buffer._data.ctypes.data
        else:
            assert isinstance(buffer, PetscMatBuffer)
            return buffer.mat.handle

    def _check_buffer_is_valid(self, orig_buffer: AbstractBuffer, new_buffer: AbstractBuffer, /) -> None:
        valid = (
            type(new_buffer) is type(orig_buffer)
            and new_buffer.size == orig_buffer.size
            and new_buffer.dtype == orig_buffer.dtype
        )
        if not valid:
            raise exc.BufferMismatchException()


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
def compile(expr: PreprocessedExpression, compiler_parameters=None):
    if not isinstance(expr, PreprocessedExpression):
        raise UnprocessedExpressionException("Expected a preprocessed expression")

    insn = expr.expression

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
    sorted_buffers = WeakValueDictionary()
    for kernel_arg in entrypoint.args:
        buffer_name = kernel_to_buffer_names[kernel_arg.name]
        sorted_buffers[kernel_arg.name] = context.global_buffers[buffer_name]

    return ModuleExecutor(translation_unit, sorted_buffers, context.global_buffer_intents, compiler_parameters)


# put into a class in transform.py?
@functools.singledispatch
def _collect_temporary_shapes(expr):
    raise TypeError(f"No handler defined for {type(expr).__name__}")


@_collect_temporary_shapes.register(InstructionList)
def _(insn_list: InstructionList, /) -> immutabledict:
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
def _(assignment: AbstractAssignment, /) -> immutabledict:
    return immutabledict()


@_collect_temporary_shapes.register
def _(call: StandaloneCalledFunction):
    return immutabledict(
        {
            arg.buffer.name: lp_arg.shape
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
):
    if strictly_all(x is None for x in {axis, path, iname_map}):
        axis = axes.root
        path = immutabledict()
        iname_map = immutabledict()

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
            # think this is now wrong, iname_replace_map_ is sufficient
            # iname_map = iname_replace_map_ | loop_indices
            axis_key = (axis.id, component.label)
            for index_exprs in axes.index_exprs:
                for axis_label, index_expr in index_exprs.get(axis_key, {}).items():
                    loop_exprs[(loop.index.id, axis_label)] = lower_expr(index_expr, [iname_replace_map_], loop_indices, codegen_context, paths=[path_])
            loop_exprs = immutabledict(loop_exprs)

            if subaxis := axes.child(axis, component):
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
                for stmt in loop.statements:
                    _compile(
                        stmt,
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

        # subarrayref nonsense/magic
        indices = []
        for s in loopy_arg.shape:
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
    if any(isinstance(arg, OpaqueBufferExpression) for arg in assignment.arguments):
        _compile_petsc_mat(assignment, loop_indices, context)
    else:
        compile_array_assignment(
            assignment,
            loop_indices,
            context,
            assignment.axis_trees,
        )


def _compile_petsc_mat(assignment: ConcretizedNonEmptyArrayAssignment, loop_indices, context):
    mat = assignment.assignee
    expr = assignment.expression

    row_axis_tree, column_axis_tree = assignment.axis_trees

    if isinstance(expr, numbers.Number):
        # If we have an expression like
        #
        #     mat[f(p), f(p)] <- 666
        #
        # then we have to convert `666` into an appropriately sized temporary
        # for Mat{Get,Set}Values to work.
        # TODO: There must be a more elegant way of doing this
        nrows = row_axis_tree.size
        ncols = column_axis_tree.size
        expr_data = np.full((nrows, ncols), expr, dtype=mat.buffer.dtype)
        array_buffer = ArrayBuffer(expr_data, constant=True)
    else:
        assert isinstance(expr, ArrayBufferExpression)
        array_buffer = expr.buffer

    if not isinstance(mat.buffer, PetscMatBuffer):
        raise NotImplementedError  # order must be different
    else:
        # We need to know whether the matrix is the assignee or not because we need
        # to know whether to put MatGetValues or MatSetValues
        setting_mat_values = True

    # tidy this up
    # if mat.mat.nested:
    #     ridx, cidx = map(just_one, just_one(mat.nest_labels))
    #     # if ridx is None:
    #     #     ridx = 0
    #     # if cidx is None:
    #     #     cidx = 0
    #
    #     if mat.mat_type[ridx, cidx] == "dat":
    #         # no preallocation is necessary
    #         if isinstance(mat, Sparsity):
    #             return
    #
    #         breakpoint()

    # now emit the right line of code, this should properly be a lp.ScalarCallable
    # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
    mat_name = context.add_buffer(assignment.assignee.buffer, assignment_type_as_intent(assignment.assignment_type))

    # NOTE: Is this always correct? It is for now.
    array_name = context.add_buffer(array_buffer, READ)

    # TODO: The following code should be done in a loop per submat.
    # blocked = mat.mat.block_shape > 1
    # if mat.nested:
    #     if len(mat.nest_labels) > 1:
    #         # Need to loop over the different nest labels and emit separate calls to
    #         # MatSetValues, maps may also be wrong.
    #         raise NotImplementedError
    #
    #     submat_name = codegen_context.unique_name("submat")
    #     ridxs, cidxs = just_one(mat.nest_labels)
    #
    #     if any(len(x) != 1 for x in {ridxs, cidxs}):
    #         raise NotImplementedError
    #
    #     (ridx,) = ridxs
    #     (cidx,) = cidxs
    #
    #     if ridx is None:
    #         ridx = 0
    #     if cidx is None:
    #         cidx = 0
    #
    #     code = textwrap.dedent(
    #         f"""
    #         Mat {submat_name};
    #         MatNestGetSubMat({mat_name}, {ridx}, {cidx}, &{submat_name});
    #         """
    #     )
    #     codegen_context.add_cinstruction(code)
    #     mat_name = submat_name

    # rmap_name = context.add_buffer(mat.row_layout.buffer, READ)
    #
    # cmap_name = context.add_buffer(mat.column_layout.buffer, READ)

    # def get_linear_size(axis_tree, path):
    #     linear_size = 1
    #     visited = axis_tree.path_with_nodes(axis_tree._node_from_path(path), and_components=True)
    #     for axis, component in visited.items():
    #         assert component.size > 0
    #         linear_size *= component.size
    #     return linear_size


    # This code figures out the sizes of the maps - it's simply the
    # size of the assignment axes but linearised for the particular
    # row and column paths.
    # For example, if we have a P2 triangle then we have row and col maps
    # with arities [3, 3] (and those values should be embedded in MatSetValues)
    # 
    # The indexed axis tree for the row (and col) looks like:
    #
    #   {closure: [{0: [(3, None)]}, {1: [(3, None)]}]}
    #   ├──➤ {dof0: [[(1, None)]]}
    #   └──➤ {dof1: [[(1, None)]]}
    #
    # where it can clearly be seen that, for each path, the size is 3.
    # rsize = get_linear_size(assignment.row_axis_tree, row_path)
    # csize = get_linear_size(assignment.column_axis_tree, col_path)
    rsize = row_axis_tree.size
    csize = column_axis_tree.size

    # these sizes can be expressions that need evaluating
    if not isinstance(rsize, numbers.Integral):
        raise NotImplementedError
        rsize_var = register_extent(
            rsize,
            loop_indices,
            context,
        )
    else:
        rsize_var = rsize

    if not isinstance(csize, numbers.Integral):
        raise NotImplementedError
        csize_var = register_extent(
            csize,
            loop_indices,
            context,
        )
    else:
        csize_var = csize

    irow = lower_expr(mat.row_layout, ((),), loop_indices, context)
    icol = lower_expr(mat.column_layout, ((),), loop_indices, context)

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
    axis=None,
    paths=None,
):
    if paths is None:
        paths = []
    if iname_replace_maps is None:
        iname_replace_maps = []

    if axis_tree is None:
        axis_tree, *axis_trees = axis_trees

        paths += [immutabledict()]
        iname_replace_maps += [immutabledict()]

        if axis_tree.is_empty or axis_tree is UNIT_AXIS_TREE or isinstance(axis, IndexedAxisTree):
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

        axis = axis_tree.root

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
            if subaxis := axis_tree.child(axis, component):
                compile_array_assignment(
                    assignment,
                    loop_indices,
                    codegen_context,
                    axis_trees,
                    iname_replace_maps=new_iname_replace_maps,
                    axis_tree=axis_tree,
                    axis=subaxis,
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
                    axis=None,
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


def lower_expr(expr, iname_maps, loop_indices, ctx, *, intent=READ, paths=None, shape=None) -> pym.Expression:
    return _lower_expr(expr, iname_maps, loop_indices, ctx, intent=intent, paths=paths, shape=shape)


@functools.singledispatch
def _lower_expr(obj: Any, /, *args, **kwargs) -> pym.Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_lower_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
    return num


@_lower_expr.register(Add)
def _(add: Add, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(add.a, *args, **kwargs) + _lower_expr(add.b, *args, **kwargs)


@_lower_expr.register(Mul)
def _(mul: Mul, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mul.a, *args, **kwargs) * _lower_expr(mul.b, *args, **kwargs)


@_lower_expr.register(AxisVar)
def _(axis_var: AxisVar, /, iname_maps, *args, **kwargs) -> pym.Expression:
    return just_one(iname_maps)[axis_var.axis_label]


@_lower_expr.register(LoopIndexVar)
def _(loop_var: LoopIndexVar, /, iname_maps, loop_indices, *args, **kwargs) -> pym.Expression:
    return loop_indices[(loop_var.loop_id, loop_var.axis_label)]


@_lower_expr.register(LinearDatArrayBufferExpression)
def _(expr: LinearDatArrayBufferExpression, /, iname_maps, loop_indices, context, *, intent, **kwargs) -> pym.Expression:
    return lower_buffer_access(expr.buffer, [expr.layout], iname_maps, loop_indices, context, intent=intent)


@_lower_expr.register(NonlinearDatArrayBufferExpression)
def _(expr: NonlinearDatArrayBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, **kwargs) -> pym.Expression:
    path = just_one(paths)
    return lower_buffer_access(expr.buffer, [expr.layouts[path]], iname_maps, loop_indices, context, intent=intent)


@_lower_expr.register(MatArrayBufferExpression)
def _(expr: MatArrayBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, shape) -> pym.Expression:
    row_path, column_path = paths
    layouts = (expr.row_layouts[row_path], expr.column_layouts[column_path])
    return lower_buffer_access(expr.buffer, layouts, iname_maps, loop_indices, context, intent=intent, shape=shape)


def lower_buffer_access(buffer: AbstractBuffer, layouts, iname_maps, loop_indices, context, *, intent, shape=None) -> pym.Expression:
    name_in_kernel = context.add_buffer(buffer, intent)

    offset_expr = 0
    strides = reversed(utils.strides(shape)) if shape else (1,)
    for stride, layout, iname_map in zip(strides, layouts, iname_maps, strict=True):
        offset_expr += stride * lower_expr(layout, [iname_map], loop_indices, context)

    indices = maybe_multiindex(buffer, offset_expr, context)
    return pym.subscript(pym.var(name_in_kernel), indices)


def maybe_multiindex(buffer, offset_expr, context):
    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    if buffer.name in context._temporary_shapes:
        shape = context._temporary_shapes[buffer.name]
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


@functools.singledispatch
def register_extent(obj: Any, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@register_extent.register(numbers.Integral)
def _(num: numbers.Integral, *args, **kwargs):
    return num


@register_extent.register(Scalar)
def _(param: Scalar, inames, loop_indices, context):
    name_in_kernel = context.add_buffer(param.buffer, READ)
    extent_name = context.add_temporary("p")
    context.add_assignment(pym.var(extent_name), pym.var(name_in_kernel)[0])
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
