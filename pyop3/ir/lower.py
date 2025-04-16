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
from typing import Any

from cachetools import cachedmethod
from petsc4py import PETSc

import loopy as lp
import numpy as np
import pymbolic as pym
from pyop3.array.dat import MatBufferExpression
from pyop3.expr_visitors import collect_axis_vars, extract_axes
from pyrsistent import freeze, pmap, PMap

import pyop2

from pyop3 import utils
from pyop3.array import Dat, _Dat, LinearDatBufferExpression, NonlinearDatBufferExpression, Parameter, Mat, AbstractMat
from pyop3.array.base import Array
from pyop3.axtree.tree import UNIT_AXIS_TREE, Add, AxisVar, IndexedAxisTree, Mul, AxisComponent
from pyop3.buffer import AbstractBuffer, ArrayBuffer, NullBuffer, PackedBuffer
from pyop3.config import config
from pyop3.dtypes import IntType
from pyop3.ir.transform import with_likwid_markers, with_petsc_event, with_attach_debugger
from pyop3.itree.tree import LoopIndexVar
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
    NonEmptyBufferAssignment,
    ContextAwareLoop,  # TODO: remove this class
    ExplicitCalledFunction,
    NonEmptyPetscMatAssignment,
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

# FIXME this needs to be synchronised with TSFC, tricky
# shared base package? or both set by Firedrake - better solution
LOOPY_TARGET = lp.CWithGNULibcTarget()
LOOPY_LANG_VERSION = (2018, 2)


# Is this still needed? Loopy may have fixed this
class OpaqueType(lp.types.OpaqueType):
    def __repr__(self) -> str:
        return f"OpaqueType('{self.name}')"


class CodegenContext(abc.ABC):
    pass


class LoopyCodegenContext(CodegenContext):
    def __init__(self):
        self._domains = []
        self._insns = []
        self._args = []
        self._subkernels = []

        self._within_inames = frozenset()
        self._last_insn_id = None

        self._name_generator = UniqueNameGenerator()

        # TODO remove
        self._dummy_names = {}

        # data argument name -> data argument
        # NOTE: If PETSc Mats were hashable then this could be a WeakSet
        self.data_arguments = weakref.WeakValueDictionary()
        # data argument name -> name in kernel
        self.kernel_arg_names = {}
        # global buffer name -> intent
        self.global_buffer_intents = {}

    @property
    def domains(self):
        return tuple(self._domains)

    @property
    def instructions(self):
        return tuple(self._insns)

    @property
    def arguments(self):
        return tuple(self._args)

    @property
    def subkernels(self):
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
        # TODO recover this functionality, in other words we should produce
        # non-renamed expressions. This means that the Renamer can also register
        # arguments so we only use the ones we actually need!

        # renamer = Renamer(self.actual_to_kernel_rename_map)
        # assignee = renamer(assignee)
        # expression = renamer(expression)

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

    # TODO wrap into add_array
    def add_dummy_argument(self, arg, dtype):
        if arg in self._dummy_names:
            name = self._dummy_names[arg]
        else:
            name = self._dummy_names.setdefault(arg, self._name_generator("dummy"))
        self._args.append(lp.ValueArg(name, dtype=dtype))

    @functools.singledispatchmethod
    def add_buffer(self, buffer: Any, intent: Intent | None = None) -> str:
        raise TypeError(f"No handler defined for {type(buffer).__name__}")

    @add_buffer.register(ArrayBuffer)
    def _(self, buffer: ArrayBuffer, intent: Intent| None = None) -> str:
        if intent is None:
            raise ValueError("Global data must declare intent")

        if buffer.name in self.data_arguments:
            if intent != self.global_buffer_intents[buffer.name]:
                raise ValueError("Cannot have mismatching intents for the same global buffer")
            return self.kernel_arg_names[buffer.name]

        self.data_arguments[buffer.name] = buffer
        self.global_buffer_intents[buffer.name] = intent

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

        kernel_name = self.unique_name("buffer")
        # If the buffer is being passed straight through to a function then we
        # have to make sure that the shapes match
        shape = self._temporary_shapes.get(buffer.name, None)
        arg = lp.GlobalArg(kernel_name, dtype=buffer.dtype, shape=shape)
        self._args.append(arg)
        self.kernel_arg_names[buffer.name] = kernel_name
        return kernel_name

    @add_buffer.register(PETSc.Mat)
    def _(self, buffer: PETSc.Mat, intent: Intent | None = None) -> str:
        if intent is None:
            raise ValueError("Global data must declare intent")

        # TODO: This is the same as for Buffer, refactor
        if buffer.name in self.data_arguments:
            if intent != self.global_buffer_intents[buffer.name]:
                raise ValueError("Cannot have mismatching intents for the same global buffer")
            return self.kernel_arg_names[buffer.name]

        self.data_arguments[buffer.name] = buffer
        self.global_buffer_intents[buffer.name] = intent

        kernel_name = self.unique_name("mat")
        arg = lp.ValueArg(kernel_name, dtype=OpaqueType("Mat"))
        self._args.append(arg)
        self.kernel_arg_names[buffer.name] = kernel_name
        return kernel_name

    @add_buffer.register(NullBuffer)
    def _(self, buffer: NullBuffer, intent: Intent | None = None) -> str:
        # NOTE: 'intent' is not important for temporaries
        if buffer.name in self.data_arguments:
            return self.kernel_arg_names[buffer.name]
        self.data_arguments[buffer.name] = buffer

        shape = self._temporary_shapes.get(buffer.name, (buffer.size,))
        name = self.add_temporary("t", buffer.dtype, shape=shape)
        self.kernel_arg_names[buffer.name] = name
        return name

    def add_temporary(self, prefix="t", dtype=IntType, *, shape=(), **kwargs) -> str:
        name = self.unique_name(prefix)
        arg = lp.TemporaryVariable(
            name,
            dtype=dtype,
            shape=shape,
            address_space=lp.AddressSpace.LOCAL,
            **kwargs,
        )
        self._args.append(arg)
        return name

    def add_parameter(self, parameter: Parameter, *, prefix="p") -> str:
        # TODO: This is the same as for Buffer, refactor
        if parameter.name in self.data_arguments:
            return self.data_arguments[parameter.name]
        self.data_arguments[parameter.name] = parameter

        name = self.unique_name(prefix)
        arg = lp.ValueArg(name, dtype=parameter.dtype)
        self._args.append(arg)
        self.kernel_arg_names[parameter.name] = name
        return name

    def add_subkernel(self, subkernel):
        self._subkernels.append(subkernel)

    # I am not sure that this belongs here, I generate names separately from adding domains etc
    def unique_name(self, prefix):
        return self._name_generator(prefix)

    @contextlib.contextmanager
    def within_inames(self, inames) -> None:
        orig_within_inames = self._within_inames
        self._within_inames |= inames
        yield
        self._within_inames = orig_within_inames

    @property
    def _depends_on(self):
        return frozenset({self._last_insn_id}) - {None}

    def _add_instruction(self, insn):
        self._insns.append(insn)
        self._last_insn_id = insn.id

    # FIXME, bad API
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes


# bad name, bit misleading as this is just the loopy bit, further optimisation
# and lowering to go...
class CodegenResult:
    # TODO: intents and datamap etc maybe all go together. All relate to the same objects
    def __init__(self, loopy_kernel, data_arguments, global_buffer_intents, compiler_parameters):
        self.loopy_kernel = loopy_kernel
        self.data_arguments = data_arguments
        self.global_buffer_intents = global_buffer_intents
        self.compiler_parameters = compiler_parameters

        # self._cache = collections.defaultdict(dict)

    # @cachedmethod(lambda self: self._cache["CodegenResult._compile"])
    # not really needed, just a @property
    def _compile(self):
        return compile_loopy(self.loopy_kernel, pyop3_compiler_parameters=self.compiler_parameters)

    def __call__(self, **kwargs):
        if not self.global_buffer_intents:
            warnings.warn(
                "Attempting to execute a kernel that does not touch any global "
                "data, skipping"
            )
            return

        executable = self._compile()

        # TODO: Check each of kwargs and make sure that the replacement is
        # valid (e.g. same size, same data type, same layout funcs).
        kernel_args = []
        for data_argument in self.data_arguments:
            data_argument = kwargs.get(data_argument.name, data_argument)
            kernel_args.append(as_kernel_arg(data_argument))

        if len(self.loopy_kernel.callables_table) > 1:
            ccode = lp.generate_code_v2(self.loopy_kernel).device_code()
            breakpoint()

        executable(*kernel_args)

    def target_code(self, target):
        raise NotImplementedError("TODO")


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
    context._insns.append(noop)

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
        # options=lp.Options(check_dep_resolution=False),
    )

    entrypoint = translation_unit.default_entrypoint
    if compiler_parameters.add_likwid_markers:
        entrypoint = with_likwid_markers(entrypoint)
    if compiler_parameters.add_petsc_event:
        entrypoint = with_petsc_event(entrypoint)
    if compiler_parameters.attach_debugger:
        entrypoint = with_attach_debugger(entrypoint)
    translation_unit = translation_unit.with_kernel(entrypoint)

    translation_unit = lp.merge((translation_unit, *context.subkernels))

    # add callables
    # tu = lp.register_callable(tu, "bsearch", BinarySearchCallable())

    # needed?
    translation_unit = translation_unit.with_entrypoints(entrypoint.name)

    data_argument_names = utils.invert_mapping(context.kernel_arg_names)
    data_arguments = []
    buffer_intents = {}
    for kernel_arg in entrypoint.args:
        data_argument_name = data_argument_names[kernel_arg.name]
        data_arguments.append(context.data_arguments[data_argument_name])

    return CodegenResult(translation_unit, data_arguments, context.global_buffer_intents, compiler_parameters)


# put into a class in transform.py?
@functools.singledispatch
def _collect_temporary_shapes(expr):
    raise TypeError(f"No handler defined for {type(expr).__name__}")


@_collect_temporary_shapes.register(InstructionList)
def _(insn_list: InstructionList, /) -> PMap:
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
def _(assignment: AbstractAssignment, /) -> PMap:
    return pmap()


@_collect_temporary_shapes.register
def _(call: ExplicitCalledFunction):
    return freeze(
        {
            arg.buffer.name: lp_arg.shape
            for lp_arg, arg in zip(
                call.function.code.default_entrypoint.args, call.arguments, strict=True
            )
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


@_compile.register(ContextAwareLoop)  # remove
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
        path = pmap()
        iname_map = pmap()

    for component in axis.components:
        path_ = path | {axis.label: component.label}

        if component._collective_size != 1:
            iname = codegen_context.unique_name("i")
            domain_var = register_extent(
                component._collective_size,
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
                for axis_label, index_expr in index_exprs.get(axis_key).items():
                    loop_exprs[(loop.index.id, axis_label)] = lower_expr(index_expr, READ, [iname_replace_map_], loop_indices, codegen_context, paths=[path_])
            loop_exprs = pmap(loop_exprs)

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
def _(call: ExplicitCalledFunction, loop_indices, ctx: LoopyCodegenContext) -> None:
    temporaries = []
    subarrayrefs = {}
    extents = {}

    # loopy args can contain ragged params too
    loopy_args = call.function.code.default_entrypoint.args[: len(call.arguments)]
    for loopy_arg, arg, spec in zip(loopy_args, call.arguments, call.argspec, strict=True):
        # this check fails because we currently assume that all arrays require packing
        # from pyop3.transform import _requires_pack_unpack
        # assert not _requires_pack_unpack(arg)
        # old names
        temporary = arg
        indexed_temp = arg

        if isinstance(arg, DummyKernelArgument):
            ctx.add_dummy_argument(arg, loopy_arg.dtype)
            name = ctx._dummy_names[arg]
            subarrayrefs[arg] = pym.var(name)
        else:
            if loopy_arg.shape is None:
                shape = (temporary.alloc_size,)
            else:
                if np.prod(loopy_arg.shape, dtype=int) != temporary.alloc_size:
                    raise RuntimeError("Shape mismatch between inner and outer kernels")
                shape = loopy_arg.shape

            temporaries.append((arg, indexed_temp, spec.intent, shape))

            # Register data
            # TODO This might be bad for temporaries
            # NOTE: not sure why I added this condition before
            # if isinstance(arg, _Dat):
            temp_name = ctx.add_buffer(arg.buffer, spec.intent)

            # this should already be done in an assignment
            # ctx.add_temporary(temporary.name, temporary.dtype, shape)

            # subarrayref nonsense/magic
            indices = []
            for s in shape:
                iname = ctx.unique_name("i")
                ctx.add_domain(iname, s)
                indices.append(pym.var(iname))
            indices = tuple(indices)

            subarrayrefs[arg] = lp.symbolic.SubArrayRef(
                indices, pym.subscript(pym.var(temp_name), indices)
            )

    # TODO this is pretty much the same as what I do in fix_intents in loopexpr.py
    # probably best to combine them - could add a sensible check there too.
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
        )
        + tuple(extents.values()),
    )

    ctx.add_function_call(assignees, expression)
    ctx.add_subkernel(call.function.code)


# FIXME this is practically identical to what we do in build_loop
@_compile.register(NonEmptyBufferAssignment)
def parse_assignment(
    assignment,
    loop_indices,
    codegen_ctx,
):
    parse_assignment_properly_this_time(
        assignment,
        loop_indices,
        codegen_ctx,
        assignment.axis_trees,
    )


@_compile.register(NonEmptyPetscMatAssignment)
def _compile_petscmat(assignment, loop_indices, context):
    mat = assignment.mat
    array = assignment.values

    assert isinstance(mat.buffer, PETSc.Mat)

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
    array_name = context.add_buffer(assignment.expression.buffer, READ)

    for row_path in assignment.row_axis_tree.leaf_paths:
        for col_path in assignment.col_axis_tree.leaf_paths:
            row_layout = mat.row_layouts[row_path]
            rmap_name = context.add_buffer(row_layout.buffer, READ)

            col_layout = mat.column_layouts[col_path]
            cmap_name = context.add_buffer(col_layout.buffer, READ)

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

    # TODO: The following code should be done in a loop per submat.

            # TODO: can be made much much nicer
            def get_linear_size(axis_tree, path):
                linear_size = 1
                visited = axis_tree.path_with_nodes(axis_tree._node_from_path(row_path), and_components=True)
                for axis, component in visited.items():
                    assert component.size > 0
                    linear_size *= component.size
                return linear_size


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
            rsize = get_linear_size(assignment.row_axis_tree, row_path)
            csize = get_linear_size(assignment.col_axis_tree, col_path)

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

            # replace inner bits with zeros
            rzeros = {var.axis_label: 0 for var in collect_axis_vars(row_layout)}
            irow = str(lower_expr(row_layout, READ, [rzeros], loop_indices, context))

            czeros = {var.axis_label: 0 for var in collect_axis_vars(col_layout)}
            icol = str(lower_expr(col_layout, READ, [czeros], loop_indices, context))

            array_row_layout = assignment.values.row_layouts[row_path]
            array_row_expr = lower_expr(array_row_layout, READ, [rzeros], loop_indices, context)
            array_col_layout = assignment.values.column_layouts[col_path]
            array_col_expr = lower_expr(array_col_layout, READ, [czeros], loop_indices, context)

            column_size = assignment.col_axis_tree.size
            array_indices = array_row_expr * column_size + array_col_expr
            array_expr = str(pym.subscript(pym.var(array_name), array_indices))

            # FIXME:
            blocked = False

            # hacky
            myargs = [
                assignment, mat_name, array_expr, rsize_var, csize_var, irow, icol, blocked
            ]
            match assignment.access_type:
                case ArrayAccessType.READ:
                    call_str = _petsc_mat_load(*myargs)
                case ArrayAccessType.WRITE:
                    call_str = _petsc_mat_store(*myargs)
                case ArrayAccessType.INC:
                    call_str = _petsc_mat_add(*myargs)
                case _:
                    raise AssertionError

            context.add_cinstruction(call_str)


def _petsc_mat_load(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}));"
    else:
        return f"MatGetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}));"


def _petsc_mat_store(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}), INSERT_VALUES);"
    else:
        return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}), INSERT_VALUES);"


def _petsc_mat_add(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}), ADD_VALUES);"
    else:
        return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}), ADD_VALUES);"

# TODO now I attach a lot of info to the context-free array, do I need to pass axes around?
def parse_assignment_properly_this_time(
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

        paths += [pmap()]
        iname_replace_maps += [pmap()]

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
        if component._collective_size != 1:
            iname = codegen_context.unique_name("i")

            extent_var = register_extent(
                component._collective_size,
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
                parse_assignment_properly_this_time(
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
                parse_assignment_properly_this_time(
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
    lexpr = lower_expr(assignment.assignee, intent, iname_replace_maps, loop_indices, codegen_context, paths=paths)
    rexpr = lower_expr(assignment.expression, READ, iname_replace_maps, loop_indices, codegen_context, paths=paths)

    if assignment.assignment_type == AssignmentType.INC:
        rexpr = lexpr + rexpr

    codegen_context.add_assignment(lexpr, rexpr)


# NOTE: There is a difference here between `lower_expr` and `lower_expr_linear` (where paths are not relevant)
@functools.singledispatch
def lower_expr(obj: Any, /, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@lower_expr.register(Add)
def _(add: Add, /, *args, **kwargs):
    return lower_expr(add.a, *args, **kwargs) + lower_expr(add.b, *args, **kwargs)


@lower_expr.register(Mul)
def _(mul: Mul, /, *args, **kwargs):
    return lower_expr(mul.a, *args, **kwargs) * lower_expr(mul.b, *args, **kwargs)


@lower_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs):
    return num


@lower_expr.register(AxisVar)
def _(axis_var: AxisVar, /, intent, iname_maps, *args, **kwargs):
    if len(iname_maps) > 1:
        raise NotImplementedError("not sure")
    else:
        iname_map = just_one(iname_maps)
    return iname_map[axis_var.axis_label]


@lower_expr.register
def _(loop_var: LoopIndexVar, intent, iname_maps, loop_indices, *args, **kwargs):
    return loop_indices[(loop_var.loop_id, loop_var.axis_label)]


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


@lower_expr.register(NonlinearDatBufferExpression)
def _(expr: NonlinearDatBufferExpression, /, intent, iname_maps, loop_indices, context, paths):
    path = just_one(paths)
    layout = expr.layouts[path]
    return lower_buffer_access(expr.buffer, layout, intent, iname_maps, loop_indices, context)


@lower_expr.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, intent, iname_maps, loop_indices, context, **kwargs):
    return lower_buffer_access(expr.buffer, expr.layout, intent, iname_maps, loop_indices, context)


def lower_buffer_access(buffer, layout, intent, iname_maps, loop_indices, context):
    name_in_kernel = context.add_buffer(buffer, intent)

    iname_map = just_one(iname_maps)
    offset_expr = lower_expr(layout, READ, [iname_map], loop_indices, context)
    indices = maybe_multiindex(buffer, offset_expr, context)

    return pym.subscript(pym.var(name_in_kernel), indices)



@lower_expr.register(MatBufferExpression)
def _(expr: MatBufferExpression, /, intent, iname_maps, loop_indices, context, paths):
    name_in_kernel = context.add_buffer(expr.buffer, intent)

    row_iname_map, col_iname_map = iname_maps
    row_path, col_path = paths

    row_layout_expr = expr.row_layouts[row_path]
    row_offset_expr = lower_expr(row_layout_expr, READ, [row_iname_map], loop_indices, context)

    col_layout_expr = expr.column_layouts[col_path]
    col_offset_expr = lower_expr(col_layout_expr, READ, [col_iname_map], loop_indices, context)

    _, column_size = expr.buffer.shape
    offset_expr = row_offset_expr * column_size + col_offset_expr
    indices = maybe_multiindex(expr.buffer, offset_expr, context)

    return pym.subscript(pym.var(name_in_kernel), indices)


# @lower_expr.register(_ExpressionDat)
# def _(dat: _ExpressionDat, /, intent, iname_maps, loop_indices, context, paths=None):
#     kernel_arg_name = context.add_buffer(dat.buffer, intent)
#
#     iname_map = just_one(iname_maps)
#     offset_expr = lower_expr(dat.layout, READ, [iname_map], loop_indices, context)
#     indices = maybe_multiindex(dat, offset_expr, context)
#
#     return pym.subscript(pym.var(kernel_arg_name), indices)


@functools.singledispatch
def register_extent(obj: Any, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(extent).__name__}")


@register_extent.register(numbers.Integral)
def _(num: numbers.Integral, *args, **kwargs):
    return num


@register_extent.register(Parameter)
def _(param: Parameter, inames, loop_indices, context):
    return context.add_parameter(param)

@register_extent.register(Dat)  # is this right? should this already be parsed?
@register_extent.register(LinearDatBufferExpression)
def _(extent, /, intent, iname_replace_map, loop_indices, context):
    expr = lower_expr(extent, READ, [iname_replace_map], loop_indices, context)
    varname = context.add_temporary("p")
    context.add_assignment(pym.var(varname), expr)
    return varname


# lives here??
@functools.singledispatch
def as_kernel_arg(obj: Any) -> numbers.Number:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


# @as_kernel_arg.register(numbers.Integral)
# def _(num: numbers.Integral) -> int:
#     return int(num)


@as_kernel_arg.register(Parameter)
def _(param: Parameter) -> np.number:
    return param.value


# @as_kernel_arg.register(np.ndarray)
# def _(array: np.ndarray) -> int:
#     return array.ctypes.data


# @as_kernel_arg.register
# def _(arg: _Dat):
#     # TODO if we use the right accessor here we modify the state appropriately
#     return as_kernel_arg(arg.buffer)


@as_kernel_arg.register
def _(arg: ArrayBuffer) -> int:
    # TODO if we use the right accessor here we modify the state appropriately
    # NOTE: Do not use .data_rw accessor here since this would trigger a halo exchange
    return arg._data.ctypes.data


# @as_kernel_arg.register
# def _(arg: PackedBuffer):
#     return as_kernel_arg(arg.array)


# @as_kernel_arg.register
# def _(array: AbstractMat):
#     return array.mat.handle


# NOTE: I think that we should probably have a MatBuffer or similar type so
# array.buffer is universal
@as_kernel_arg.register(PETSc.Mat)
def _(mat):
    return mat.handle


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
