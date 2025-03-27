from __future__ import annotations

import abc
import collections
import contextlib
import dataclasses
import enum
import functools
import numbers
import textwrap
from functools import cached_property
from typing import Any
import weakref

from cachetools import cachedmethod
from petsc4py import PETSc

import loopy as lp
import numpy as np
import pymbolic as pym
from pyrsistent import freeze, pmap, PMap

import pyop2

from pyop3.array import Dat, _Dat, _ExpressionDat, _ConcretizedDat, _ConcretizedMat
from pyop3.array.base import Array
from pyop3.array.petsc import Mat, AbstractMat
from pyop3.axtree.tree import UNIT_AXIS_TREE, Add, AxisVar, IndexedAxisTree, Mul, AxisComponent
from pyop3.buffer import Buffer, NullBuffer, PackedBuffer
from pyop3.config import config
from pyop3.dtypes import IntType
from pyop3.ir.transform import with_likwid_markers, with_petsc_event
from pyop3.itree.tree import LoopIndexVar
from pyop3.lang import (
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    NA,
    READ,
    RW,
    AbstractAssignment,
    NullInstruction,
    parse_compiler_parameters,
    WRITE,
    AssignmentType,
    NonEmptyBufferAssignment,
    ContextAwareLoop,  # TODO: remove this class
    DirectCalledFunction,
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
    Parameter,
    PrettyTuple,
    StrictlyUniqueDict,
    UniqueNameGenerator,
    as_tuple,
    checked_zip,
    just_one,
    merge_dicts,
    single_valued,
    strictly_all,
    Identified,
)

# FIXME this needs to be synchronised with TSFC, tricky
# shared base package? or both set by Firedrake - better solution
LOOPY_TARGET = lp.CWithGNULibcTarget()
LOOPY_LANG_VERSION = (2018, 2)


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

        self.actual_to_kernel_rename_map = {}

        self._within_inames = frozenset()
        self._last_insn_id = None

        self._name_generator = UniqueNameGenerator()

        # TODO remove
        self._dummy_names = {}

        self._seen_arrays = set()

        self.datamap = weakref.WeakValueDictionary()

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

    @property
    def kernel_to_actual_rename_map(self):
        return {
            kernel: actual
            for actual, kernel in self.actual_to_kernel_rename_map.items()
        }

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

    # TODO we pass a lot more data here than we need I think, need to use unique *buffers*
    def add_array(self, array: Array) -> None:
        if array.name in self._seen_arrays:
            return
        self._seen_arrays.add(array.name)

        self.datamap[array.name] = array

        if isinstance(array.buffer, NullBuffer):
            name = self.unique_name("t")
            shape = self._temporary_shapes.get(array.name, (array.alloc_size,))
            arg = lp.TemporaryVariable(
                name, dtype=array.dtype, shape=shape, read_only=array.constant, address_space=lp.AddressSpace.LOCAL,
            )
        # NOTE: It only makes sense to inject *buffers*, not, say, Dats
        # except that isn't totally true. One could have dat[::2] as a constant and only
        # want to inject the used entries
        elif array.constant and array.buffer.size < config["max_static_array_size"]:
            name = self.unique_name("t")
            arg = lp.TemporaryVariable(
                name,
                dtype=array.dtype,
                initializer=array.buffer.data_ro,
                address_space=lp.AddressSpace.LOCAL,
                read_only=True,
            )
        elif isinstance(array.buffer, Buffer):
            name = self.unique_name("buffer")
            arg = lp.GlobalArg(name, dtype=self._dtype(array), shape=None)
        else:
            assert isinstance(array.buffer, PETSc.Mat)
            name = self.unique_name("mat")
            arg = lp.ValueArg(name, dtype=self._dtype(array))

        self.actual_to_kernel_rename_map[array.name] = name
        self._args.append(arg)

    # can this now go? no, not all things are arrays
    def add_temporary(self, name, dtype=IntType, shape=()):
        temp = lp.TemporaryVariable(name, dtype=dtype, shape=shape)
        self._args.append(temp)

    def add_parameter(self, parameter: Parameter) -> str:
        self.datamap[parameter.id] = parameter

        name = self.unique_name("p")
        self.actual_to_kernel_rename_map[parameter.id] = name

        loopy_arg = lp.ValueArg(name, dtype=parameter.box.dtype)
        self._args.append(loopy_arg)

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

    # TODO Perhaps this should be made more public so external users can register
    # arguments. I don't want to make it a property to attach to the objects since
    # that would "tie us in" to loopy more than I would like.
    @functools.singledispatchmethod
    def _dtype(self, array):
        """Return the dtype corresponding to a given kernel argument.

        This function is required because we need to distinguish between the
        dtype of the data stored in the array and that of the array itself. For
        basic arrays loopy can figure this out but for complex types like `PetscMat`
        it would otherwise get this wrong.

        """
        raise TypeError(f"No handler provided for {type(array).__name__}")

    @_dtype.register(_Dat)
    def _(self, array):
        return self._dtype(array.buffer)

    @_dtype.register(Buffer)
    def _(self, array):
        return array.dtype

    @_dtype.register
    def _(self, array: PackedBuffer):
        return self._dtype(array.array)

    @_dtype.register
    def _(self, array: AbstractMat):
        return OpaqueType("Mat")

    @_dtype.register(PETSc.Mat)
    def _(self, mat):
        return OpaqueType("Mat")

    def _add_instruction(self, insn):
        self._insns.append(insn)
        self._last_insn_id = insn.id

    # FIXME, bad API
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes


# bad name, bit misleading as this is just the loopy bit, further optimisation
# and lowering to go...
class CodegenResult:
    def __init__(self, ir, arg_replace_map, datamap, compiler_parameters):
        self.ir = ir
        self.arg_replace_map = arg_replace_map
        self.datamap = datamap
        self.compiler_parameters = compiler_parameters

        # self._cache = collections.defaultdict(dict)

    # @cachedmethod(lambda self: self._cache["CodegenResult._compile"])
    # not really needed, just a @property
    def _compile(self):
        return compile_loopy(self.ir, pyop3_compiler_parameters=self.compiler_parameters)

    def __call__(self, **kwargs):
        # TODO: Check each of kwargs and make sure that the replacement is
        # valid (e.g. same size, same data type, same layout funcs).
        data_args = []
        for kernel_arg in self.ir.default_entrypoint.args:
            actual_arg_name = self.arg_replace_map[kernel_arg.name]
            array = kwargs.get(actual_arg_name, self.datamap[actual_arg_name])
            data_args.append(as_kernel_arg(array))

        if len(self.ir.callables_table) > 1:
            breakpoint()

        if len(data_args) > 0:
            executable = self._compile()
            executable(*data_args)

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

    ctx = LoopyCodegenContext()
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
            ctx.set_temporary_shapes(_collect_temporary_shapes(e))
            _compile(e, loop_indices, ctx)

    # add a no-op instruction touching all of the kernel arguments so they are
    # not silently dropped
    noop = lp.CInstruction(
        (),
        "",
        read_variables=frozenset({a.name for a in ctx.arguments}),
        within_inames=frozenset(),
        within_inames_is_final=True,
        depends_on=ctx._depends_on,
    )
    ctx._insns.append(noop)

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
        ctx.domains,
        ctx.instructions,
        ctx.arguments,
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
    translation_unit = translation_unit.with_kernel(entrypoint)

    translation_unit = lp.merge((translation_unit, *ctx.subkernels))

    # add callables
    # tu = lp.register_callable(tu, "bsearch", BinarySearchCallable())

    # needed?
    translation_unit = translation_unit.with_entrypoints(entrypoint.name)

    return CodegenResult(translation_unit, ctx.kernel_to_actual_rename_map, ctx.datamap, compiler_parameters)


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
def _(call: DirectCalledFunction):
    return freeze(
        {
            arg.name: lp_arg.shape
            for lp_arg, arg in checked_zip(
                call.function.code.default_entrypoint.args, call.arguments
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
                    # loop_exprs[(loop.index.id, axis_label)] = lower_expr(index_expr, [iname_map], loop_indices, codegen_context, paths=[path_])
                    loop_exprs[(loop.index.id, axis_label)] = lower_expr(index_expr, [iname_replace_map_], loop_indices, codegen_context, paths=[path_])
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
def _(call: DirectCalledFunction, loop_indices, ctx: LoopyCodegenContext) -> None:
    temporaries = []
    subarrayrefs = {}
    extents = {}

    # loopy args can contain ragged params too
    loopy_args = call.function.code.default_entrypoint.args[: len(call.arguments)]
    for loopy_arg, arg, spec in checked_zip(loopy_args, call.arguments, call.argspec):
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

            temporaries.append((arg, indexed_temp, spec.access, shape))

            # Register data
            # TODO This might be bad for temporaries
            if isinstance(arg, _Dat):
                ctx.add_array(arg)

            # this should already be done in an assignment
            # ctx.add_temporary(temporary.name, temporary.dtype, shape)

            # subarrayref nonsense/magic
            indices = []
            for s in shape:
                iname = ctx.unique_name("i")
                ctx.add_domain(iname, s)
                indices.append(pym.var(iname))
            indices = tuple(indices)

            temp_name = ctx.actual_to_kernel_rename_map[temporary.name]
            subarrayrefs[arg] = lp.symbolic.SubArrayRef(
                indices, pym.subscript(pym.var(temp_name), indices)
            )

    # TODO this is pretty much the same as what I do in fix_intents in loopexpr.py
    # probably best to combine them - could add a sensible check there too.
    assignees = tuple(
        subarrayrefs[arg]
        for arg, spec in checked_zip(call.arguments, call.argspec)
        # if spec.access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE, NA}
        if spec.access in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}
    )
    expression = pym.primitives.Call(
        pym.var(call.function.code.default_entrypoint.name),
        tuple(
            subarrayrefs[arg]
            for arg, spec in checked_zip(call.arguments, call.argspec)
            if spec.access in {READ, RW, INC, MIN_RW, MAX_RW, NA}
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
def _compile_petscmat(assignment, loop_indices, codegen_context):
    mat = assignment.mat
    array = assignment.values

    assert isinstance(mat, _ConcretizedMat)

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
    codegen_context.add_array(mat)
    codegen_context.add_array(array)

    mat_name = codegen_context.actual_to_kernel_rename_map[mat.name]
    array_name = codegen_context.actual_to_kernel_rename_map[array.name]

    for row_path in assignment.row_axis_tree.leaf_paths:
        for col_path in assignment.col_axis_tree.leaf_paths:
            row_layout = mat.row_layouts[row_path]
            codegen_context.add_array(row_layout)
            rmap_name = codegen_context.actual_to_kernel_rename_map[row_layout.name]

            col_layout = mat.col_layouts[col_path]
            codegen_context.add_array(col_layout)
            cmap_name = codegen_context.actual_to_kernel_rename_map[col_layout.name]

            blocked = mat.mat.block_shape > 1
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

            raxes = row_layout.dat.axes
            row_subtree = raxes.subtree(raxes.child(raxes.root))
            rsize = row_subtree.size

            caxes = col_layout.dat.axes
            col_subtree = caxes.subtree(caxes.child(caxes.root))
            csize = col_subtree.size

            # these sizes can be expressions that need evaluating
            if not isinstance(rsize, numbers.Integral):
                raise NotImplementedError
                rsize_var = register_extent(
                    rsize,
                    loop_indices,
                    codegen_context,
                )
            else:
                rsize_var = rsize

            if not isinstance(csize, numbers.Integral):
                raise NotImplementedError
                csize_var = register_extent(
                    csize,
                    loop_indices,
                    codegen_context,
                )
            else:
                csize_var = csize

            # replace inner bits with zeros
            rzeros = {axis.label: 0 for axis in row_layout.dat.axes.nodes if axis is not row_layout.dat.axes.root}
            irow = str(lower_expr(row_layout, [rzeros], loop_indices, codegen_context))

            czeros = {axis.label: 0 for axis in col_layout.dat.axes.nodes if axis is not col_layout.dat.axes.root}
            icol = str(lower_expr(col_layout, [czeros], loop_indices, codegen_context))

            array_row_layout = assignment.values.row_layouts[row_path]
            array_row_expr = lower_expr(array_row_layout, [rzeros], loop_indices, codegen_context)
            array_col_layout = assignment.values.col_layouts[col_path]
            array_col_expr = lower_expr(array_col_layout, [czeros], loop_indices, codegen_context)

            array_indices = array_row_expr * csize + array_col_expr
            array_expr = str(pym.subscript(pym.var(array_name), array_indices))

            # hacky
            myargs = [
                assignment, mat_name, array_expr, rsize_var, csize_var, irow, icol, blocked
            ]
            access_type = assignment.access_type
            if access_type == ArrayAccessType.READ:
                call_str = _petsc_mat_load(*myargs)
            elif access_type == ArrayAccessType.WRITE:
                call_str = _petsc_mat_store(*myargs)
            else:
                assert access_type == ArrayAccessType.INC
                call_str = _petsc_mat_add(*myargs)

            codegen_context.add_cinstruction(call_str)


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
    lexpr = lower_expr(assignment.assignee, iname_replace_maps, loop_indices, codegen_context, paths=paths)
    rexpr = lower_expr(assignment.expression, iname_replace_maps, loop_indices, codegen_context, paths=paths)

    if assignment.assignment_type == AssignmentType.WRITE:
        pass
    else:
        assert assignment.assignment_type == AssignmentType.INC
        rexpr = lexpr + rexpr

    codegen_context.add_assignment(lexpr, rexpr)


# NOTE: This could really just be lower_expr itself
def make_array_expr(array, path, inames, ctx):
    assert False, "old code"
    # TODO: This should be propagated as an option - we don't always want to optimise
    # TODO: Disabled optimising for now since I can't get it to work without a
    # symbolic language. That has to be future work.

    # ultimately this can go when everything is just lower_expr
    ctx.add_array(array)  # (lower_expr registers the rest)

    array_offset = lower_expr(
        array.layouts[path],
        inames,
        ctx,
    )

    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    if array.name in ctx._temporary_shapes:
        shape = ctx._temporary_shapes[array.name]
        assert shape is not None
        rank = len(shape)
        extra_indices = (0,) * (rank - 1)

        # also has to be a scalar, not an expression
        temp_offset_name = ctx.unique_name("j")
        temp_offset_var = pym.var(temp_offset_name)
        ctx.add_temporary(temp_offset_name)
        ctx.add_assignment(temp_offset_var, array_offset)
        indices = extra_indices + (temp_offset_var,)
    else:
        indices = (array_offset,)

    name = ctx.actual_to_kernel_rename_map[array.name]
    return pym.subscript(pym.var(name), indices)


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
def _(axis_var: AxisVar, /, iname_maps, loop_indices, context, paths=None):
    if len(iname_maps) > 1:
        raise NotImplementedError("not sure")
    else:
        iname_map = just_one(iname_maps)
    return iname_map[axis_var.axis_label]


@lower_expr.register
def _(loop_var: LoopIndexVar, iname_maps, loop_indices, context, paths=None):
    return loop_indices[(loop_var.loop_id, loop_var.axis_label)]


def maybe_multiindex(dat, offset_expr, context):
    # hack to handle the fact that temporaries can have shape but we want to
    # linearly index it here
    if dat.name in context._temporary_shapes:
        shape = context._temporary_shapes[dat.name]
        assert shape is not None
        rank = len(shape)
        extra_indices = (0,) * (rank - 1)

        # also has to be a scalar, not an expression
        temp_offset_name = context.unique_name("j")
        temp_offset_var = pym.var(temp_offset_name)
        context.add_temporary(temp_offset_name)
        context.add_assignment(temp_offset_var, offset_expr)
        indices = extra_indices + (temp_offset_var,)
    else:
        indices = (offset_expr,)

    return indices


@lower_expr.register(_ConcretizedDat)
def _(dat: _ConcretizedDat, /, iname_maps, loop_indices, context, paths):
    iname_map = just_one(iname_maps)
    path = just_one(paths)

    context.add_array(dat)

    new_name = context.actual_to_kernel_rename_map[dat.name]

    layout_expr = dat.layouts[path]
    offset_expr = lower_expr(layout_expr, [iname_map], loop_indices, context)
    indices = maybe_multiindex(dat, offset_expr, context)

    return pym.subscript(pym.var(new_name), indices)


@lower_expr.register(_ConcretizedMat)
def _(mat: _ConcretizedMat, /, iname_maps, loop_indices, context, paths):
    context.add_array(mat)

    new_name = context.actual_to_kernel_rename_map[mat.name]

    row_iname_map, col_iname_map = iname_maps
    row_path, col_path = paths

    row_layout_expr = mat.row_layouts[row_path]
    col_layout_expr = mat.col_layouts[col_path]

    row_offset_expr = lower_expr(row_layout_expr, [row_iname_map], loop_indices, context)
    col_offset_expr = lower_expr(col_layout_expr, [col_iname_map], loop_indices, context)

    offset_expr = row_offset_expr * mat.mat.caxes.size + col_offset_expr
    indices = maybe_multiindex(mat, offset_expr, context)

    rexpr = pym.subscript(pym.var(new_name), indices)
    return rexpr


@lower_expr.register(_ExpressionDat)
def _(dat: _ExpressionDat, /, iname_maps, loop_indices, context, paths=None):
    iname_map = just_one(iname_maps)

    context.add_array(dat)

    new_name = context.actual_to_kernel_rename_map[dat.name]

    offset_expr = lower_expr(dat.layout, [iname_map], loop_indices, context)
    indices = maybe_multiindex(dat, offset_expr, context)
    rexpr = pym.subscript(pym.var(new_name), indices)
    return rexpr


@functools.singledispatch
def register_extent(obj: Any, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(extent).__name__}")


@register_extent.register(numbers.Integral)
def _(num: numbers.Integral, *args, **kwargs):
    return num


@register_extent.register(Parameter)
def _(param: Parameter, inames, loop_indices, context):
    return context.add_parameter(param)

@register_extent.register(Dat)
@register_extent.register(_ExpressionDat)
def _(extent, iname_replace_map, loop_indices, ctx):
    expr = lower_expr(extent, [iname_replace_map], loop_indices, ctx)
    varname = ctx.unique_name("p")
    ctx.add_temporary(varname)
    ctx.add_assignment(pym.var(varname), expr)
    return varname


# lives here??
@functools.singledispatch
def as_kernel_arg(obj: Any) -> int:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@as_kernel_arg.register(numbers.Integral)
def _(num: numbers.Integral) -> int:
    return int(num)


@as_kernel_arg.register(Parameter)
def _(param: Parameter) -> int:
    return as_kernel_arg(param.value)


@as_kernel_arg.register(np.ndarray)
def _(array: np.ndarray) -> int:
    return array.ctypes.data


@as_kernel_arg.register
def _(arg: _Dat):
    # TODO if we use the right accessor here we modify the state appropriately
    return as_kernel_arg(arg.buffer)


@as_kernel_arg.register
def _(arg: Buffer):
    # TODO if we use the right accessor here we modify the state appropriately
    # NOTE: Do not use .data_rw accessor here since this would trigger a halo exchange
    return as_kernel_arg(arg._data)


@as_kernel_arg.register
def _(arg: PackedBuffer):
    return as_kernel_arg(arg.array)


@as_kernel_arg.register
def _(array: AbstractMat):
    return array.mat.handle


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
    import ctypes, os
    from pyop2.utils import get_petsc_dir
    from pyop2.compilation import load

    code = lp.generate_code_v2(translation_unit).device_code()
    argtypes = [ctypes.c_voidp for _ in translation_unit.default_entrypoint.args]
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
