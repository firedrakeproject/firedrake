from __future__ import annotations

import abc
import contextlib
import functools
import numbers
import os
from typing import Any

import loopy as lp
import numpy as np
import pymbolic as pym
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.buffer
import pyop3.cache
import pyop3.config
import pyop3.constants
import pyop3.dtypes
import pyop3.expr
from pyop3 import mpi, utils
from pyop3.axis_tree.tree import (
    UNIT_AXIS_TREE,
    IndexedAxisTree,
)
from pyop3.buffer import (
    AbstractBuffer,
    NullBuffer,
    PetscMatBuffer,
)
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE
from pyop3.dtypes import IntType
from pyop3.insn.base import (
    AbstractAssignment,
    AssignmentType,
    Exscan,
    InstructionList,
    Loop,
    NonEmptyArrayAssignment,
    NullInstruction,
    StandaloneCalledFunction,
    assignment_type_as_intent,
)

from pyop3.lower.context import CodegenContext

# TODO: import other way around?
from pyop3.lower.transform import (
    with_attach_debugger,
    with_likwid_markers,
    with_petsc_event,
)
# FIXME this needs to be synchronised with TSFC, tricky
# shared base package? or both set by Firedrake - better solution
LOOPY_TARGET = lp.CWithGNULibcTarget()
LOOPY_LANG_VERSION = (2018, 2)

class LoopyCodegenContext(CodegenContext):
    def __init__(self, *, propagate_negatives: bool, mask_array_accesses: bool) -> None:
        super().__init__(
            propagate_negatives=propagate_negatives,
            mask_array_accesses=mask_array_accesses
        )
        self._within_inames = frozenset()

        # initializer hash -> temporary name
        self._reusable_temporaries: dict[int, str] = {}

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

    def add_subkernel(self, subkernel): 
        self._subkernels.append(subkernel)
    
    def var(self, iname: str, *args) -> pym.primitives.Variable:
        return pym.var(iname)

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

    def add_function_call(self, code, args, prefix="insn"): 
        input_refs = []
        output_refs = []
        loopy_args = code.default_entrypoint.args
        for loopy_arg, (arg, spec) in zip(loopy_args, args, strict=True):
            name_in_kernel = self.add_buffer(arg.buffer_view, spec.intent)
            if isinstance(loopy_arg, lp.ArrayArg):
                # array arguments to an inner kernel require all strides to be defined
                indices = []
                for s in loopy_arg.shape:
                    iname = self.unique_name("i")
                    self.add_domain(iname, s)
                    indices.append(pym.var(iname))
                indices = tuple(indices)
                ref = lp.symbolic.SubArrayRef(
                    indices, pym.var(name_in_kernel)[indices]
                )
            else:
                assert isinstance(loopy_arg, lp.ValueArg)
                ref = pym.var(name_in_kernel)

            if isinstance(loopy_arg.dtype, lp.types.OpaqueType):
                # no packing, passthrough arg, don't treat as written
                input_refs.append(ref)
            else:
                if spec.intent in {READ, RW, INC, MIN_RW, MAX_RW}:
                    input_refs.append(ref)
                if spec.intent in {WRITE, RW, INC, MIN_RW, MIN_WRITE, MAX_RW, MAX_WRITE}:
                    output_refs.append(ref)

        assignees = tuple(output_refs)
        expression = pym.primitives.Call(
            pym.var(code.default_entrypoint.name), tuple(input_refs)
        )

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

    def add_buffer(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        intent: pyop3.constants.Intent | None = None,
    ) -> str:
        # TODO: This should check to make sure that we do not encounter any
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

        buffer = buffer_view.buffer
        if isinstance(buffer, NullBuffer):
            assert not buffer_view.nest_indices
            # Note that intent is not important for temporaries
            try:
                return self.kernel_names[buffer_view]
            except KeyError:
                shape = self._temporary_shapes.get(buffer, (buffer.size,))
                assert isinstance(shape, tuple) and all(isinstance(s, numbers.Integral) for s in shape)
                name_in_kernel = self.add_temporary("t", buffer.dtype, shape=shape)
                return self.kernel_names.setdefault(buffer_view, name_in_kernel)

        else:
            if intent is None:
                raise ValueError("Global data must declare intent")

            # Inject constant buffer data into the generated code if sufficiently small
            # TODO: Enable this in an earlier pass (insert literals) (but have to make absolutely sure
            # that it is correctly included in the cache key).
            # if isinstance(handle, pyop3.buffer.ArrayBuffer):
            #     that it is correctly included in the cache key).
            #     Inject constant buffer data into the generated code if sufficiently small
            #     if (
            #         buffer.rank_equal
            #         and isinstance(buffer.size, numbers.Integral)
            #         and buffer.size < CONFIG.max_static_array_size
            #     ):
            #         return self.add_temporary(
            #             "t",
            #             buffer.dtype,
            #             initializer=buffer.data_ro,
            #             shape=buffer.data_ro.shape,
            #             read_only=True,
            #         )

            if buffer_view in self.kernel_names:
                if intent != self.buffer_intents[buffer]:
                    # We are accessing a buffer with different intents so have to
                    # pessimally claim RW access
                    self.buffer_intents[buffer] = RW
                return self.kernel_names[buffer_view]

            # Extract the underlying data as that is what we need to generate code
            handle = buffer_view.handle
            if isinstance(handle, np.ndarray):
                if isinstance(handle.dtype, np.dtypes.IntDType):
                    name_in_kernel = self.unique_name("idat")
                else:
                    name_in_kernel = self.unique_name("dat")

                # If the buffer is being passed straight through to a function then we
                # have to make sure that the shapes match
                shape = self._temporary_shapes.get(buffer, None)  # TODO: should be handle not buffer here?
                loopy_arg = lp.GlobalArg(name_in_kernel, dtype=handle.dtype, shape=shape)
            else:
                assert isinstance(handle, PETSc.Mat)
                assert handle.type not in {"nest", "python"}
                name_in_kernel = self.unique_name("mat")
                loopy_arg = lp.ValueArg(name_in_kernel, dtype=pyop3.dtypes.OpaqueType("Mat"))

            self.buffer_intents[buffer] = intent
            self._arguments.append(loopy_arg)
            return self.kernel_names.setdefault(buffer_view, name_in_kernel)

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

    def add_opaque(self, opaque: OpaqueTerminal, intent) -> str:
        if opaque in self.kernel_names:
            return self.kernel_names[opaque]

        name_in_kernel = self.unique_name("opaque")
        loopy_arg = lp.ValueArg(name_in_kernel, dtype=opaque.dtype)

        self.buffer_intents[opaque] = intent
        self._arguments.append(loopy_arg)
        self.kernel_names[opaque] = name_in_kernel
        return name_in_kernel

    @contextlib.contextmanager
    def within_inames(self, inames) -> None:
        orig_within_inames = self._within_inames
        self._within_inames |= inames
        yield
        self._within_inames = orig_within_inames

    # FIXME, bad API but it is context-dependent
    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes
        
    def lower_buffer_access(
        self,
        buffer_view: pyop3.buffer.IndexedBuffer,
        layouts,
        iname_maps,
        loop_indices,
        *,
        intent,
    ) -> pym.Expression: 
        name_in_kernel = self.add_buffer(buffer_view, intent)

        buffer = buffer_view.buffer
        if isinstance(buffer, PetscMatBuffer):
            buffer = buffer_view.denested.getPythonContext().buffer

        # At this point we know how to address each axis of the underlying buffer.
        # This is sufficient to address a flat buffer, but for a buffer with more
        # dimensions (i.e. a matrix) we have to do more work. As an example
        # consider accessing a 2D buffer with shape (5, 5) using layout functions
        # '2*i+1' and 'j+2' for the rows and columns respectively, where
        # '0<=i<2' and '0<=j<3'. The offset expression that we want from this is:
        #
        #     5*(2*i+1) + (j+2)
        #
        # Which we can only determine from knowing the underlying buffer shape.
        offset_expr = sum(
            stride * self.lower_expr(layout, [iname_map], loop_indices)
            for stride, layout, iname_map in zip(
                utils.strides(buffer.shape),
                layouts,
                iname_maps,
                strict=True
            )
        )

        # Add some leading zeros to make loopy happy
        indices = self.maybe_multiindex(buffer_view.buffer, offset_expr)

        subscript = pym.subscript(pym.var(name_in_kernel), indices)
        if self.propagate_negatives and intent == READ:
            idx = indices[-1]  # only the final index has meaning
            is_negative = pym.primitives.Comparison(idx, "<", 0)
            return pym.primitives.If(is_negative, -1, subscript)
        else:
            return subscript
    
    def maybe_multiindex(self, buffer, offset_expr):
        # hack to handle the facbuffer.t that temporaries can have shape but we want to
        # linearly index it here
        if buffer in self._temporary_shapes:
            shape = self._temporary_shapes[buffer]
            rank = len(shape)
            extra_indices = (0,) * (rank - 1)

            # also has to be a scalar, not an expression
            temp_offset_name = self.add_temporary("j")
            temp_offset_var = pym.var(temp_offset_name)
            self.add_assignment(temp_offset_var, offset_expr)
            indices = extra_indices + (temp_offset_var,)
        else:
            indices = (offset_expr,)

        return indices

    # NOTE: This could probably be refactored
    def add_leaf_assignment(
        self,
        assignee,
        expression,
        assignment_type,
        paths,
        iname_replace_maps,
        loop_indices,
    ):
        intent = assignment_type_as_intent(assignment_type)
        lexpr = self.lower_expr(
            assignee,
            iname_replace_maps,
            loop_indices,
            intent=intent,
            paths=paths,
        )
        rexpr = self.lower_expr(
            expression,
            iname_replace_maps,
            loop_indices,
            paths=paths,
        )

        match assignment_type:
            case AssignmentType.WRITE:
                pass
            case AssignmentType.INC:
                rexpr = lexpr + rexpr
            case AssignmentType.MAX:
                rexpr = pym.Variable("max")(lexpr, rexpr)
            case AssignmentType.MIN:
                rexpr = pym.Variable("min")(lexpr, rexpr)
            case _:
                raise NotImplementedError

        if self.mask_array_accesses:
            # a[off_a] = off_b < 0 ? a[off_a] : b[off_b]
            offset_expr = _min_subscript_offset(rexpr)
            # if there are no subcripts then the mask is pointless
            if offset_expr is not None:
                cond = pym.primitives.Comparison(offset_expr, "<", 0)
                rexpr = pym.primitives.If(cond, lexpr, rexpr)

        self.add_assignment(lexpr, rexpr)

    def lower_expr(self, expr, iname_maps, loop_indices, *, intent=READ, paths=None) -> pym.Expression:
        return _lower_expr(expr, iname_maps, loop_indices, intent=intent, paths=paths, context=self)

    def finalize_kernel(self, function_name, compiler_parameters):
        preambles = [
            ("20_debug", "#include <stdio.h>"),  # dont always inject
            ("30_petsc", "#include <petsc.h>"),  # perhaps only if petsc callable used?
        ]

        # add a no-op instruction touching all of the kernel arguments so they are
        # not silently dropped
        noop = lp.CInstruction(
            (),
            "",
            read_variables=frozenset({a.name for a in self.arguments}),
            within_inames=frozenset(),
            within_inames_is_final=True,
            depends_on=self._depends_on,
        )
        self._instructions.append(noop)

        translation_unit = lp.make_kernel(
            self.domains,
            self.instructions,
            self.arguments,
            name=function_name,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
            preambles=preambles,
        )
        translation_unit = lp.merge((translation_unit, *self.subkernels))

        entrypoint = translation_unit.default_entrypoint
        if compiler_parameters.add_likwid_markers:
            entrypoint = with_likwid_markers(entrypoint)
        if compiler_parameters.add_petsc_event:
            entrypoint = with_petsc_event(entrypoint)
        if compiler_parameters.attach_debugger:
            entrypoint = with_attach_debugger(entrypoint)
        translation_unit = translation_unit.with_kernel(entrypoint)
        
        return translation_unit.with_kernel(entrypoint)

    @functools.singledispatchmethod
    def register_extent(self, obj: Any, *args, **kwargs):
        raise TypeError(f"No handler defined for {type(obj).__name__}")


    @register_extent.register(numbers.Integral)
    def _(self, num: numbers.Integral, *args, **kwargs):
        return num

    @register_extent.register(pyop3.expr.Expression)
    def _(self, expr: pyop3.expr.Expression, inames, loop_indices):
        pym_expr = self.lower_expr(expr, [inames], loop_indices)
        extent_name = self.add_temporary("p")
        self.add_assignment(pym.var(extent_name), pym_expr)
        return extent_name


# TODO: use overloadedexpressionevaluator
@functools.singledispatch
def _lower_expr(obj: Any, /, *args, **kwargs) -> pym.Expression:
    raise TypeError(f"No handler defined for {type(obj).__name__}")

@_lower_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
    return num

@_lower_expr.register(pyop3.expr.Add)
def _(add: pyop3.expr.Add, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(add.a, *args, **kwargs) + _lower_expr(add.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Sub)
def _(sub: pyop3.expr.Sub, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(sub.a, *args, **kwargs) - _lower_expr(sub.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Mul)
def _(mul: pyop3.expr.Mul, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mul.a, *args, **kwargs) * _lower_expr(mul.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Modulo)
def _(mod: pyop3.expr.Mod, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(mod.a, *args, **kwargs) % _lower_expr(mod.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Or)
def _(or_: pyop3.expr.Or, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.LogicalOr((_lower_expr(or_.a, *args, **kwargs), _lower_expr(or_.b, *args, **kwargs)))


@_lower_expr.register(pyop3.expr.Neg)
def _(neg: pyop3.expr.Neg, /, *args, **kwargs) -> pym.Expression:
    return -_lower_expr(neg.a, *args, **kwargs)


@_lower_expr.register(pyop3.expr.FloorDiv)
def _(fdiv: pyop3.expr.FloorDiv, /, *args, **kwargs) -> pym.Expression:
    return _lower_expr(fdiv.a, *args, **kwargs) // _lower_expr(fdiv.b, *args, **kwargs)


@_lower_expr.register(pyop3.expr.Comparison)
def _(cond, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.Comparison(
        _lower_expr(cond.a, *args, **kwargs),
        cond._symbol,
        _lower_expr(cond.b, *args, **kwargs),
    )

@_lower_expr.register(pyop3.expr.Conditional)
def _(cond: pyop3.expr.Conditional, /, *args, **kwargs) -> pym.Expression:
    return pym.primitives.If(_lower_expr(cond.a, *args, **kwargs), _lower_expr(cond.b, *args, **kwargs), _lower_expr(cond.c, *args, **kwargs))

@_lower_expr.register(pyop3.expr.AxisVar)
def _(axis_var: pyop3.expr.AxisVar, /, iname_maps, *args, **kwargs) -> pym.Expression:
    return utils.just_one(iname_maps)[axis_var.axis.label]


@_lower_expr.register(pyop3.expr.LoopIndexVar)
def _(loop_var: pyop3.expr.LoopIndexVar, /, iname_maps, loop_indices, *args, **kwargs) -> pym.Expression:
    return loop_indices[(loop_var.loop_index.id, loop_var.axis.label)]


@_lower_expr.register(pyop3.expr.ScalarBufferExpression)
def _(
    expr: pyop3.expr.ScalarBufferExpression,
    /,
    iname_maps,
    loop_indices,
    *,
    intent,
    context,
    **kwargs,
) -> pym.ExpressionNode:
    return context.lower_buffer_access(expr.buffer_view, [0], iname_maps, loop_indices, intent=intent)


@_lower_expr.register(pyop3.expr.LinearDatBufferExpression)
def _(expr: pyop3.expr.LinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, context, **kwargs) -> pym.Expression:
    return context.lower_buffer_access(expr.buffer_view, [expr.layout], iname_maps, loop_indices, intent=intent)


@_lower_expr.register(pyop3.expr.NonlinearDatBufferExpression)
def _(expr: pyop3.expr.NonlinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context, **kwargs) -> pym.Expression:
    path = utils.just_one(paths)
    return context.lower_buffer_access(
        expr.buffer_view,
        [expr.layouts[path]],
        iname_maps,
        loop_indices,
        intent=intent,
    )


@_lower_expr.register(pyop3.expr.MatPetscMatBufferExpression)
def _(mat_expr: pyop3.expr.MatPetscMatBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context) -> pym.Expression:
    row_path, column_path = paths
    layouts = (
        mat_expr.row_layout.linearize(row_path),
        mat_expr.column_layout.linearize(column_path),
    )
    return context.lower_buffer_access(
        mat_expr.buffer_view,
        layouts,
        iname_maps,
        loop_indices,
        intent=intent,
    )


@_lower_expr.register(pyop3.expr.MatArrayBufferExpression)
def _(expr: pyop3.expr.MatArrayBufferExpression, /, iname_maps, loop_indices, *, intent, paths, context) -> pym.Expression:
    row_path, column_path = paths
    layouts = (expr.row_layouts[row_path], expr.column_layouts[column_path])
    return context.lower_buffer_access(
        expr.buffer_view,
        layouts,
        iname_maps,
        loop_indices,
        intent=intent,
    )

class _MinSubscriptOffsetMapper(pym.mapper.IdentityMapper):

    def __init__(self):
        self.subscript_found = False
        super().__init__()

    def map_sum(self, expr):
        assert len(expr.children) == 2
        a, b = map(self.rec, expr.children)
        return pym.primitives.If(pym.primitives.Comparison(a, "<", b), a, b)

    def map_subscript(self, expr):
        self.subscript_found = True
        # do not recurse
        return utils.just_one(expr.index_tuple)


def _min_subscript_offset(expr: pym.ExpressionNode) -> pym.ExpressionNode | None:
    """Return an expression for the minimum subscript offset in an expression.

    This is important because we sometimes need to be able to check if we are
    indexing with negative values (and hence might want to mask the access).

    If no subscripts are found then `None` is returned.

    """
    mapper = _MinSubscriptOffsetMapper()
    mapped_expr = mapper(expr)
    if mapper.subscript_found:
        return mapped_expr
    else:
        return None

class LACallable(lp.ScalarCallable, metaclass=abc.ABCMeta):
    """
    The LACallable (Linear algebra callable)
    replaces loopy.CallInstructions to linear algebra functions
    like solve or inverse by LAPACK calls.
    """
    def __init__(self, name=None, arg_id_to_dtype=None,
                 arg_id_to_descr=None, name_in_target=None):
        if name is not None:
            assert name == self.name

        name_in_target = name_in_target if name_in_target else self.name
        super().__init__(
            self.name,
            arg_id_to_dtype=arg_id_to_dtype,
            arg_id_to_descr=arg_id_to_descr,
            name_in_target=name_in_target
        )

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractmethod
    def generate_preambles(self, target):
        pass

    def with_types(self, arg_id_to_dtype, callables_table):
        dtypes = {}
        for i in range(len(arg_id_to_dtype)):
            if arg_id_to_dtype.get(i) is None:
                # the types provided aren't mature enough to specialize the
                # callable
                return (self.copy(arg_id_to_dtype=arg_id_to_dtype),
                        callables_table)
            else:
                mat_dtype = arg_id_to_dtype[i].numpy_dtype
                dtypes[i] = lp.types.NumpyType(mat_dtype)
        dtypes[-1] = lp.types.NumpyType(dtypes[0].dtype)

        return (self.copy(name_in_target=self.name_in_target,
                arg_id_to_dtype=idict(dtypes)),
                callables_table)

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        assert self.is_ready_for_codegen()
        assert isinstance(insn, lp.CallInstruction)

        parameters = insn.expression.parameters

        parameters = list(parameters)
        par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]

        parameters.append(insn.assignees[-1])
        par_dtypes.append(self.arg_id_to_dtype[0])

        mat_descr = self.arg_id_to_descr[0]
        arg_c_parameters = [
            expression_to_code_mapper(
                par,
                pym.mapper.stringifier.PREC_NONE,
                lp.expression.dtype_to_type_context(target, par_dtype),
                par_dtype
            ).expr
            for par, par_dtype in zip(parameters, par_dtypes)
        ]
        c_parameters = [arg_c_parameters[-1]]
        c_parameters.extend([arg for arg in arg_c_parameters[:-1]])
        c_parameters.append(np.int32(mat_descr.shape[1]))  # n
        return pym.var(self.name_in_target)(*c_parameters), False


# Read c files  for linear algebra callables in on import
if mpi.COMM_WORLD.rank == 0:
    with open(os.path.dirname(__file__)+"/inverse.c", "r") as myfile:
        inverse_preamble = myfile.read()
    with open(os.path.dirname(__file__)+"/solve.c", "r") as myfile:
        solve_preamble = myfile.read()
else:
    solve_preamble = None
    inverse_preamble = None

inverse_preamble = mpi.COMM_WORLD.bcast(inverse_preamble, root=0)
solve_preamble = mpi.COMM_WORLD.bcast(solve_preamble, root=0)


class INVCallable(LACallable):
    """
    The InverseCallable replaces loopy.CallInstructions to "inverse"
    functions by LAPACK getri.
    """
    name = "inverse"

    def generate_preambles(self, target):
        assert isinstance(target, type(target))
        yield ("inverse", inverse_preamble)


class SolveCallable(LACallable):
    """
    The SolveCallable replaces loopy.CallInstructions to "solve"
    functions by LAPACK getrs.
    """
    name = "solve"

    def generate_preambles(self, target):
        assert isinstance(target, type(target))
        yield ("solve", solve_preamble)
