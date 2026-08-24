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

import pyop3.axis_tree
import pyop3.cache
import pyop3.config
import pyop3.dtypes
import pyop3.expr
from pyop3 import utils, mpi
from pyop3.cache import memory_and_disk_cache
from pyop3.expr import NonlinearDatBufferExpression
from pyop3.expr.visitors import collect_axis_vars, replace
from pyop3.axis_tree.tree import UNIT_AXIS_TREE, IndexedAxisTree, AxisComponent, relabel_path
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.dtypes import IntType
from pyop3.lower.context import CodegenContext
from pyop3.lower.transform import with_likwid_markers, with_petsc_event, with_attach_debugger
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
    WRITE,
    AssignmentType,
    ConcretizedNonEmptyArrayAssignment,
    StandaloneCalledFunction,
    Loop,
    InstructionList,
)

# TODO: import other way around?
from pyop3.insn.exec import parse_compiler_parameters

import pyop3.debug_flags # Sam for debugging
LOOPY_TARGET = lp.CWithGNULibcTarget()
LOOPY_LANG_VERSION = (2018, 2)

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
        super(LACallable, self).__init__(self.name,
                                         arg_id_to_dtype=arg_id_to_dtype,
                                         arg_id_to_descr=arg_id_to_descr,
                                         name_in_target=name_in_target)

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

class LoopyCodegenContext(CodegenContext):
    def __init__(self, *, check_negatives: bool):
        super().__init__(check_negatives=check_negatives)

        self._within_inames = frozenset()

        # initializer hash -> temporary name 
        self._reusable_temporaries: dict[int, str] = {}

    ''' Lowering path from PyOP3 to pymbolic '''
    def lower_expr(self, expr, iname_maps, loop_indices, intent=READ, paths=None):
        return self._lower_expr(expr, iname_maps, loop_indices, intent=intent, paths=paths)

    @functools.singledispatchmethod
    def _lower_expr(self, obj, iname_maps, loop_indices, *, intent, paths):
        raise TypeError(f"No Loopy handler defined for {type(obj).__name__}")

    @_lower_expr.register(numbers.Number)
    def _(self, num, *args, **kwargs) -> numbers.Number: 
        return num

    @_lower_expr.register(pyop3.expr.Add)
    def _(self, add: pyop3.expr.Add, /, *args, **kwargs) -> pym.Expression:
        return self._lower_expr(add.a, *args, **kwargs) + self._lower_expr(add.b, *args, **kwargs)


    @_lower_expr.register(pyop3.expr.Sub)
    def _(self, sub: pyop3.expr.Sub, /, *args, **kwargs) -> pym.Expression:
        return self._lower_expr(sub.a, *args, **kwargs) - self._lower_expr(sub.b, *args, **kwargs)


    @_lower_expr.register(pyop3.expr.Mul)
    def _(self, mul: pyop3.expr.Mul, /, *args, **kwargs) -> pym.Expression:
        return self._lower_expr(mul.a, *args, **kwargs) * self._lower_expr(mul.b, *args, **kwargs)


    @_lower_expr.register(pyop3.expr.Modulo)
    def _(self, mod: pyop3.expr.Modulo, /, *args, **kwargs) -> pym.Expression:
        return self._lower_expr(mod.a, *args, **kwargs) % self._lower_expr(mod.b, *args, **kwargs)


    @_lower_expr.register(pyop3.expr.Or)
    def _(self, or_: pyop3.expr.Or, /, *args, **kwargs) -> pym.Expression:
        return pym.primitives.LogicalOr((self._lower_expr(or_.a, *args, **kwargs), self._lower_expr(or_.b, *args, **kwargs)))


    @_lower_expr.register(pyop3.expr.Neg)
    def _(self, neg: pyop3.expr.Neg, /, *args, **kwargs) -> pym.Expression:
        return -self._lower_expr(neg.a, *args, **kwargs)


    @_lower_expr.register(pyop3.expr.FloorDiv)
    def _(self, neg: pyop3.expr.Neg, /, *args, **kwargs) -> pym.Expression:
        return self._lower_expr(neg.a, *args, **kwargs) // self._lower_expr(neg.b, *args, **kwargs)


    @_lower_expr.register(pyop3.expr.Comparison)
    def _(self, cond, /, *args, **kwargs) -> pym.Expression:
        return pym.primitives.Comparison(
            self._lower_expr(cond.a, *args, **kwargs),
            cond._symbol,
            self._lower_expr(cond.b, *args, **kwargs),
        )
    
    @_lower_expr.register(pyop3.expr.AxisVar)
    def _(self, axis_var, iname_maps, *args, **kwargs):
        return utils.just_one(iname_maps)[axis_var.axis.label]

    @_lower_expr.register(pyop3.expr.LoopIndexVar)
    def _(self, loop_var: pyop3.expr.LoopIndexVar, /, iname_maps, loop_indices, *args, **kwargs) -> pym.Expression:
        return loop_indices[(loop_var.loop_index.id, loop_var.axis.label)]
    
    @_lower_expr.register(pyop3.expr.Scalar)
    def _(self, scalar, iname_maps, loop_indices, *, intent, **kwargs):
        name = self.add_buffer(scalar.buffer, intent)
        return pym.subscript(pym.var(name), (0,))

    @_lower_expr.register(pyop3.expr.ScalarBufferExpression)
    def _(self, expr: pyop3.expr.ScalarBufferExpression, /, iname_maps, loop_indices, *, intent, **kwargs) -> pym.Expression:
        return self.lower_buffer_access(expr.buffer, [0], iname_maps, loop_indices, intent=intent)

    @_lower_expr.register(pyop3.expr.LinearDatBufferExpression)
    def _(self, expr, iname_maps, loop_indices, *, intent, **kwargs):
        return self.lower_buffer_access(expr.buffer, [expr.layout], iname_maps, loop_indices, intent)

    @_lower_expr.register(pyop3.expr.NonlinearDatBufferExpression)
    def _(self, expr: pyop3.expr.NonlinearDatBufferExpression, /, iname_maps, loop_indices, *, intent, paths, **kwargs) -> pym.Expression:
        path = utils.just_one(paths)
        return self.lower_buffer_access(expr.buffer, [expr.layouts[path]], iname_maps, loop_indices, intent=intent)
    
    @_lower_expr.register(pyop3.expr.MatPetscMatBufferExpression)
    def _(self, mat_expr, iname_maps, loop_indices, *, intent, paths, **kwargs):
        row_path, col_path = paths
        layouts = (mat_expr.row_layout.linearize(row_path), mat_expr.column_layout.linearize(col_path))
        return self.lower_buffer_access(mat_expr.buffer, layouts, iname_maps, loop_indices, intent)

    @_lower_expr.register(pyop3.expr.Conditional)
    def _(self, cond, iname_maps, loop_indices, **kwargs):
        return pym.primitives.If(
            self._lower_expr(cond.a, iname_maps, loop_indices, **kwargs),
            self._lower_expr(cond.b, iname_maps, loop_indices, **kwargs),
            self._lower_expr(cond.c, iname_maps, loop_indices, **kwargs)
        )

    @_lower_expr.register(pyop3.expr.MatArrayBufferExpression)
    def _(self, expr: pyop3.expr.MatArrayBufferExpression, /, iname_maps, loop_indices, *, intent, paths) -> pym.Expression:
        row_path, column_path = paths
        layouts = (expr.row_layouts[row_path], expr.column_layouts[column_path])
        return self.lower_buffer_access(expr.buffer, layouts, iname_maps, loop_indices, intent=intent)

    def lower_buffer_access(self, buffer: AbstractBuffer, layouts, iname_maps, loop_indices, intent):
        name_in_kernel = self.add_buffer(buffer, intent)

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
        indices = self.maybe_multiindex(buffer, offset_expr)

        subscript = pym.subscript(pym.var(name_in_kernel), indices)
        if self.check_negatives and intent == Intent.READ:
            idx = indices[-1]  # only the final index has meaning
            is_negative = pym.primitives.Comparison(idx, "<", 0)
            return pym.primitives.If(is_negative, -1, subscript)
        else:
            return subscript

    def add_leaf_assignment(self, assignment, paths, iname_maps, loop_indices):
        intent = assignment_type_as_intent(assignment.assignment_type)
        lexpr = self.lower_expr(assignment.assignee, iname_maps, loop_indices, intent=intent, paths=paths)
        rexpr = self.lower_expr(assignment.expression, iname_maps, loop_indices, paths=paths)
        if assignment.assignment_type == AssignmentType.INC:
            rexpr = lexpr + rexpr  
        self.add_assignment(lexpr, rexpr)

    def maybe_multiindex(self, buffer_ref, offset_expr):
        # hack to handle the facbuffer.t that temporaries can have shape but we want to
        # linearly index it here
        buffer_key = (buffer_ref.name, buffer_ref.nest_indices)
        if buffer_key in self._temporary_shapes:
            shape = self._temporary_shapes[buffer_key]
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

    def add_buffer(self, buffer, intent: Intent | None = None) -> str:
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

        if buffer.is_nested:
            raise NotImplementedError("Currently handle nesting outside the generated code")

        buffer_key = (buffer.name, buffer.nest_indices)
        if isinstance(buffer, NullBuffer):
            assert not buffer.nest_indices
            # 'intent' is not important for temporaries
            if buffer_key in self._kernel_names:
                return self._kernel_names[buffer_key]
            shape = self._temporary_shapes.get(buffer_key, (buffer.size,))
            assert isinstance(shape, tuple) and all(isinstance(s, numbers.Integral) for s in shape)
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

            if isinstance(buffer.handle, np.ndarray):
                # TODO: Enable this in an earlier pass (insert literals) (but have to make absolutely sure
                # that it is correctly included in the cache key).
                # Inject constant buffer data into the generated code if sufficiently small
                # if (
                #     buffer.rank_equal
                #     and isinstance(buffer.size, numbers.Integral)
                #     and buffer.size < CONFIG.max_static_array_size
                # ):
                #     return self.add_temporary(
                #         "t",
                #         buffer.dtype,
                #         initializer=buffer.data_ro,
                #         shape=buffer.data_ro.shape,
                #         read_only=True,
                #     )

                if isinstance(buffer.dtype, np.dtypes.IntDType):
                    name_in_kernel = self.unique_name("idat")
                else:
                    name_in_kernel = self.unique_name("dat")

                # If the buffer is being passed straight through to a function then we
                # have to make sure that the shapes match
                shape = self._temporary_shapes.get(buffer_key, None)
                loopy_arg = lp.GlobalArg(name_in_kernel, dtype=buffer.dtype, shape=shape)
            else:
                assert isinstance(buffer, PetscMatBuffer)
                assert buffer.mat_type not in {"nest", "python"}

                name_in_kernel = self.unique_name("mat")
                loopy_arg = lp.ValueArg(name_in_kernel, dtype=pyop3.dtypes.OpaqueType("Mat"))
            
            self.global_buffers[buffer_key] = buffer
            self.global_buffer_intents[buffer_key] = intent
            self._arguments.append(loopy_arg)

        self._kernel_names[buffer_key] = name_in_kernel
        return name_in_kernel

    def add_temporary(self, prefix="t", dtype=IntType, *, shape=(), initializer: np.ndarray =None, read_only=False) -> str:
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
        if opaque in self._kernel_names:
            return self._kernel_names[opaque]

        name_in_kernel = self.unique_name("opaque")
        loopy_arg = lp.ValueArg(name_in_kernel, dtype=opaque.dtype)

        self.global_buffers[opaque] = opaque
        self.global_buffer_intents[opaque] = intent
        self._arguments.append(loopy_arg)
        self._kernel_names[opaque] = name_in_kernel
        return name_in_kernel

    def add_subkernel(self, subkernel): 
        self._subkernels.append(subkernel)

    @contextlib.contextmanager
    def within_inames(self, inames):
        orig = self._within_inames
        self._within_inames |= inames
        yield
        self._within_inames = orig

    def set_temporary_shapes(self, shapes):
        self._temporary_shapes = shapes

    def compile_standalone_function(
        self, 
        call: StandaloneCalledFunction, 
        loop_indices
    ) -> None:
        subarrayrefs = {}
        loopy_args = call.function.code.default_entrypoint.args
        for loopy_arg, arg, spec in zip(loopy_args, call.arguments, call.argspec, strict=True):
            name_in_kernel = self.add_buffer(arg.buffer, spec.intent)
            if isinstance(loopy_arg, lp.ArrayArg):
                # array arguments to an inner kernel require all strides to be defined
                indices = []
                for s in loopy_arg.shape:
                    iname = self.unique_name("i")
                    self.add_domain(iname, s)
                    indices.append(pym.var(iname))
                indices = tuple(indices)
                subarrayrefs[arg] = lp.symbolic.SubArrayRef(
                    indices, pym.var(name_in_kernel)[indices]
                )
            else:
                assert isinstance(loopy_arg, lp.ValueArg)
                subarrayrefs[arg] = pym.var(name_in_kernel)

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

        self.add_function_call(assignees, expression)
        subkernel = call.function.code.with_entrypoints(frozenset())
        self.add_subkernel(subkernel)

    def compile_petsc_mat(
            self, 
            assignment: ConcretizedNonEmptyArrayAssignment, 
            loop_indices
        ) -> None:
        # We need to know whether the matrix is the assignee or not because we need
        # to know whether to put MatGetValues or MatSetValues
        if isinstance(assignment.assignee.buffer, PetscMatBuffer):
            mat = assignment.assignee
            expr = assignment.expression
            setting_mat_values = True
        else:
            mat = assignment.expression
            expr = assignment.assignee
            setting_mat_values = False


        row_axis_tree, column_axis_tree = assignment.axis_trees

        assert isinstance(expr, pyop3.expr.BufferExpression)
        array_buffer = expr.buffer

        # now emit the right line of code, this should properly be a lp.ScalarCallable
        # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
        mat_name = self.add_buffer(mat.buffer, assignment_type_as_intent(assignment.assignment_type))

        # NOTE: Is this always correct? It is for now.
        array_name = self.add_buffer(array_buffer, READ)

        rsize = row_axis_tree.size
        csize = column_axis_tree.size

        # these sizes can be expressions that need evaluating
        rsize_var = self.register_extent(
            rsize,
            {},
            loop_indices,
        )

        csize_var = self.register_extent(
            csize,
            {},
            loop_indices,
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
            subst_sublayout = layout.layouts[idict()]
            subst_layout = pyop3.expr.LinearDatBufferExpression(layout.buffer, subst_sublayout)
            layout_expr = self.lower_expr(subst_layout, ((),), loop_indices)
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
                    call_str = self._petsc_mat_store(*myargs)
                case AssignmentType.INC:
                    call_str = self._petsc_mat_add(*myargs)
                case _:
                    raise AssertionError
        else:
            call_str = self._petsc_mat_load(*myargs)

        self.add_cinstruction(call_str)

    def _petsc_mat_load(self, assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
        if blocked:
            return f"MatGetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"
        else:
            return f"MatGetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"


    def _petsc_mat_store(self, assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
        if blocked:
            return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"
        else:
            return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"


    def _petsc_mat_add(self, assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
        if blocked:
            return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"
        else:
            return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"


    def compile_exscan(
            self, 
            exscan: Exscan, 
            loop_indices
        ) -> None:
        assert isinstance(exscan, Exscan)

        if exscan.scan_type != "+":
            raise NotImplementedError
        domain_var = self.register_extent(
            exscan.extent,
            {},
            loop_indices
        )
        iname = self.unique_name("i")
        self.add_domain(iname, domain_var)

        lexpr = self.lower_expr(exscan.assignee, [{exscan.scan_axis.label: pym.var(iname)+1}], loop_indices, intent=WRITE)
        lexpr2 = self.lower_expr(exscan.assignee, [{exscan.scan_axis.label: pym.var(iname)}], loop_indices)
        rexpr = self.lower_expr(exscan.expression, [{exscan.scan_axis.label: pym.var(iname)}], loop_indices)

        rexpr = lexpr2 + rexpr
        self.add_assignment(lexpr, rexpr)

    def finalize_kernel(self, function_name, compiler_parameters):
        preambles = [
            ("20_debug", "#include <stdio.h>"), # dont always inject
            ("30_petsc", "#include <petsc.h>"), # perhaps only inject if petsc callable used
        ]

        # Add noop
        noop = lp.CInstruction((), "", read_variables=frozenset({a.name for a in self.arguments}),
                               within_inames=frozenset(), within_inames_is_final=True, depends_on=self._depends_on)
        self._instructions.append(noop)
        
        translation_unit = lp.make_kernel(
            self.domains, 
            self.instructions, 
            self.arguments, 
            name=function_name,
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
            preambles=preambles
        )
        translation_unit = lp.merge((translation_unit, *self.subkernels))
        
        entrypoint = translation_unit.default_entrypoint
        if compiler_parameters.add_likwid_markers: 
            entrypoint = with_likwid_markers(entrypoint)
        if compiler_parameters.add_petsc_event: 
            entrypoint = with_petsc_event(entrypoint)
        if compiler_parameters.attach_debugger: 
            entrypoint = with_attach_debugger(entrypoint)
        
        return translation_unit.with_kernel(entrypoint), self
