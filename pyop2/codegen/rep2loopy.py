import ctypes
import numpy
from dataclasses import dataclass

from immutabledict import immutabledict
import loopy
from loopy.symbolic import SubArrayRef
from loopy.expression import dtype_to_type_context
from pymbolic.mapper.stringifier import PREC_NONE
from pymbolic import var
from loopy.types import NumpyType, OpaqueType
import abc

import islpy as isl
import pymbolic.primitives as pym

from collections import OrderedDict, defaultdict
from functools import singledispatch, reduce, partial
import itertools
import operator

from pyop2.codegen.node import traversal, Node, Memoizer, reuse_if_untouched

from pyop2.types.access import READ, WRITE
from pyop2.datatypes import as_ctypes

from pyop2.codegen.optimise import index_merger, rename_nodes

from pyop2.codegen.representation import (Index, FixedIndex, RuntimeIndex,
                                          MultiIndex, Extent, Indexed,
                                          BitShift, BitwiseNot, BitwiseAnd, BitwiseOr,
                                          Conditional, Comparison, DummyInstruction,
                                          LogicalNot, LogicalAnd, LogicalOr,
                                          Materialise, Accumulate, FunctionCall, When,
                                          Argument, Variable, Literal, NamedLiteral,
                                          Symbol, Zero, Sum, Min, Max, Product,
                                          Quotient, FloorDiv, Remainder)
from pyop2.codegen.representation import (PackInst, UnpackInst, KernelInst, PreUnpackInst)
from pytools import ImmutableRecord
from pyop2.codegen.loopycompat import _match_caller_callee_argument_dimension_
from pyop2.configuration import target

from petsc4py import PETSc


# Read c files  for linear algebra callables in on import
import os
from pyop2.mpi import COMM_WORLD
if COMM_WORLD.rank == 0:
    with open(os.path.dirname(__file__)+"/c/inverse.c", "r") as myfile:
        inverse_preamble = myfile.read()
    with open(os.path.dirname(__file__)+"/c/solve.c", "r") as myfile:
        solve_preamble = myfile.read()
else:
    solve_preamble = None
    inverse_preamble = None

inverse_preamble = COMM_WORLD.bcast(inverse_preamble, root=0)
solve_preamble = COMM_WORLD.bcast(solve_preamble, root=0)


class Bag(object):
    pass


def symbol_mangler(kernel, name):
    if name in {"ADD_VALUES", "INSERT_VALUES"}:
        return loopy.types.to_loopy_type(numpy.int32), name
    return None


class PetscCallable(loopy.ScalarCallable):

    def with_types(self, arg_id_to_dtype, callables_table):
        new_arg_id_to_dtype = dict(arg_id_to_dtype)
        return (self.copy(
            name_in_target=self.name,
            arg_id_to_dtype=immutabledict(new_arg_id_to_dtype)), callables_table)

    def with_descrs(self, arg_id_to_descr, callables_table):
        from loopy.kernel.function_interface import ArrayArgDescriptor
        from loopy.kernel.array import FixedStrideArrayDimTag
        new_arg_id_to_descr = dict(arg_id_to_descr)
        for i, des in arg_id_to_descr.items():
            # petsc takes 1D arrays as arguments
            if isinstance(des, ArrayArgDescriptor):
                dim_tags = tuple(FixedStrideArrayDimTag(stride=int(numpy.prod(des.shape[i+1:])),
                                                        layout_nesting_level=len(des.shape)-i-1)
                                 for i in range(len(des.shape)))
                new_arg_id_to_descr[i] = des.copy(dim_tags=dim_tags)

        return (self.copy(arg_id_to_descr=immutabledict(new_arg_id_to_descr)),
                callables_table)

    def generate_preambles(self, target):
        assert isinstance(target, type(target))
        yield ("00_petsc", "#include <petsc.h>")
        return


petsc_functions = set()


def register_petsc_function(name):
    petsc_functions.add(name)


class LACallable(loopy.ScalarCallable, metaclass=abc.ABCMeta):
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
                dtypes[i] = NumpyType(mat_dtype)
        dtypes[-1] = NumpyType(dtypes[0].dtype)

        return (self.copy(name_in_target=self.name_in_target,
                arg_id_to_dtype=immutabledict(dtypes)),
                callables_table)

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        assert self.is_ready_for_codegen()
        assert isinstance(insn, loopy.CallInstruction)

        parameters = insn.expression.parameters

        parameters = list(parameters)
        par_dtypes = [self.arg_id_to_dtype[i] for i, _ in enumerate(parameters)]

        parameters.append(insn.assignees[-1])
        par_dtypes.append(self.arg_id_to_dtype[0])

        mat_descr = self.arg_id_to_descr[0]
        arg_c_parameters = [
            expression_to_code_mapper(
                par,
                PREC_NONE,
                dtype_to_type_context(target, par_dtype),
                par_dtype
            ).expr
            for par, par_dtype in zip(parameters, par_dtypes)
        ]
        c_parameters = [arg_c_parameters[-1]]
        c_parameters.extend([arg for arg in arg_c_parameters[:-1]])
        c_parameters.append(numpy.int32(mat_descr.shape[1]))  # n
        return var(self.name_in_target)(*c_parameters), False


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


class _PreambleGen(ImmutableRecord):
    fields = set(("preamble", ))

    def __init__(self, preamble):
        self.preamble = preamble

    def __call__(self, preamble_info):
        yield ("0", self.preamble)


@dataclass(frozen=True, init=False)
class PyOP2KernelCallable(loopy.ScalarCallable):
    """Handles PyOP2 Kernel passed in as a string
    """

    init_arg_names = ("name", "parameters", "arg_id_to_dtype", "arg_id_to_descr", "name_in_target")

    parameters: tuple

    def __init__(self, name, parameters, arg_id_to_dtype=None, arg_id_to_descr=None, name_in_target=None):
        super().__init__(name, arg_id_to_dtype, arg_id_to_descr, name_in_target)
        object.__setattr__(self, "parameters", tuple(parameters))

    def with_types(self, arg_id_to_dtype, callables_table):
        new_arg_id_to_dtype = dict(arg_id_to_dtype)
        return self.copy(
            name_in_target=self.name,
            arg_id_to_dtype=immutabledict(new_arg_id_to_dtype)), callables_table

    def with_descrs(self, arg_id_to_descr, callables_table):
        from loopy.kernel.function_interface import ArrayArgDescriptor
        from loopy.kernel.array import FixedStrideArrayDimTag
        new_arg_id_to_descr = dict(arg_id_to_descr)
        for i, des in arg_id_to_descr.items():
            # 1D arrays
            if isinstance(des, ArrayArgDescriptor):
                dim_tags = tuple(
                    FixedStrideArrayDimTag(
                        stride=int(numpy.prod(des.shape[i+1:])),
                        layout_nesting_level=len(des.shape)-i-1
                    )
                    for i in range(len(des.shape))
                )
                new_arg_id_to_descr[i] = des.copy(dim_tags=dim_tags)
        return (self.copy(arg_id_to_descr=immutabledict(new_arg_id_to_descr)), callables_table)

    def emit_call_insn(self, insn, target, expression_to_code_mapper):
        # reorder arguments, e.g. a,c = f(b,d) to f(a,b,c,d)
        par_dtypes = tuple(expression_to_code_mapper.infer_type(p) for p in self.parameters)

        from loopy.expression import dtype_to_type_context
        from pymbolic.mapper.stringifier import PREC_NONE
        from pymbolic import var

        c_parameters = [
            expression_to_code_mapper(
                par, PREC_NONE, dtype_to_type_context(target, par_dtype),
                par_dtype).expr
            for par, par_dtype in zip(self.parameters, par_dtypes)]

        assignee_is_returned = False
        return var(self.name_in_target)(*c_parameters), assignee_is_returned


@singledispatch
def replace_materialise(node, self):
    raise AssertionError("Unhandled node type %r" % type(node))


replace_materialise.register(Node)(reuse_if_untouched)


@replace_materialise.register(Materialise)
def replace_materialise_materialise(node, self):
    v = Variable(node.name, node.shape, node.dtype)
    inits = list(map(self, node.children))
    label = node.label
    accs = []
    for rvalue, indices in zip(*(inits[0::2], inits[1::2])):
        lvalue = Indexed(v, indices)
        if isinstance(rvalue, When):
            when, rvalue = rvalue.children
            acc = When(when, Accumulate(label, lvalue, rvalue))
        else:
            acc = Accumulate(label, lvalue, rvalue)
        accs.append(acc)
    self.initialisers.append(tuple(accs))
    return v


def runtime_indices(expressions):
    indices = []
    for node in traversal(expressions):
        if isinstance(node, RuntimeIndex):
            indices.append(node.name)
    # use a dict as an ordered set
    return {i: None for i in indices}


def imperatives(exprs):
    for op in traversal(exprs):
        if isinstance(op, (Accumulate, FunctionCall)):
            yield op


def loop_nesting(instructions, deps, outer_inames, kernel_name):
    nesting = {}

    for insn in imperatives(instructions):
        if isinstance(insn, Accumulate):
            if isinstance(insn.children[1], (Zero, Literal)):
                nesting[insn] = outer_inames
            else:
                nesting[insn] = runtime_indices([insn]) | runtime_indices(insn.label.within_inames)
        else:
            assert isinstance(insn, FunctionCall)
            if insn.name in (petsc_functions | {kernel_name}):
                nesting[insn] = outer_inames
            else:
                nesting[insn] = runtime_indices([insn])

    # take care of dependencies. e.g. t1[i] = A[i], t2[j] = B[t1[j]], then t2 should depends on {i, j}
    name_to_insn = dict((n, i) for i, (n, _) in deps.items())
    for insn, (name, _deps) in deps.items():
        s = set(_deps)
        while s:
            d = s.pop()
            nesting[insn] = nesting[insn] | nesting[name_to_insn[d]]
            s = s | set(deps[name_to_insn[d]][1]) - set([name])

    # boost inames, if one instruction is inside inner inames (free indices),
    # it should be inside the outer inames as dictated by other instructions.
    index_nesting = defaultdict(dict)  # free index -> {runtime indices}
    for insn in instructions:
        if isinstance(insn, When):
            key = insn.children[1]
        else:
            key = insn
        for fi in traversal([insn]):
            if isinstance(fi, Index):
                index_nesting[fi] |= nesting[key]

    for insn in imperatives(instructions):
        outer = reduce(operator.or_,
                       iter(index_nesting[fi] for fi in traversal([insn]) if isinstance(fi, Index)),
                       {})
        nesting[insn] = nesting[insn] | outer

    return nesting


def instruction_dependencies(instructions, initialisers):
    deps = {}
    names = {}
    instructions_by_type = defaultdict(list)
    c = itertools.count()
    for op in imperatives(instructions):
        name = "statement%d" % next(c)
        names[op] = name
        instructions_by_type[type(op.label)].append(op)
        deps[op] = frozenset()

    # read-write dependencies in packing instructions
    def variables(exprs):
        for op in traversal(exprs):
            if isinstance(op, (Argument, Variable)):
                yield op

    def bounds(exprs):
        for op in traversal(exprs):
            if isinstance(op, RuntimeIndex):
                for v in variables(op.extents):
                    yield v

    writers = defaultdict(list)
    for op in instructions_by_type[PackInst]:
        assert isinstance(op, Accumulate)
        lvalue, _ = op.children
        # Only writes to the outer-most variable
        writes = next(variables([lvalue]))
        if isinstance(writes, Variable):
            writers[writes].append(names[op])

    for op in instructions_by_type[PackInst]:
        _, rvalue = op.children
        deps[op] |= frozenset(x for x in itertools.chain(*(
            writers[r] for r in itertools.chain(variables([rvalue]), bounds([op]))
        )))
        deps[op] -= frozenset(names[op])

    for typ, depends_on in [(KernelInst, [PackInst]),
                            (PreUnpackInst, [KernelInst]),
                            (UnpackInst, [KernelInst, PreUnpackInst])]:
        for op in instructions_by_type[typ]:
            ops = itertools.chain(*(instructions_by_type[t] for t in depends_on))
            deps[op] |= frozenset(names[o] for o in ops)

    # add sequential instructions in the initialisers
    for inits in initialisers:
        for i, parent in enumerate(inits[1:], 1):
            for p in imperatives([parent]):
                deps[p] |= frozenset(names[c] for c in imperatives(inits[:i])) - frozenset([name])

    # add name to deps
    return dict((op, (names[op], dep)) for op, dep in deps.items())


def generate(builder, wrapper_name=None):
    # Reset all terminal counters to avoid generated code becoming different across ranks
    Argument._count = defaultdict(partial(itertools.count))
    Index._count = itertools.count()
    Materialise._count = itertools.count()
    RuntimeIndex._count = itertools.count()

    # use a dict as an ordered set
    outer_inames = {builder._loop_index.name: None}
    if builder.layer_index is not None:
        outer_inames.update({builder.layer_index.name: None})

    instructions = list(builder.emit_instructions())

    parameters = Bag()
    parameters.domains = OrderedDict()
    parameters.assumptions = OrderedDict()
    parameters.wrapper_arguments = builder.wrapper_args
    parameters.layer_start = builder.layer_extents[0].name
    parameters.layer_end = builder.layer_extents[1].name
    parameters.conditions = []
    parameters.kernel_data = list(None for _ in parameters.wrapper_arguments)
    parameters.temporaries = {}
    parameters.kernel_name = builder.kernel.name

    # replace Materialise
    mapper = Memoizer(replace_materialise)
    mapper.initialisers = []
    instructions = list(mapper(i) for i in instructions)

    # merge indices
    merger = index_merger(instructions)
    instructions = list(merger(i) for i in instructions)
    initialiser = list(itertools.chain(*mapper.initialisers))
    merger = index_merger(initialiser)
    initialiser = list(merger(i) for i in initialiser)
    instructions = instructions + initialiser
    mapper.initialisers = [tuple(merger(i) for i in inits) for inits in mapper.initialisers]

    def name_generator(prefix):
        yield from (f"{prefix}{i}" for i in itertools.count())

    # rename indices and nodes (so that the counters start from zero)
    node_names = {}
    node_namers = dict((cls, name_generator(prefix))
                       for cls, prefix in [(Index, "i"), (Variable, "t")])

    def renamer(expr):
        if isinstance(expr, Argument):
            if expr._name is not None:
                # Some arguments have given names
                return expr._name
            else:
                # Otherwise generate one with their given prefix.
                namer = node_namers.setdefault((type(expr), expr.prefix),
                                               name_generator(expr.prefix))
        else:
            namer = node_namers[type(expr)]
        try:
            return node_names[expr]
        except KeyError:
            return node_names.setdefault(expr, next(namer))

    instructions = rename_nodes(instructions, renamer)
    mapper.initialisers = [rename_nodes(inits, renamer)
                           for inits in mapper.initialisers]
    parameters.wrapper_arguments = rename_nodes(parameters.wrapper_arguments, renamer)
    s, e = rename_nodes([mapper(e) for e in builder.layer_extents], renamer)
    parameters.layer_start = s.name
    parameters.layer_end = e.name

    # scheduling and loop nesting
    deps = instruction_dependencies(instructions, mapper.initialisers)
    within_inames = loop_nesting(instructions, deps, outer_inames, parameters.kernel_name)

    # used to avoid disadvantageous loop interchanges
    loop_priorities = set()
    for iname_nest in within_inames.values():
        if len(iname_nest) > 1:
            loop_priorities.add(tuple(iname_nest.keys()))
    loop_priorities = frozenset(loop_priorities)

    # generate loopy
    context = Bag()
    context.parameters = parameters
    context.within_inames = {k: frozenset(v.keys()) for k, v in within_inames.items()}
    context.conditions = []
    context.index_ordering = []
    context.instruction_dependencies = deps
    context.kernel_parameters = {}

    statements = list(statement(insn, context) for insn in instructions)
    # remove the dummy instructions (they were only used to ensure
    # that the kernel knows about the outer inames).
    statements = list(s for s in statements if not isinstance(s, DummyInstruction))

    domains = list(parameters.domains.values())
    if builder.single_cell:
        new_domains = []
        for d in domains:
            if d.get_dim_name(isl.dim_type.set, 0) == builder._loop_index.name:
                # n = start
                new_domains.append(d.add_constraint(isl.Constraint.eq_from_names(d.space, {"n": 1, "start": -1})))
            else:
                new_domains.append(d)
        domains = new_domains
        if builder.extruded:
            new_domains = []
            for d in domains:
                if d.get_dim_name(isl.dim_type.set, 0) == builder.layer_index.name:
                    # layer = t1 - 1
                    t1 = parameters.layer_end
                    new_domains.append(d.add_constraint(isl.Constraint.eq_from_names(d.space, {"layer": 1, t1: -1, 1: 1})))
                else:
                    new_domains.append(d)
        domains = new_domains

    assumptions, = reduce(operator.and_,
                          parameters.assumptions.values()).params().get_basic_sets()
    options = loopy.Options(check_dep_resolution=True, ignore_boostable_into=True)

    # sometimes masks are not used, but we still need to create the function arguments
    for i, arg in enumerate(parameters.wrapper_arguments):
        if parameters.kernel_data[i] is None:
            arg = loopy.GlobalArg(arg.name, dtype=arg.dtype, shape=arg.shape,
                                  strides=loopy.auto)
            parameters.kernel_data[i] = arg

    if wrapper_name is None:
        wrapper_name = "wrap_%s" % builder.kernel.name

    pwaffd = isl.affs_from_space(assumptions.get_space())
    assumptions = assumptions & pwaffd["start"].ge_set(pwaffd[0])
    if builder.single_cell:
        assumptions = assumptions & pwaffd["start"].lt_set(pwaffd["end"])
    else:
        assumptions = assumptions & pwaffd["start"].le_set(pwaffd["end"])
    if builder.extruded:
        assumptions = assumptions & pwaffd[parameters.layer_start].le_set(pwaffd[parameters.layer_end])
    assumptions = reduce(operator.and_, assumptions.get_basic_sets())

    wrapper = loopy.make_kernel(domains,
                                statements,
                                kernel_data=parameters.kernel_data,
                                target=target,
                                temporary_variables=parameters.temporaries,
                                symbol_manglers=[symbol_mangler],
                                options=options,
                                assumptions=assumptions,
                                lang_version=(2018, 2),
                                name=wrapper_name,
                                loop_priority=loop_priorities)

    # register kernel
    kernel = builder.kernel
    headers = set(kernel.headers)
    headers = headers | set(["#include <math.h>", "#include <complex.h>", "#include <petsc.h>"])
    if PETSc.Log.isActive():
        headers = headers | set(["#include <petsclog.h>"])
    preamble = "\n".join(sorted(headers))

    if isinstance(kernel.code, loopy.TranslationUnit):
        knl = kernel.code
        wrapper = loopy.merge([wrapper, knl])
        # remove the local kernel from the available entrypoints
        wrapper = wrapper.copy(entrypoints=wrapper.entrypoints-{kernel.name})
        wrapper = _match_caller_callee_argument_dimension_(wrapper, kernel.name)
    else:
        # kernel is a string, add it to preamble
        assert isinstance(kernel.code, str)
        code = kernel.code
        wrapper = loopy.register_callable(
            wrapper,
            kernel.name,
            PyOP2KernelCallable(name=kernel.name,
                                parameters=context.kernel_parameters[kernel.name]))
        preamble = preamble + "\n" + code

    wrapper = loopy.register_preamble_generators(wrapper, [_PreambleGen(preamble)])

    # register petsc functions
    for identifier in petsc_functions:
        wrapper = loopy.register_callable(wrapper, identifier, PetscCallable(name=identifier))

    return wrapper


def argtypes(kernel):
    args = []
    for arg in kernel.args:
        if isinstance(arg, loopy.ValueArg):
            args.append(as_ctypes(arg.dtype))
        elif isinstance(arg, loopy.ArrayArg):
            args.append(ctypes.c_voidp)
        else:
            raise ValueError("Unhandled arg type '%s'" % type(arg))
    return args


@singledispatch
def statement(expr, context):
    raise AssertionError("Unhandled statement type '%s'" % type(expr))


@statement.register(DummyInstruction)
def statement_dummy(expr, context):
    new_children = tuple(expression(c, context.parameters) for c in expr.children)
    return DummyInstruction(expr.label, new_children)


@statement.register(When)
def statement_when(expr, context):
    condition, stmt = expr.children
    context.conditions.append(expression(condition, context.parameters))
    stmt = statement(stmt, context)
    context.conditions.pop()
    return stmt


@statement.register(Accumulate)
def statement_assign(expr, context):
    lvalue, _ = expr.children
    if isinstance(lvalue, Indexed):
        context.index_ordering.append(tuple(i.name for i in lvalue.index_ordering()))
    lvalue, rvalue = tuple(expression(c, context.parameters) for c in expr.children)
    within_inames = context.within_inames[expr]

    id, depends_on = context.instruction_dependencies[expr]
    predicates = frozenset(context.conditions)
    return loopy.Assignment(lvalue, rvalue, within_inames=within_inames,
                            within_inames_is_final=True,
                            predicates=predicates,
                            id=id,
                            depends_on=depends_on, depends_on_is_final=True)


@statement.register(FunctionCall)
def statement_functioncall(expr, context):
    parameters = context.parameters

    # We cannot reconstruct the correct calling convention for C-string kernels
    # without providing some additional context about the argument ordering.
    # This is processed inside the ``emit_call_insn`` method of
    # :class:`.PyOP2KernelCallable`.
    context.kernel_parameters[expr.name] = []

    free_indices = set(i.name for i in expr.free_indices)
    writes = []
    reads = []
    for access, child in zip(expr.access, expr.children):
        var = expression(child, parameters)
        if isinstance(var, pym.Subscript):
            # tensor argument
            sweeping_indices = []
            for index in var.index_tuple:
                if isinstance(index, pym.Variable) and index.name in free_indices:
                    sweeping_indices.append(index)
            arg = SubArrayRef(tuple(sweeping_indices), var)
        else:
            # scalar argument or constant
            arg = var
        context.kernel_parameters[expr.name].append(arg)

        if access is READ or (isinstance(child, Argument) and isinstance(child.dtype, OpaqueType)):
            reads.append(arg)
        elif access is WRITE:
            writes.append(arg)
        else:
            reads.append(arg)
            writes.append(arg)

    within_inames = context.within_inames[expr]
    predicates = frozenset(context.conditions)
    id, depends_on = context.instruction_dependencies[expr]

    call = pym.Call(pym.Variable(expr.name), tuple(reads))

    return loopy.CallInstruction(tuple(writes), call,
                                 within_inames=within_inames,
                                 within_inames_is_final=True,
                                 predicates=predicates,
                                 id=id,
                                 depends_on=depends_on, depends_on_is_final=True)


@singledispatch
def expression(expr, parameters):
    raise AssertionError("Unhandled expression type '%s'" % type(expr))


@expression.register(Index)
def expression_index(expr, parameters):
    name = expr.name
    if name not in parameters.domains:
        vars = isl.make_zero_and_vars([name])
        zero = vars[0]
        domain = (vars[name].ge_set(zero) & vars[name].lt_set(zero + expr.extent))
        parameters.domains[name] = domain
    return pym.Variable(name)


@expression.register(FixedIndex)
def expression_fixedindex(expr, parameters):
    return expr.value


@expression.register(RuntimeIndex)
def expression_runtimeindex(expr, parameters):
    @singledispatch
    def translate(expr, vars):
        raise AssertionError("Unhandled type '%s' in domain translation" % type(expr))

    @translate.register(Sum)
    def translate_sum(expr, vars):
        return operator.add(*(translate(c, vars) for c in expr.children))

    @translate.register(Argument)
    def translate_argument(expr, vars):
        expr = expression(expr, parameters)
        return vars[expr.name]

    @translate.register(Variable)
    def translate_variable(expr, vars):
        return vars[expr.name]

    @translate.register(Zero)
    def translate_zero(expr, vars):
        assert expr.shape == ()
        return vars[0]

    @translate.register(LogicalAnd)
    def translate_logicaland(expr, vars):
        a, b = (translate(c, vars) for c in expr.children)
        return a & b

    @translate.register(Comparison)
    def translate_comparison(expr, vars):
        a, b = (translate(c, vars) for c in expr.children)
        fn = {">": "gt_set",
              ">=": "ge_set",
              "==": "eq_set",
              "!=": "ne_set",
              "<": "lt_set",
              "<=": "le_set"}[expr.operator]
        return getattr(a, fn)(b)

    name = expr.name
    if name not in parameters.domains:
        lo, hi, constraint = expr.children
        params = list(v.name for v in traversal([lo, hi]) if isinstance(v, (Argument, Variable)))
        vars = isl.make_zero_and_vars([name], params)
        domain = (vars[name].ge_set(translate(lo, vars))
                  & vars[name].lt_set(translate(hi, vars)))
        parameters.domains[name] = domain
        if constraint is not None:
            parameters.assumptions[name] = translate(constraint, vars)
    return pym.Variable(name)


@expression.register(MultiIndex)
def expression_multiindex(expr, parameters):
    return tuple(expression(c, parameters) for c in expr.children)


@expression.register(Extent)
def expression_extent(expr, parameters):
    multiindex, = expr.children
    # TODO: If loopy eventually gains the ability to vectorise
    # functions that use this, we will need a symbolic node for the
    # index extent.
    return int(numpy.prod(tuple(i.extent for i in multiindex)))


@expression.register(Symbol)
def expression_symbol(expr, parameters):
    return pym.Variable(expr.name)


@expression.register(Argument)
def expression_argument(expr, parameters):
    name = expr.name
    shape = expr.shape
    dtype = expr.dtype
    if shape == ():
        arg = loopy.ValueArg(name, dtype=dtype)
    else:
        arg = loopy.GlobalArg(name,
                              dtype=dtype,
                              shape=shape,
                              strides=loopy.auto)
    idx = parameters.wrapper_arguments.index(expr)
    parameters.kernel_data[idx] = arg
    return pym.Variable(name)


@expression.register(Variable)
def expression_variable(expr, parameters):
    name = expr.name
    shape = expr.shape
    dtype = expr.dtype
    if name not in parameters.temporaries:
        parameters.temporaries[name] = loopy.TemporaryVariable(name,
                                                               dtype=dtype,
                                                               shape=shape,
                                                               address_space=loopy.auto)
    return pym.Variable(name)


@expression.register(Zero)
def expression_zero(expr, parameters):
    assert expr.shape == ()
    return 0


@expression.register(Literal)
def expression_literal(expr, parameters):
    assert expr.shape == ()
    if expr.casting:
        return loopy.symbolic.TypeCast(expr.dtype, expr.value)
    return expr.value


@expression.register(NamedLiteral)
def expression_namedliteral(expr, parameters):
    name = expr.name
    val = loopy.TemporaryVariable(name,
                                  dtype=expr.dtype,
                                  shape=expr.shape,
                                  address_space=loopy.AddressSpace.LOCAL,
                                  read_only=True,
                                  initializer=expr.value)
    parameters.temporaries[name] = val

    return pym.Variable(name)


@expression.register(Conditional)
def expression_conditional(expr, parameters):
    return pym.If(*(expression(c, parameters) for c in expr.children))


@expression.register(Comparison)
def expression_comparison(expr, parameters):
    l, r = (expression(c, parameters) for c in expr.children)
    return pym.Comparison(l, expr.operator, r)


@expression.register(LogicalNot)
@expression.register(BitwiseNot)
def expression_uop(expr, parameters):
    child, = (expression(c, parameters) for c in expr.children)
    return {LogicalNot: pym.LogicalNot,
            BitwiseNot: pym.BitwiseNot}[type(expr)](child)


@expression.register(Sum)
@expression.register(Product)
@expression.register(Quotient)
@expression.register(FloorDiv)
@expression.register(Remainder)
@expression.register(LogicalAnd)
@expression.register(LogicalOr)
@expression.register(BitwiseAnd)
@expression.register(BitwiseOr)
def expression_binop(expr, parameters):
    children = tuple(expression(c, parameters) for c in expr.children)
    if type(expr) in {Quotient, FloorDiv, Remainder}:
        return {Quotient: pym.Quotient,
                FloorDiv: pym.FloorDiv,
                Remainder: pym.Remainder}[type(expr)](*children)
    else:
        return {Sum: pym.Sum,
                Product: pym.Product,
                LogicalOr: pym.LogicalOr,
                LogicalAnd: pym.LogicalAnd,
                BitwiseOr: pym.BitwiseOr,
                BitwiseAnd: pym.BitwiseAnd}[type(expr)](children)


@expression.register(Min)
@expression.register(Max)
def expression_minmax(expr, parameters):
    children = tuple(expression(c, parameters) for c in expr.children)
    return {Min: pym.Variable("min"),
            Max: pym.Variable("max")}[type(expr)](*children)


@expression.register(BitShift)
def expression_bitshift(expr, parameters):
    children = (expression(c, parameters) for c in expr.children)
    return {"<<": pym.LeftShift,
            ">>": pym.RightShift}[expr.direction](*children)


@expression.register(Indexed)
def expression_indexed(expr, parameters):
    aggregate, multiindex = (expression(c, parameters) for c in expr.children)
    return pym.Subscript(aggregate, multiindex)
