from collections import OrderedDict

from ufl.corealg.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view, Delta, Index, Division)
from gem import indices as make_indices
from gem.node import Memoizer
from gem.node import pre_traversal as traverse_dags

from functools import singledispatch
import firedrake.slate.slate as sl
import loopy as lp
from loopy.transform.callable import merge
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401
from firedrake.parameters import target
from tsfc.loopy import profile_insns
from petsc4py import PETSc


class RemoveRestrictions(MultiFunction):
    """UFL MultiFunction for removing any restrictions on the
    integrals of forms.
    """
    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


def slate_to_gem(expression, options):
    """Convert a slate expression to gem.

    :arg expression: A slate expression.
    :returns: A singleton list of gem expressions and a mapping from
        gem variables to UFL "terminal" forms.
    """

    mapper, var2terminal = slate2gem(expression, options)
    return mapper, var2terminal


@singledispatch
def _slate2gem(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
@_slate2gem.register(sl.BlockAssembledVector)
def _slate2gem_tensor(expr, self):
    shape = expr.shape if not len(expr.shape) == 0 else (1, )
    name = f"T{len(self.var2terminal)}"
    assert expr not in self.var2terminal.values()
    var = Variable(name, shape)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Block)
def _slate2gem_block(expr, self):
    child, = map(self, expr.children)
    child_shapes = expr.children[0].shapes
    offsets = tuple(sum(shape[:idx]) for shape, (idx, *_)
                    in zip(child_shapes.values(), expr._indices))
    return view(child, *(slice(idx, idx+extent) for idx, extent in zip(offsets, expr.shape)))


@_slate2gem.register(sl.DiagonalTensor)
def _slate2gem_diagonal(expr, self):
    if not self.matfree:
        A, = map(self, expr.children)
        assert A.shape[0] == A.shape[1]
        i, j = (Index(extent=s) for s in A.shape)
        return ComponentTensor(Product(Indexed(A, (i, i)), Delta(i, j)), (i, j))
    else:
        raise NotImplementedError("Diagonals on Slate expressions are \
                                   not implemented in a matrix-free manner yet.")


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    tensor, = expr.children
    if expr.diagonal:
        # optimise inverse on diagonal tensor by translating to
        # matrix which contains the reciprocal values of the diagonal tensor
        A, = map(self, expr.children)
        i, j = (Index(extent=s) for s in A.shape)
        return ComponentTensor(Product(Division(Literal(1), Indexed(A, (i, i))),
                                       Delta(i, j)), (i, j))
    else:
        return Inverse(self(tensor))


@_slate2gem.register(sl.Reciprocal)
def _slate2gem_reciprocal(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    return ComponentTensor(Division(Literal(1.), Indexed(child, indices)), indices)


@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    return Solve(*map(self, expr.children))


@_slate2gem.register(sl.Transpose)
def _slate2gem_transpose(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    return ComponentTensor(Indexed(child, indices), tuple(indices[::-1]))


@_slate2gem.register(sl.Negative)
def _slate2gem_negative(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    return ComponentTensor(Product(Literal(-1),
                           Indexed(child, indices)),
                           indices)


@_slate2gem.register(sl.Add)
def _slate2gem_add(expr, self):
    A, B = map(self, expr.children)
    indices = tuple(make_indices(len(A.shape)))
    return ComponentTensor(Sum(Indexed(A, indices),
                           Indexed(B, indices)),
                           indices)


@_slate2gem.register(sl.Mul)
def _slate2gem_mul(expr, self):
    A, B = map(self, expr.children)
    *i, k = tuple(make_indices(len(A.shape)))
    _, *j = tuple(make_indices(len(B.shape)))
    ABikj = Product(Indexed(A, tuple(i + [k])),
                    Indexed(B, tuple([k] + j)))
    return ComponentTensor(IndexSum(ABikj, (k, )), tuple(i + j))


@_slate2gem.register(sl.Factorization)
def _slate2gem_factorization(expr, self):
    A, = map(self, expr.children)
    return A


def slate2gem(expression, options):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    mapper.matfree = options["replace_mul"]
    return mapper(expression), mapper.var2terminal


def depth_first_search(graph, node, visited, schedule):
    """A recursive depth-first search (DFS) algorithm for
    traversing a DAG consisting of Slate expressions.

    :arg graph: A DAG whose nodes (vertices) are Slate expressions
                with edges connected to dependent expressions.
    :arg node: A starting vertex.
    :arg visited: A set keeping track of visited nodes.
    :arg schedule: A list of reverse-postordered nodes. This list is
                   used to produce a topologically sorted list of
                   Slate nodes.
    """
    if node not in visited:
        visited.add(node)

        for n in graph[node]:
            depth_first_search(graph, n, visited, schedule)

        schedule.append(node)


def topological_sort(exprs):
    """Topologically sorts a list of Slate expressions. The
    expression graph is constructed by relating each Slate
    node with a list of dependent Slate nodes.

    :arg exprs: A list of Slate expressions.
    """
    graph = OrderedDict((expr, set(traverse_dags([expr])) - {expr})
                        for expr in exprs)

    schedule = []
    visited = set()
    for n in graph:
        depth_first_search(graph, n, visited, schedule)

    return schedule


def merge_loopy(slate_loopy, output_arg, builder, var2terminal, name):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs = builder.collect_coefficients()
    constants = builder.collect_constants()
    builder.bag = SlateWrapperBag(coeffs, constants)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(var2terminal, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: (x.terminal and not x.assembled), var2terminal.values()))
    calls_and_kernels_and_events = tuple((c, k, e) for terminal in terminal_tensors
                                         for c, k, e in builder.generate_tsfc_calls(terminal, tensor2temp[terminal]))
    if calls_and_kernels_and_events:  # tsfc may not give a kernel back
        tsfc_calls, tsfc_kernels, tsfc_events = zip(*calls_and_kernels_and_events)
    else:
        tsfc_calls = ()
        tsfc_kernels = ()

    args, tmp_args = builder.generate_wrapper_kernel_args(tensor2temp)
    kernel_args = [output_arg] + args
    loopy_args = [output_arg.loopy_arg] + [a.loopy_arg for a in args] + tmp_args

    # Add profiling for inits
    inits, slate_init_event, preamble_init = profile_insns("inits_"+name, inits, PETSc.Log.isActive())

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns.append(builder.slate_call(slate_loopy, tensor2temp.values()))

    # Add profiling for the whole kernel
    insns, slate_wrapper_event, preamble = profile_insns(name, insns, PETSc.Log.isActive())

    # Add a no-op touching all kernel arguments to make sure they are not
    # silently dropped
    noop = lp.CInstruction(
        (), "", read_variables=frozenset({a.name for a in loopy_args}),
        within_inames=frozenset(), within_inames_is_final=True)
    insns.append(noop)

    # Inames come from initialisations + loopyfying kernel args and lhs
    domains = builder.bag.index_creator.domains

    # Generates the loopy wrapper kernel
    preamble = preamble_init+preamble if preamble else []
    slate_wrapper = lp.make_function(domains, insns, loopy_args, name=name,
                                     seq_dependencies=True, target=target,
                                     lang_version=(2018, 2), preambles=preamble)

    # Generate program from kernel, so that one can register kernels
    from loopy.kernel.function_interface import CallableKernel

    for tsfc_loopy in tsfc_kernels:
        slate_wrapper = merge([slate_wrapper, tsfc_loopy])
    slate_wrapper = merge([slate_wrapper, slate_loopy])

    # At this point the individual subkernels are no longer callable, we
    # only want to access the generated code via the wrapper.
    slate_wrapper = slate_wrapper.with_entrypoints({name})

    for tsfc_loopy in tsfc_kernels:
        for name in tsfc_loopy.callables_table:
            if isinstance(slate_wrapper.callables_table[name], CallableKernel):
                slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)
    for name in slate_loopy.callables_table:
        if isinstance(slate_wrapper.callables_table[name], CallableKernel):
            slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)

    events = tsfc_events + (slate_wrapper_event, slate_init_event) if PETSc.Log.isActive() else ()
    return slate_wrapper, tuple(kernel_args), events


# Everything in this file was formerly in pyop2/codegen/loopycompat.py
#
# Everything in this file was formerly in loopy/transform/callable.py
# but was removed in https://github.com/inducer/loopy/pull/327. It has
# been kept here for compatibility but should be phased out.

# Note that since this code is copypasted, the linter has been turned off.

# flake8: noqa

from loopy.kernel.instruction import CallInstruction, MultiAssignmentBase, \
    CInstruction, _DataObliviousInstruction
from loopy.symbolic import CombineMapper, IdentityMapper
from loopy.symbolic import simplify_via_aff
from loopy.kernel.function_interface import CallableKernel
from loopy.translation_unit import TranslationUnit


# Tools to match caller to callee args by (guessed) automatic reshaping
#
# (This is undocumented and not recommended, but it is currently needed
# to support Firedrake.)

class DimChanger(IdentityMapper):
    """
    Mapper to change the dimensions of an argument.
    .. attribute:: callee_arg_dict
        A mapping from the argument name (:class:`str`) to instances of
        :class:`loopy.kernel.array.ArrayBase`.
    .. attribute:: desried_shape
        A mapping from argument name (:class:`str`) to an instance of
        :class:`tuple`.
    """
    def __init__(self, callee_arg_dict, desired_shape):
        self.callee_arg_dict = callee_arg_dict
        self.desired_shape = desired_shape
        super().__init__()

    def map_subscript(self, expr):
        if expr.aggregate.name not in self.callee_arg_dict:
            return super().map_subscript(expr)
        callee_arg_dim_tags = self.callee_arg_dict[expr.aggregate.name].dim_tags
        flattened_index = sum(dim_tag.stride*idx for dim_tag, idx in
                zip(callee_arg_dim_tags, expr.index_tuple))
        new_indices = []

        from operator import mul
        from functools import reduce
        stride = reduce(mul, self.desired_shape[expr.aggregate.name], 1)

        for length in self.desired_shape[expr.aggregate.name]:
            stride /= length
            ind = flattened_index // int(stride)
            flattened_index -= (int(stride) * ind)
            new_indices.append(simplify_via_aff(ind))

        return expr.aggregate[tuple(new_indices)]


def _match_caller_callee_argument_dimension_for_single_kernel(
        caller_knl, callee_knl):
    """
    :returns: a copy of *caller_knl* with the instance of
        :class:`loopy.kernel.function_interface.CallableKernel` addressed by
        *callee_function_name* in the *caller_knl* aligned with the argument
        dimensions required by *caller_knl*.
    """
    from loopy.kernel.array import ArrayBase
    from loopy.kernel.data import auto

    for insn in caller_knl.instructions:
        if not isinstance(insn, CallInstruction) or (
                insn.expression.function.name !=
                callee_knl.name):
            # Call to a callable kernel can only occur through a
            # CallInstruction.
            continue

        def _shape_1_if_empty(shape_caller, shape_callee):
            assert isinstance(shape_caller, tuple)
            if shape_caller == () and shape_caller!=shape_callee:
                return (1,)
            else:
                return shape_caller

        from loopy.kernel.function_interface import (
                ArrayArgDescriptor, get_arg_descriptor_for_expression,
                get_kw_pos_association)
        _, pos_to_kw = get_kw_pos_association(callee_knl)
        arg_id_to_shape = {}
        for arg_id, arg in insn.arg_id_to_arg().items():
            arg_id = pos_to_kw[arg_id]

            arg_descr = get_arg_descriptor_for_expression(caller_knl, arg)
            if isinstance(arg_descr, ArrayArgDescriptor):
                arg_id_to_shape[arg_id] = arg_descr.shape
            else:
                arg_id_to_shape[arg_id] = (1, )

        dim_changer = DimChanger(
                callee_knl.arg_dict,
                arg_id_to_shape)

        new_callee_insns = []
        for callee_insn in callee_knl.instructions:
            if isinstance(callee_insn, MultiAssignmentBase):
                new_callee_insns.append(callee_insn
                        .with_transformed_expressions(dim_changer))

            elif isinstance(callee_insn, (CInstruction,
                    _DataObliviousInstruction)):
                # The layout of the args to a CInstructions is not going to be matched to the caller_kernel,
                # they are appended with unmatched args.
                # We only use Cinstructions exceptionally, e.g. for adding profile instructions,
                # without arguments that required to be matched, so this is ok.
                new_callee_insns.append(callee_insn)
            else:
                raise NotImplementedError("Unknown instruction %s." %
                        type(insn))

        new_args = [arg if not isinstance(arg, ArrayBase)
                    else arg.copy(shape=arg_id_to_shape[arg.name],
                                  dim_tags=None, strides=auto, order="C")
                    for arg in callee_knl.args]

        # subkernel with instructions adjusted according to the new dimensions
        new_callee_knl = callee_knl.copy(instructions=new_callee_insns,
                                         args=new_args)

        return new_callee_knl


class _FunctionCalledChecker(CombineMapper):
    def __init__(self, func_name):
        self.func_name = func_name
        super().__init__()

    def combine(self, values):
        return any(values)

    def map_call(self, expr):
        if expr.function.name == self.func_name:
            return True
        return self.combine(
                tuple(
                    self.rec(child) for child in expr.parameters)
                )

    map_call_with_kwargs = map_call

    def map_constant(self, expr):
        return False

    def map_type_cast(self, expr):
        return self.rec(expr.child)

    def map_algebraic_leaf(self, expr):
        return False

    def map_kernel(self, kernel):
        return any(self.rec(insn.expression) for insn in kernel.instructions if
                isinstance(insn, MultiAssignmentBase))


def _match_caller_callee_argument_dimension_(program, callee_function_name):
    """
    Returns a copy of *program* with the instance of
    :class:`loopy.kernel.function_interface.CallableKernel` addressed by
    *callee_function_name* in the *program* aligned with the argument
    dimensions required by *caller_knl*.
    .. note::
        The callee kernel addressed by *callee_function_name*, should be
        called at only one location throughout the program, as multiple
        invocations would demand complex renaming logic which is not
        implemented yet.
    """
    assert isinstance(program, TranslationUnit)
    assert isinstance(callee_function_name, str)
    assert callee_function_name not in program.entrypoints
    assert callee_function_name in program.callables_table

    is_invoking_callee = _FunctionCalledChecker(
            callee_function_name).map_kernel

    caller_knl,  = [in_knl_callable.subkernel for in_knl_callable in
            program.callables_table.values() if isinstance(in_knl_callable,
                CallableKernel) and
            is_invoking_callee(in_knl_callable.subkernel)]

    from pymbolic.primitives import Call
    assert len([insn for insn in caller_knl.instructions if (isinstance(insn,
        CallInstruction) and isinstance(insn.expression, Call) and
        insn.expression.function.name == callee_function_name)]) == 1
    new_callee_kernel = _match_caller_callee_argument_dimension_for_single_kernel(
            caller_knl, program[callee_function_name])
    return program.with_kernel(new_callee_kernel)
