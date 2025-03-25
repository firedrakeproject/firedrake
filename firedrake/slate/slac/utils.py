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
    from pyop2.codegen.loopycompat import _match_caller_callee_argument_dimension_
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
