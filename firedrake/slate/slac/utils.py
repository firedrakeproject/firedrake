from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view)
from gem import indices as make_indices
from gem.node import Memoizer
from gem.node import pre_traversal as traverse_dags

from functools import singledispatch
import firedrake.slate.slate as sl
import loopy as lp
from loopy.program import make_program
from loopy.transform.callable import inline_callable_kernel, register_callable_kernel
import itertools


class RemoveRestrictions(MultiFunction):
    """UFL MultiFunction for removing any restrictions on the
    integrals of forms.
    """
    expr = MultiFunction.reuse_if_untouched

    def positive_restricted(self, o):
        return self(o.ufl_operands[0])


class SymbolWithFuncallIndexing(ast.Symbol):
    """A functionally equivalent representation of a `coffee.Symbol`,
    with modified output for rank calls. This is syntactically necessary
    when referring to symbols of Eigen::MatrixBase objects.
    """

    def _genpoints(self):
        """Parenthesize indices during loop assignment"""
        pt = lambda p: "%s" % p
        pt_ofs = lambda p, o: "%s*%s+%s" % (p, o[0], o[1])
        pt_ofs_stride = lambda p, o: "%s+%s" % (p, o)
        result = []

        if not self.offset:
            for p in self.rank:
                result.append(pt(p))
        else:
            for p, ofs in zip(self.rank, self.offset):
                if ofs == (1, 0):
                    result.append(pt(p))
                elif ofs[0] == 1:
                    result.append(pt_ofs_stride(p, ofs[1]))
                else:
                    result.append(pt_ofs(p, ofs))
        result = ', '.join(i for i in result)

        return "(%s)" % result


class Transformer(Visitor):
    """Replaces all out-put tensor references with a specified
    name of :type: `Eigen::Matrix` with appropriate shape. This
    class is primarily for COFFEE acrobatics, jumping through
    nodes and redefining where appropriate.

    The default name of :data:`"A"` is assigned, otherwise a
    specified name may be passed as the :data:`name` keyword
    argument when calling the visitor.
    """

    def visit_object(self, o, *args, **kwargs):
        """Visits an object and returns it.

        e.g. string ---> string
        """
        return o

    def visit_list(self, o, *args, **kwargs):
        """Visits an input of COFFEE objects and returns
        the complete list of said objects.
        """
        newlist = [self.visit(e, *args, **kwargs) for e in o]
        if all(newo is e for newo, e in zip(newlist, o)):
            return o

        return newlist

    visit_Node = Visitor.maybe_reconstruct

    def visit_FunDecl(self, o, *args, **kwargs):
        """Visits a COFFEE FunDecl object and reconstructs
        the FunDecl body and header to generate
        ``Eigen::MatrixBase`` C++ template functions.

        Creates a template function for each subkernel form.

        .. code-block:: c++

            template <typename Derived>
            static inline void foo(Eigen::MatrixBase<Derived> const & A, ...)
            {
              [Body...]
            }
        """
        name = kwargs.get("name", "A")
        new = self.visit_Node(o, *args, **kwargs)
        ops, okwargs = new.operands()
        if all(new is old for new, old in zip(ops, o.operands()[0])):
            return o

        ret, kernel_name, kernel_args, body, pred, headers, template = ops

        body_statements, _ = body.operands()
        decl_init = "const_cast<Eigen::MatrixBase<Derived> &>(%s_);\n" % name
        new_dec = ast.Decl(typ="Eigen::MatrixBase<Derived> &", sym=name,
                           init=decl_init)
        new_body = [new_dec] + body_statements
        eigen_template = "template <typename Derived>"

        new_ops = (ret, kernel_name, kernel_args,
                   new_body, pred, headers, eigen_template)

        return o.reconstruct(*new_ops, **okwargs)

    def visit_Decl(self, o, *args, **kwargs):
        """Visits a declared tensor and changes its type to
        :template: result `Eigen::MatrixBase<Derived>`.

        i.e. double A[n][m] ---> const Eigen::MatrixBase<Derived> &A_
        """
        name = kwargs.get("name", "A")
        if o.sym.symbol != name:
            return o
        newtype = "const Eigen::MatrixBase<Derived> &"

        return o.reconstruct(newtype, ast.Symbol("%s_" % name))

    def visit_Symbol(self, o, *args, **kwargs):
        """Visits a COFFEE symbol and redefines it as a Symbol with
        FunCall indexing.

        i.e. A[j][k] ---> A(j, k)
        """
        name = kwargs.get("name", "A")
        if o.symbol != name:
            return o

        return SymbolWithFuncallIndexing(o.symbol, o.rank, o.offset)


def slate_to_gem(expression):
    """Convert a slate expression to gem.

        :arg expression: A slate expression.
        :returns: A singleton list of gem expressions and
        a mapping from gem variables to UFL "terminal" forms.
    """

    mapper, var2terminal = slate2gem(expression)
    return mapper, var2terminal


@singledispatch
def _slate2gem(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
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


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    return Inverse(*map(self, expr.children))


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


def slate2gem(expression):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    return mapper(expression), mapper.var2terminal


def eigen_tensor(expr, temporary, index):
    """Returns an appropriate assignment statement for populating a particular
    `Eigen::MatrixBase` tensor. If the tensor is mixed, then access to the
    :meth:`block` of the eigen tensor is provided. Otherwise, no block
    information is needed and the tensor is returned as is.

    :arg expr: a `slate.Tensor` node.
    :arg temporary: the associated temporary of the expr argument.
    :arg index: a tuple of integers used to determine row and column
                information. This is provided by the SplitKernel
                associated with the expr.
    """
    if expr.is_mixed:
        try:
            row, col = index
        except ValueError:
            row = index[0]
            col = 0
        rshape = expr.shapes[0][row]
        rstart = sum(expr.shapes[0][:row])
        try:
            cshape = expr.shapes[1][col]
            cstart = sum(expr.shapes[1][:col])
        except KeyError:
            cshape = 1
            cstart = 0

        tensor = ast.FlatBlock("%s.block<%d, %d>(%d, %d)" % (temporary,
                                                             rshape, cshape,
                                                             rstart, cstart))
    else:
        tensor = temporary

    return tensor


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


def merge_loopy(slate_loopy, builder, var2terminal):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs = builder.collect_coefficients()
    builder.bag = SlateWrapperBag(coeffs)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(var2terminal, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: isinstance(x, sl.Tensor), var2terminal.values()))
    tsfc_calls, tsfc_kernels = zip(*itertools.chain.from_iterable(
                                   (builder.generate_tsfc_calls(terminal, tensor2temp[terminal])
                                    for terminal in terminal_tensors)))

    # Construct args
    args = slate_loopy.args.copy() + builder.generate_wrapper_kernel_args(tensor2temp, tsfc_kernels)

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns += builder.slate_call(slate_loopy)

    # Inames come from initialisations + loopyfying kernel args and lhs
    domains = builder.bag.index_creator.domains

    # Generates the loopy wrapper kernel
    slate_wrapper = lp.make_function(domains, insns, args, name="slate_wrapper",
                                     seq_dependencies=True, target=lp.CTarget())

    # Generate program from kernel, so that one can register and inline kernels
    prg = make_program(slate_wrapper)
    for tsfc_loopy in tsfc_kernels:
        prg = register_callable_kernel(prg, tsfc_loopy)
        prg = inline_callable_kernel(prg, tsfc_loopy.name)
    prg = register_callable_kernel(prg, slate_loopy)
    prg = inline_callable_kernel(prg, slate_loopy.name)

    return prg
