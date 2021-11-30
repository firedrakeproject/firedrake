from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 Solve, Inverse, Variable, view, Delta, Index, Division,
                 Action)
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
@_slate2gem.register(sl.TensorShell)
def _slate2gem_tensor(expr, self):
    shape = expr.shape if not len(expr.shape) == 0 else (1, )
    assert expr not in self.var2terminal.values()
    var = Variable(None, shape)
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


@_slate2gem.register(sl.Action)
def _slate2gem_action(expr, self):
    assert expr not in self.gem2slate.values()
    children = list(map(self, expr.children))
    var = Action(*children, expr.pick_op)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    if expr.matfree:
        assert expr not in self.gem2slate.values()
        var = Solve(*map(self, expr.children), expr.matfree, self(expr.Aonx), self(expr.Aonp))
        self.gem2slate[var.name] = expr
        return var
    else:
        return Solve(*map(self, expr.children))


@_slate2gem.register(sl.Transpose)
def _slate2gem_transpose(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Indexed(child, indices), tuple(indices[::-1]))
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Negative)
def _slate2gem_negative(expr, self):
    child, = map(self, expr.children)
    indices = tuple(make_indices(len(child.shape)))
    var = ComponentTensor(Product(Literal(-1),
                          Indexed(child, indices)),
                          indices)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Add)
def _slate2gem_add(expr, self):
    A, B = map(self, expr.children)
    indices = tuple(make_indices(len(A.shape)))
    var = ComponentTensor(Sum(Indexed(A, indices),
                          Indexed(B, indices)),
                          indices)
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Mul)
def _slate2gem_mul(expr, self):
    A, B = map(self, expr.children)
    *i, k = tuple(make_indices(len(A.shape)))
    _, *j = tuple(make_indices(len(B.shape)))
    ABikj = Product(Indexed(A, tuple(i + [k])),
                    Indexed(B, tuple([k] + j)))
    var = ComponentTensor(IndexSum(ABikj, (k, )), tuple(i + j))
    self.gem2slate[var] = expr
    return var


@_slate2gem.register(sl.Factorization)
def _slate2gem_factorization(expr, self):
    A, = map(self, expr.children)
    return A


def slate2gem(expression, options):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    mapper.gem2slate = OrderedDict()
    mapper.matfree = options["replace_mul"]
    m = mapper(expression)
    # WIP actually make use of the fact that we do have two different dicts now
    mapper.var2terminal.update(mapper.gem2slate)
    return m, mapper.var2terminal


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


def merge_loopy(slate_loopy, output_arg, builder, var2terminal, wrapper_name, ctx_g2l, strategy="terminals_first", slate_expr=None, tsfc_parameters=None, slate_parameters=None):
    """ Merges tsfc loopy kernels and slate loopy kernel into a wrapper kernel."""

    if strategy == "terminals_first":
        slate_loopy_prg = slate_loopy
        slate_loopy = slate_loopy[builder.slate_loopy_name]
        tensor2temp, tsfc_kernels, insns, builder = assemble_terminals_first(builder, var2terminal, slate_loopy)
        # Construct args
        args, tmp_args = builder.generate_wrapper_kernel_args(tensor2temp)
        kernel_args = [output_arg] + args
        args = [output_arg.loopy_arg] + [a.loopy_arg for a in args] + tmp_args
        for a in slate_loopy.args:
            if a.name not in [arg.name for arg in args] and a.name.startswith("S"):
                ac = a.copy(address_space=lp.AddressSpace.LOCAL)
                args.append(ac)

        # Inames come from initialisations + loopyfying kernel args and lhs
        domains = slate_loopy.domains + builder.bag.index_creator.domains

        # Help scheduling by setting within_inames_is_final on everything
        insns_new = []
        for i, insn in enumerate(insns):
            if insn:
                insns_new.append(insn.copy(depends_on=frozenset({}),
                                 priority=len(insns)-i,
                                 within_inames_is_final=True))

        # Generates the loopy wrapper kernel
        slate_wrapper = lp.make_function(domains, insns_new, args, name=wrapper_name,
                                         seq_dependencies=True, target=lp.CTarget())

        # Prevent loopy interchange by loopy
        slate_wrapper = lp.prioritize_loops(slate_wrapper, ",".join(builder.bag.index_creator.inames.keys()))

        # Register kernels
        loop = []
        for k in tsfc_kernels:
            if k:
                loop += [k.items()]
        loop += [{slate_loopy.name: slate_loopy_prg}.items()]

        for l in loop:
            (name, knl), = tuple(l)
            if knl:
                slate_wrapper = lp.merge([slate_wrapper, knl])
                slate_wrapper = _match_caller_callee_argument_dimension_(slate_wrapper, name)
        return slate_wrapper, tuple(kernel_args)

    elif strategy == "when_needed":
        tensor2temp, builder, slate_loopy = assemble_when_needed(builder, var2terminal,
                                                                 slate_loopy, slate_expr,
                                                                 ctx_g2l, tsfc_parameters,
                                                                 slate_parameters, True, {}, output_arg)
        return slate_loopy


def assemble_terminals_first(builder, var2terminal, slate_loopy):
    from firedrake.slate.slac.kernel_builder import SlateWrapperBag
    coeffs, _ = builder.collect_coefficients(artificial=False)
    builder.bag = SlateWrapperBag(coeffs, name=slate_loopy.name)

    # In the initialisation the loopy tensors for the terminals are generated
    # Those are the needed again for generating the TSFC calls
    inits, tensor2temp = builder.initialise_terminals(var2terminal, builder.bag.coefficients)
    terminal_tensors = list(filter(lambda x: isinstance(x, sl.Tensor), var2terminal.values()))
    tsfc_calls, tsfc_kernels = zip(*itertools.chain.from_iterable(
                                   (builder.generate_tsfc_calls(terminal, tensor2temp[terminal])
                                    for terminal in terminal_tensors)))

    # Add profiling for inits
    inits, slate_init_event, preamble_init = profile_insns("inits_"+name, inits, PETSc.Log.isActive())

    # Munge instructions
    insns = inits
    insns.extend(tsfc_calls)
    insns.append(builder.slate_call(slate_loopy, tensor2temp.values()))
    
    return tensor2temp, tsfc_kernels, insns, builder

def assemble_when_needed(builder, var2terminal, slate_loopy, slate_expr, ctx_g2l, tsfc_parameters, slate_parameters, init_temporaries=True, tensor2temp={}, output_arg=None, matshell=False):
    # FIXME This function needs some refactoring
    # Essentially there are 4 codepath: 1) insn is no matrix-free special insn
    #                                   2) insn is an Action
    #                                   3) insn is a Solve
    #                                       a) the matrix is terminal
    #                                       b) the matrix is a TensorShell
    # My idea would be to subclass CallInstruction from loopy to ActionInstruction, SolveInstruction, ...
    # and then we can use a dispatcher

    # need imports here bc m partially initialized module error
    from firedrake.slate.slac.kernel_builder import LocalLoopyKernelBuilder
    from firedrake.slate.slac.compiler import gem_to_loopy

    insns = []
    tensor2temps = tensor2temp
    knl_list = {}
    gem2pym = ctx_g2l.gem_to_pymbolic

    # invert dict
    pyms = [pyms.name if isinstance(pyms, pym.Variable) else pyms.assignee_name for pyms in gem2pym.values()]
    pym2gem = OrderedDict(zip(pyms, gem2pym.keys()))
    slate_loopy_name = builder.slate_loopy_name
    for insn in slate_loopy[slate_loopy_name].instructions:
        # TODO specialise the call instruction node and dispatch based on its type
        if (not isinstance(insn, lp.kernel.instruction.CallInstruction)
            or not (insn.expression.function.name.startswith("action")
                    or insn.expression.function.name.startswith("mtf"))):
            # normal instructions can stay as they are
            insns.append(insn)
        else:
            pass
    if init_temporaries:
        # We need to do initialise the temporaries at the end, when we collected all the ones we need
        builder, tensor2temps, inits = initialise_temps(builder, var2terminal, tensor2temps)
        for i in inits:
            insns.insert(0, i)

    slate_loopy = update_wrapper_kernel(builder, insns, output_arg, tensor2temps, knl_list, slate_loopy)
    return tensor2temps, builder, slate_loopy
