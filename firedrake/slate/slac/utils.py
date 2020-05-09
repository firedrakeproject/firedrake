from coffee import base as ast
from coffee.visitor import Visitor
from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction
import ufl

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 FlexiblyIndexed, Solve, Inverse, Variable, view)
from gem import indices as make_indices
from gem.node import Memoizer, post_traversal
from gem.node import pre_traversal as traverse_dags



from functools import singledispatch, update_wrapper
import firedrake.slate.slate as sl
import loopy as lp
from loopy.program import make_program
from loopy.transform.callable import inline_callable_kernel, register_callable_kernel
from loopy.kernel.creation import add_sequential_dependencies
from islpy import BasicSet

import numpy as np
import islpy as isl
import pymbolic.primitives as pym
from numbers import Integral


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


# singledispatch for class functions
def classsingledispatch(func):
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


class SlateTranslator():
    """Multifunction for translating Slate -> GEM.  """

    def __init__(self, builder):
        self.builder = builder
        self.decomp_dict = OrderedDict()
        self.var2terminal = OrderedDict()
        self.mapper = None

    def slate_to_gem_translate(self):  
        mapper, var2terminal = slate2gem(self.builder.expression)
        if not (type(mapper) == Indexed or type(mapper) == FlexiblyIndexed):
            mapper = Indexed(mapper, make_indices(len(self.builder.shape(mapper))))
        self.mapper = mapper
        self.var2terminal = var2terminal
        return mapper


def get_shape(tensor):
    """ A helper method to retrieve tensor shape information.
    In particular needed for the right shape of scalar tensors.
    """
    if tensor.shape == ():
        return (1, )  # scalar tensor
    else:
        return tensor.shape

@singledispatch
def _slate2gem(expr, self):
        raise AssertionError("Cannot handle terminal type: %s" % type(expr))

@_slate2gem.register(sl.Tensor)
@_slate2gem.register(sl.AssembledVector)
def _slate2gem_tensor(expr, self):
    shape = get_shape(expr)
    name = f"T{len(self.var2terminal)}"
    assert expr not in self.var2terminal.values()
    var = Variable(name, shape)
    self.var2terminal[var] = expr
    return var


@_slate2gem.register(sl.Block)
def _slate2gem_block(expr, self):
    raise NotImplementedError("blocks")
    # slices = []
    # child, = map(self, expr.operands)
    # indices = expr._indices
    # tmp = list(zip(list((indices,)), expr.operands[0].shapes.values()))
    # for idxs, extents in tmp:
    #     sidx = idxs[0]
    #     eidx = idxs[-1]
    #     slices.append(slice(sum(extents[:sidx]), sum(extents[:eidx+1])))
    # return view(child, *slices)


@_slate2gem.register(sl.Inverse)
def _slate2gem_inverse(expr, self):
    return Inverse(*map(self, expr.operands))


@_slate2gem.register(sl.Solve)
def _slate2gem_solve(expr, self):
    ops = (expr.operands[0].operands[0], expr.operands[1])
    return Solve(*map(self, ops))


@_slate2gem.register(sl.Transpose)
def _slate2gem_transpose(expr, self):
    child, = map(self, expr.operands)
    indices = tuple(make_indices(len(child.shape)))
    return ComponentTensor(Indexed(child, indices), tuple(indices[::-1]))


@_slate2gem.register(sl.Negative)
def _slate2gem_negative(expr, self):
    child, = map(self, expr.operands)
    indices = tuple(make_indices(len(child.shape)))
    return ComponentTensor(Product(Literal(-1),
                                           Indexed(child, indices)),
                               indices)


@_slate2gem.register(sl.Add)
def _slate2gem_add(expr, self):
    A, B = map(self, expr.operands)
    indices = tuple(make_indices(len(A.shape)))
    return ComponentTensor(Sum(Indexed(A, indices),
                                       Indexed(B, indices)),
                               indices)


@_slate2gem.register(sl.Mul)
def _slate2gem_mul(expr, self):
    A, B = map(self, expr.operands)
    *i, k = tuple(make_indices(len(A.shape)))
    _, *j = tuple(make_indices(len(B.shape)))
    ABikj = Product(Indexed(A, tuple(i + [k])),
                        Indexed(B, tuple([k] + j)))
    return ComponentTensor(IndexSum(ABikj, (k, )), tuple(i + j))

@_slate2gem.register(sl.Factorization)
def slate2gem_factorization(self, tensor):
    return []


def slate2gem(expressions):
    mapper = Memoizer(_slate2gem)
    mapper.var2terminal = OrderedDict()
    return mapper(expressions), mapper.var2terminal


def eigen_tensor(expr, temporary, index):
    """Returns an appropriate assignment statement for populating a particular
    `Eigen::MatrixBase` tensor. If the tensor is mixed, then access to the
    :meth:`block` of the eigen tensor is provided. Otherwise, no block
    information is needed and the tensor is returned as is.

    :arg expr: a `sl.Tensor` node.
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


#If you append global args later you need to specify dimtags
# Append global args and temporaries
def generate_kernel_arguments(builder, loopy_outer):
    args = []
    for loopy_inner in builder.templated_subkernels:
        for arg in loopy_inner.args[1:]:
                if arg.name == builder.coordinates_arg or\
                    arg.name == builder.cell_orientations_arg or\
                        arg.name == builder.cell_size_arg:
                    if arg not in args:
                        args.append(arg)
    
    for coeff in builder.coefficients.values():
        if isinstance(coeff[0], tuple):
            for c_, (name,extent) in coeff:
                arg = lp.GlobalArg(name, shape=extent, dtype="double", dim_tags=["N0"])
                args.append(arg)
        else:
            (name, extent) = coeff
            arg = lp.GlobalArg(name, shape=extent, dtype="double", dim_tags=["N0"])
            args.append(arg)

    if builder.needs_cell_facets:
        # Arg for is exterior (==0)/interior (==1) facet or not
        args.append(lp.GlobalArg(builder.cell_facets_arg,
                                    shape=(builder.num_facets, 2),
                                    dtype=np.int8,
                                    dim_tags=["N1","N0"]))
        
        loopy_outer.temporary_variables[builder.local_facet_array_arg] = lp.TemporaryVariable(builder.local_facet_array_arg,
                                        shape=(builder.num_facets,),
                                        dtype=np.uint32,
                                        address_space=lp.AddressSpace.LOCAL,
                                        read_only=True,
                                        initializer=np.arange(builder.num_facets, dtype=np.uint32))

    if builder.needs_mesh_layers:
        loopy_outer.temporary_variables["layer"] = lp.TemporaryVariable("layer", shape=(), dtype=np.int32, address_space=lp.AddressSpace.GLOBAL)

    for tensor_temp in builder.tensor2temp.values():
        loopy_outer.temporary_variables[tensor_temp.name] = tensor_temp

    return args
# Fix domains
# We have to add additional indices coming from
# A) subarrayreffing the subkernel args
# B) inames from initialisations
def create_domains(indices):
    domains = []
    for var, extent in indices.items():
        name = var.name
        if not isinstance(extent, Integral) and len(extent)==1:
            extent = extent[0]
        inames = isl.make_zero_and_vars([name], [])
        domains.append(BasicSet(str((inames[0].le_set(inames[name])) & (inames[name].lt_set(inames[0] + extent)))))
    return domains

def merge_loopy(loopy_outer, builder, translator):

    # Builder takes care of data management, i.e. loopifying arguments for tsfc call
    builder._setup(translator.var2terminal)
    
    # Munge instructions
    inits = builder.inits
    kitting_insn = []
    for integral_type in builder.assembly_calls:
        kitting_insn += builder.assembly_calls[integral_type]
    loopy_merged = loopy_outer.copy(instructions=inits+kitting_insn+loopy_outer.instructions)

    # Munge args, dependencies and domains
    args = generate_kernel_arguments(builder, loopy_outer)
    loopy_merged = loopy_merged.copy(args=loopy_merged.args+args)
    loopy_merged = add_sequential_dependencies(loopy_merged)
    # Remove priorities generated from tsfc compile call (TODO do we need this?)
    for insn in loopy_merged.instructions[-len(loopy_outer.instructions):]:
        loopy_merged = lp.set_instruction_priority(loopy_merged, "id:"+insn.id, None)
    domains = list(create_domains(builder.inames))
    loopy_merged = loopy_merged.copy(domains=domains+loopy_merged.domains)

    # Generate program from kernel, register inner kernel and inline inner kernel
    prg = make_program(loopy_merged)
    for loopy_inner in builder.templated_subkernels:
        prg = register_callable_kernel(prg, loopy_inner)
        prg = inline_callable_kernel(prg, loopy_inner.name)
    return prg