from coffee import base as ast
from coffee.visitor import Visitor
from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Sum, Product, Indexed, ComponentTensor, IndexSum,
                 FlexiblyIndexed, Solve, Inverse, Variable)


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

    def slate_to_gem_translate(self):
        translated_nodes = OrderedDict()
        traversed_dag = list(post_traversal([self.builder.expression]))

        # Tree traversal
        for tensor in traversed_dag[:len(traversed_dag)-1]:
            translated_nodes.setdefault(tensor, self.slate_to_gem(tensor, translated_nodes))

        # Last root contains the whole tree
        last_tensor = traversed_dag[len(traversed_dag)-1]
        last_loopy_tensor = self.slate_to_gem(last_tensor, translated_nodes)
        out_indices = self.builder.create_index(last_loopy_tensor.shape, "gem")
        if not (type(last_loopy_tensor) == Indexed or type(last_loopy_tensor) == FlexiblyIndexed):
            last_loopy_tensor = Indexed(last_loopy_tensor, out_indices)
        return last_loopy_tensor

    @classsingledispatch
    def slate_to_gem(self, tensor, node_dict):
        """Translates slate tensors into GEM.
        :returns: GEM translation of the modified terminal
        """
        raise AssertionError("Cannot handle terminal type: %s" % type(tensor))

    @slate_to_gem.register(sl.Tensor)
    def slate_to_gem_tensor(self, tensor, node_dict):
        idx = self.builder.create_index(self.builder.shape(tensor), "gem")
        return ComponentTensor(self.index_tensor(self.builder.temps[tensor], idx), idx)

    @slate_to_gem.register(sl.AssembledVector)
    def slate_to_gem_vector(self, tensor, node_dict):
        ret = None
        idx = self.builder.create_index(tensor.shape, "gem")
        # Not mixed tensor can just be translated into the right gem Node saved in builder
        if len(tensor.shapes) == 1 and not tensor.is_mixed:
            coeffs = self.builder.coefficient_vecs[((tensor.shapes[0][0], False), 0)]
            for coeff in coeffs:
                if coeff.vector == tensor:
                    assert ret is None, "This vector as already been assembled."
                    ret = ComponentTensor(self.index_tensor(coeff.local_temp, idx), idx)
        else:
            tensor_found = False
            for cinfo_list in self.builder.coefficient_vecs.values():
                for cinfo in cinfo_list:
                    if cinfo.vector == tensor and not tensor_found:
                        var = cinfo.local_temp
                        tensor_found = True
            ret = Indexed(var, idx)
            ret = ComponentTensor(ret, idx)
        return ret

    @slate_to_gem.register(sl.Add)
    def slate_to_gem_add(self, tensor, node_dict):
        A, B = tensor.operands  # slate tensors
        _A, _B = node_dict[A], node_dict[B]  # gem representations
        new_indices = self.builder.create_index(A.shape, "gem")
        _A = self.index_tensor(_A, new_indices)
        _B = self.index_tensor(_B, new_indices)
        return ComponentTensor(Sum(_A, _B), new_indices)

    @slate_to_gem.register(sl.Negative)
    def slate_to_gem_negative(self, tensor, node_dict):
        A, = tensor.operands
        new_indices = self.builder.create_index(A.shape, "gem")
        var_A = self.index_tensor(node_dict[A], new_indices)
        return ComponentTensor(Product(Literal(-1), var_A), new_indices)

    @slate_to_gem.register(sl.Transpose)
    def slate_to_gem_transpose(self, tensor, node_dict):
        A, = tensor.operands
        _A = node_dict[A]
        new_indices = self.builder.create_index(A.shape, "gem")
        var_A = self.index_tensor(_A, new_indices)
        ret = ComponentTensor(var_A, new_indices[::-1])
        return ret

    @slate_to_gem.register(sl.Mul)
    def slate_to_gem_mul(self, tensor, node_dict):
        A, B = tensor.operands
        var_A, var_B = node_dict[A], node_dict[B]  # gem representations

        # New indices are necessary in case as Tensor gets multiplied with itself.
        new_indices_A = self.builder.create_index(A.shape, "gem")
        new_indices_B = self.builder.create_index(B.shape, "gem")

        if len(A.shape) == len(B.shape) and A.shape[1] == B.shape[0]:
            var_A = self.index_tensor(var_A, new_indices_A)
            var_B = self.index_tensor(var_B, (new_indices_A[1], new_indices_B[1]))

            prod = Product(var_A, var_B)
            sum = IndexSum(prod, (new_indices_A[1],))
            return ComponentTensor(sum, (new_indices_A[0], new_indices_B[1]))

        elif len(A.shape) > len(B.shape) and A.shape[1] == B.shape[0]:
            var_A = self.index_tensor(var_A, new_indices_A)
            var_B = self.index_tensor(var_B, (new_indices_A[1],))

            prod = Product(var_A, var_B)
            sum = IndexSum(prod, (new_indices_A[1],))
            return ComponentTensor(sum, (new_indices_A[0],))
        else:
            return[]

    @slate_to_gem.register(sl.Block)
    def slate_to_gem_blocks(self, tensor, node_dict):

        A, = tensor.operands
        node = node_dict[A]

        # get first block while handling ranges handle ranges
        first_ind = ()
        for index in tensor._indices:
            if type(index) != range:
                first_ind += (index,)
            else:
                first_ind += (index.start,)
        first_block = tuple(range(ten, ten+1) for ten in first_ind)
        index_offset = ()

        # i points to the dim
        # idx points to the shape of that block matrix in that dim
        for i, idx in enumerate(first_block):
            if idx.start == 0:
                index_offset += (0, )
            else:
                index_offset += ((sum(A.shapes[i][:idx.start])), )

        index_extent = tuple(sum(shape) for shape in tensor.shapes.values())
        gem_index = self.builder.create_index(index_extent, "gem")

        dim2idxs = []
        for i, dim in enumerate(tensor.shapes.keys()):
            dim2idxs += ((index_offset[i], ((gem_index[i], 1), )),)

        return ComponentTensor(FlexiblyIndexed(node, tuple(dim2idxs)), gem_index)

    def index_tensor(self, var, idx):
        """ This method is a helper to:
        - pick up free indices from a variable in the right
        order
        - if needed (it is for IndexSum and Indexed),
        pull up a scalar variable to a tensor thing with this indices
        - index it with new indices :arg idx.
        """
        if type(var) == Variable or type(var) == ComponentTensor or type(var) == Inverse or type(var) == Solve:
            var = Indexed(var, idx)
        else:
            assert True, "Variable type is "+str(type(var))+". Must be a type that has shape."
        return var

    @slate_to_gem.register(sl.Solve)
    def slate_to_gem_solve(self, tensor, node_dict):
        fac, B = tensor.operands  # TODO is first operand always factorization?
        A, = fac.operands
        A_indices = self.builder.create_index(A.shape, "gem")
        B_indices = self.builder.create_index(B.shape, "gem")
        ret_A = ComponentTensor(self.index_tensor(node_dict[A], A_indices), A_indices)
        ret_B = ComponentTensor(self.index_tensor(node_dict[B], B_indices), B_indices)
        ret = Solve(ret_A, ret_B)
        return ret

    @slate_to_gem.register(sl.Inverse)
    def slate_to_gem_inverse(self, tensor, node_dict):
        A, = tensor.operands
        A_indices = self.builder.create_index(A.shape, "gem")
        ret = ComponentTensor(self.index_tensor(node_dict[A], A_indices), A_indices)
        ret = Inverse(ret)
        return ret

    # TODO how do we deal with surpressed factorization nodes,
    # maybe populate decompdict and pass through to loopy later?
    @slate_to_gem.register(sl.Factorization)
    def slate_to_gem_factorization(self, tensor, node_dict):
        self.decomp_dict.setdefault(tensor, tensor.decomposition)
        return []


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


# Adapted from tsfc.gem.node.py
def post_traversal(expression_dags):
    """Post-order traversal of the nodes of expression DAGs."""
    seen = set()
    lifo = []
    # Some roots might be same, but they must be visited only once.
    # Keep the original ordering of roots, for deterministic code
    # generation.
    for root in expression_dags:
        if root not in seen:
            seen.add(root)
            lifo.append((root, list(root.operands)))

    while lifo:
        node, deps = lifo[-1]
        for i, dep in enumerate(deps):
            if dep is not None and dep not in seen:
                lifo.append((dep, list(dep.operands)))
                deps[i] = None
                break
        else:
            yield node
            seen.add(node)
            lifo.pop()


def traverse_dags(exprs):
    """Traverses a set of DAGs and returns each node.

    :arg exprs: An iterable of Slate expressions.
    """
    seen = set()
    container = []
    for tensor in exprs:
        if tensor not in seen:
            seen.add(tensor)
            container.append(tensor)
    while container:
        tensor = container.pop()
        yield tensor

        for operand in tensor.operands:
            if operand not in seen:
                seen.add(operand)
                container.append(operand)


def merge_loopy(loopy_outer, loopy_inner_list, builder):
    # Generate initilisation instructions for all tensor temporaries
    inits = []
    c = 0
    for slate_tensor, gem_indexed in builder.temps.items():
        # Create new indices for inits and save with indexed (gem) key instead of slate tensor
        extent = builder.shape(slate_tensor)
        indices = builder.create_index(extent, "loopy")
        builder.save_index(indices, extent)
        loopy_tensor = builder.gem_loopy_dict[gem_indexed]
        inames = {var.name for var in indices}
        inits.append(lp.Assignment(pym.Subscript(pym.Variable(loopy_tensor.name), indices), 0.0, id="init%d" % c, within_inames=frozenset(inames)))
        c += 1

    # Generate initilisation instructions for all coefficent temporaries,
    # which are in an AssembledVector
    # Same difficulty as adding those as global args (see comment in compiler)
    coeff_shape_list = []
    coeff_function_list = []
    coeff_tensor_list = []
    coeff_offset_list = []
    for v in builder.coefficient_vecs.values():
        for coeff_info in v:
            coeff_shape_list.append(coeff_info.shape)
            coeff_function_list.append(coeff_info.vector._function)
            coeff_tensor_list.append(coeff_info.local_temp)
            coeff_offset_list.append(coeff_info.offset_index)

    coeff_no = 0
    for ordered_coeff in builder.expression.coefficients():
        try:
            indices = [i for i, x in enumerate(coeff_function_list) if x == ordered_coeff]
            for func_index in indices:
                loopy_tensor = builder.gem_loopy_dict[coeff_tensor_list[func_index]]
                loopy_outer.temporary_variables[loopy_tensor.name] = loopy_tensor
                extent = coeff_shape_list[func_index]
                indices = builder.create_index(extent, "loopy")
                builder.save_index(indices, (extent,))
                inames = {var.name for var in indices}
                inits.append(lp.Assignment(pym.Subscript(pym.Variable(loopy_tensor.name), (pym.Sum((coeff_offset_list[func_index], indices[0])),)), pym.Subscript(pym.Variable("coeff%d" % coeff_no), indices), id="init%d" % c, within_inames=frozenset(inames)))
                c += 1
                coeff_no += 1
        except ValueError:
            pass

    # Generate temp e.g. for plexmesh_exterior_local_facet_number (maps from global to local facets)
    if builder.needs_cell_facets:
        loopy_outer.temporary_variables["facet_array"] = lp.TemporaryVariable(builder.local_facet_array_arg,
                                                                              shape=(builder.num_facets, 2),
                                                                              dtype=np.uint32,
                                                                              address_space=lp.AddressSpace.LOCAL,
                                                                              read_only=True,
                                                                              initializer=np.arange(builder.num_facets, dtype=np.uint32))

    # Get the CallInstruction for each kernel from builder
    kitting_insn = []
    for integral_type in builder.assembly_calls:
        kitting_insn += builder.assembly_calls[integral_type]

    loopy_merged = loopy_outer.copy(instructions=inits+kitting_insn+loopy_outer.instructions)

    # Generate dependencies
    loopy_merged = add_sequential_dependencies(loopy_merged)

    # Remove priority generated from tsfc compile call
    for insn in loopy_merged.instructions[-len(loopy_outer.instructions):]:
        loopy_merged = lp.set_instruction_priority(loopy_merged, "id:"+insn.id, None)

    # Fix domains
    # We have to add additional indices coming from
    # A) subarrayreffing the subkernel args
    # B) initialisation from above
    def create_domains(indices):
        for var, extent in indices.items():
            name = var.name
            extent = extent
            isl.make_zero_and_vars([name], [])
            yield BasicSet("{ ["+name+"]: 0<="+name+"<"+str(extent)+"}")

    domains = list(create_domains(builder.inames))
    loopy_merged = loopy_merged.copy(domains=domains+loopy_merged.domains)

    # Generate program from kernel, register inner kernel and inline inner kernel
    prg = make_program(loopy_merged)
    for loopy_inner in loopy_inner_list:
        prg = register_callable_kernel(prg, loopy_inner)
        prg = inline_callable_kernel(prg, loopy_inner.name)
    return prg


# My own dependecy generation
# Out of date, I use TSFCs dependency generation now
def my_dependency_generation(loopy_outer, loopy_merged, kitting_insn, builder, inits):
    noi_outer = len(loopy_outer.instructions)
    noi_inits = len(inits)
    # # add dependencies dynamically
    if len(loopy_outer.instructions) > 1:
        for i in range(len(kitting_insn)):
            # add dep from first insn of outer kernel to all subkernels
            loopy_merged = lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer].id, "id:"+loopy_merged.instructions[-noi_outer-i-1].id)

            # dep from subkernel to the according init
            # loopy_merged= lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[noi_inits+i].id,  "id:"+loopy_merged.instructions[noi_inits-i-1].id)
            loopy_merged = lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[noi_inits+i].id, "id:"+loopy_merged.instructions[i].id)

    elif not len(kitting_insn) == 0:
        for i, assembly_call in enumerate(kitting_insn):
            # add dep from first insn of outer kernel to the subkernel in first loop
            # then from subkernel to subkernel
            loopy_merged = lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer-i].id, "id:"+loopy_merged.instructions[-noi_outer-i-1].id)

    # # dep from first subkernel to the according init# TODO do we need this?
    loopy_merged = lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer-len(kitting_insn)].id, "id:"+loopy_merged.instructions[0].id)

    # # link to initilisaton of vectemps, TODO: this is not robust
    for k, v in builder.coefficient_vecs.items():
        loopy_merged = lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer+len(builder.temps)].id, "id:"+loopy_merged.instructions[len(builder.temps)].id)
