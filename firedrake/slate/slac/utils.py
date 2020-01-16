from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Zero, Identity, Sum, Product, Division,
                 Power, MathFunction, MinValue, MaxValue, Comparison,
                 LogicalNot, LogicalAnd, LogicalOr, Conditional,
                 Index, Indexed, ComponentTensor, IndexSum,
                 ListTensor,Variable,index_sum)#,Inverse,Solve,)


from functools import singledispatch,update_wrapper
import firedrake
import firedrake.slate.slate as sl
import loopy as lp
from loopy.kernel.instruction import CallInstruction
from loopy.program import make_program
from loopy.transform.callable import inline_callable_kernel, register_callable_kernel
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

#singledispatch for second argument
def classsingledispatch(func):
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

class SlateTranslator():
    """Multifunction for translating UFL -> GEM.  """

    def __init__(self,builder):
        # Need context during translation!
        self.tensor_to_variable=builder.temps
        self.coeff_vecs=builder.coefficient_vecs
        self.traversed_slate_expr_dag=builder.expression_dag
        self.builder=builder

    def slate_to_gem_translate(self):
        gem_expression_dag=[]
        for tensor in self.traversed_slate_expr_dag:#tensor hier is actually TensorBase
            # Terminal tensors/Assembled Vectors are already defined
            #just redirect to allocated memory, how??
            if isinstance(tensor, sl.Tensor):
                gem_expression_dag.append(self.tensor_to_variable[tensor])

            elif isinstance(tensor, sl.AssembledVector):
                gem_expression_dag.append(self.coeff_vecs[3][0].local_temp)

            #other tensor types are translated into gem nodes
            else:
                gem_expression_dag.append(self.slate_to_gem(tensor))
        return list(gem_expression_dag)

    
    @classsingledispatch
    def slate_to_gem(self,tensor):
        """Translates slate tensors into GEM.
        :returns: GEM translation of the modified terminal
        """
        raise AssertionError("Cannot handle terminal type: %s" % type(tensor))

    @slate_to_gem.register(firedrake.slate.slate.Add)
    def slate_to_gem_add(self,tensor):
        A, B = tensor.operands #slate tensors
        _A,_B=self.tensor_to_variable[A],self.tensor_to_variable[B]#gem representations
        return Sum(_A,_B)

    @slate_to_gem.register(firedrake.slate.slate.Negative)
    def slate_to_gem_negative(self,tensor):
        A,=tensor.operands
        return Product(Literal(-1), self.tensor_to_variable[A])

    @slate_to_gem.register(firedrake.slate.slate.Transpose)
    def slate_to_gem_transpose(self,tensor):
        A, = tensor.operands
        indices =self.builder.create_index(A.shape,tensor)
        A_indices=self.builder.gem_indices[tensor]
        ret=Indexed(ComponentTensor(Indexed(self.tensor_to_variable[A].children[0],A_indices),tuple(reversed(A_indices))),A_indices)
        return ret

    #@TODO: actually more complicated because used for mixed tensors?
    @slate_to_gem.register(firedrake.slate.slate.Block)
    def slate_to_gem_block(self,tensor,slice_indices):
        A,=tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        ret=ComponentTensor(Indexed(self.tensor_to_variable[A],A_indices),slice_indices)

        print("ret multiiindex: ",ret.multiindex)
        print("ret freeindex: ",ret.free_indices)
        print("ret children: ",ret.children)
        print("ret: ",ret)
        return ret
    
    @slate_to_gem.register(firedrake.slate.slate.Mul)
    def slate_to_gem_mul(self,tensor):
        A, B = tensor.operands

        #TODO:We have a problem with multiplying the same operand!"

        indices =self.builder.create_index(A.shape,A)
        A_indices=self.builder.gem_indices[A]

        indices =self.builder.create_index(B.shape,B)
        B_indices=self.builder.gem_indices[B]

        if len(A.shape)==len(B.shape) and A.shape[1]==B.shape[0]:
            prod=Product(Indexed(self.tensor_to_variable[A].children[0],A_indices),Indexed(self.tensor_to_variable[B].children[0],(A_indices[1],B_indices[1])))
            
            sum_indices=(A_indices[0],B_indices[1])
            sum=IndexSum(prod,(A_indices[1],))

            new_indices =self.builder.create_index((A.shape[0],B.shape[1]),tensor)
            new_indices=self.builder.gem_indices[tensor]
            return Indexed(ComponentTensor(sum,sum_indices),new_indices)

        elif len(A.shape)>len(B.shape) and A.shape[1]==B.shape[0]:
            vec_gem=self.coeff_vecs[3][0].local_temp

            prod=Product(Indexed(self.tensor_to_variable[A].children[0],A_indices),Indexed(vec_gem.children[0],(A_indices[1],)))

            sum_indices=(A_indices[0],)
            sum=IndexSum(prod,(A_indices[1],))

            new_indices =self.builder.create_index((A.shape[0],),tensor)
            new_indices=self.builder.gem_indices[tensor]
            return Indexed(ComponentTensor(sum,sum_indices),new_indices)

        return ret

    #call gem nodes for inverse and solve
    #@TODO: see questions on that in gem
    @slate_to_gem.register(firedrake.slate.slate.Inverse)
    def slate_to_gem_inverse(self,tensor,context):
        return Inverse(self.tensor_to_variable[A])

    @slate_to_gem.register(firedrake.slate.slate.Solve)
    def slate_to_gem_solve(self,tensor,context):
        raise Solve(self.tensor_to_variable[A])         




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


def merge_loopy(loopy_outer,loopy_inner_list,builder):
    #generate initilisation instructions for all tensor temporaries
    inits=[]
    c=0
    for slate_tensor,gem_indexed in builder.temps.items():
        #create new indices for inits and save with indexed (gem) key instead of slate tensor
        indices =builder.create_index(slate_tensor.shape,gem_indexed)
        loopy_tensor=builder.gem_loopy_dict[gem_indexed]
        indices=builder.loopy_indices[gem_indexed]
        inames={var.name for var in indices}
        inits.append(lp.Assignment(pym.Subscript(pym.Variable(loopy_tensor.name),indices), 0.0,id="init%d"%c, within_inames=frozenset(inames)))
        c+=1
    
    #generate initilisation instructions for all coefficent temporaries
    for k,v in builder.coefficient_vecs.items():
        loopy_tensor=builder.gem_loopy_dict[v[0].local_temp]
        loopy_outer.temporary_variables[loopy_tensor.name]=loopy_tensor
        indices=builder.loopy_indices[v[0].vector]
        inames={var.name for var in indices}
        inits.append(lp.Assignment(pym.Subscript(pym.Variable(loopy_tensor.name),indices), pym.Subscript(pym.Variable("coeff"),indices),id="init%d"%c, within_inames=frozenset(inames)))
        c+=1

    loopy_outer.temporary_variables["facet_array"] = lp.TemporaryVariable(builder.local_facet_array_arg,
                                                                     shape=(builder.num_facets,),
                                                                     dtype=np.uint32,
                                                                     address_space=lp.AddressSpace.GLOBAL,
                                                                     read_only=True,
                                                                    initializer=np.arange(builder.num_facets, dtype=np.uint32))


    # indices =builder.create_index((builder.num_facets,),builder.local_facet_array_arg+"_init")
    # inames={var.name for var in indices}
    # inits.append(lp.Assignment(pym.Subscript(pym.Variable(builder.local_facet_array_arg),indices), 0.0,id="init%d"%c, within_inames=frozenset(inames)))
        
        
    # loopy_outer.temporary_variables["cell_facets"] = lp.TemporaryVariable(builder.cell_facets_arg,
    #                                                                 shape=(builder.num_facets,2),
    #                                                                 dtype=np.uint8,
    #                                                                 address_space=lp.AddressSpace.LOCAL,
    #                                                                 read_only=True)

    #get the CallInstruction from builder and include at beginning of outer kernel
    kitting_insn=builder.assembly_calls["exterior_facet"]
    loopy_merged = loopy_outer.copy(instructions=inits+kitting_insn+loopy_outer.instructions)
    
    noi_outer=len(loopy_outer.instructions)
    noi_inits=len(inits)
    #add dependencies dynamically
    if len(loopy_outer.instructions)>1:
        for i in range(len(kitting_insn)):
            #add dep from second insn of outer kernel to all subkernels
            loopy_merged= lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer+1].id,  "id:"+loopy_merged.instructions[-noi_outer-i-1].id)
            #dep from subkernel to the according init       
            loopy_merged= lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[noi_inits+i].id,  "id:"+loopy_merged.instructions[i].id)
    
    elif not len(kitting_insn)==0:
        #add dep from first insn of outer kernel to the subkernel
        loopy_merged= lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer].id,  "id:"+loopy_merged.instructions[-noi_outer-1].id)
        
        #dep from subkernel to the according init       
        loopy_merged= lp.add_dependency(loopy_merged, "id:"+loopy_merged.instructions[-noi_outer-1].id,  "id:"+loopy_merged.instructions[0].id)

    #remove priority generate from tsfc compile call
    for insn in loopy_merged.instructions[-noi_outer:]:
        loopy_merged=lp.set_instruction_priority(loopy_merged,"id:"+insn.id, None)

  
    #add arguments of the subkernel
    #TODO check if is the dimtag right
    #TODO maybe we need to run over subkernels
    #    inner_args=builder.templated_subkernels[0].args
    #if len(builder.templated_subkernels)>0:
    #    loopy_merged = loopy_merged.copy(args=list(loopy_merged.args)+inner_args[1:]) #first variable is A which gets replaced by slate name of tensor anyways

    #fix domains (add additional indices coming from calling the subkernel)
    def create_domains(gem_indices):

        for tuple_index in gem_indices:
            for i in tuple_index:
                name=i.name
                extent=i.extent
                vars = isl.make_zero_and_vars([name], [])
                yield BasicSet("{ ["+name+"]: 0<="+name+"<"+str(extent)+"}")
    domains = list(create_domains(builder.gem_indices.values()))
    loopy_merged= loopy_merged.copy(domains=domains+loopy_merged.domains)


    print(loopy_merged)
    #generate program from kernel, register inner kernel and inline inner kernel
    prg=make_program(loopy_merged)
    for loopy_inner in loopy_inner_list:
        prg = register_callable_kernel(prg, loopy_inner)
        prg=inline_callable_kernel(prg,loopy_inner.name)

    return prg