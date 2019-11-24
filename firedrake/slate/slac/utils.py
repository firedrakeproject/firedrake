from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Zero, Identity, Sum, Product, Division,
                 Power, MathFunction, MinValue, MaxValue, Comparison,
                 LogicalNot, LogicalAnd, LogicalOr, Conditional,
                 Index, Indexed, ComponentTensor, IndexSum,
                 ListTensor,Variable)#,Inverse,Solve,)


from functools import singledispatch,update_wrapper
import firedrake
import firedrake.slate.slate as sl
import loopy as lp
from loopy.transform.callable import _inline_call_instruction as inline
from loopy.kernel.instruction import CallInstruction

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

    def __init__(self, tensor_to_variable):
        # Need context during translation!
        self.tensor_to_variable=tensor_to_variable

    def slate_to_gem_translate(self, slate_expression_dag):
        gem_expression_dag=[]
        for tensor in slate_expression_dag:#tensor hier is actually TensorBase
            
            # Terminal tensors/Assembled Vectors are already defined
            #just redirect to allocated memory, how??
            if isinstance(tensor, sl.Tensor):
                gem_expression_dag.append(self.tensor_to_variable[tensor])
                print(self.tensor_to_variable[tensor])

            elif isinstance(tensor, sl.AssembledVector):
                gem_expression_dag.append(self.tensor_to_variable(tensor))

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

    @slate_to_gem.register(firedrake.slate.slate.Mul)
    def slate_to_gem_mul(self,tensor):
        A, B = tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        B_indices=tuple(Index(extent=B.shape[i]) for i in range(len(B.shape)))
        

        #@TODO: No need for all this, because extents have to be set

        if not tensor.shape:
            pass
            #@TODO: DO WE NEVER HAVE THIS CASE?
            #try:
            #    #1x3 dot 3x1
            #    ret=IndexSum(Product(Indexed(self.tensor_to_variable[A],A_indices),Indexed(self.tensor_to_variable[B],B_indices)),(A_indices[1],))
            #except:
            #    #1x1 dot 1x1
            #     Product(self.tensor_to_variable[A],self.tensor_to_variable[B])
        elif tensor.shape==(A.shape[0],):
            try:
                #3x3 dot 3x1
                prod=IndexSum(Product(Indexed(self.tensor_to_variable[A],A_indices),Indexed(self.tensor_to_variable[B],B_indices)),(A_indices[1],))
                ret=ComponentTensor(prod,(A_indices[0],))
            except:
                #3x1 dot 1x1
                prod=Product(Indexed(self.tensor_to_variable[A],A_indices),self.tensor_to_variable[B])
                ret=ComponentTensor(prod,(A_indices[0],))
        #@TODO: DO WE NEVER HAVE THIS CASE?
        #    elif tensor.shape==tuple(,B.shape[1]):
        #        try:
        #            #1x3 dot 3x3
        #            prod=IndexSum(Product(Indexed(self.tensor_to_variable[A],A_indices),Indexed(self.tensor_to_variable[B],B_indices)),A_indices[1])
        #            ret=ComponentTensor(prod,(A_indices[0],))
        #        except:
        #            #1x1 dot 1x3
        #            prod=Product(self.tensor_to_variable[A],Indexed(self.tensor_to_variable[B],B_indices))
        #            ret=ComponentTensor(prod,(,B_indices[1]))
        #3x3 dot 3x3
        elif tensor.shape==(A.shape[0],B.shape[1]):
            prod=IndexSum(Product(Indexed(self.tensor_to_variable[A],A_indices),Indexed(self.tensor_to_variable[B],B_indices)),(A_indices[1],))
            ret=ComponentTensor(prod,(A_indices[0],B_indices[1]))
        #this all cases?
        else:
            raise Exception("Not valid matrix multiplication")


        print("A indices: ",A_indices)
        print("B indices: ",B_indices)
        print("A shape: ",A.shape)
        print("B shape: ",B.shape)
        print("Tensor shape: ",tensor.shape)
        print("IndexSum multiiindex: ",prod.multiindex)
        print("IndexSum freeindex: ",prod.free_indices)
        print("ret multiiindex: ",ret.multiindex)
        print("ret freeindex: ",ret.free_indices)
        print("ret: ",ret)
        return ret

    @slate_to_gem.register(firedrake.slate.slate.Add)
    def slate_to_gem_add(self,tensor):
        A, B = tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        B_indices=tuple(Index(extent=A.shape[i]) for i in range(len(B.shape)))
        print(tuple(A_indices[i].extent for i in range(len(B.shapes))))
        print(tuple(B_indices[i].extent for i in range(len(B.shapes))))
        print("aaa",A_indices)
        _A=Indexed(self.tensor_to_variable[A],A_indices)
        _B=Indexed(self.tensor_to_variable[B],A_indices)
        print("ttt:",self.tensor_to_variable[A].shape)

        ret=ComponentTensor(Sum(_A,_B),A_indices)
        print("A multiiindex: ",_A.multiindex)
        print("ret freeindex: ",ret.free_indices)
        print("ret: ",ret)
        return ret

    @slate_to_gem.register(firedrake.slate.slate.Negative)
    def slate_to_gem_negative(tensor,self):
        A,=tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        ret=ComponentTensor(Product(Literal(-1), Indexed(self.tensor_to_variable[A],A_indices)),A_indices)

        print("ret multiiindex: ",ret.multiindex)
        print("ret freeindex: ",ret.free_indices)
        print("ret children: ",ret.children)
        print("ret: ",ret)
        return ret

    @slate_to_gem.register(firedrake.slate.slate.Transpose)
    def slate_to_gem_transpose(tensor,self):
        A, = tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        ret=ComponentTensor(Indexed(self.tensor_to_variable[A],A_indices),tuple(reversed(A_indices)))
        print("ret multiiindex: ",ret.multiindex)
        print("ret freeindex: ",ret.free_indices)
        print("ret children: ",ret.children)
        print("ret: ",ret)
        return ret

    #@TODO: actually more complicated because used for mixed tensors?
    @slate_to_gem.register(firedrake.slate.slate.Block)
    def slate_to_gem_block(tensor,self,slice_indices):
        A,=tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        ret=ComponentTensor(Indexed(self.tensor_to_variable[A],A_indices),slice_indices)

        print("ret multiiindex: ",ret.multiindex)
        print("ret freeindex: ",ret.free_indices)
        print("ret children: ",ret.children)
        print("ret: ",ret)
        return ret

    #call gem nodes for inverse and solve
    #@TODO: see questions on that in gem
    @slate_to_gem.register(firedrake.slate.slate.Inverse)
    def slate_to_gem_inverse(tensor,self,context):
        return Inverse(self.tensor_to_variable[A])

    @slate_to_gem.register(firedrake.slate.slate.Solve)
    def slate_to_gem_solve(tensor,self,context):
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

def my_merge_loopy(loopy_outer,loopy_inner):

    print("MERGE ROUTINE:")
    #create kitting code
    kitting_code=lp.make_kernel("{ [v,w]: 0<=v<3 and 0<=w<3}","T0[v,w] =A[v,w] {id=insn}")
    kitting_code_knl=kitting_code.callables_table.resolved_functions["loopy_kernel"].subkernel

    #kitting arguments
    data = list(loopy_outer.args)#outer can only have args
    #@TODO: I think you need to convert args into temporary variables
    data.extend(loopy_inner.args)#inner can have args or temps
    data.extend(list(loopy_inner.temporary_variables.values()))    
    data.extend(list(loopy_outer.temporary_variables.values()))    
    print("DATA:",data)

    #kitting domains
    domains_inner=loopy_inner.domains
    domains_outer= loopy_outer.domains
    #domains=loopy_inner.domains+loopy_outer.domains
    domains=loopy_inner.domains+kitting_code_knl.domains+loopy_outer.domains
    print("DOMAIN:",domains)

    #kitting the instructions
    instructions=[]
    instructions.extend(loopy_inner.instructions)
    A=loopy_inner.args[0]
    T=loopy_outer.temporary_variables["T0"]
    #kitting_code_instr = lp.fix_parameters(kittigng_code_instr ,n=loopy_inner.args[0].shape[0])
    #lp.Assignment(p.Subscript(p.Variable(loopy_outer.temporary_variables["T0"].name), p.Variable(tt)),p.Subscript(p.Variable(loopy_inner.args[0].name), p.Variable(ss)))    
    instructions.extend(kitting_code_knl.instructions)   
    instructions.append(loopy_outer.instructions[0])
    print(instructions)

    #fix the ids/prios of instrutions for right scheduling
    #@TODO: am I messing up my dependencies with this?
    #@TODO: maybe sth like a substitution would be better?
    insn_new = []
    for i, insn in enumerate(instructions):
        if i==0:
            insn_new.append(insn.copy(id="insn",priority=len(instructions) - i))
        else:
            insn_new.append(insn.copy(id="insn_"+str(i-1),priority=len(instructions) - i))
    print(insn_new)
    
    print("\n data\n:",data)

    # Create loopy kernel
    knl = lp.make_function(domains,insn_new, data, name="glueloopy", target=lp.CTarget(),
                           seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])

    # Prevent loopy interchange by loopy
    #knl = lp.prioritize_loops(knl, ",".join(ctx.index_extent.keys()))

    return knl


def merge_loopy(loopy_outer,loopy_inner):
    #@TODO: we need an instruction in outer kernel to call inner kernel
    repl=loopy_outer.instructions[0]
    print(repl)
    assert isinstance(repl, CallInstruction)

    from loopy.program import make_program
    prg=make_program(loopy_outer)
    from loopy.transform.callable import inline_callable_kernel, register_callable_kernel


    wrapper = register_callable_kernel(prg, loopy_inner)
    inlined_prg=inline_callable_kernel(prg,"subkernel0_cell_to_00_cell_integral_otherwise")


    print("inlined prg: ",inlined_prg)

    return inlined_prg
