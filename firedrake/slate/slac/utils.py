from coffee import base as ast
from coffee.visitor import Visitor

from collections import OrderedDict

from ufl.algorithms.multifunction import MultiFunction

from gem import (Literal, Zero, Identity, Sum, Product, Division,
                 Power, MathFunction, MinValue, MaxValue, Comparison,
                 LogicalNot, LogicalAnd, LogicalOr, Conditional,
                 Index, Indexed, ComponentTensor, IndexSum,
                 ListTensor,Variable)#,Inverse,Solve,)


from functools import singledispatch
import firedrake
import firedrake.slate.slate as sl

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
                gem_expression_dag.append(tensor_to_coefficient(tensor))

            #other tensor types are translated into gem nodes
            else:
                gem_expression_dag.append(self.slate_to_gem_add(tensor))

        return gem_expression_dag            

    @singledispatch
    def slate_to_gem(self,tensor):
        """Translates slate tensors into GEM.
        :returns: GEM translation of the modified terminal
        """
        raise AssertionError("Cannot handle terminal type: %s" % type(tensor))

    @slate_to_gem.register(firedrake.slate.slate.Mul)
    def slate_to_gem_mul(self,tensor):
        A, B = tensor.operands
        #@TODO: loop over shape rather than implementing hard her
        #also in all the following statements
        i,j,k=Index(extent=A.shape[0]),Index(extent=A.shape[1]),Index(extent=B.shape[1])
        return ComponentTensor(IndexSum(Indexed(self.tensor_to_variable(A),(i,j)),Indexed(self.tensor_to_variable(B),(j,k)),j),(i,k))

    @slate_to_gem.register(firedrake.slate.slate.Add)
    def slate_to_gem_add(self,tensor):
        A, B = tensor.operands
        A_indices=tuple(Index(extent=A.shape[i]) for i in range(len(A.shape)))
        B_indices=tuple(Index(extent=A.shape[i]) for i in range(len(B.shape)))
        print(tuple(A_indices[i].extent for i in range(len(B.shapes))))
        print(tuple(B_indices[i].extent for i in range(len(B.shapes))))
        _A=Indexed(self.tensor_to_variable[A],A_indices)
        _B=Indexed(self.tensor_to_variable[B],A_indices)
        print(self.tensor_to_variable[A].shape)

        ret=ComponentTensor(Sum(_A,_B),A_indices)
        print(ret.multiindex)
        print(ret.free_indices)
        print(ret.children)
        return ret
    @slate_to_gem.register(firedrake.slate.slate.Negative)
    def slate_to_gem_negative(self,tensor):
        A,=tensor.operands
        i,j=Index(extent=A.shape[0]),Index(extent=A.shape[1])
        return ComponentTensor(Product(Literal(-1), Indexed(self.tensor_to_variable(A),(i,j))),(i,j))
        
    @slate_to_gem.register(firedrake.slate.slate.Transpose)
    def slate_to_gem_transpose(self,tensor):
        A, = tensor.operands
        i,j=Index(extent=A.shape[0]),Index(extent=A.shape[1])
        return ComponentTensor(Indexed(self.tensor_to_variable(A), (i,j)),(j,i))

    #@TODO: actually more complicated because used for mixed tensors?
    @slate_to_gem.register(firedrake.slate.slate.Block)
    def slate_to_gem_block(self,tensor,indices):
        A,=tensor.operands
        return ComponentTensor(Indexed(self.tensor_to_variable(A),indices),indices)

    #call gem nodes for inverse and solve
    #@TODO: see questions on that in gem
    @slate_to_gem.register(firedrake.slate.slate.Inverse)
    def slate_to_gem_inverse(self,tensor,context):
        return Inverse(self.tensor_to_variable(A))

    @slate_to_gem.register(firedrake.slate.slate.Solve)
    def slate_to_gem_solve(self,tensor,context):
        raise Solve(self.tensor_to_variable(A))





    


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
