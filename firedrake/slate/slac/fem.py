
import firedrake.slate.slate as slate
import tsfc.fem

from gem import (Literal, Zero, Identity, Sum, Product, Division,
                 Power, MathFunction, MinValue, MaxValue, Comparison,
                 LogicalNot, LogicalAnd, LogicalOr, Conditional,
                 Index, Indexed, ComponentTensor, IndexSum,
                 ListTensor,Inverse,Solve,Variable)


class SlateTranslator(MultiFunction):
    """Multifunction for translating UFL -> GEM.  """

    def __init__(self, context):
        # MultiFunction.__init__ does not call further __init__
        # methods
        MultiFunction.__init__(self)

        # Need context during translation!
        self.context = context


    def slate_to_gem_translate(self, tensor):
        #should I do stuff on the tensor before already?
        return slate_to_gem(tensor, self.context)

@singledispatch
def slate_to_gem(tensor,context):
    """Translates slate tensors into GEM.
    :returns: GEM translation of the modified terminal
    """
    raise AssertionError("Cannot handle terminal type: %s" % type(tensor))

@slate_to_gem.register(slate.Mul)
def slate_to_gem_mul(tensor,context):
    _A, _B = tensor.operands
    A, B = _A.form,_B.form
    ###
    #like this:
    ####
    i,j,k=Index(extent=_A.shape[0]),Index(extent=_A.shape[1]),Index(extent=_B.shape[1])
    return ComponentTensor(IndexSum(Indexed(A,(i,j)),Indexed(B,(j,k)),j),(i,k))
    ###
    # @TODO: or call functions of mixin object of miklos??
    # we cannot do this because miklos stuff is pointwise only??
    ###

### the tensor in the following is still slate tensor
### @TODO: adjust after deciding the above
@slate_to_gem.register(slate.Add)
def slate_to_gem_add(tensor,context):
    A, B = tensor.operands
    i,j=Index(extent=A.shape[0]),Index(extent=A.shape[1])
    return ComponentTensor(Sum(Indexed(A,(i,j)),Indexed(B,(i,j))),(i,j))

@slate_to_gem.register(slate.Negative)
def slate_to_gem_negative(tensor,context):
    A,=tensor.operands
    i,j=Index(extent=A.shape[0]),Index(extent=A.shape[1])
    return ComponentTensor(Product(Literal(-1), Indexed(A,(i,j))),(i,j))
    
@slate_to_gem.register(slate.Transpose)
def slate_to_gem_transpose(tensor,context):
    A, = tensor.operands
    i,j=Index(extent=A.shape[0]),Index(extent=A.shape[1])
    return ComponentTensor(Indexed(A, (i,j)),(j,i))

#@TODO: tensor and assembled vector simply translate into variable??
@slate_to_gem.register(slate.Tensor)
def slate_to_gem_tensor(tensor,context):
    A,=tensor.operands
    #i,j=Index(extent=tensor.shape[0]),Index(extent=tensor.shape[1])
    #return ComponentTensor(Indexed(A,(i,j)),(i,j))
    return Variable(A.name,A.shape)

@slate_to_gem.register(slate.AssembledVector)
def slate_to_gem_assembledvector(tensor,context):
    A,=tensor.operands
    return Variable(A.name,A.shape)

#@TODO: actually more complicated because used for mixed tensors?
@slate_to_gem.register(slate.Block)
def slate_to_gem_block(tensor,indices,context):
    A,=tensor.operands
    return ComponentTensor(Indexed(A,indices),indices)

#call gem nodes for inverse and solve
#@TODO: see questions on that in gem
@slate_to_gem.register(slate.Inverse)
def slate_to_gem_inverse(tensor,context):
    return Inverse(tensor)

@slate_to_gem.register(slate.Solve)
def slate_to_gem_solve(tensor,context):
    raise Solve(tensor)





    


def slate_to_cpp(expr, temps, prec=None):
    """Translates a Slate expression into its equivalent representation in
    GEM syntax. This function is combining compile_ufl of tsfc, 
    but considers the additional functionalities of slate originally considered in slate_to_cpp.



    :arg expr: a :class:`slate.TensorBase` expression.
    :arg temps: a `dict` of temporaries which map a given expression to its
        corresponding representation as a `coffee.Symbol` object.
    :arg prec: an argument dictating the order of precedence in the linear
        algebra operations. This ensures that parentheticals are placed
        appropriately and the order in which linear algebra operations
        are performed are correct.

    Returns:
        a `string` which represents the C/C++ code representation of the
        `slate.TensorBase` expr.
    """
    # If the tensor is terminal, it has already been declared.
    # Coefficients defined as AssembledVectors will have been declared
    # by now, as well as any other nodes with high reference count or
    # matrix factorizations.
    if expr in temps:
        return temps[expr].gencode()

    elif isinstance(expr, slate.Transpose):
        tensor, = expr.operands
        return "(%s).transpose()" % slate_to_cpp(tensor, temps)

    elif isinstance(expr, slate.Inverse):
        tensor, = expr.operands
        return "(%s).inverse()" % slate_to_cpp(tensor, temps)

    elif isinstance(expr, slate.Negative):
        tensor, = expr.operands
        result = "-%s" % slate_to_cpp(tensor, temps, expr.prec)
        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, (slate.Add, slate.Mul)):
        op = {slate.Add: '+',
              slate.Mul: '*'}[type(expr)]
        A, B = expr.operands
        result = "%s %s %s" % (slate_to_cpp(A, temps, expr.prec),
                               op,
                               slate_to_cpp(B, temps, expr.prec))

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, slate.Block):
        tensor, = expr.operands
        indices = expr._indices
        try:
            ridx, cidx = indices
        except ValueError:
            ridx, = indices
            cidx = 0
        rids = as_tuple(ridx)
        cids = as_tuple(cidx)

        # Check if indices are non-contiguous
        if not all(all(ids[i] + 1 == ids[i + 1] for i in range(len(ids) - 1))
                   for ids in (rids, cids)):
            raise NotImplementedError("Non-contiguous blocks not implemented")

        rshape = expr.shape[0]
        rstart = sum(tensor.shapes[0][:min(rids)])
        if expr.rank == 1:
            cshape = 1
            cstart = 0
        else:
            cshape = expr.shape[1]
            cstart = sum(tensor.shapes[1][:min(cids)])

        result = "(%s).block<%d, %d>(%d, %d)" % (slate_to_cpp(tensor,
                                                              temps,
                                                              expr.prec),
                                                 rshape, cshape,
                                                 rstart, cstart)

        return parenthesize(result, expr.prec, prec)

    elif isinstance(expr, slate.Solve):
        A, B = expr.operands
        result = "%s.solve(%s)" % (slate_to_cpp(A, temps, expr.prec),
                                   slate_to_cpp(B, temps, expr.prec))

        return parenthesize(result, expr.prec, prec)

    else:
        raise NotImplementedError("Type %s not supported.", type(expr))

