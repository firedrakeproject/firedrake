from gem.node import MemoizerArg
from functools import singledispatch
from itertools import repeat
from firedrake.slate.slate import *
from contextlib import contextmanager
from collections import namedtuple

""" ActionBag class
:arg coeff: what we contract with.
:arg pick_op:   decides which argument in Tensor is exchanged against the coefficient
                and also in which operand the action has to be pushed,
                basically determins if we pre or postmultiply
"""
ActionBag = namedtuple("ActionBag", ["coeff", "pick_op"])

def optimise(expression, parameters):
    """Optimises a Slate expression, by pushing blocks and multiplications
    inside the expression and by removing double transposes.

    :arg expression: A (potentially unoptimised) Slate expression.
    :arg parameters: A dict of compiler parameters.

    Returns: An optimised Slate expression
    """
    # 1) Block optimisation
    expression = push_block(expression)

    # 2) Multiplication optimisation
    if expression.rank < 2:
        expression = push_mul(expression, None, parameters)

    # 3) Transpose optimisation
    expression = drop_double_transpose(expression)

    return expression

def push_block(expression):
    """Executes a Slate compiler optimisation pass.
    The optimisation is achieved by pushing blocks from the outside to the inside of an expression.
    Without the optimisation the local TSFC kernels are assembled first
    and then the result of the assembly kernel gets indexed in the Slate kernel
    (and further linear algebra operations maybe done on it).
    The optimisation pass essentially changes the order of assembly and indexing.

    :arg expression: A (potentially unoptimised) Slate expression.

    Returns: An optimised Slate expression, where Blocks are terminal whereever possible.
    """
    mapper = MemoizerArg(_push_block)
    ret = mapper(expression, ())
    return ret


@singledispatch
def _push_block(expr, self, indices):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_push_block.register(Transpose)
def _push_block_transpose(expr, self, indices):
    """Indices of the Blocks are transposed if Block is pushed into a Transpose."""
    return Transpose(*map(self, expr.children, repeat(indices[::-1]))) if indices else expr


@_push_block.register(Add)
@_push_block.register(Negative)
def _push_block_distributive(expr, self, indices):
    """Distributes Blocks for these nodes"""
    return type(expr)(*map(self, expr.children, repeat(indices))) if indices else expr


@_push_block.register(Factorization)
@_push_block.register(Inverse)
@_push_block.register(Solve)
@_push_block.register(Mul)
def _push_block_stop(expr, self, indices):
    """Blocks cannot be pushed further into this set of nodes."""
    return Block(expr, indices) if indices else expr


@_push_block.register(Tensor)
def _push_block_tensor(expr, self, indices):
    """Turns a Block on a Tensor into a Tensor of an indexed form."""
    return Tensor(Block(expr, indices).form) if indices else expr


@_push_block.register(AssembledVector)
def _push_block_assembled_vector(expr, self, indices):
    """Turns a Block on an AssembledVector into the  specialized node BlockAssembledVector."""
    return BlockAssembledVector(expr._function, Block(expr, indices).form, indices) if indices else expr


@_push_block.register(Block)
def _push_block_block(expr, self, indices):
    """Inlines Blocks into each other.
    Note that the indices are inlined from the ouside.
    Example: If we have got the Slate expression A.blocks[:3,:3].blocks[0,0], we encounter the (0,0)-blocks first.
    The first time round the indices are empty and the (0,0) are in expr._indices.
    The second time round, (0,0) is stored in indices and the slices are in expr._indices.
    So in the following line we basically say indices = ((0,1,2)[0], (0,1,2)[0])
    """
    reindexed = tuple(big[slice(small[0], small[-1]+1)] for big, small in zip(expr._indices, indices))
    indices = expr._indices if not indices else reindexed
    block, = map(self, expr.children, repeat(indices))
    return block


def drop_double_transpose(expr):
    """Remove double transposes from optimised Slate expression."""
    from gem.node import Memoizer
    mapper = Memoizer(_drop_double_transpose)
    a = mapper(expr)
    return a


@singledispatch
def _drop_double_transpose(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_drop_double_transpose.register(Tensor)
@_drop_double_transpose.register(AssembledVector)
@_drop_double_transpose.register(Block)
def _drop_double_transpose_terminals(expr, self):
    """Terminal expression is encountered."""
    return expr


@_drop_double_transpose.register(Factorization)
def _drop_double_transpose_factorization(expr, self):
    """ Drop any factorisations. """
    return self(*expr.children)


@_drop_double_transpose.register(Transpose)
def _drop_double_transpose_transpose(expr, self):
    """When the expression and its child are transposes the grandchild is returned,
    because A=A.T.T."""
    child, = expr.children
    if isinstance(child, Transpose):
        grandchild, = child.children
        return self(grandchild)
    else:
        return type(expr)(*map(self, expr.children))


@_drop_double_transpose.register(Negative)
@_drop_double_transpose.register(Add)
@_drop_double_transpose.register(Mul)
@_drop_double_transpose.register(Solve)
@_drop_double_transpose.register(Inverse)
def _drop_double_transpose_distributive(expr, self):
    """Distribute into the children of the expression. """
    return type(expr)(*map(self, expr.children))


def push_mul(tensor, coeff, options):
    """Compute the action of a form on a Coefficient.

    This works simply by replacing the last Argument
    with a Coefficient on the same function space (element).
    The form returned will thus have one Argument less
    and one additional Coefficient at the end if no
    Coefficient has been provided.
    """

    from gem.node import MemoizerArg
    mapper = MemoizerArg(_push_mul)
    mapper.action = options["replace_mul"]
    a = mapper(tensor, ActionBag(coeff, 1))
    return a


@singledispatch
def _push_mul(expr, self, state):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))

@_push_mul.register(Tensor)
@_push_mul.register(Block)
def _push_mul_tensor(expr, self, state):
    if not self.action:
        if state.coeff:
            return Mul(expr, state.coeff) if state.pick_op == 1 else Mul(state.coeff, expr)
        else:
            return expr
    else:
        assert "Actions in Slate are not yet supported."

@_push_mul.register(AssembledVector)
def _push_mul_vector(expr, self, state):
    """Do not push into AssembledVectors."""
    return expr

@_push_mul.register(Negative)
@_push_mul.register(Add)
def _push_mul_distributive(expr, self, state):
    """Distribute the multiplication into the children of the expression. """
    return type(expr)(*map(self, expr.children, (state,)*len(expr.children)))

@_push_mul.register(Inverse)
def _push_mul_inverse(expr, self, state):
    """ Rewrites the multiplication of Inverse
    with a coefficient into a Solve via A.inv*b = A.solve(b)
    or b*A^{-1}= (A.T.inv*b.T).T = A.T.solve(b.T).T ."""
    child, = expr.children
    return Solve(child, state.coeff) if state.pick_op \
           else Transpose(Solve(Transpose(child), Transpose(state.coeff)))

@_push_mul.register(Transpose)
def _push_mul_transpose(expr, self, state):
    """ Pushes a multiplication through a transpose.
        This works with help of A.T*x = (x.T*A).T.
        Another example for  expr:=C*A.solve(B.T) : (C*A.solve(B.T)).T * y = (y.T * (C*A.solve(B.T))).T

        :arg expr: a Transpose
        :arg self: a MemoizerArg object.
        :arg state: state carries a coefficient in .coeff,
                    and information if multiply from front (0) or back (1) in .pick_op
        :returns: an optimised transposed expression
    """
    if expr.rank == 2:
        pushed_expr = self(*expr.children,              # push mul into A
                           ActionBag(Transpose(state.coeff),
                                     state.pick_op^1))  # but switch the multiplication order with pick_op 
        return self(Transpose(pushed_expr),             # then Transpose the end result
                    ActionBag(state.coeff, state.pick_op))
    else:
        return expr

@_push_mul.register(Solve)
def _push_mul_solve(expr, self, state):
    """ Pushes a multiplication through a solve.
        case 1) child 1 is matrix, child2 is vector
        case 2) child 1 is matrix, child2 is matrix
                a) multiplication from front
                b) multiplication from back
    """
    if expr.rank == 2 and state.pick_op == 0:
        """
        case 2) child 1 is matrix, child2 is matrix and a coefficient is passed through
                a)  multiplication from front
                    We exploit A.T*x = (x.T*A).T and previously used A.inv*b=A.solve(b).
                    e.g.            (1) T3*T3.inv*T3.T              
                    transforms to   (2) T3.T.solve(C3.T).T*T3.T     
                                    (3) (T_3*T3.T.solve(C3.T)).T    
                    From (2) to (3) we need to swap rhs of the solve with the coefficient.

                    or              (1) (C*(A.solve(A.solve(A))))
                    is              (2) (A.solve(A).T*A.T.solve(C.T)).T
                    transforms to   (3) (A.T.solve(C.T)).T*A.solve(A)).T.T
                                    (4) (A.T*A.T.solve((A.T.solve(C.T)))
                                    (5) (A.T*A.T.solve(A.T.solve(C.T))).T
                                    (6) A.T.solve(A.T.solve(C.T)).T*A
        """
        mat = Transpose(expr.children[state.pick_op])
        swapped_op = Transpose(expr.children[state.pick_op^1])
        new_rhs = Transpose(state.coeff)
        pushed_child = self(Solve(mat, new_rhs), ActionBag(None, state.pick_op^1))
        return Transpose(self(swapped_op, ActionBag(pushed_child, state.pick_op^1)))
    else:
        """
        case 1) a)  child 1 is matrix, child2 is vector and there is no coefficient passed through
                b)  child 2 is matrix, child2 is matrix and there is a coefficient passed through
                    ->  multiplication from back
                        A.solve(B)*x = A.inv*B*x = A.inv*(B*x) = A.solve(Bx)
                We always push into the right hand side of the solve.
        """
        mat, rhs = expr.children
        if (rhs.rank == 1 and state.coeff):
            return expr
        else:
            coeff = self(rhs, state)
            return Solve(mat, self(coeff, state))


@_push_mul.register(Mul)
def _push_mul_mul(expr, self, state):
    """ Pushes an multiplication by a coefficent through a multiplication to the innermost node.
        e.g. (A1*A2)*b = A1*(A2*b)

        case 1) child 1 is matrix, child2 is vector or other way around
        case 2) child 1 is matrix, child2 is matrix

        :arg expr: a Multiplication
        :arg self: a MemoizerArg object.
        :arg state: state carries a coefficient in .coeff,
                    and information if multiply from front (0) or back (1) in .pick_op
        :returns: an optimised Multiplication
    """
    if expr.rank == 1:
        """
        case 1) child 1 is matrix, child2 is vector or other way around
                In this case the Mul expression is already partially optimised.
                a) The child with rank 1 must contain the coefficient.
                Both childs may still leave space for optimisation.
                Optimise child of rank 1 first and use result as coefficient
                to push the Mul into the other child.
                No coefficient would be multiplied on the outside of the Mul.
                e.g. (A0*A1)*((A2+A3)*y) = (A0*A1)*(A2*y+A3*y) = A0*(A1*(A2*y+A3*y))
                b) A coefficient is multiplied from the outside.
                Both childs may still leave space for optimisation.
                Optimise child of rank 1 first but do not use result as coefficient.
                Push coefficient from the outside into the Mul of the other child.
                e.g. (y.T*(A0+A1)*B)*y = ((y.T*A0+y.T*A1)*B)*y = (y.T*A0+y.T*A1)*(B*y) 
        """
        rank1expr, = tuple(filter(lambda child: child.rank == 1, expr.children))        # a) (A2+A3)*y              b) (y.T*A0+y.T*A1)
        coeff = self(rank1expr, state)                                                  # a) (A2*y+A3*y)            b) y.T*A0+y.T*A1)
        pick_op = expr.children.index(rank1expr)
        other_child = expr.children[pick_op^1]                                          # a) A0*A1                  b) B             
        return self(other_child, ActionBag(coeff, pick_op))                             # a) A0*(A1*(A2*y+A3*y))    b)(y.T*A0+y.T*A1)*(B*y)
    elif expr.rank == 2:
            """
            case 2) child 1 is matrix, child2 is matrix

            EXAMPLE 1: (A*B*C)*y
            –––––––––––––––––––––––––––––––––––––––––––––––––––––––
                    outmost_mul(A*B*C,y)   ---->    A*B*outmost_mul(C,y)   ---->    A*(B*outmost_mul(C,y))
                
                        outmost_mul1                Mul2                                     Mul3
                        /     \                   /      \                                  /    \ 
                    Mul2       y   ---->      Mul3        outmost_mul1     ---->           A      Mul2
                    /   \                    /    \       /    \                                 /    \
                Mul3     C                  A      B     C      y                               B    outmost_mul
                /   \                                                                                /         \
               A     B                                                                             C            y
            
            EXAMPLE 2: y.T*TensorOp(op1, op2)
            –––––––––––––––––––––––––––––––––––––––––––––––––––––––
                    outmost_mul                               TensorOp
                    /     \                                  /        \
                y.T       TensorOp        ---->    outmost_mul       op2
                         /       \                   /   \
                        op1      op2               y.T   op1

            EXAMPLE 3: C is (3,4) A is (4,4), B is (3,4), y is (3,1)
            ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

                    (y.T * (C*A.solve(B.T)).T                               ->{(1,3)*[(3,4)*(4,4)*(4,3)]}.T
            -->     ((y.T*C)*A.solve(B.T)).T =(y_new*(A.inv*(B.T))).T       ->{(1,4)*[((4,4)*(4,3)]}.T
            -->     (((A.inv*(B.T)).T*y_new.T).T.T                          ->{[(4,4)*(4,3)].T*(4,1)}.T.T
            -->     (B*A.inv.T*(y_new.T)).T.T = B*A.T.solve(y_new.T)       -> (3,4)*[(4,4)*(4,1)]

            """
            # We optimise the first child first if multiplication happens from front,    EXAMPLE 3
            # otw. we we walk into second child first.
            other_child = expr.children[state.pick_op^1]                                 # A.solve(B.T)
            prio_child = expr.children[state.pick_op]                                    # C
            coeff = self(prio_child, state)                                              # y.T*C
            return self(other_child, ActionBag(coeff, state.pick_op))                    # push_mul(A.solve(B), y.T*C)

@_push_mul.register(Factorization)
def _push_mul_factorization(expr, self, state):
    """ Drop any factorisations. """
    return self(*expr.children, state)

