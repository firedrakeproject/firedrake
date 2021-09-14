from gem.node import MemoizerArg
from functools import singledispatch
from itertools import repeat
from firedrake.slate.slate import *
from contextlib import contextmanager
from collections import namedtuple
from firedrake import Function


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
    # Optimise expression which is already partially optimised
    # by optimising a subexpression that is not optimised yet
    # the non optimised expression is a Mul
    # and has at least one AssembledVector as child
    partially_optimised = not (isinstance(expression, Mul)
                               or any(isinstance(child, AssembledVector)
                                      for child in expression.children))
    if partially_optimised:
        expression = push_mul(expression, None, parameters)
    else:
        expression = push_mul(*expression.children, parameters)

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
    mapper.swapc = SwapController()
    mapper.action = options["replace_mul"]
    a = mapper(tensor, ActionBag(coeff, None, 1))
    return a


@singledispatch
def _push_mul(expr, self, state):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))

@_push_mul.register(Tensor)
@_push_mul.register(Block)
def _push_mul_tensor(expr, self, state):
    if not self.action:
        return Mul(expr, state.coeff) if state.pick_op == 1 else Mul(state.coeff, expr)
    else:
        assert "Actions in Slate are not yet supported."

@_push_mul.register(AssembledVector)
def _push_mul_vector(expr, self, state):
    """Do not push into AssembledVectors."""
    return expr

@_push_mul.register(Inverse)
def _push_mul_inverse(expr, self, state):
    """ Rewrites the multiplication of Inverse
    with a coefficient into a Solve via A^{-1}*b = A.solve(b)."""
    child, = expr.children
    return Solve(child, state.coeff)

@_push_mul.register(Solve)
def _push_mul_solve(expr, self, state):
    """ Pushes an action through a multiplication.
        We explot A.T*x = (x.T*A).T,
        e.g.            (y_new*(A.inv*(B.T))).T           -> {(1,4)*[((4,4)*(4,3)]}.T
        transforms to   (((A.inv*(B.T)).T*y_new.T).T.T    -> {[(4,4)*(4,3)].T*(4,1)}.T.T
                        = (B*A.inv.T*y_new.T).T.T         -> {(3,4)*[(4,4)*(4,1)]}
                        = B*A.T.solve(y_new.T)
    """
    assert expr.rank != 0, "You cannot do actions on 0 forms"
    if expr.rank == 1:
        if not state.coeff:
            # expression is already partially optimised
            # meaning coefficient is not multiplied on the outside of it
            # so the rhs of the solve must contain the coefficient
            expr1, expr2 = expr.children
            assert expr2.rank == 1
            coeff = self(expr2, state)
            arbitrary_coeff_x = AssembledVector(Function(expr1.arg_function_spaces[state.pick_op]))
            arbitrary_coeff_p = AssembledVector(Function(expr1.arg_function_spaces[state.pick_op]))
            Aonx = self(expr1, ActionBag(arbitrary_coeff_x, None, state.pick_op))
            Aonp = self(expr1, ActionBag(arbitrary_coeff_p, None, state.pick_op))
            if not isinstance(expr1, Tensor): # non terminal node 
                mat = TensorShell(expr1)
            else:
                mat = expr1
            return Solve(mat, coeff, matfree=expr.is_matfree, Aonx=Aonx, Aonp=Aonp)
    else:
        # swap operands if we are currently premultiplying due to a former transpose
        if state.pick_op == 0:
            rhs = state.swap_op
            mat = Transpose(expr.children[state.pick_op])
            swapped_op = Transpose(expr.children[state.pick_op^1])#, ActionBag(state.coeff, None, state.pick_op))
            # FIXME
            assert not(isinstance(rhs, Solve) and rhs.rank==2), "We need to fix the case where \
                                                                the rhs in a  Solve is a result of a Solve"
            return swapped_op, Solve(mat, self(rhs, ActionBag(state.coeff, None, state.pick_op^1)),
                                    matfree=expr.is_matfree)
        else:
            rhs = expr.children[state.pick_op]
            mat = expr.children[state.pick_op^1]
            # always push into the right hand side of the solve
            return Solve(mat, self(rhs, state), matfree=expr.is_matfree)

@_push_mul.register(Transpose)
def _push_mul_transpose(expr, self, state):
    """ Pushes a multiplication through a transpose.
        This works with help of A.T*x = (x.T*A).T.
        Another example for  expr:=C*A.solve(B.T) : (C*A.solve(B.T)).T * y = (y.T * (C*A.solve(B.T))).T

        :arg expr: a Transpose
        :arg self: a MemoizerArg object.
        :arg state: state carries a coefficient in .coeff,
                    information about argument swapping in .swap_op,
                    and information if multiply from front (0) or back (1) in .pick_op
        :returns: a transposed expression
    """
    if expr.rank == 2:
        pushed_expr = self(*expr.children,              # push mul into A
                           ActionBag(state.coeff,
                                     state.swap_op,
                                     state.pick_op^1))  # but switch the multiplication order with pick_op 
        return self(Transpose(pushed_expr,              # then Transpose the end result
                    ActionBag(state.coeff, state.swap_op, state.pick_op)))
    else:
        return expr


@_push_mul.register(Negative)
@_push_mul.register(Add)
def _push_mul_distributive(expr, self, state):
    """Distribute the multiplication into the children of the expression. """
    return type(expr)(*map(self, expr.children, (state,)*len(expr.children)))

@_push_mul.register(Mul)
def _push_mul_mul(expr, self, state):
    """ Pushes an action through a multiplication.

        EXAMPLE 1: (A*B*C)*y
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––
                action(A*B*C,y)   ---->    A*B*action(C,y)       |-> A*B*new_coeff  ---->    action(A*B, new_coeff)
            
                    action1                     Mul2                                                action2
                    /     \                   /      \                                              /      \ 
                Mul2       y   ---->      Mul3        action1    |                  ---->       Mul3        new_coeff
                /   \                    /    \       /    \     |->new_coeff                  /    \ 
            Mul3    op3               op1      op2 op3      y    |                          op1       op2
            /   \
        op1      op2
        
        EXAMPLE 2: TensorOp(op1, op2)*y
        –––––––––––––––––––––––––––––––––––––––––––––––––––––––
                action                           TensorOp
                /     \                         /        \
             y.T       TensorOp        ---->  action       op2
                       /       \              /   \
                    op1          op2         y.T   op1

        EXAMPLE 3: C is (3,4) A is (4,4), B is (3,4), x is (3,1)
        ––––––––––––––––––––––––––––––––––––––––––––––––––––––––

                (y.T * (C*A.solve(B.T)).T                               ->{(1,3)*[(3,4)*(4,4)*(4,3)]}.T
        -->     ((y.T*C)*A.solve(B.T)).T =(y_new*(A.inv*(B.T))).T       ->{(1,4)*[((4,4)*(4,3)]}.T
        -->     (((A.inv*(B.T)).T*y_new.T).T.T                          ->{[(4,4)*(4,3)].T*(4,1)}.T.T
        -->     (B*A.inv.T*(y_new.T)).T.T = B*A.T.solve(y_new.T).T.T    -> (3,4)*[(4,4)*(4,1)]

    :arg expr: a Mul Slate node.
    :arg self: MemoizerArg.
    :arg state:  1: if we need to transpose this node, 0 will contain an operand
                    which needs to be swapped through
                 0: coefficient
                 2: pick op
    :returns: an action of this node on the coefficient.
    """
    assert expr.rank != 0, "You cannot do actions on 0 forms"
    if expr.rank == 2:
            other_child = expr.children[state.pick_op^1]
            prio_child = expr.children[state.pick_op]
            if self.swapc.should_swap(prio_child, state):
                with self.swapc.swap_ops_bag(state, Transpose(other_child)) as new_state:
                    other_child, pushed_prio_child = self(prio_child, new_state)
            else:
                pushed_prio_child = self(prio_child, state)

            # Then action the leftover operator onto the thing where the action got pushed into
            # solve needs special case because we need to swap args if we premultiply with a vec
            if self.swapc.should_swap(other_child, state):
                with self.swapc.swap_ops_bag(state, Transpose(pushed_prio_child)) as new_state:
                    swapped_op, pushed_other_child = self(other_child, new_state)
                coeff = pushed_other_child
                return self(Transpose(self(swapped_op, ActionBag(coeff, state.swap_op, state.pick_op^1))), ActionBag(coeff, state.swap_op, state.pick_op))
            else:
                coeff = pushed_prio_child
                return self(other_child, ActionBag(coeff, state.swap_op, state.pick_op))
    elif expr.rank == 1:
        # expression is already partially optimised
        # meaning the coefficient is not multiplied on the outside of it
        if not state.coeff:
            # optimise the expression which is rank1
            # because it is the one that must contain a coefficient
            rank1expr, = tuple(filter(lambda child: child.rank == 1, expr.children))
            coeff = self(rank1expr, state)
            pick_op = expr.children.index(rank1expr)
            return self(expr.children[pick_op^1], ActionBag(coeff, state.swap_op, pick_op))
        else:
            return expr

@_push_mul.register(Factorization)
def _push_mul_factorization(expr, self, state):
    """ Drop any factorisations. """
    return self(*expr.children, state)

""" ActionBag class
:arg coeff: what we contract with.
:arg swap_op:   holds an operand that needs to be swapped with the child of another operand
                needed to deal with solves which get premultiplied by a vector.
:arg pick_op:   decides which argument in Tensor is exchanged against the coefficient
                and also in which operand the action has to be pushed,
                basically determins if we pre or postmultiply
"""
ActionBag = namedtuple("ActionBag", ["coeff", "swap_op",  "pick_op"])

class SwapController(object):

    def should_swap(self, child, state):
        return isinstance(child, Solve) and state.pick_op == 0 and child.rank == 2

    @contextmanager
    def swap_ops_bag(self, state, swap_op):
        """Provides a context to swap operand swap_op with a node from a level down.
        :arg state: current state.
        :arg op: operand to be swapped.
        :returns: the modified code generation context."""
        yield ActionBag(state.coeff, swap_op, state.pick_op)
