
from functools import singledispatch
from contextlib import contextmanager
from collections import namedtuple
from firedrake.slate.slate import *

@singledispatch
def _action(expr, self, state):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))

@_action.register(Action)
def _action_action(expr, self, state):
    return expr

@_action.register(Tensor)
def _action_tensor(expr, self, state):
    if not self.action:
        return Mul(expr, state.coeff) if state.pick_op == 1 else Mul(state.coeff, expr)
    else:
        return Action(expr, state.coeff, state.pick_op)

@_action.register(AssembledVector)
def _action_block(expr, self, state):
    raise AssertionError("You cannot push into this node.")

@_action.register(Inverse)
def _action_inverse(expr, self, state):
    return Solve(expr.children[0], state.coeff, matfree=True)

@_action.register(Solve)
def _action_solve(expr, self, state):
    """ Pushes an action through a multiplication.
        We explot A.T*x = (x.T*A).T,
        e.g.            (y_new*(A.inv*(B.T))).T           -> {(1,4)*[((4,4)*(4,3)]}.T
        transforms to   (((A.inv*(B.T)).T*y_new.T).T.T    -> {[(4,4)*(4,3)].T*(4,1)}.T.T
                        = (B*A.inv.T*y_new.T).T.T         -> {(3,4)*[(4,4)*(4,1)]}
                        = B*A.T.solve(y_new.T)
    """
    assert expr.rank != 0, "You cannot do actions on 0 forms"
    if expr.rank == 1:
        return expr

    
    # swap operands if we are currently premultiplying due to a former transpose
    if state.pick_op == 0:
        rhs = state.swap_op
        mat = Transpose(expr.children[state.pick_op])
        swapped_op = Transpose(expr.children[state.pick_op^1])
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

@_action.register(Transpose)
def _action_transpose(expr, self, state):
    """ Pushes an action through a multiplication.
        Considers A.T*x = (x.T*A).T,
        e.g. (C*A.solve(B.T)).T * y = ((y.T * (C*A.solve(B.T))).T

        :arg expr: a Mul Slate node.
        :arg self: MemoizerArg.
        :arg state:  1: if we need to transpose this node, 0 will contain an operand
                        which needs to be swapped through
                    0: coefficient
                    2: pick op
        :returns: an action of this node on the coefficient.
    """
    if expr.rank == 2:
        return Transpose(self(*expr.children, ActionBag(state.coeff, state.swap_op, state.pick_op^1)))
    else:
        return expr


@_action.register(Negative)
@_action.register(Add)
@_action.register(Block)
def _action_distributive(expr, self, state):
    return type(expr)(*map(self, expr.children, (state,)*len(expr.children)))

@_action.register(Mul)
def _action_mul(expr, self, state):
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
    assert expr.rank != 1, "Action should not be pushed into a multiplication by vector"
    if expr.rank == 2:
            other_child = expr.children[state.pick_op^1]
            prio_child = expr.children[state.pick_op]
            if self.swapc.should_swap(prio_child, state):
                with self.swapc.swap_ops_bag(state, Transpose(other_child)) as new_state:
                    other_child, pushed_prio_child = self(prio_child, new_state)
            else:
                pushed_prio_child = self(prio_child, state)

            # Assemble new coefficient
            # FIXME: this is a temporary solutions we jump out of local assembly
            # into global assembly and back to local assembly here.
            # We need to stack tsfc calls instead.
            from firedrake import assemble

            # Then action the leftover operator onto the thing where the action got pushed into
            # solve needs special case because we need to swap args if we premultiply with a vec
            if self.swapc.should_swap(other_child, state):
                with self.swapc.swap_ops_bag(state, Transpose(pushed_prio_child)) as new_state:
                    swapped_op, pushed_other_child = self(other_child, new_state)
                coeff = pushed_other_child
                return Transpose(self(swapped_op, ActionBag(coeff, state.swap_op, state.pick_op^1)))
            else:
                coeff = pushed_prio_child
                return self(other_child, ActionBag(coeff, state.swap_op, state.pick_op))

@_action.register(Factorization)
def _action_factorization(expr, self, state):
    return Factorization(*map(self, expr.children, state))

def push_mul(tensor, coeff, options):
    """Compute the action of a form on a Coefficient.

    This works simply by replacing the last Argument
    with a Coefficient on the same function space (element).
    The form returned will thus have one Argument less
    and one additional Coefficient at the end if no
    Coefficient has been provided.
    """

    from gem.node import MemoizerArg
    mapper = MemoizerArg(_action)
    mapper.swapc = SwapController()
    mapper.action = options["replace_mul_with_action"]
    a = mapper(tensor, ActionBag(coeff, None, 1))
    return a

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

def optimise(expr, tsfc_parameters):
    if isinstance(expr, Mul):
        return push_mul(*expr.children, tsfc_parameters)
    else:
        return expr