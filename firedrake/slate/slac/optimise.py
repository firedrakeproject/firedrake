
from functools import singledispatch
from contextlib import contextmanager
from collections import namedtuple
from firedrake.slate.slate import *
from firedrake import Function
from gem.node import pre_traversal as traverse_dags


def drop_double_transpose(expr):
    """ Remove double transposes from optimised Slate expression.
        Remember A = A.T.T
    """
    from gem.node import Memoizer
    mapper = Memoizer(_drop_double_transpose)
    a = mapper(expr)
    return a


@singledispatch
def _drop_double_transpose(expr, self):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_drop_double_transpose.register(Tensor)
@_drop_double_transpose.register(AssembledVector)
@_drop_double_transpose.register(TensorShell)
@_drop_double_transpose.register(Block)
def _drop_double_transpose_terminals(expr, self):
    return expr


@_drop_double_transpose.register(Transpose)
def _drop_double_transpose_transpose(expr, self):
    child, = expr.children
    if isinstance(child, Transpose):
        grandchild, = child.children
        return self(grandchild)
    return type(expr)(*map(self, expr.children))


@_drop_double_transpose.register(Negative)
@_drop_double_transpose.register(Add)
@_drop_double_transpose.register(Mul)
def _drop_double_transpose_distributive(expr, self):
    return type(expr)(*map(self, expr.children))


@_drop_double_transpose.register(Action)
def _drop_double_transpose_action(expr, self):
    return type(expr)(*map(self, expr.children), expr.pick_op)


@_drop_double_transpose.register(Solve)
def _drop_double_transpose_action(expr, self):
    return type(expr)(*map(self, expr.children), matfree=expr.is_matfree, Aonx=expr._Aonx, Aonp=expr._Aonp)


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


@_action.register(Block)
def _action_tensor(expr, self, state):
    if not self.action:
        return Mul(expr, state.coeff) if state.pick_op == 1 else Mul(state.coeff, expr)
    else:
        tensor, = expr.children  # drop the block node
        self.block_indices = expr._indices
        return Action(tensor, state.coeff, state.pick_op)

@_action.register(AssembledVector)
def _action_block(expr, self, state):
    return expr

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
        if not state.coeff:
            # expression is already partially optimised
            # meaning coefficient is not multiplied on the outside of it
            # so the rhs of the solve must contain the coefficient
            expr1, expr2 = expr.children
            assert expr2.rank == 1
            coeff = self(expr2, state)
            arbitrary_coeff_x = AssembledVector(Function(expr1.arg_function_spaces[state.pick_op]))
            arbitrary_coeff_p = AssembledVector(Function(expr1.arg_function_spaces[state.pick_op]))
            Aonx = self(expr1, ActionBag(arbitrary_coeff_x, None, state.pick_op, state.block_indices))
            Aonp = self(expr1, ActionBag(arbitrary_coeff_p, None, state.pick_op, state.block_indices))
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
            return swapped_op, Solve(mat, self(rhs, ActionBag(state.coeff, None, state.pick_op^1, state.block_indices)),
                                    matfree=expr.is_matfree)
        else:
            rhs = expr.children[state.pick_op]
            mat = expr.children[state.pick_op^1]
            if isinstance(mat, Factorization):
                mat, = mat.children
            if isinstance(mat, Block):
                mat, = mat.children
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
        return self(Transpose(self(*expr.children,
                                    ActionBag(state.coeff, state.swap_op, state.pick_op^1, state.block_indices))),
                                    ActionBag(state.coeff, state.swap_op, state.pick_op, state.block_indices))
    else:
        return expr


@_action.register(Negative)
@_action.register(Add)
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
                return self(Transpose(self(swapped_op,
                                           ActionBag(coeff, state.swap_op, state.pick_op^1, state.block_indices))),
                                           ActionBag(coeff, state.swap_op, state.pick_op, state.block_indices))
            else:
                coeff = pushed_prio_child
                # new_state = ActionBag(coeff, state.swap_op, state.pick_op, state.block_indices)
                # state = check_children_are_blocks((other_child, pushed_prio_child), new_state)
                # return blockify(self(other_child, state))
                return self(other_child, ActionBag(coeff, state.swap_op, state.pick_op, state.block_indices))

    elif expr.rank == 1:
        # expression is already partially optimised
        # meaning the coefficient is not multiplied on the outside of it
        if not state.coeff:
            # optimise the expression which is rank1
            # because it is the one that must contain a coefficient
            rank1expr, = tuple(filter(lambda child: child.rank == 1, expr.children))
            coeff = self(rank1expr, state)
            pick_op = expr.children.index(rank1expr)
            return self(expr.children[pick_op^1], ActionBag(coeff, state.swap_op, pick_op, state.block_indices))
        else:
            return expr

@_action.register(Factorization)
def _action_factorization(expr, self, state):
    return self(*expr.children, state)

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
    mapper.block_indices = ()
    # FIXME blocking only from the outside may not be sufficient
    # when subexpressions are already optimised by user
    state = ActionBag(coeff, None, 1, ())
    if tensor.children:
        block_state = check_children_are_blocks(tensor.children, state)
        a = mapper(tensor, block_state[0]) 
        return blockify(a, block_state[0])
    else:
        return mapper(tensor, state)


""" ActionBag class
:arg coeff: what we contract with.
:arg swap_op:   holds an operand that needs to be swapped with the child of another operand
                needed to deal with solves which get premultiplied by a vector.
:arg pick_op:   decides which argument in Tensor is exchanged against the coefficient
                and also in which operand the action has to be pushed,
                basically determins if we pre or postmultiply
"""
ActionBag = namedtuple("ActionBag", ["coeff", "swap_op",  "pick_op", "block_indices"])

def blockify(child, state):
    if state.block_indices:
        return Block(child, (state.block_indices[state.pick_op^1],))
    else:
        return child

def check_children_are_blocks(children, state):
    # For Blocks the coefficient has to be pulled inside the expression
    # which has to go along with a change of shape of the coefficient
    # so a new coefficient needs to be generated on the non-indexed FS
    # For Solve nodes e.g. we don't have a corresponding coefficient attached,
    # so we need to generate one here
    # this only has to happen once!
    def is_block(node): return isinstance(node, Block)
    def needs_new_coeff(last, new): return (not last or (last and (not last[0] == new)))
    states = ()
    coeff = state.coeff
    for node in children:
        indices = ()
        if is_block(node):
            if not states or (states and needs_new_coeff(states[0].block_indices, node._indices)):
                tensor, = node.children
                if state.coeff.shape[0] != tensor.shape[1]:
                    coeff = AssembledVector(Function(tensor.arg_function_spaces[state.pick_op]))
                    indices = node._indices
        states += (ActionBag(coeff, state.swap_op, state.pick_op, indices), )
    return states


class SwapController(object):

    def should_swap(self, child, state):
        return isinstance(child, Solve) and state.pick_op == 0 and child.rank == 2

    @contextmanager
    def swap_ops_bag(self, state, swap_op):
        """Provides a context to swap operand swap_op with a node from a level down.
        :arg state: current state.
        :arg op: operand to be swapped.
        :returns: the modified code generation context."""
        yield ActionBag(state.coeff, swap_op, state.pick_op, state.block_indices)

def optimise(expr, tsfc_parameters):
    # Optimise expression which is already partially optimised
    # by optimising a subexpression that is not optimised yet
    # the non optimised expression is a Mul
    # and has at least one AssembledVector as child
    partially_optimised = not (isinstance(expr, Mul)
                               or any(isinstance(child, AssembledVector)
                                      for child in expr.children))
    if partially_optimised:
        # for partially optimised exppresions we pass no coefficient to act on 
        return drop_double_transpose(push_mul(expr, None, tsfc_parameters))
    else:
        return drop_double_transpose(push_mul(*expr.children, tsfc_parameters))
