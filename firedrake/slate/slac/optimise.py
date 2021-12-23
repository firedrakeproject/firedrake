from gem.node import MemoizerArg
from functools import singledispatch
from itertools import repeat
from firedrake.slate.slate import *
from collections import namedtuple
from firedrake.ufl_expr import adjoint

""" ActionBag class
:arg coeff:     This is the object b in Action(A, b) or Mul(A, b).
:arg pick_op:   Pick_op decides which argument in Tensor is exchanged against the coefficient
                and also in which operand the action has to be pushed,
                it basically determines if we pre or postmultiply.
"""
ActionBag = namedtuple("ActionBag", ["coeff", "pick_op"])


def flip(pick_op):
    """Flip an index. Using this function essentially reverses the order of multiplication."""
    return pick_op ^ 1


def optimise(expression, parameters):
    """Optimises a Slate expression, by pushing blocks and multiplications
    inside the expression and by removing double transposes.

    :arg expression: A (potentially unoptimised) Slate expression.
    :arg parameters: A dict of compiler parameters.

    Returns: An optimised Slate expression
    """
    # 0) Block optimisation
    expression = push_block(expression)

    # 1) DiagonalTensor optimisation
    expression = push_diag(expression)

    # 2) Multiplication optimisation
    if expression.rank < 2:
        expression = push_mul(expression, parameters)

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
    return (Transpose(*map(self, expr.children, repeat(reversed(indices))))
            if indices else Transpose(*map(self, expr.children, repeat(indices))))


@_push_block.register(Add)
@_push_block.register(Negative)
@_push_block.register(Reciprocal)
def _push_block_distributive(expr, self, indices):
    """Distributes Blocks for these nodes"""
    return type(expr)(*map(self, expr.children, repeat(indices)))


@_push_block.register(TensorShell)
def _push_block_shell(expr, self, indices):
    """Distributes Blocks into TensorShell node"""
    child, = expr.children
    # Drop TensorShell nodes if the child is terminal
    # maybe we don't ever get into that state, should be asserted earlier
    return self(child, indices) if child.terminal else type(expr)(self(child, indices))

@_push_block.register(DiagonalTensor)
def _push_block_diag(expr, self, indices):
    """Distributes Blocks for these nodes"""
    return type(expr)(*map(self, expr.children, repeat(indices)), expr.vec)


@_push_block.register(Factorization)
@_push_block.register(Inverse)
@_push_block.register(Mul)
def _push_block_stop(expr, self, indices):
    """Blocks cannot be pushed further into this set of nodes."""
    expr = type(expr)(*map(self, expr.children, repeat(tuple())))
    return Block(expr, indices) if indices else expr


@_push_block.register(Solve)
def _push_block_solve(expr, self, indices):
    """Blocks cannot be pushed further into this set of nodes."""
    expr = type(expr)(*map(self, expr.children, repeat(tuple())), matfree=expr.matfree,
                           Aonx=expr.Aonx, Aonp=expr.Aonp,
                           preconditioner=expr.preconditioner, Ponr=expr.Ponr,
                           diag_prec=expr.diag_prec)
    return Block(expr, indices) if indices else expr


@_push_block.register(Action)
def _push_block_action(expr, self, indices):
    """Blocks cannot be pushed further into Action nodes."""
    expr = type(expr)(*map(self, expr.children, repeat(tuple())), expr.pick_op)
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


def push_diag(expression):
    """Executes a Slate compiler optimisation pass.
    The optimisation is achieved by pushing DiagonalTensor from the outside to the inside of an expression.

    :arg expression: A (potentially unoptimised) Slate expression.

    Returns: An optimised Slate expression, where DiagonalTensors are sitting
    on terminal tensors whereever possible.
    """
    mapper = MemoizerArg(_push_diag)
    mapper.vec = False
    return mapper(expression, False)


@singledispatch
def _push_diag(expr, self, diag, vec):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_push_diag.register(Transpose)
@_push_diag.register(Add)
@_push_diag.register(Negative)
def _push_diag_distributive(expr, self, diag):
    """Distributes the DiagonalTensors into these nodes"""
    return type(expr)(*map(self, expr.children, repeat(diag)))


@_push_diag.register(Factorization)
@_push_diag.register(Inverse)
@_push_diag.register(Mul)
@_push_diag.register(Tensor)
@_push_diag.register(TensorShell)
def _push_diag_stop(expr, self, diag):
    """Diagonal Tensors cannot be pushed further into this set of nodes."""
    expr = type(expr)(*map(self, expr.children, repeat(False))) if not expr.terminal else expr
    return DiagonalTensor(expr, self.vec) if diag else expr


@_push_diag.register(Block)
def _push_diag_block(expr, self, diag):
    """Diagonal Tensors cannot be pushed further into this set of nodes."""
    expr = type(expr)(*map(self, expr.children, repeat(False)), expr._indices) if not expr.terminal else expr
    return DiagonalTensor(expr, self.vec) if diag else expr


@_push_diag.register(AssembledVector)
@_push_diag.register(Reciprocal)
@_push_diag.register(Action)
@_push_diag.register(Solve)
def _push_diag_vectors(expr, self, diag):
    """DiagonalTensors should not be pushed onto rank-1 tensors."""
    if diag:
        raise AssertionError("It is not legal to define DiagonalTensors on rank-1 tensors.")
    else:
        return expr


@_push_diag.register(DiagonalTensor)
def _push_diag_diag(expr, self, diag):
    """DiagonalTensors are either pushed down or ignored when wrapped into another DiagonalTensor."""
    self.vec = expr.vec
    return self(*expr.children, not diag)


def push_mul(tensor, options):
    """Executes a Slate compiler optimisation pass.
    The optimisation is achieved by pushing coefficients from
    the outside to the inside of an expression.
    The optimisation pass essentially changes the order of operations
    in the expressions so that only matrix-vector products are executed.

    :arg tensor: A (potentially unoptimised) Slate expression.
    :arg options: Optimisation pass options,
                  e.g. if the multiplication should be replaced by an action.

    Returns: An optimised Slate expression,
             where only matrix-vector products are executed whereever possible.
    """

    from gem.node import MemoizerArg
    mapper = MemoizerArg(_push_mul)
    mapper.action = options["replace_mul"]
    a = mapper(tensor, ActionBag(None, 1))
    return a


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
@_drop_double_transpose.register(TensorShell)
def _drop_double_transpose_terminals(expr, self):
    """Terminal expression is encountered."""
    return expr


@_drop_double_transpose.register(Factorization)
def _drop_double_transpose_factorization(expr, self):
    """Drop any factorisations. This is needed because whenever
    a Solve node is created so is a Factorization node inside it.
    The Factorization nodes are only needed for the Eigen Backend however."""
    return self(*expr.children)


@_drop_double_transpose.register(Transpose)
def _drop_double_transpose_transpose(expr, self):
    """When the expression and its child are transposes the grandchild is returned,
    because A=A.T.T."""
    child, = expr.children
    if isinstance(child, Transpose):
        grandchild, = child.children
        return self(grandchild)
    elif child.terminal and child.rank > 1:
        return Tensor(adjoint(child.form))
    else:
        return type(expr)(*map(self, expr.children))


@_drop_double_transpose.register(Negative)
@_drop_double_transpose.register(Add)
@_drop_double_transpose.register(Mul)
@_drop_double_transpose.register(Inverse)
@_drop_double_transpose.register(Reciprocal)
def _drop_double_transpose_distributive(expr, self):
    """Distribute into the children of the expression. """
    return type(expr)(*map(self, expr.children))


@_drop_double_transpose.register(Action)
def _drop_double_transpose_action(expr, self):
    return type(expr)(*map(self, expr.children), expr.pick_op)


@_drop_double_transpose.register(Solve)
def _drop_double_transpose_solve(expr, self):
    return type(expr)(*map(self, expr.children), matfree=expr.matfree,
                           Aonx=expr.Aonx, Aonp=expr.Aonp,
                           preconditioner=expr.preconditioner, Ponr=expr.Ponr,
                           diag_prec=expr.diag_prec)

@_drop_double_transpose.register(DiagonalTensor)
def _drop_double_transpose_diag(expr, self):
    return type(expr)(*map(self, expr.children), expr.vec)


@singledispatch
def _push_mul(expr, self, state):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_push_mul.register(Tensor)
@_push_mul.register(Block)
@_push_mul.register(TensorShell)
def _push_mul_tensor(expr, self, state):
    if not self.action:
        if state.coeff:
            return Mul(expr, state.coeff) if state.pick_op == 1 else Mul(state.coeff, expr)
        else:
            return expr
    else:
        return Action(expr, state.coeff, state.pick_op)


@_push_mul.register(AssembledVector)
@_push_mul.register(DiagonalTensor)
@_push_mul.register(Reciprocal)
@_push_mul.register(Hadamard)
def _push_mul_vector(expr, self, state):
    """Do not push into these nodes."""
    return expr


@_push_mul.register(Action)
def _push_mul_action(expr, self, state):
    """Drop TensorShells inside actions if needed and push on if not terminal."""
    tensor, rhs = expr.children
    if isinstance(tensor, TensorShell):
        tensor, = tensor.children
    return (Action(tensor, rhs, state.pick_op)
            if tensor.terminal else self(tensor, ActionBag(rhs, state.pick_op)))


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
    if expr.diagonal:
        # Don't optimise further so that the translation to gem at a later can just spill ]1/a_ii[
        return (Action(expr, state.coeff, state.pick_op)
                if self.action and child.children[0].terminal and False
                else expr * state.coeff if state.pick_op else state.coeff * expr)
    else:
        if (self.action and state.coeff                                              # in matrix-free mode
            and isinstance(child, Mul)                                               # when lhs == P.inv * A * x  
            and (isinstance(state.coeff, Mul) or isinstance(state.coeff, Action))):  # and rhs == P.inv * b 
            # turn the inverse into a preconditioned matrix-free solve    
            assert state.pick_op == 1, "This case is not considered in the optimiser yet."
            preconditioner_l, mat = child.children
            preconditioner_r, coeff = state.coeff.children
            assert preconditioner_l == preconditioner_r, "If you want to use a local precondtioner, \
                                                          make sure you multiply with the same operator \
                                                          on the left and on the right."
        else:
            preconditioner_l = None
            mat = child
            coeff = state.coeff
        expr = (Solve(mat, coeff, preconditioner=preconditioner_l) if state.pick_op
                else Transpose(Solve(Transpose(mat), Transpose(coeff), preconditioner=preconditioner_l)))
        # sometimes the solve constructor returns inverses (when the tensors are small enough)
        # so then we do not want to recurse futher into the node
        return expr if isinstance(expr, Mul) else self(expr, ActionBag(None, 1))


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
        # switch the multiplication order with pick_op
        transposed_state = ActionBag(Transpose(state.coeff), flip(state.pick_op))
        # push mul into A with new state
        pushed_expr = self(*expr.children, transposed_state)
        # transpose the end result
        return self(Transpose(pushed_expr), state)
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
    from firedrake import Function

    def make_action(expr, pick_op, matfree):
        # This is a use-case where we generate actions outside of the matrix-free solve
        # reason for which is that we need to let the optimiser run on the actions too
        arbitrary_coeff = AssembledVector(Function(expr.arg_function_spaces[pick_op]))
        A = Action(expr, arbitrary_coeff, pick_op) if matfree else None
        return A

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
        rhs = expr.children[flip(state.pick_op)]
        Aonx = make_action(expr.children[state.pick_op], state.pick_op, self.action)
        Aonp = make_action(expr.children[state.pick_op], state.pick_op, self.action)
        Ponr_pickop = state.pick_op if expr.preconditioner.rank > 1 else 0
        Ponr = make_action(expr.preconditioner, Ponr_pickop, self.action) if expr.preconditioner else None

        swapped_op = Transpose(rhs)
        new_rhs = Transpose(state.coeff)
        pushed_child = self(Solve(mat, new_rhs, matfree=self.action, Aonx=Aonx, Aonp=Aonp,
                                  preconditioner=expr.preconditioner, Ponr=Ponr),
                            ActionBag(None, flip(state.pick_op)))
        return Transpose(self(swapped_op, ActionBag(pushed_child, flip(state.pick_op))))
    else:
        """
        case 1) a)  child 1 is matrix, child2 is vector and there is no coefficient passed through
                b)  child 2 is matrix, child2 is matrix and there is a coefficient passed through
                    ->  multiplication from back
                        A.solve(B)*x = A.inv*B*x = A.inv*(B*x) = A.solve(Bx)
                We always push into the right hand side of the solve.
        """
        mat, rhs = expr.children
        Aonx = make_action(mat, state.pick_op, self.action)
        Aonp = make_action(mat, state.pick_op, self.action)
        if expr.preconditioner:
            Ponr_pickop = state.pick_op if expr.preconditioner.rank > 1 else 0
            Ponr = make_action(expr.preconditioner, Ponr_pickop, self.action)
        else:
            Ponr = None
        return Solve(mat, self(self(rhs, state), state), matfree=self.action, Aonx=Aonx, Aonp=Aonp,
                               preconditioner=expr.preconditioner, Ponr=Ponr)


@_push_mul.register(Mul)
def _push_mul_mul(expr, self, state):
    """ Pushes an multiplication by a coefficent through a multiplication to the innermost node.
        e.g. (A1*A2)*b = A1*(A2*b)

--------case 1) child 1 is matrix, child2 is vector or other way around
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
e.g. ((y.T*(A0+A1))*B)*y = ((y.T*A0+y.T*A1)*B)*y = (y.T*A0+y.T*A1)*(B*y)

--------case 2) child 1 is matrix, child2 is matrix
Examples:

a) (A*B*C)*y
–––––––––––––––––––––––––––––––––––––––––––––––––––––––
        outmost_mul(A*B*C,y)   ---->    A*B*outmost_mul(C,y)   ---->    A*(B*outmost_mul(C,y))

            outmost_mul1                     Mul2                                Mul3
            /      |                      /     |                               /   |
        Mul2       y   ---->           Mul3     outmost_mul1     ---->        A     Mul2
        /  |                          /   |       /    |                           /   |
    Mul3   C                         A    B      C     y                          B    outmost_mul
    /  |                                                                              /         |
    A   B                                                                             C          y

b) y.T*TensorOp(op1, op2)
–––––––––––––––––––––––––––––––––––––––––––––––––––––––
        outmost_mul                               TensorOp
        /      |                                 /       |
    y.T       TensorOp        ---->    outmost_mul      op2
                /    |                    /    |
            op1      op2                y.T   op1

        :arg expr: a Multiplication
        :arg self: a MemoizerArg object.
        :arg state: state carries a coefficient in .coeff,
                    and information if multiply from front (0) or back (1) in .pick_op
        :returns: an optimised Multiplication
    """
    if expr.rank == 1:
        # Optimise the child first that must contain a coefficient
        prio_child, = tuple(filter(lambda child: child.rank == 1, expr.children))
        pick_op = expr.children.index(prio_child)
    else:
        # Optimise the first child first if multiplication happens from front,
        # otw. we we walk into second child first.
        prio_child = expr.children[state.pick_op]
        pick_op = state.pick_op
    other_child = expr.children[flip(pick_op)]
    if state.coeff and expr.rank == 1:
        assert "So far Slate cannot express linear algebra as in case 1b. If that changes, remove the assertion."
        # Optimise child of rank 1 first but do not use result as coefficient.
        coeff = self(prio_child, ActionBag(None, pick_op))
        pushed_other_child = self(other_child, ActionBag(state.coeff, pick_op))
        type = Action if self.action else Mul
        return (type)(pushed_other_child, prio_child) if pick_op else (type)(prio_child, pushed_other_child)
    else:
        # Optimise child of rank 1 first and use result as coefficient
        coeff = self(prio_child, state)
        return self(other_child, ActionBag(coeff, pick_op))


@_push_mul.register(Factorization)
def _push_mul_factorization(expr, self, state):
    """Drop any factorisations. This is needed because whenever
    a Solve node is created so is a Factorization node inside it.
    The Factorization nodes are only needed for the Eigen Backend however."""
    return self(*expr.children, state)
