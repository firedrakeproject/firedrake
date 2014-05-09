import ufl
from ufl.algorithms import ReuseTransformer
from ufl.constantvalue import ConstantValue, Zero, IntValue
from ufl.indexing import MultiIndex
from ufl.operatorbase import Operator
from ufl.mathfunctions import MathFunction

import pyop2.coffee.ast_base as ast
from pyop2 import op2

import constant
import function
import functionspace

_to_sum = lambda o: ast.Sum(_ast(o[0]), _to_sum(o[1:])) if len(o) > 1 else _ast(o[0])
_to_prod = lambda o: ast.Prod(_ast(o[0]), _to_sum(o[1:])) if len(o) > 1 else _ast(o[0])

_ast_map = {
    MathFunction: (lambda e: ast.FunCall(e._name, _ast(e._argument)), None),
    ufl.algebra.Sum: (lambda e: _to_sum(e._operands)),
    ufl.algebra.Product: (lambda e: _to_prod(e._operands)),
    ufl.algebra.Division: (lambda e: ast.Div(_ast(e._a), _ast(e._b))),
    ufl.algebra.Abs: (lambda e: ast.FunCall("abs", _ast(e._a))),
    ufl.constantvalue.ScalarValue: (lambda e: ast.Symbol(e._value)),
    ufl.constantvalue.Zero: (lambda e: ast.Symbol(0))
}


def _ast(expr):
    """Convert expr to a PyOP2 ast."""

    try:
        return expr.ast
    except AttributeError:
        for t, f in _ast_map.iteritems():
            if isinstance(expr, t):
                return f(expr)
        raise TypeError("No ast handler for %s" % str(type(expr)))


class DummyFunction(ufl.Coefficient):

    """A dummy object to take the place of a :class:`.Function` in the
    expression. This has the sole role of producing the right strings
    when the expression is unparsed and when the arguments are
    formatted.
    """

    def __init__(self, function, argnum, intent=op2.READ):
        ufl.Coefficient.__init__(self, function._element)

        self.argnum = argnum
        self.function = function

        # All arguments in expressions are read, except those on the
        # LHS of augmented assignment operators. In those cases, the
        # operator will have to change the intent.
        self.intent = intent

    def __str__(self):
        if isinstance(self.function, constant.Constant):
            if len(self.function.ufl_element().value_shape()) == 0:
                return "fn_%d[0]" % self.argnum
            else:
                return "fn_%d[dim]" % self.argnum
        if isinstance(self.function.function_space(),
                      functionspace.VectorFunctionSpace):
            return "fn_%d[dim]" % self.argnum
        else:
            return "fn_%d[0]" % self.argnum

    @property
    def arg(self):
        argtype = self.function.dat.ctype + "*"
        name = " fn_%r" % self.argnum

        return ast.Decl(argtype, ast.Symbol(name))

    @property
    def ast(self):
        # Constant broadcasts across functions if it's a scalar
        if isinstance(self.function, constant.Constant) and \
           len(self.function.ufl_element().value_shape()) == 0:
            return ast.Symbol("fn_%d" % self.argnum, (0, ))
        return ast.Symbol("fn_%d" % self.argnum, ("dim",))


class AssignmentBase(Operator):

    """Base class for UFL augmented assignments."""
    __slots__ = ("_operands", "_symbol", "_ast", "_visit")

    _identity = Zero()

    def __init__(self, lhs, rhs):
        self._operands = map(ufl.as_ufl, (lhs, rhs))

        # Sub function assignment, we've put a Zero in the lhs
        # indicating we should do nothing.
        if type(lhs) is Zero:
            return
        if not (isinstance(lhs, function.Function)
                or isinstance(lhs, DummyFunction)):
            raise TypeError("Can only assign to a Function")

    def operands(self):
        """Return the list of operands."""
        return self._operands

    def __str__(self):
        return (" %s " % self._symbol).join(map(str, self._operands))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(repr(o) for o in self._operands))

    @property
    def ast(self):

        return self._ast(_ast(self._operands[0]), _ast(self._operands[1]))


class Assign(AssignmentBase):

    """A UFL assignment operator."""
    _symbol = "="
    _ast = ast.Assign

    def _visit(self, transformer):
        lhs = self._operands[0]

        transformer._result = lhs

        try:
            # If lhs is int the dictionary, this indicates that it is
            # also on the RHS and therefore needs to be RW.
            new_lhs = transformer._args[lhs]
            new_lhs.intent = op2.RW

        except KeyError:
            if transformer._function_space is None:
                transformer._function_space = lhs._function_space
            elif transformer._function_space != lhs._function_space:
                raise ValueError("Expression has incompatible function spaces")
            transformer._args[lhs] = DummyFunction(lhs, len(transformer._args),
                                                   intent=op2.WRITE)
            new_lhs = transformer._args[lhs]

        return [new_lhs, self._operands[1]]

# UFL class mangling hack
Assign._uflclass = Assign


class AugmentedAssignment(AssignmentBase):

    """Base for the augmented assignment operators `+=`, `-=,` `*=`, `/=`"""

    def _visit(self, transformer):
        lhs = self._operands[0]

        transformer._result = lhs

        try:
            new_lhs = transformer._args[lhs]
        except KeyError:
            if transformer._function_space is None:
                transformer._function_space = lhs._function_space
            elif transformer._function_space != lhs._function_space:
                raise ValueError("Expression has incompatible function spaces")
            transformer._args[lhs] = DummyFunction(lhs, len(transformer._args))
            new_lhs = transformer._args[lhs]

        new_lhs.intent = op2.RW

        return [new_lhs, self._operands[1]]


class IAdd(AugmentedAssignment):

    """A UFL `+=` operator."""
    _symbol = "+="
    _ast = ast.Incr

# UFL class mangling hack
IAdd._uflclass = IAdd


class ISub(AugmentedAssignment):

    """A UFL `-=` operator."""
    _symbol = "-="
    _ast = ast.Decr

# UFL class mangling hack
ISub._uflclass = ISub


class IMul(AugmentedAssignment):

    """A UFL `*=` operator."""
    _symbol = "*="
    _ast = ast.IMul
    _identity = IntValue(1)

# UFL class mangling hack
IMul._uflclass = IMul


class IDiv(AugmentedAssignment):

    """A UFL `/=` operator."""
    _symbol = "/="
    _ast = ast.IDiv
    _identity = IntValue(1)

# UFL class mangling hack
IDiv._uflclass = IDiv


class Power(ufl.algebra.Power):

    """Subclass of :class:`ufl.algebra.Power` which prints pow(x,y)
    instead of x**y."""

    def __str__(self):
        return "pow(%s, %s)" % (str(self._a), str(self._b))

    @property
    def ast(self):
        return ast.FunCall("pow", _ast(self._a), _ast(self._b))


class Ln(ufl.mathfunctions.Ln):

    """Subclass of :class:`ufl.mathfunctions.Ln` which prints log(x)
    instead of ln(x)."""

    def __str__(self):
        return "log(%s)" % str(self._argument)

    @property
    def ast(self):
        return ast.FunCall("log", _ast(self._argument))


class ComponentTensor(ufl.tensors.ComponentTensor):
    """Subclass of :class:`ufl.tensors.ComponentTensor` which only prints the
    first operand."""

    def __str__(self):
        return str(self.operands()[0])

    @property
    def ast(self):
        return _ast(self.operands()[0])


class Indexed(ufl.indexed.Indexed):
    """Subclass of :class:`ufl.indexed.Indexed` which only prints the first
    operand."""

    def __str__(self):
        return str(self.operands()[0])

    @property
    def ast(self):
        return _ast(self.operands()[0])


class ExpressionSplitter(ReuseTransformer):
    """Split an expression tree into a subtree for each component of the
    appropriate :class:`.FunctionSpaceBase`."""

    def split(self, expr):
        """Split the given expression."""
        self._identity = expr._identity
        self._trees = None
        lhs, rhs = expr.operands()
        # If the expression is not an assignment, the function spaces for both
        # operands have to match
        if not isinstance(expr, AssignmentBase) and \
                lhs.function_space() != rhs.function_space():
            raise ValueError("Operands of %r must have the same FunctionSpace" % expr)
        self._fs = lhs.function_space()
        return [expr.reconstruct(*ops) for ops in zip(*map(self.visit, (lhs, rhs)))]

    def indexed(self, o, *operands):
        """Reconstruct the :class:`ufl.indexed.Indexed` only if the coefficient
        is defined on a :class:`.VectorFunctionSpace`."""
        def reconstruct_if_vec(coeff, idx, i):
            # If the MultiIndex contains a FixedIndex we only want to return
            # the indexed coefficient if its position matches the FixedIndex
            # Since we don't split VectorFunctionSpaces, we have to
            # reconstruct the fixed index expression for those (and only those)
            if isinstance(idx._indices[0], ufl.indexing.FixedIndex):
                if idx._indices[0]._value != i:
                    return self._identity
                elif isinstance(coeff.function_space(), functionspace.VectorFunctionSpace):
                    return o.reconstruct(coeff, idx)
            return coeff
        return [reconstruct_if_vec(*ops, i=i)
                for i, ops in enumerate(zip(*operands))]

    def component_tensor(self, o, *operands):
        """Only return the first operand."""
        return operands[0]

    def terminal(self, o):
        if isinstance(o, function.Function):
            # A function must either be defined on the same function space
            # we're assigning to, in which case we split it into components
            if o.function_space() == self._fs:
                return o.split()
            # Otherwise the function space must be indexed and we
            # return the Function for the indexed component and the
            # identity for this assignment for every other
            idx = o.function_space().index
            # LHS is indexed
            if self._fs.index is not None:
                # RHS indexed, indexed RHS function space must match
                # indexed LHS function space.
                if idx is not None and self._fs._fs != o.function_space()._fs:
                    raise ValueError("Mismatching indexed function spaces")
                # RHS not indexed, RHS function space must match
                # indexed LHS function space
                elif idx is None and self._fs._fs != o.function_space():
                    raise ValueError("Mismatching function spaces")
                # OK, everything checked out. Return RHS
                return (o,)
            # LHS not indexed, RHS must be indexed and isn't
            if idx is None:
                raise ValueError("Coefficient %r is not indexed" % o)
            # RHS indexed, parent function space must match LHS function space
            if self._fs != o.function_space()._parent:
                raise ValueError("Mismatching function spaces")
            # Return RHS in index slot in expression and
            # identity otherwise.
            return tuple(o if i == idx else self._identity
                         for i, _ in enumerate(self._fs))
        # We replicate ConstantValue and MultiIndex for each component
        elif isinstance(o, (constant.Constant, ConstantValue, MultiIndex)):
            # If LHS is indexed, only return a scalar result
            if self._fs.index is not None:
                return (o,)
            return tuple(o for _ in self._fs)
        raise NotImplementedError("Don't know what to do with %r" % o)

    def product(self, o, *operands):
        """Reconstruct a product on each of the component spaces."""
        return [op0 * op1 for op0, op1 in zip(*operands)]

    def operator(self, o, *operands):
        """Reconstruct an operator on each of the component spaces."""
        ret = []
        for ops in zip(*operands):
            # Don't try to reconstruct if we've just got the identity
            # Stops domain errors when calling Log on Zero (for example)
            if len(ops) == 1 and type(ops[0]) is type(self._identity):
                ret.append(ops[0])
            else:
                ret.append(o.reconstruct(*ops))
        return ret


class ExpressionWalker(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)

        self._args = {}
        self._function_space = None
        self._result = None

    def walk(self, expr):
        """Walk the given expression and return a tuple of the transformed
        expression, the list of arguments sorted by their count and the
        function space the expression is defined on."""
        return (self.visit(expr),
                sorted(self._args.values(), key=lambda c: c.count()),
                self._function_space)

    def coefficient(self, o):

        if isinstance(o, function.Function):
            if self._function_space is None:
                self._function_space = o._function_space
            elif self._function_space.index is not None:
                # If the LHS is indexed, check compatibility with the
                # underlying fs
                sfs = self._function_space._fs
                ofs = o._function_space
                if o._function_space.index is not None:
                    ofs = self._function_space._fs
                if sfs != ofs:
                    raise ValueError("Expression has incompatible function spaces")
            elif self._function_space != o._function_space:
                raise ValueError("Expression has incompatible function spaces")

            try:
                arg = self._args[o]
                if arg.intent == op2.WRITE:
                    # arg occurs on both the LHS and RHS of an assignment.
                    arg.intent = op2.RW
                return arg
            except KeyError:
                self._args[o] = DummyFunction(o, len(self._args))
                return self._args[o]

        elif isinstance(o, constant.Constant):
            if self._function_space is None:
                raise NotImplementedError("Cannot assign to Constant coefficients")
            else:
                # Constant shape has to match if the constant is not a scalar
                # If it is a scalar, it gets broadcast across all of
                # the values of the function.
                if len(o.ufl_element().value_shape()) > 0:
                    for fs in self._function_space:
                        if fs.ufl_element().value_shape() != o.ufl_element().value_shape():
                            raise ValueError("Constant has mismatched shape for expression function space")
            try:
                arg = self._args[o]
                if arg.intent == op2.WRITE:
                    arg.intent = op2.RW
                return arg
            except KeyError:
                self._args[o] = DummyFunction(o, len(self._args))
                return self._args[o]
        elif isinstance(o, DummyFunction):
            # Idempotency.
            return o

        else:
            raise TypeError("Operand %s is of unsupported type" % o)

    # Prevent AlgebraOperators falling through to the Operator case.
    algebra_operator = ReuseTransformer.reuse_if_possible
    conditional = ReuseTransformer.reuse_if_possible
    condition = ReuseTransformer.reuse_if_possible
    math_function = ReuseTransformer.reuse_if_possible

    def power(self, o, *operands):
        # Need to convert notation to c for exponents.
        return Power(*operands)

    def ln(self, o, *operands):
        # Need to convert notation to c.
        return Ln(*operands)

    def component_tensor(self, o, *operands):
        """Override string representation to only print first operand."""
        return ComponentTensor(*operands)

    def indexed(self, o, *operands):
        """Override string representation to only print first operand."""
        return Indexed(*operands)

    def operator(self, o):

        # Need pre-traversal of operators so as to correctly set the
        # intent of the lhs function of Assignments.
        if isinstance(o, AssignmentBase):
            operands = o._visit(self)
            # The left operand is special-cased in the assignment
            # visit method. The general visitor is applied to the RHS.
            operands = [operands[0], self.visit(operands[1])]

        else:
            # For all other operators, just visit the children.
            operands = map(self.visit, o.operands())

        return o.reconstruct(*operands)


def expression_kernel(expr, args):
    """Produce a :class:`pyop2.Kernel` from the processed UFL expression
    expr and the corresponding args."""

    fs = args[0].function.function_space()

    d = ast.Symbol("dim")
    if isinstance(fs, functionspace.VectorFunctionSpace):
        ast_expr = _ast(expr)
    else:
        ast_expr = ast.FlatBlock(str(expr) + ";")
    body = ast.Block(
        (
            ast.Decl("int", d),
            ast.For(ast.Assign(d, ast.Symbol(0)),
                    ast.Less(d, ast.Symbol(fs.dof_dset.cdim)),
                    ast.Incr(d, ast.Symbol(1)),
                    ast_expr)
        )
    )

    return op2.Kernel(ast.FunDecl("void", "expression",
                                  [arg.arg for arg in args], body),
                      "expression")


def evaluate_preprocessed_expression(expr, args, subset=None):

    # Empty slot indicating assignment to indexed LHS, so don't do anything
    if type(expr) is Zero:
        return
    kernel = expression_kernel(expr, args)

    # We need to splice the args according to the components of the
    # MixedFunctionSpace if we have one
    for j, dats in enumerate(zip(*tuple(a.function.dat for a in args))):

        itset = subset or args[0].function._function_space[j].node_set
        parloop_args = [dat(args[i].intent) for i, dat in enumerate(dats)]
        op2.par_loop(kernel, itset, *parloop_args)


def evaluate_expression(expr, subset=None):
    """Evaluates UFL expressions on :class:`.Function`\s."""

    for tree in ExpressionSplitter().split(expr):
        e, args, _ = ExpressionWalker().walk(tree)
        evaluate_preprocessed_expression(e, args, subset)


def assemble_expression(expr, subset=None):
    """Evaluates UFL expressions on :class:`.Function`\s pointwise and assigns
    into a new :class:`.Function`."""

    result = function.Function(ExpressionWalker().walk(expr)[2])
    evaluate_expression(Assign(result, expr), subset)
    return result
