import ufl
from ufl.algorithms import ReuseTransformer
from ufl.operatorbase import Operator
from pyop2 import op2
import core_types
import cgen


class DummyFunction(ufl.Coefficient):

    """A dummy object to take the place of a Function in the expression. This
    has the sole role of producing the right strings when the expression is
    unparsed and when the arguments are formatted."""

    def __init__(self, function, argnum, intent=op2.READ):
        ufl.Coefficient.__init__(self, function._element)

        self.argnum = argnum
        self.function = function

        # All arguments in expressions are read, except those on the
        # LHS of augmented assignment operators. In those cases, the
        # operator will have to change the intent.
        self.intent = intent

    def __str__(self):
        if isinstance(self.function.function_space(),
                      core_types.VectorFunctionSpace):
            return "fn_%d[dim]" % self.argnum
        else:
            return "fn_%d[0]" % self.argnum

    @property
    def arg(self):
        argtype = self.function.dat.ctype + "*"
        name = " fn_%r" % self.argnum

        return cgen.Value(argtype, name)


class AssignmentBase(Operator):

    """Base class for UFL augmented assignments."""
    __slots__ = ("_operands", "_symbol", "_visit")

    def __init__(self, lhs, rhs):
        self._operands = map(ufl.as_ufl, (lhs, rhs))

        if not (isinstance(lhs, core_types.Function)
                or isinstance(lhs, DummyFunction)):
            raise TypeError("Can only assign to a Function")

    def __str__(self):
        return (" %s " % self._symbol).join(map(str, self._operands))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(repr(o) for o in self._operands))


class Assign(AssignmentBase):

    """A UFL assignment operator."""
    _symbol = "="

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

# UFL class mangling hack
IAdd._uflclass = IAdd


class ISub(AugmentedAssignment):

    """A UFL `-=` operator."""
    _symbol = "-="

# UFL class mangling hack
ISub._uflclass = ISub


class IMul(AugmentedAssignment):

    """A UFL `*=` operator."""
    _symbol = "*="

# UFL class mangling hack
IMul._uflclass = IMul


class IDiv(AugmentedAssignment):

    """A UFL `/=` operator."""
    _symbol = "/="

# UFL class mangling hack
IDiv._uflclass = IDiv


class Power(ufl.algebra.Power):

    """Subclass of :class:`ufl.algebra.Power` which prints pow(x,y)
    instead of x**y."""

    def __str__(self):
        return "pow(%s, %s)" % (str(self._a), str(self._b))


class Ln(ufl.mathfunctions.Ln):

    """Subclass of :class:`ufl.mathfunctions.Ln` which prints log(x)
    instead of ln(x)."""

    def __str__(self):
        return "log(%s)" % str(self._argument)


class ExpressionWalker(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)

        self._args = {}
        self._function_space = None
        self._result = None

    def coefficient(self, o):

        if isinstance(o, core_types.Function):
            if self._function_space is None:
                self._function_space = o._function_space
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

        elif isinstance(o, DummyFunction):
            # Idempotency.
            return o

        else:
            raise TypeError("Operand ", str(o), " is of unsupported type")

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
            operands = map(self.visit, o._operands)

        return o.reconstruct(*operands)


def expression_kernel(expr, args):
    """Produce a :class:`PyOP2.Kernel` from the processed UFL expression
    expr and the corresponding args."""

    fs = args[0].function.function_space()

    body = cgen.Block()

    if isinstance(fs, core_types.VectorFunctionSpace):
        body.extend(
            [cgen.Value("int", "dim"),
             cgen.For("dim = 0",
                      "dim < %s" % fs.dof_dset.cdim,
                      "++dim",
                      cgen.Line(str(expr) + ";")
                      )
             ]
        )
    else:
        body.append(cgen.Line(str(expr) + ";"))

    fdecl = cgen.FunctionDeclaration(
        cgen.Value("void", "expression"),
        [arg.arg for arg in args])

    return op2.Kernel(str(cgen.FunctionBody(fdecl, body)), "expression")


def evaluate_preprocessed_expression(expr, args, subset=None):

    kernel = expression_kernel(expr, args)

    fs = args[0].function._function_space

    parloop_args = [kernel, subset or fs.node_set] +\
        [arg.function.dat(arg.intent)
         for arg in args]

    op2.par_loop(*parloop_args)


def evaluate_expression(expr, subset=None):
    """Evaluates UFL expressions on Functions."""

    e = ExpressionWalker()
    processed_expr = e.visit(expr)

    evaluate_preprocessed_expression(processed_expr, e._args.values(), subset)


def assemble_expression(expr, subset=None):
    """Evaluates UFL expressions on Functions pointwise and assigns
    into a new :class:`Function`."""

    e = ExpressionWalker()
    assign_expr = e.visit(expr)

    result = core_types.Function(e._function_space)

    assign_expr = Assign(result, assign_expr)

    # Revisit to process the Assign.
    assign_expr = e.visit(assign_expr)

    evaluate_preprocessed_expression(assign_expr, e._args.values(), subset)

    return result
