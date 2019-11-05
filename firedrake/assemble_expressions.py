import weakref

import ufl
from ufl.algorithms import ReuseTransformer
from ufl.corealg.map_dag import MultiFunction, map_expr_dag
from ufl.constantvalue import ConstantValue, Zero, IntValue
from ufl.core.multiindex import MultiIndex
from ufl.core.operator import Operator
from ufl.mathfunctions import MathFunction
from ufl.core.ufl_type import ufl_type as orig_ufl_type
from ufl import classes
from collections import defaultdict
import itertools

import loopy
import pymbolic.primitives as p
from pyop2 import op2
from pyop2.profiling import timed_function

from firedrake import constant
from firedrake import function
from firedrake import utils

from functools import singledispatch


def ufl_type(*args, **kwargs):
    r"""Decorator mimicing :func:`ufl.core.ufl_type.ufl_type`.

    Additionally adds the class decorated to the appropriate set of ufl classes."""
    def decorator(cls):
        orig_ufl_type(*args, **kwargs)(cls)
        classes.all_ufl_classes.add(cls)
        if cls._ufl_is_abstract_:
            classes.abstract_classes.add(cls)
        else:
            classes.ufl_classes.add(cls)
        if cls._ufl_is_terminal_:
            classes.terminal_classes.add(cls)
        else:
            classes.nonterminal_classes.add(cls)
        return cls
    return decorator


class IndexRelabeller(MultiFunction):

    def __init__(self):
        super().__init__()
        self._reset()
        self.index_cache = defaultdict(lambda: classes.Index(next(self.count)))

    def _reset(self):
        self.count = itertools.count()

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return type(o)(tuple(self.index_cache[i] if isinstance(i, classes.Index) else i for i in o._indices))


class DummyFunction(ufl.Coefficient):

    r"""A dummy object to take the place of a :class:`.Function` in the
    expression. This has the sole role of producing the right strings
    when the expression is unparsed and when the arguments are
    formatted.
    """

    def __init__(self, function, argnum, intent=op2.READ):
        ufl.Coefficient.__init__(self, function.ufl_function_space())

        self.argnum = argnum
        self.name = "fn_{0}".format(argnum)
        self.function = function

        # All arguments in expressions are read, except those on the
        # LHS of augmented assignment operators. In those cases, the
        # operator will have to change the intent.
        self.intent = intent

    @property
    def arg(self):
        return loopy.GlobalArg(self.name, dtype=self.function.dat.dtype, shape=loopy.auto)


class AssignmentBase(Operator):

    r"""Base class for UFL augmented assignments."""

    __slots__ = ("ufl_shape",)
    _identity = Zero()

    def __init__(self, lhs, rhs):
        operands = list(map(ufl.as_ufl, (lhs, rhs)))
        super(AssignmentBase, self).__init__(operands)
        self.ufl_shape = lhs.ufl_shape
        # Sub function assignment, we've put a Zero in the lhs
        # indicating we should do nothing.
        if type(lhs) is Zero:
            return
        if not (isinstance(lhs, function.Function)
                or isinstance(lhs, DummyFunction)):
            raise TypeError("Can only assign to a Function")

    def __str__(self):
        return (" %s " % self._symbol).join(map(str, self.ufl_operands))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join(repr(o) for o in self.ufl_operands))


@ufl_type(num_ops=2, is_abstract=False, is_index_free=True, is_shaping=False)
class Assign(AssignmentBase):

    r"""A UFL assignment operator."""
    _symbol = "="
    __slots__ = ("ufl_shape",)

    def _visit(self, transformer):
        lhs = self.ufl_operands[0]

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

        return [new_lhs, self.ufl_operands[1]]


class AugmentedAssignment(AssignmentBase):

    r"""Base for the augmented assignment operators `+=`, `-=,` `*=`, `/=`"""
    __slots__ = ()

    def _visit(self, transformer):
        lhs = self.ufl_operands[0]

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

        return [new_lhs, self.ufl_operands[1]]


@ufl_type(num_ops=2, is_abstract=False, is_index_free=True, is_shaping=False)
class IAdd(AugmentedAssignment):

    r"""A UFL `+=` operator."""
    _symbol = "+="
    __slots__ = ()


@ufl_type(num_ops=2, is_abstract=False, is_index_free=True, is_shaping=False)
class ISub(AugmentedAssignment):

    r"""A UFL `-=` operator."""
    _symbol = "-="
    __slots__ = ()


@ufl_type(num_ops=2, is_abstract=False, is_index_free=True, is_shaping=False)
class IMul(AugmentedAssignment):

    r"""A UFL `*=` operator."""
    _symbol = "*="
    _identity = IntValue(1)
    __slots__ = ()


@ufl_type(num_ops=2, is_abstract=False, is_index_free=True, is_shaping=False)
class IDiv(AugmentedAssignment):

    r"""A UFL `/=` operator."""
    _symbol = "/="
    _identity = IntValue(1)
    __slots__ = ()


class ExpressionSplitter(ReuseTransformer):
    r"""Split an expression tree into a subtree for each component of the
    appropriate :class:`.FunctionSpace`."""

    def split(self, expr):
        r"""Split the given expression."""
        self._identity = expr._identity
        self._trees = None
        lhs, rhs = expr.ufl_operands
        # If the expression is not an assignment, the function spaces for both
        # operands have to match
        if not isinstance(expr, AssignmentBase) and \
                lhs.function_space() != rhs.function_space():
            raise ValueError("Operands of %r must have the same FunctionSpace" % expr)
        self._fs = lhs.function_space()
        return [expr._ufl_expr_reconstruct_(*ops) for ops in zip(*map(self.visit, (lhs, rhs)))]

    def indexed(self, o, *operands):
        r"""Reconstruct the :class:`ufl.indexed.Indexed` only if the coefficient
        is defined on a :class:`.FunctionSpace` with rank 1."""
        def reconstruct_if_vec(coeff, idx, i):
            # If the MultiIndex contains a FixedIndex we only want to return
            # the indexed coefficient if its position matches the FixedIndex
            # Since we don't split rank-1 function spaces, we have to
            # reconstruct the fixed index expression for those (and only those)
            if isinstance(idx._indices[0], ufl.core.multiindex.FixedIndex):
                if idx._indices[0]._value != i:
                    return self._identity
                elif coeff.function_space().rank == 1:
                    return o._ufl_expr_reconstruct_(coeff, idx)
                elif coeff.function_space().rank >= 2:
                    raise NotImplementedError("Not implemented for tensor spaces")
            return coeff
        return [reconstruct_if_vec(*ops, i=i)
                for i, ops in enumerate(zip(*operands))]

    def component_tensor(self, o, *operands):
        r"""Only return the first operand."""
        return operands[0]

    def terminal(self, o):
        if isinstance(o, function.Function):
            # A function must either be defined on the same function space
            # we're assigning to, in which case we split it into components
            if o.function_space() == self._fs:
                return o.split()
            # If the function space we're assigning into is /not/
            # Mixed, o must be indexed and the functionspace component
            # much match us.
            if len(self._fs) == 1 and self._fs.index is None:
                idx = o.function_space().index
                if idx is None:
                    raise ValueError("Coefficient %r is not indexed" % o)
                if o.function_space() != self._fs:
                    raise ValueError("Mismatching function spaces")
                return (o,)
            # Otherwise the function space must be indexed and we
            # return the Function for the indexed component and the
            # identity for this assignment for every other
            idx = o.function_space().index
            # LHS is indexed
            if self._fs.index is not None:
                # RHS indexed, indexed RHS function space must match
                # indexed LHS function space.
                if idx is not None and self._fs != o.function_space():
                    raise ValueError("Mismatching indexed function spaces")
                # RHS not indexed, RHS function space must match
                # indexed LHS function space
                elif idx is None and self._fs != o.function_space():
                    raise ValueError("Mismatching function spaces")
                # OK, everything checked out. Return RHS
                return (o,)
            # LHS not indexed, RHS must be indexed and isn't
            if idx is None:
                raise ValueError("Coefficient %r is not indexed" % o)
            # RHS indexed, parent function space must match LHS function space
            if self._fs != o.function_space().parent:
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
            # LHS is mixed and Constant has same shape, use each
            # component in turn to assign to each component of the
            # mixed space.
            if len(self._fs) > 1 and \
               isinstance(o, constant.Constant) and \
               o.ufl_element().value_shape() == self._fs.ufl_element().value_shape():
                offset = 0
                consts = []
                val = o.dat.data_ro
                for fs in self._fs:
                    shp = fs.ufl_element().value_shape()
                    if len(shp) == 0:
                        c = constant.Constant(val[offset], domain=o.ufl_domain())
                        offset += 1
                    elif len(shp) == 1:
                        c = constant.Constant(val[offset:offset+shp[0]],
                                              domain=o.ufl_domain())
                        offset += shp[0]
                    else:
                        raise NotImplementedError("Broadcasting Constant to TFS not implemented")
                    consts.append(c)
                return consts
            # Broadcast value across sub spaces.
            return tuple(o for _ in self._fs)
        raise NotImplementedError("Don't know what to do with %r" % o)

    def product(self, o, *operands):
        r"""Reconstruct a product on each of the component spaces."""
        return [op0 * op1 for op0, op1 in zip(*operands)]

    def operator(self, o, *operands):
        r"""Reconstruct an operator on each of the component spaces."""
        ret = []
        for ops in zip(*operands):
            # Don't try to reconstruct if we've just got the identity
            # Stops domain errors when calling Log on Zero (for example)
            if len(ops) == 1 and type(ops[0]) is type(self._identity):
                ret.append(ops[0])
            else:
                ret.append(o._ufl_expr_reconstruct_(*ops))
        return ret


class ExpressionWalker(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)

        self._args = {}
        self._function_space = None
        self._result = None

    def walk(self, expr):
        r"""Walk the given expression and return a tuple of the transformed
        expression, the list of coefficients sorted by their count and the
        function space the expression is defined on."""
        return (self.visit(expr),
                sorted(self._args.values(), key=lambda c: c.count()),
                self._function_space)

    def coefficient(self, o):

        if isinstance(o, function.Function):
            if self._function_space is None:
                self._function_space = o.function_space()
            else:
                # Peel out (potentially indexed) function space of LHS
                # and RHS to check for compatibility.
                sfs = self._function_space
                ofs = o.function_space()
                if sfs != ofs:
                    raise ValueError("Expression has incompatible function spaces %s and %s" %
                                     (sfs, ofs))
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
            operands = list(map(self.visit, o.ufl_operands))

        return o._ufl_expr_reconstruct_(*operands)


class Bag():
    r"""An empty class which will be used to store arbitrary properties."""
    pass


def expression_kernel(expr, args):
    r"""Produce a :class:`pyop2.Kernel` from the processed UFL expression
    expr and the corresponding args."""

    # Empty slot indicating assignment to indexed LHS, so don't do anything
    if type(expr) is Zero:
        return

    fs = args[0].function.function_space()

    import islpy as isl
    inames = isl.make_zero_and_vars(["d"])
    domain = (inames[0].le_set(inames["d"])) & (inames["d"].lt_set(inames[0] + fs.dof_dset.cdim))

    context = Bag()
    context.within_inames = frozenset(["d"])
    context.indices = (p.Variable("d"),)

    insn = loopy_instructions(expr, context)
    data = [arg.arg for arg in args]
    knl = loopy.make_function([domain], [insn], data, name="expression", silenced_warnings=["summing_if_branches_ops"])

    return op2.Kernel(knl, "expression")


def evaluate_preprocessed_expression(kernel, args, subset=None):
    # We need to splice the args according to the components of the
    # MixedFunctionSpace if we have one
    for j, dats in enumerate(zip(*tuple(a.function.dat for a in args))):

        itset = subset or args[0].function._function_space[j].node_set
        parloop_args = [dat(args[i].intent) for i, dat in enumerate(dats)]
        op2.par_loop(kernel, itset, *parloop_args)


relabeller = IndexRelabeller()


def relabel_indices(expr):
    relabeller._reset()
    return map_expr_dag(relabeller, expr, compress=True)


@utils.known_pyop2_safe
def evaluate_expression(expr, subset=None):
    r"""Evaluates UFL expressions on :class:`.Function`\s."""

    # We cache the generated kernel and the argument list on the
    # result function, keyed on the hash of the expression
    # (implemented by UFL).  Since the argument list references
    # objects that we may want collected, we do a little magic in the
    # cache.  The "function" slot in the DummyFunction argument is
    # replaced by a weakref to the function in the cached arglist.
    # This ensures that we don't leak objects.  However, sometimes, it
    # means that the proxy will have been collected and will be out of
    # date.  So we catch this error and fall back to the slow code
    # path in that case.
    result = expr.ufl_operands[0]
    if result._expression_cache is not None:
        try:
            # Fast path, look for the expression itself
            key = hash(expr)
            vals = result._expression_cache[key]
        except KeyError:
            # Now relabel indices and check
            key2 = hash(relabel_indices(expr))
            try:
                vals = result._expression_cache[key2]
                result._expression_cache[key] = vals
            except KeyError:
                vals = None
        if vals:
            try:
                for k, args in vals:
                    evaluate_preprocessed_expression(k, args, subset=subset)
                return
            except ReferenceError:
                pass
    vals = []
    for tree in ExpressionSplitter().split(expr):
        e, args, _ = ExpressionWalker().walk(tree)
        k = expression_kernel(e, args)
        evaluate_preprocessed_expression(k, args, subset)
        # Replace function slot by weakref to avoid leaking objects
        for a in args:
            a.function = weakref.proxy(a.function)
        vals.append((k, args))
    if result._expression_cache is not None:
        result._expression_cache[key] = vals
        result._expression_cache[key2] = vals


@timed_function("AssembleExpression")
def assemble_expression(expr, subset=None):
    r"""Evaluates UFL expressions on :class:`.Function`\s pointwise and assigns
    into a new :class:`.Function`."""

    result = function.Function(ExpressionWalker().walk(expr)[2])
    evaluate_expression(Assign(result, expr), subset)
    return result


@singledispatch
def loopy_instructions(expr, context):
    raise AssertionError("Unhandled statement type '%s'" % type(expr))


@loopy_instructions.register(Assign)
def loopy_inst_assign(expr, context):
    lhs, rhs = expr.ufl_operands
    lhs = loopy_instructions(lhs, context)
    rhs = loopy_instructions(rhs, context)
    return loopy.Assignment(lhs, rhs, within_inames=context.within_inames)


@loopy_instructions.register(IAdd)
@loopy_instructions.register(ISub)
@loopy_instructions.register(IMul)
@loopy_instructions.register(IDiv)
def loopy_inst_aug_assign(expr, context):
    lhs, rhs = [loopy_instructions(o, context) for o in expr.ufl_operands]
    import operator
    op = {IAdd: operator.add,
          ISub: operator.sub,
          IMul: operator.mul,
          IDiv: operator.truediv}[type(expr)]
    return loopy.Assignment(lhs, op(lhs, rhs), within_inames=context.within_inames)


@loopy_instructions.register(DummyFunction)
def loopy_inst_func(expr, context):
    if (isinstance(expr.function, constant.Constant) and len(expr.function.ufl_element().value_shape()) == 0):
        # Broadcast if constant
        return p.Variable(expr.name).index((0,))
    return p.Variable(expr.name).index(context.indices)


@loopy_instructions.register(ufl.constantvalue.Zero)
def loopy_inst_zero(expr, context):
    # Shape doesn't matter because this turns into a scalar assignment
    # to an indexed expression in loopy.
    return 0


@loopy_instructions.register(ufl.constantvalue.ScalarValue)
def loopy_inst_scalar(expr, context):
    return expr._value


@loopy_instructions.register(ufl.algebra.Product)
@loopy_instructions.register(ufl.algebra.Sum)
@loopy_instructions.register(ufl.algebra.Division)
def loopy_inst_binary(expr, context):
    left, right = [loopy_instructions(o, context) for o in expr.ufl_operands]
    import operator
    op = {ufl.algebra.Sum: operator.add,
          ufl.algebra.Product: operator.mul,
          ufl.algebra.Division: operator.truediv}[type(expr)]
    return op(left, right)


@loopy_instructions.register(MathFunction)
def loopy_inst_mathfunc(expr, context):
    children = [loopy_instructions(o, context) for o in expr.ufl_operands]
    if expr._name == "ln":
        name = "log"
    else:
        name = expr._name
    return p.Variable(name)(*children)


@loopy_instructions.register(ufl.algebra.Power)
def loopy_inst_power(expr, context):
    children = [loopy_instructions(o, context) for o in expr.ufl_operands]
    return p.Power(*children)


@loopy_instructions.register(ufl.algebra.Abs)
def loopy_inst_abs(expr, context):
    child, = [loopy_instructions(o, context) for o in expr.ufl_operands]
    return p.Variable("abs")(child)


@loopy_instructions.register(ufl.classes.MaxValue)
def loopy_inst_max(expr, context):
    children = [loopy_instructions(o, context) for o in expr.ufl_operands]
    return p.Variable("max")(*children)


@loopy_instructions.register(ufl.classes.MinValue)
def loopy_inst_min(expr, context):
    children = [loopy_instructions(o, context) for o in expr.ufl_operands]
    return p.Variable("min")(*children)


@loopy_instructions.register(ufl.classes.Conditional)
def loopy_inst_conditional(expr, context):
    children = [loopy_instructions(o, context) for o in expr.ufl_operands]
    return p.If(*children)


@loopy_instructions.register(ufl.classes.EQ)
@loopy_instructions.register(ufl.classes.NE)
@loopy_instructions.register(ufl.classes.LT)
@loopy_instructions.register(ufl.classes.LE)
@loopy_instructions.register(ufl.classes.GT)
@loopy_instructions.register(ufl.classes.GE)
def loopy_inst_compare(expr, context):
    left, right = [loopy_instructions(o, context) for o in expr.ufl_operands]
    op = expr._name
    return p.Comparison(left, op, right)


@loopy_instructions.register(ufl.classes.ComponentTensor)
@loopy_instructions.register(ufl.classes.Indexed)
def loopy_inst_component_tensor(expr, context):
    # The expression walker just needs the tensor operand for these.
    # The indices are handled elsewhere.
    return loopy_instructions(expr.ufl_operands[0], context)
