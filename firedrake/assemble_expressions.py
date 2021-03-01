import itertools
import weakref
from collections import OrderedDict, defaultdict
from functools import singledispatch

import gem
import loopy
import numpy
import ufl
from gem.impero_utils import compile_gem, preprocess_gem
from gem.node import MemoizerArg
from gem.node import traversal as gem_traversal
from pyop2 import op2
from pyop2.sequential import Arg
from tsfc import ufl2gem
from tsfc.loopy import generate
from tsfc.ufl_utils import ufl_reuse_if_untouched
from ufl.algorithms.apply_algebra_lowering import LowerCompoundAlgebra
from ufl.classes import (Coefficient, ComponentTensor, ConstantValue, Expr,
                         Index, Indexed, MultiIndex, Terminal)
from ufl.corealg.map_dag import map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.traversal import unique_pre_traversal as ufl_traversal

import firedrake
from firedrake.utils import ScalarType, cached_property, known_pyop2_safe


def extract_coefficients(expr):
    return tuple(e for e in ufl_traversal(expr) if isinstance(e, ufl.Coefficient))


class Translator(MultiFunction, ufl2gem.Mixin):
    def __init__(self):
        self.varmapping = OrderedDict()
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)

    # Override shape-based things
    # Need to inspect GEM shape not UFL shape, due to Coefficients changing shape.
    def sum(self, o, *ops):
        shape, = set(o.shape for o in ops)
        indices = gem.indices(len(shape))
        return gem.ComponentTensor(gem.Sum(*[gem.Indexed(op, indices) for op in ops]),
                                   indices)

    def real(self, o, expr):
        indices = gem.indices(len(expr.shape))
        return gem.ComponentTensor(gem.MathFunction('real', gem.Indexed(expr, indices)),
                                   indices)

    def imag(self, o, expr):
        indices = gem.indices(len(expr.shape))
        return gem.ComponentTensor(gem.MathFunction('imag', gem.Indexed(expr, indices)),
                                   indices)

    def conj(self, o, expr):
        indices = gem.indices(len(expr.shape))
        return gem.ComponentTensor(gem.MathFunction('conj', gem.Indexed(expr, indices)),
                                   indices)

    def abs(self, o, expr):
        indices = gem.indices(len(expr.shape))
        return gem.ComponentTensor(gem.MathFunction('abs', gem.Indexed(expr, indices)),
                                   indices)

    def conditional(self, o, condition, then, else_):
        assert condition.shape == ()
        shape, = set([then.shape, else_.shape])
        indices = gem.indices(len(shape))
        return gem.ComponentTensor(gem.Conditional(condition, gem.Indexed(then, indices),
                                                   gem.Indexed(else_, indices)),
                                   indices)

    def indexed(self, o, aggregate, index):
        return gem.Indexed(aggregate, index[:len(aggregate.shape)])

    def index_sum(self, o, summand, indices):
        index, = indices
        indices = gem.indices(len(summand.shape))
        return gem.ComponentTensor(gem.IndexSum(gem.Indexed(summand, indices), (index,)),
                                   indices)

    def component_tensor(self, o, expression, index):
        index = tuple(i for i in index if i in expression.free_indices)
        return gem.ComponentTensor(expression, index)

    def expr(self, o):
        raise ValueError(f"Expression of type {type(o)} unsupported in pointwise expressions")

    def coefficient(self, o):
        # Because we act on dofs, the ufl_shape is not the right thing to check
        shape = o.dat.dim
        try:
            var = self.varmapping[o]
        except KeyError:
            name = f"C{len(self.varmapping)}"
            var = gem.Variable(name, shape)
            self.varmapping[o] = var
        if o.ufl_shape == ():
            assert shape == (1, )
            return gem.Indexed(var, (0, ))
        else:
            return var


class IndexRelabeller(MultiFunction):
    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        count = itertools.count()
        self.index_cache = defaultdict(lambda: Index(next(count)))

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return type(o)(tuple(self.index_cache[i] if isinstance(i, Index) else i
                             for i in o.indices()))


def flatten(shape):
    if shape == ():
        return shape
    else:
        return (numpy.prod(shape, dtype=int), )


def reshape(expr, shape):
    if numpy.prod(expr.ufl_shape, dtype=int) != numpy.prod(shape, dtype=int):
        raise ValueError(f"Can't reshape from {expr.ufl_shape} to {shape}")
    if shape == expr.ufl_shape:
        return expr
    if shape == ():
        return expr
    else:
        expr = numpy.asarray([expr[i] for i in numpy.ndindex(expr.ufl_shape)])
        return ufl.as_tensor(expr.reshape(shape))


@singledispatch
def _split(o, self, inct):
    raise AssertionError(f"Unhandled expression type {type(o)} in splitting")


@_split.register(Expr)
def _split_expr(o, self, inct):
    return tuple(ufl_reuse_if_untouched(o, *ops)
                 for ops in zip(*(self(op, inct) for op in o.ufl_operands)))


@_split.register(Coefficient)
def _split_coefficient(o, self, inct):
    if isinstance(o, firedrake.Constant):
        return tuple(o for _ in range(self.n))
    else:
        split = o.split()
        assert len(split) == self.n
        # Reshaping to handle tensor/vector confusion.
        return tuple(reshape(s, flatten(s.ufl_shape)) for s in split)


@_split.register(Terminal)
def _split_terminal(o, self, inct):
    return tuple(o for _ in range(self.n))


@_split.register(ComponentTensor)
def _split_component_tensor(o, self, inct):
    expressions, multiindices = (self(op, True) for op in o.ufl_operands)
    result = []
    shape_indices = set(i.count() for i in multiindices[0].indices())
    for expression, multiindex in zip(expressions, multiindices):
        if shape_indices <= set(expression.ufl_free_indices):
            result.append(ufl_reuse_if_untouched(o, expression, multiindex))
        else:
            result.append(expression)
    return tuple(result)


@_split.register(Indexed)
def _split_indexed(o, self, inct):
    aggregate, multiindex = o.ufl_operands
    indices = multiindex.indices()
    result = []
    for agg in self(aggregate, False):
        ncmp = len(agg.ufl_shape)
        if ncmp == 0:
            result.append(agg)
        elif not inct:
            idx = indices[:ncmp]
            indices = indices[ncmp:]
            mi = multiindex if multiindex.indices() == idx else MultiIndex(idx)
            result.append(ufl_reuse_if_untouched(o, agg, mi))
        else:
            # shape and inct
            aggshape = (flatten(agg.ufl_shape)
                        + tuple(itertools.repeat(1, len(aggregate.ufl_shape) - 1)))
            agg = reshape(agg, aggshape)
            result.append(ufl_reuse_if_untouched(o, agg, multiindex))
    return tuple(result)


class Assign(object):
    """Representation of a pointwise assignment expression."""
    relabeller = IndexRelabeller()
    symbol = "="

    __slots__ = ("lvalue", "rvalue", "__dict__", "__weakref__")

    def __init__(self, lvalue, rvalue):
        """
        :arg lvalue: The coefficient to assign into.
        :arg rvalue: The pointwise expression.
        """
        if not isinstance(lvalue, ufl.Coefficient):
            raise ValueError("lvalue for pointwise assignment must be a coefficient")
        self.lvalue = lvalue
        self.rvalue = ufl.as_ufl(rvalue)
        n = len(self.lvalue.function_space())
        if n > 1:
            self.splitter = MemoizerArg(_split)
            self.splitter.n = n

    def __str__(self):
        return f"{self.lvalue} {self.symbol} {self.rvalue}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.lvalue!r}, {self.rvalue!r})"

    @cached_property
    def coefficients(self):
        """Tuple of coefficients involved in the assignment."""
        return (self.lvalue, ) + tuple(c for c in self.rcoefficients if c.dat != self.lvalue.dat)

    @cached_property
    def rcoefficients(self):
        """Coefficients appearing in the rvalue."""
        return extract_coefficients(self.rvalue)

    @cached_property
    def split(self):
        """A tuple of assignment expressions, separated by subspace for mixed spaces."""
        V = self.lvalue.function_space()
        if len(V) > 1:
            # rvalue cases we handle for mixed:
            # 1. rvalue is a scalar constant (broadcast to all subspaces)
            # 2. rvalue is a function in the same mixed space (actually
            #    handled by copy special-case in function.assign)
            # 3. rvalue is has indexed subspaces and all indices are
            #    the same (assign to that subspace of the output mixed
            #    space)
            # 4. rvalue is an expression only over mixed spaces and
            #    the spaces match (split and evaluate subspace-wise).
            spaces = tuple(c.function_space() for c in self.rcoefficients)
            indices = set(s.index for s in spaces if s is not None)
            if len(indices) == 0:
                # rvalue is some combination of constants
                if self.rvalue.ufl_shape != ():
                    raise ValueError("Can only broadcast scalar constants to "
                                     "mixed spaces in pointwise assignment")
                return tuple(type(self)(s, self.rvalue) for s in self.lvalue.split())
            else:
                if indices == set([None]):
                    if len((set(spaces) | {V}) - {None}) != 1:
                        # Check that there were no unindexed coefficients
                        raise ValueError("Saw indexed coefficients in rvalue, "
                                         "perhaps you meant to index the lvalue with .sub(...)")
                    rvalues = self.splitter(self.rvalue, False)
                    return tuple(type(self)(lvalue, rvalue)
                                 for lvalue, rvalue in zip(self.lvalue.split(), rvalues))
                elif indices & set([None]):
                    raise ValueError("Either all or non of the rvalue coefficients must have "
                                     "a .sub(...) index")
                try:
                    index, = indices
                except ValueError:
                    raise ValueError("All rvalue coefficients must have the same .sub(...) index")
                return (type(self)(self.lvalue.sub(index), self.rvalue), )
        else:
            return (weakref.proxy(self), )

    @property
    @known_pyop2_safe
    def args(self):
        """Tuple of par_loop arguments for the expression."""
        args = []
        if self.lvalue in self.rcoefficients:
            args.append(Arg(weakref.ref(self.lvalue.dat), access=op2.RW))
        else:
            args.append(Arg(weakref.ref(self.lvalue.dat), access=op2.WRITE))
        for c in self.rcoefficients:
            if c.dat == self.lvalue.dat:
                continue
            args.append(Arg(weakref.ref(c.dat), access=op2.READ))
        return tuple(args)

    @cached_property
    def iterset(self):
        return weakref.proxy(self.lvalue.node_set)

    @cached_property
    def fast_key(self):
        """A fast lookup key for this expression."""
        return (type(self), hash(self.lvalue), hash(self.rvalue))

    @cached_property
    def slow_key(self):
        """A slow lookup key for this expression (relabelling UFL indices)."""
        self.relabeller._reset()
        rvalue, = map_expr_dags(self.relabeller, [self.rvalue])
        return (type(self), hash(self.lvalue), hash(rvalue))

    @cached_property
    def par_loop_args(self):
        """Arguments for a parallel loop to evaluate this expression.

        If the expression is over a mixed space, this merges kernels
        for subspaces with the same node_set (resulting in fewer
        par_loop calls).
        """
        result = []
        grouping = OrderedDict()
        for e in self.split:
            grouping.setdefault(e.lvalue.node_set, []).append(e)
        for iterset, exprs in grouping.items():
            k, args = pointwise_expression_kernel(exprs, ScalarType)
            result.append((k, iterset, tuple(args)))
        return tuple(result)


class AugmentedAssign(Assign):
    """Base class for augmented pointwise assignment."""


class IAdd(AugmentedAssign):
    symbol = "+="


class ISub(AugmentedAssign):
    symbol = "-="


class IMul(AugmentedAssign):
    symbol = "*="


class IDiv(AugmentedAssign):
    symbol = "/="


def compile_to_gem(expr, translator):
    """Compile a single pointwise expression to GEM.

    :arg expr: The expression to compile.
    :arg translator: a :class:`Translator` instance.
    :returns: A (lvalue, rvalue) pair of preprocessed GEM."""
    if not isinstance(expr, Assign):
        raise ValueError(f"Don't know how to assign expression of type {type(expr)}")
    spaces = tuple(c.function_space() for c in expr.coefficients)
    if any(type(s.ufl_element()) is ufl.MixedElement for s in spaces if s is not None):
        raise ValueError("Not expecting a mixed space at this point, "
                         "did you forget to index a function with .sub(...)?")
    if len(set(s.ufl_element() for s in spaces if s is not None)) != 1:
        raise ValueError("All coefficients must be defined on the same space")
    lvalue = expr.lvalue
    rvalue = expr.rvalue
    broadcast = isinstance(rvalue, (firedrake.Constant, ConstantValue)) and rvalue.ufl_shape == ()
    if not broadcast and lvalue.ufl_shape != rvalue.ufl_shape:
        try:
            rvalue = reshape(rvalue, lvalue.ufl_shape)
        except ValueError:
            raise ValueError("Mismatching shapes between lvalue and rvalue in pointwise assignment")
    rvalue, = map_expr_dags(LowerCompoundAlgebra(), [rvalue])
    try:
        lvalue, rvalue = map_expr_dags(translator, [lvalue, rvalue])
    except (AssertionError, ValueError):
        raise ValueError("Mismatching shapes in pointwise assignment. "
                         "For intrinsically vector-/tensor-valued spaces make "
                         "sure you're not using shaped Constants or literals.")

    indices = gem.indices(len(lvalue.shape))
    if not broadcast:
        if rvalue.shape != lvalue.shape:
            raise ValueError("Mismatching shapes in pointwise assignment. "
                             "For intrinsically vector-/tensor-valued spaces make "
                             "sure you're not using shaped Constants or literals.")
        rvalue = gem.Indexed(rvalue, indices)
    lvalue = gem.Indexed(lvalue, indices)
    if isinstance(expr, IAdd):
        rvalue = gem.Sum(lvalue, rvalue)
    elif isinstance(expr, ISub):
        rvalue = gem.Sum(lvalue, gem.Product(gem.Literal(-1), rvalue))
    elif isinstance(expr, IMul):
        rvalue = gem.Product(lvalue, rvalue)
    elif isinstance(expr, IDiv):
        rvalue = gem.Division(lvalue, rvalue)
    return preprocess_gem([lvalue, rvalue])


def pointwise_expression_kernel(exprs, scalar_type):
    """Compile a kernel for pointwise expressions.

    :arg exprs: List of expressions, all on the same iteration set.
    :arg scalar_type: Default scalar type (numpy.dtype).
    :returns: a PyOP2 kernel for evaluation of the expressions."""
    if len(set(e.lvalue.node_set for e in exprs)) > 1:
        raise ValueError("All expressions must have same node layout.")
    translator = Translator()
    assignments = tuple(compile_to_gem(expr, translator) for expr in exprs)
    prefix_ordering = tuple(OrderedDict.fromkeys(itertools.chain.from_iterable(
        node.index_ordering()
        for node in gem_traversal([v for v, _ in assignments])
        if isinstance(node, gem.Indexed))))
    impero_c = compile_gem(assignments, prefix_ordering=prefix_ordering,
                           remove_zeros=False, emit_return_accumulate=False)
    coefficients = translator.varmapping
    args = []
    plargs = []
    for expr in exprs:
        for c, arg in zip(expr.coefficients, expr.args):
            try:
                var = coefficients.pop(c)
            except KeyError:
                continue
            plargs.append(arg)
            args.append(loopy.GlobalArg(var.name, shape=var.shape, dtype=c.dat.dtype))
    assert len(coefficients) == 0
    knl = generate(impero_c, args, scalar_type, kernel_name="expression_kernel",
                   return_increments=False)
    return firedrake.op2.Kernel(knl, knl.name), plargs


class dereffed(object):
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        for a in self.args:
            data = a.data()
            if data is None:
                raise ReferenceError
            a.data = a.data()
        return self.args

    def __exit__(self, *args, **kwargs):
        for a in self.args:
            a.data = weakref.ref(a.data)


@known_pyop2_safe
def evaluate_expression(expr, subset=None):
    """Evaluate a pointwise expression.

    :arg expr: The expression to evaluate.
    :arg subset: An optional subset to apply the expression on.
    :returns: The lvalue in the provided expression."""
    lvalue = expr.lvalue
    cache = lvalue._expression_cache
    if cache is not None:
        fast_key = expr.fast_key
        try:
            arguments = cache[fast_key]
        except KeyError:
            slow_key = expr.slow_key
            try:
                arguments = cache[slow_key]
                cache[fast_key] = arguments
            except KeyError:
                arguments = None
        if arguments is not None:
            try:
                for kernel, iterset, args in arguments:
                    with dereffed(args) as args:
                        firedrake.op2.par_loop(kernel, subset or iterset, *args)
                return lvalue
            except ReferenceError:
                # TODO: Is there a situation where some of the kernels
                # succeed and others don't?
                pass
    arguments = expr.par_loop_args
    if cache is not None:
        cache[slow_key] = arguments
        cache[fast_key] = arguments
    for kernel, iterset, args in arguments:
        with dereffed(args) as args:
            firedrake.op2.par_loop(kernel, subset or iterset, *args)
    return lvalue


def assemble_expression(expr, subset=None):
    """Evaluate a UFL expression pointwise and assign it to a new
    :class:`~.Function`.

    :arg expr: The UFL expression.
    :arg subset: Optional subset to apply the expression on.
    :returns: A new function."""
    try:
        coefficients = extract_coefficients(expr)
        V, = set(c.function_space() for c in coefficients) - {None}
    except ValueError:
        raise ValueError("Cannot deduce correct target space from pointwise expression")
    result = firedrake.Function(V)
    return evaluate_expression(Assign(result, expr), subset)
