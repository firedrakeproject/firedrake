"""Generate loopy kernel from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from math import isnan

import numpy
from functools import singledispatch
from collections import defaultdict, OrderedDict

from gem import gem, impero as imp

import islpy as isl
import loopy as lp

import pymbolic.primitives as p

from pytools import UniqueNameGenerator

from tsfc.parameters import is_complex


class LoopyContext(object):
    def __init__(self):
        self.indices = {}  # indices for declarations and referencing values, from ImperoC
        self.active_indices = {}  # gem index -> pymbolic variable
        self.index_extent = OrderedDict()  # pymbolic variable for indices -> extent
        self.gem_to_pymbolic = {}  # gem node -> pymbolic variable
        self.name_gen = UniqueNameGenerator()

    def pym_multiindex(self, multiindex):
        indices = []
        for index in multiindex:
            if isinstance(index, gem.Index):
                indices.append(self.active_indices[index])
            elif isinstance(index, gem.VariableIndex):
                indices.append(expression(index.expression, self))
            else:
                assert isinstance(index, int)
                indices.append(index)
        return tuple(indices)

    def pymbolic_variable(self, node):
        try:
            pym = self.gem_to_pymbolic[node]
        except KeyError:
            name = self.name_gen(node.name)
            pym = p.Variable(name)
            self.gem_to_pymbolic[node] = pym
        if node in self.indices:
            indices = self.pym_multiindex(self.indices[node])
            if indices:
                return p.Subscript(pym, indices)
            else:
                return pym
        else:
            return pym

    def active_inames(self):
        # Return all active indices
        return frozenset([i.name for i in self.active_indices.values()])


def generate(impero_c, args, precision, scalar_type, kernel_name="loopy_kernel", index_names=[]):
    """Generates loopy code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg args: list of loopy.GlobalArgs
    :arg precision: floating-point precision for printing
    :arg scalar_type: type of scalars as C typename string
    :arg kernel_name: function name of the kernel
    :arg index_names: pre-assigned index names
    :returns: loopy kernel
    """
    ctx = LoopyContext()
    ctx.indices = impero_c.indices
    ctx.index_names = defaultdict(lambda: "i", index_names)
    ctx.precision = precision
    ctx.scalar_type = scalar_type
    ctx.epsilon = 10.0 ** (-precision)

    # Create arguments
    data = list(args)
    for i, temp in enumerate(impero_c.temporaries):
        name = "t%d" % i
        if isinstance(temp, gem.Constant):
            data.append(lp.TemporaryVariable(name, shape=temp.shape, dtype=temp.array.dtype, initializer=temp.array, address_space=lp.AddressSpace.LOCAL, read_only=True))
        else:
            shape = tuple([i.extent for i in ctx.indices[temp]]) + temp.shape
            data.append(lp.TemporaryVariable(name, shape=shape, dtype=numpy.float64, initializer=None, address_space=lp.AddressSpace.LOCAL, read_only=False))
        ctx.gem_to_pymbolic[temp] = p.Variable(name)

    # Create instructions
    instructions = statement(impero_c.tree, ctx)

    # Create domains
    domains = []
    for idx, extent in ctx.index_extent.items():
        inames = isl.make_zero_and_vars([idx])
        domains.append(((inames[0].le_set(inames[idx])) & (inames[idx].lt_set(inames[0] + extent))))

    if not domains:
        domains = [isl.BasicSet("[] -> {[]}")]

    # Create loopy kernel
    knl = lp.make_function(domains, instructions, data, name=kernel_name, target=lp.CTarget(),
                           seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])

    # Prevent loopy interchange by loopy
    knl = lp.prioritize_loops(knl, ",".join(ctx.index_extent.keys()))

    # Help loopy in scheduling by assigning priority to instructions
    insn_new = []
    for i, insn in enumerate(knl.instructions):
        insn_new.append(insn.copy(priority=len(knl.instructions) - i))
    knl = knl.copy(instructions=insn_new)

    return knl


@singledispatch
def statement(tree, ctx):
    """Translates an Impero (sub)tree into a loopy instructions corresponding
    to a C statement.

    :arg tree: Impero (sub)tree
    :arg ctx: miscellaneous code generation data
    :returns: list of loopy instructions
    """
    raise AssertionError("cannot generate loopy from %s" % type(tree))


@statement.register(imp.Block)
def statement_block(tree, ctx):
    from itertools import chain
    return list(chain(*(statement(child, ctx) for child in tree.children)))


@statement.register(imp.For)
def statement_for(tree, ctx):
    extent = tree.index.extent
    assert extent
    idx = ctx.name_gen(ctx.index_names[tree.index])
    ctx.active_indices[tree.index] = p.Variable(idx)
    ctx.index_extent[idx] = extent

    statements = statement(tree.children[0], ctx)

    ctx.active_indices.pop(tree.index)
    return statements


@statement.register(imp.Initialise)
def statement_initialise(leaf, ctx):
    return [lp.Assignment(expression(leaf.indexsum, ctx), 0.0, within_inames=ctx.active_inames())]


@statement.register(imp.Accumulate)
def statement_accumulate(leaf, ctx):
    lhs = expression(leaf.indexsum, ctx)
    rhs = lhs + expression(leaf.indexsum.children[0], ctx)
    return [lp.Assignment(lhs, rhs, within_inames=ctx.active_inames())]


@statement.register(imp.Return)
def statement_return(leaf, ctx):
    lhs = expression(leaf.variable, ctx)
    rhs = lhs + expression(leaf.expression, ctx)
    return [lp.Assignment(lhs, rhs, within_inames=ctx.active_inames())]


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, ctx):
    lhs = expression(leaf.variable, ctx)
    rhs = lhs + expression(leaf.indexsum.children[0], ctx)
    return [lp.Assignment(lhs, rhs, within_inames=ctx.active_inames())]


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, ctx):
    expr = leaf.expression
    if isinstance(expr, gem.ListTensor):
        ops = []
        var = ctx.pymbolic_variable(expr)
        index = ()
        if isinstance(var, p.Subscript):
            var, index = var.aggregate, var.index_tuple
        for multiindex, value in numpy.ndenumerate(expr.array):
            ops.append(lp.Assignment(p.Subscript(var, index + multiindex), expression(value, ctx), within_inames=ctx.active_inames()))
        return ops
    elif isinstance(expr, gem.Constant):
        return []
    else:
        return [lp.Assignment(ctx.pymbolic_variable(expr), expression(expr, ctx, top=True), within_inames=ctx.active_inames())]


def expression(expr, ctx, top=False):
    """Translates GEM expression into a pymbolic expression

    :arg expr: GEM expression
    :arg ctx: miscellaneous code generation data
    :arg top: do not generate temporary reference for the root node
    :returns: pymbolic expression
    """
    if not top and expr in ctx.gem_to_pymbolic:
        return ctx.pymbolic_variable(expr)
    else:
        return _expression(expr, ctx)


@singledispatch
def _expression(expr, parameters):
    raise AssertionError("cannot generate expression from %s" % type(expr))


@_expression.register(gem.Failure)
def _expression_failure(expr, parameters):
    raise expr.exception


@_expression.register(gem.Product)
def _expression_product(expr, ctx):
    return p.Product(tuple(expression(c, ctx) for c in expr.children))


@_expression.register(gem.Sum)
def _expression_sum(expr, ctx):
    return p.Sum(tuple(expression(c, ctx) for c in expr.children))


@_expression.register(gem.Division)
def _expression_division(expr, ctx):
    return p.Quotient(*(expression(c, ctx) for c in expr.children))


@_expression.register(gem.Power)
def _expression_power(expr, ctx):
    return p.Variable("pow")(*(expression(c, ctx) for c in expr.children))


@_expression.register(gem.MathFunction)
def _expression_mathfunction(expr, ctx):

    from tsfc.coffee import math_table

    math_table = math_table.copy()
    math_table['abs'] = ('abs', 'cabs')

    complex_mode = int(is_complex(ctx.scalar_type))

    # Bessel functions
    if expr.name.startswith('cyl_bessel_'):
        if complex_mode:
            msg = "Bessel functions for complex numbers: missing implementation"
            raise NotImplementedError(msg)
        nu, arg = expr.children
        nu_thunk = lambda: expression(nu, ctx)
        arg_loopy = expression(arg, ctx)
        if expr.name == 'cyl_bessel_j':
            if nu == gem.Zero():
                return p.Variable("j0")(arg_loopy)
            elif nu == gem.one:
                return p.Variable("j1")(arg_loopy)
            else:
                return p.Variable("jn")(nu_thunk(), arg_loopy)
        if expr.name == 'cyl_bessel_y':
            if nu == gem.Zero():
                return p.Variable("y0")(arg_loopy)
            elif nu == gem.one:
                return p.Variable("y1")(arg_loopy)
            else:
                return p.Variable("yn")(nu_thunk(), arg_loopy)

        # Modified Bessel functions (C++ only)
        #
        # These mappings work for FEniCS only, and fail with Firedrake
        # since no Boost available.
        if expr.name in ['cyl_bessel_i', 'cyl_bessel_k']:
            name = 'boost::math::' + expr.name
            return p.Variable(name)(nu_thunk(), arg_loopy)

        assert False, "Unknown Bessel function: {}".format(expr.name)

    # Other math functions
    name = math_table[expr.name][complex_mode]
    if name is None:
        raise RuntimeError("{} not supported in complex mode".format(expr.name))

    return p.Variable(name)(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.MinValue)
def _expression_minvalue(expr, ctx):
    return p.Variable("min")(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.MaxValue)
def _expression_maxvalue(expr, ctx):
    return p.Variable("max")(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.Comparison)
def _expression_comparison(expr, ctx):
    left, right = [expression(c, ctx) for c in expr.children]
    return p.Comparison(left, expr.operator, right)


@_expression.register(gem.LogicalNot)
def _expression_logicalnot(expr, ctx):
    return p.LogicalNot(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.LogicalAnd)
def _expression_logicaland(expr, ctx):
    return p.LogicalAnd(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.LogicalOr)
def _expression_logicalor(expr, ctx):
    return p.LogicalOr(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.Conditional)
def _expression_conditional(expr, ctx):
    return p.If(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.Constant)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    v = expr.value
    if isnan(v):
        return p.Variable("NAN")
    r = round(v, 1)
    if r and abs(v - r) < parameters.epsilon:
        return r
    return v


@_expression.register(gem.Variable)
def _expression_variable(expr, ctx):
    return ctx.pymbolic_variable(expr)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, ctx):
    rank = ctx.pym_multiindex(expr.multiindex)
    var = expression(expr.children[0], ctx)
    if isinstance(var, p.Subscript):
        rank = var.index + rank
        var = var.aggregate
    return p.Subscript(var, rank)


@_expression.register(gem.FlexiblyIndexed)
def _expression_flexiblyindexed(expr, ctx):
    var = expression(expr.children[0], ctx)

    rank = []
    for off, idxs in expr.dim2idxs:
        for index, stride in idxs:
            assert isinstance(index, gem.Index)

        rank_ = [off]
        for index, stride in idxs:
            rank_.append(p.Product((ctx.active_indices[index], stride)))
        rank.append(p.Sum(tuple(rank_)))

    return p.Subscript(var, tuple(rank))
