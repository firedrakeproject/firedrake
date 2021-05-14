"""Generate loopy kernel from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

import numpy
from functools import singledispatch, partial
from collections import defaultdict, OrderedDict

from gem import gem, impero as imp
from gem.node import Memoizer

import islpy as isl
import loopy as lp

import pymbolic.primitives as p
from loopy.symbolic import SubArrayRef

from pytools import UniqueNameGenerator

from tsfc.parameters import is_complex

from contextlib import contextmanager


maxtype = partial(numpy.find_common_type, [])


@singledispatch
def _assign_dtype(expression, self):
    return maxtype(map(self, expression.children))


@_assign_dtype.register(gem.Terminal)
def _assign_dtype_terminal(expression, self):
    return self.scalar_type


@_assign_dtype.register(gem.Zero)
@_assign_dtype.register(gem.Identity)
@_assign_dtype.register(gem.Delta)
def _assign_dtype_real(expression, self):
    return self.real_type


@_assign_dtype.register(gem.Literal)
def _assign_dtype_identity(expression, self):
    return expression.array.dtype


@_assign_dtype.register(gem.Power)
def _assign_dtype_power(expression, self):
    # Conservative
    return self.scalar_type


@_assign_dtype.register(gem.MathFunction)
def _assign_dtype_mathfunction(expression, self):
    if expression.name in {"abs", "real", "imag"}:
        return self.real_type
    elif expression.name == "sqrt":
        return self.scalar_type
    else:
        return maxtype(map(self, expression.children))


@_assign_dtype.register(gem.MinValue)
@_assign_dtype.register(gem.MaxValue)
def _assign_dtype_minmax(expression, self):
    # UFL did correctness checking
    return self.real_type


@_assign_dtype.register(gem.Conditional)
def _assign_dtype_conditional(expression, self):
    return maxtype(map(self, expression.children[1:]))


@_assign_dtype.register(gem.Comparison)
@_assign_dtype.register(gem.LogicalNot)
@_assign_dtype.register(gem.LogicalAnd)
@_assign_dtype.register(gem.LogicalOr)
def _assign_dtype_logical(expression, self):
    return numpy.int8


def assign_dtypes(expressions, scalar_type):
    """Assign numpy data types to expressions.

    Used for declaring temporaries when converting from Impero to lower level code.

    :arg expressions: List of GEM expressions.
    :arg scalar_type: Default scalar type.

    :returns: list of tuples (expression, dtype)."""
    mapper = Memoizer(_assign_dtype)
    mapper.scalar_type = scalar_type
    if scalar_type.kind == "c":
        mapper.real_type = numpy.finfo(scalar_type).dtype
    else:
        mapper.real_type = scalar_type
    return [(e, mapper(e)) for e in expressions]


class LoopyContext(object):
    def __init__(self):
        self.indices = {}  # indices for declarations and referencing values, from ImperoC
        self.active_indices = {}  # gem index -> pymbolic variable
        self.index_extent = OrderedDict()  # pymbolic variable for indices -> extent
        self.gem_to_pymbolic = {}  # gem node -> pymbolic variable
        self.name_gen = UniqueNameGenerator()

    def fetch_multiindex(self, multiindex):
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

    # Generate index from gem multiindex
    def gem_to_pym_multiindex(self, multiindex):
        indices = []
        for index in multiindex:
            assert index.extent
            if not index.name:
                name = self.name_gen(self.index_names[index])
            else:
                name = index.name
            self.index_extent[name] = index.extent
            indices.append(p.Variable(name))
        return tuple(indices)

    # Generate index from shape
    def pymbolic_multiindex(self, shape):
        indices = []
        for extent in shape:
            name = self.name_gen(self.index_names[extent])
            self.index_extent[name] = extent
            indices.append(p.Variable(name))
        return tuple(indices)

    # Generate pym variable from gem
    def pymbolic_variable_and_destruct(self, node):
        pym = self.pymbolic_variable(node)
        if isinstance(pym, p.Subscript):
            return pym.aggregate, pym.index_tuple
        else:
            return pym, ()

    # Generate pym variable or subscript
    def pymbolic_variable(self, node):
        pym = self._gem_to_pym_var(node)
        if node in self.indices:
            indices = self.fetch_multiindex(self.indices[node])
            if indices:
                return p.Subscript(pym, indices)
        return pym

    def _gem_to_pym_var(self, node):
        try:
            pym = self.gem_to_pymbolic[node]
        except KeyError:
            name = self.name_gen(node.name)
            pym = p.Variable(name)
            self.gem_to_pymbolic[node] = pym
        return pym

    def active_inames(self):
        # Return all active indices
        return frozenset([i.name for i in self.active_indices.values()])


@contextmanager
def active_indices(mapping, ctx):
    """Push active indices onto context.
   :arg mapping: dict mapping gem indices to pymbolic index expressions
   :arg ctx: code generation context.
   :returns: new code generation context."""
    ctx.active_indices.update(mapping)
    yield ctx
    for key in mapping:
        ctx.active_indices.pop(key)


def generate(impero_c, args, scalar_type, kernel_name="loopy_kernel", index_names=[],
             return_increments=True):
    """Generates loopy code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg args: list of loopy.GlobalArgs
    :arg scalar_type: type of scalars as C typename string
    :arg kernel_name: function name of the kernel
    :arg index_names: pre-assigned index names
    :arg return_increments: Does codegen for Return nodes increment the lvalue, or assign?
    :returns: loopy kernel
    """
    ctx = LoopyContext()
    ctx.indices = impero_c.indices
    ctx.index_names = defaultdict(lambda: "i", index_names)
    ctx.epsilon = numpy.finfo(scalar_type).resolution
    ctx.scalar_type = scalar_type
    ctx.return_increments = return_increments

    # Create arguments
    data = list(args)
    for i, (temp, dtype) in enumerate(assign_dtypes(impero_c.temporaries, scalar_type)):
        name = "t%d" % i
        if isinstance(temp, gem.Constant):
            data.append(lp.TemporaryVariable(name, shape=temp.shape, dtype=dtype, initializer=temp.array, address_space=lp.AddressSpace.LOCAL, read_only=True))
        else:
            shape = tuple([i.extent for i in ctx.indices[temp]]) + temp.shape
            data.append(lp.TemporaryVariable(name, shape=shape, dtype=dtype, initializer=None, address_space=lp.AddressSpace.LOCAL, read_only=False))
        ctx.gem_to_pymbolic[temp] = p.Variable(name)

    # Create instructions
    instructions = statement(impero_c.tree, ctx)

    # Create domains
    domains = create_domains(ctx.index_extent.items())

    # Create loopy kernel
    knl = lp.make_function(domains, instructions, data, name=kernel_name, target=lp.CTarget(),
                           seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"],
                           lang_version=(2018, 2))

    # Prevent loopy interchange by loopy
    knl = lp.prioritize_loops(knl, ",".join(ctx.index_extent.keys()))

    return knl


def create_domains(indices):
    """ Create ISL domains from indices

    :arg indices: iterable of (index_name, extent) pairs
    :returns: A list of ISL sets representing the iteration domain of the indices."""

    domains = []
    for idx, extent in indices:
        inames = isl.make_zero_and_vars([idx])
        domains.append(((inames[0].le_set(inames[idx])) & (inames[idx].lt_set(inames[0] + extent))))

    if not domains:
        domains = [isl.BasicSet("[] -> {[]}")]
    return domains


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
    ctx.index_extent[idx] = extent
    with active_indices({tree.index: p.Variable(idx)}, ctx) as ctx_active:
        return statement(tree.children[0], ctx_active)


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
    rhs = expression(leaf.expression, ctx)
    if ctx.return_increments:
        rhs = lhs + rhs
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
        var, index = ctx.pymbolic_variable_and_destruct(expr)
        for multiindex, value in numpy.ndenumerate(expr.array):
            ops.append(lp.Assignment(p.Subscript(var, index + multiindex), expression(value, ctx), within_inames=ctx.active_inames()))
        return ops
    elif isinstance(expr, gem.Constant):
        return []
    elif isinstance(expr, gem.ComponentTensor):
        idx = ctx.gem_to_pym_multiindex(expr.multiindex)
        var, sub_idx = ctx.pymbolic_variable_and_destruct(expr)
        lhs = p.Subscript(var, idx + sub_idx)
        with active_indices(dict(zip(expr.multiindex, idx)), ctx) as ctx_active:
            return [lp.Assignment(lhs, expression(expr.children[0], ctx_active), within_inames=ctx_active.active_inames())]
    elif isinstance(expr, gem.Inverse):
        idx = ctx.pymbolic_multiindex(expr.shape)
        var = ctx.pymbolic_variable(expr)
        lhs = (SubArrayRef(idx, p.Subscript(var, idx)),)

        idx_reads = ctx.pymbolic_multiindex(expr.children[0].shape)
        var_reads = ctx.pymbolic_variable(expr.children[0])
        reads = (SubArrayRef(idx_reads, p.Subscript(var_reads, idx_reads)),)
        rhs = p.Call(p.Variable("inverse"), reads)

        return [lp.CallInstruction(lhs, rhs, within_inames=ctx.active_inames())]
    elif isinstance(expr, gem.Solve):
        idx = ctx.pymbolic_multiindex(expr.shape)
        var = ctx.pymbolic_variable(expr)
        lhs = (SubArrayRef(idx, p.Subscript(var, idx)),)

        reads = []
        for child in expr.children:
            idx_reads = ctx.pymbolic_multiindex(child.shape)
            var_reads = ctx.pymbolic_variable(child)
            reads.append(SubArrayRef(idx_reads, p.Subscript(var_reads, idx_reads)))
        rhs = p.Call(p.Variable("solve"), tuple(reads))

        return [lp.CallInstruction(lhs, rhs, within_inames=ctx.active_inames())]
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
def _expression(expr, ctx):
    raise AssertionError("cannot generate expression from %s" % type(expr))


@_expression.register(gem.Failure)
def _expression_failure(expr, ctx):
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
    if expr.name.startswith('cyl_bessel_'):
        # Bessel functions
        if is_complex(ctx.scalar_type):
            raise NotImplementedError("Bessel functions for complex numbers: "
                                      "missing implementation")
        nu, arg = expr.children
        nu_ = expression(nu, ctx)
        arg_ = expression(arg, ctx)
        # Modified Bessel functions (C++ only)
        #
        # These mappings work for FEniCS only, and fail with Firedrake
        # since no Boost available.
        if expr.name in {'cyl_bessel_i', 'cyl_bessel_k'}:
            name = 'boost::math::' + expr.name
            return p.Variable(name)(nu_, arg_)
        else:
            # cyl_bessel_{jy} -> {jy}
            name = expr.name[-1:]
            if nu == gem.Zero():
                return p.Variable(f"{name}0")(arg_)
            elif nu == gem.one:
                return p.Variable(f"{name}1")(arg_)
            else:
                return p.Variable(f"{name}n")(nu_, arg_)
    else:
        if expr.name == "ln":
            name = "log"
        else:
            name = expr.name
        # Not all mathfunctions apply to complex numbers, but this
        # will be picked up in loopy. This way we allow erf(real(...))
        # in complex mode (say).
        return p.Variable(name)(*(expression(c, ctx) for c in expr.children))


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
def _expression_scalar(expr, ctx):
    assert not expr.shape
    v = expr.value
    if numpy.isnan(v):
        return p.Variable("NAN")
    r = numpy.round(v, 1)
    if r and numpy.abs(v - r) < ctx.epsilon:
        return r
    return v


@_expression.register(gem.Variable)
def _expression_variable(expr, ctx):
    return ctx.pymbolic_variable(expr)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, ctx):
    rank = ctx.fetch_multiindex(expr.multiindex)
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
