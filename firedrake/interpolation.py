from __future__ import absolute_import

import numpy
from functools import partial

import FIAT
import ufl

from coffee import base as ast
from pyop2 import op2

import firedrake
from firedrake import utils, conditional

__all__ = ("interpolate", "Interpolator")


def interpolate(expr, V, subset=None, access=None):
    """Interpolate an expression onto a new function in V.

    :arg expr: an :class:`.Expression`.
    :arg V: the :class:`.FunctionSpace` to interpolate into (or else
        an existing :class:`.Function`).
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
        interpolation over.

    Returns a new :class:`.Function` in the space ``V`` (or ``V`` if
    it was a Function).

    .. note::

       If you find interpolating the same expression again and again
       (for example in a time loop) you may find you get better
       performance by using a :class:`Interpolator` instead.
    """
    return Interpolator(expr, V, subset=subset, access=access).interpolate()


class Interpolator(object):
    """A reusable interpolation object.

    :arg expr: The expression to interpolate.
    :arg V: The :class:`.FunctionSpace` or :class:`.Function` to
        interpolate into.

    This object can be used to carry out the same interpolation
    multiple times (for example in a timestepping loop).

    .. note::

       The :class:`Interpolator` holds a reference to the provided
       arguments (such that they won't be collected until the
       :class:`Interpolator` is also collected).
    """
    def __init__(self, expr, V, subset=None, access=None):
        self.callable = make_interpolator(expr, V, subset, access)

    @utils.known_pyop2_safe
    def interpolate(self):
        """Compute the interpolation.

        :returns: The resulting interpolated :class:`.Function`.
        """
        return self.callable()


class SubExpression(object):
    """A helper class for interpolating onto mixed functions.

    Allows using the user arguments from a provided expression, but
    overrides the code to pull out.
    """
    def __init__(self, expr, idx, shape):
        self._expr = expr
        self.code = numpy.array(expr.code[idx]).flatten()
        self._shape = shape
        self.ufl_shape = shape

    def value_shape(self):
        return self._shape

    def rank(self):
        return len(self.ufl_shape)

    def __getattr__(self, name):
        return getattr(self._expr, name)


def make_interpolator(expr, V, subset, access=None):
    assert isinstance(expr, ufl.classes.Expr)

    if isinstance(V, firedrake.Function):
        f = V
        V = f.function_space()
    else:
        f = firedrake.Function(V)

    # Make sure we have an expression of the right length i.e. a value for
    # each component in the value shape of each function space
    dims = [numpy.prod(fs.ufl_element().value_shape(), dtype=int)
            for fs in V]
    loops = []
    if numpy.prod(expr.ufl_shape, dtype=int) != sum(dims):
        raise RuntimeError('Expression of length %d required, got length %d'
                           % (sum(dims), numpy.prod(expr.ufl_shape, dtype=int)))

    if not isinstance(expr, firedrake.Expression):
        if len(V) > 1:
            raise NotImplementedError(
                "UFL expressions for mixed functions are not yet supported.")
        if access is not None:
            # check upward propagation of these functions
            # raise appropriate not implemented errors if not in MIN/Max Case
            from operator import lt, gt
            if access not in [op2.MIN, op2.MAX]:
                raise NotImplementedError("Only MIN and MAX are supported accesses")
            op = {op2.MIN: lt, op2.MAX: gt}[access]
            expr = conditional(op(f, expr), f, expr)
        loops.append(_interpolator(V, f.dat, expr, subset, access=access))
    elif access is not None:
        raise NotImplementedError("Access is only defined for UFL expressions")
    elif hasattr(expr, 'eval'):
        if len(V) > 1:
            raise NotImplementedError(
                "Python expressions for mixed functions are not yet supported.")
        loops.append(_interpolator(V, f.dat, expr, subset))
    else:
        # Slice the expression and pass in the right number of values for
        # each component function space of this function
        d = 0
        for fs, dat, dim in zip(V, f.dat, dims):
            idx = d if fs.rank == 0 else slice(d, d+dim)
            loops.append(_interpolator(fs, dat,
                                       SubExpression(expr, idx, fs.ufl_element().value_shape()),
                                       subset))
            d += dim

    def callable(loops, f):
        for l in loops:
            l()
        return f

    return partial(callable, loops, f)


def _interpolator(V, dat, expr, subset, access=None):
    to_element = V.fiat_element
    to_pts = []

    if V.ufl_element().mapping() != "identity":
        raise NotImplementedError("Can only interpolate onto elements "
                                  "with affine mapping. Try projecting instead")

    for dual in to_element.dual_basis():
        if not isinstance(dual, FIAT.functional.PointEvaluation):
            raise NotImplementedError("Can only interpolate onto point "
                                      "evaluation operators. Try projecting instead")
        to_pts.append(dual.pt_dict.keys()[0])

    if len(expr.ufl_shape) != len(V.ufl_element().value_shape()):
        raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                           % (len(expr.ufl_shape), len(V.ufl_element().value_shape())))

    if expr.ufl_shape != V.ufl_element().value_shape():
        raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                           % (expr.ufl_shape, V.ufl_element().value_shape()))

    mesh = V.ufl_domain()
    coords = mesh.coordinates

    if not isinstance(expr, (firedrake.Expression, SubExpression)):
        kernel, oriented, coefficients = compile_ufl_kernel(expr, to_pts, to_element, V)
        flatten = True
        indexed = True
    elif hasattr(expr, "eval"):
        kernel, oriented, coefficients = compile_python_kernel(expr, to_pts, to_element, V, coords)
        flatten = False
        indexed = False
    elif expr.code is not None:
        kernel, oriented, coefficients = compile_c_kernel(expr, to_pts, to_element, V, coords)
        flatten = False
        indexed = True
    else:
        raise RuntimeError("Attempting to evaluate an Expression which has no value.")

    cell_set = coords.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset
    args = [kernel, cell_set]

    if indexed:
        args.append(dat(op2.WRITE, V.cell_node_map()[op2.i[0]]))
    else:
        args.append(dat(op2.WRITE, V.cell_node_map()))
    if oriented:
        co = mesh.cell_orientations()
        args.append(co.dat(op2.READ, co.cell_node_map(), flatten=flatten))
    for coefficient in coefficients:
        args.append(coefficient.dat(op2.READ, coefficient.cell_node_map(), flatten=flatten))

    return partial(op2.par_loop, *args)


def compile_ufl_kernel(expression, to_pts, to_element, fs):
    import collections
    from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
    from ufl.algorithms.apply_derivatives import apply_derivatives
    from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
    from ufl.algorithms import extract_arguments, extract_coefficients
    from gem import gem, impero_utils
    from tsfc import fem, ufl_utils
    from tsfc.coffee import generate as generate_coffee
    from tsfc.kernel_interface import (KernelBuilderBase,
                                       needs_cell_orientations,
                                       cell_orientations_coffee_arg)

    # Imitate the compute_form_data processing pipeline
    #
    # Unfortunately, we cannot call compute_form_data here, since
    # we only have an expression, not a form
    expression = apply_algebra_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_function_pullbacks(expression)
    expression = apply_geometry_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_geometry_lowering(expression)
    expression = apply_derivatives(expression)

    # Replace coordinates (if any)
    if expression.ufl_domain():
        assert fs.mesh() == expression.ufl_domain()
        expression = ufl_utils.replace_coordinates(expression, fs.mesh().coordinates)

    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    builder = KernelBuilderBase()
    args = []

    coefficients = extract_coefficients(expression)
    for i, coefficient in enumerate(coefficients):
        args.append(builder.coefficient(coefficient, "w_%d" % i))

    point_index = gem.Index(name='p')
    ir = fem.compile_ufl(expression,
                         cell=fs.mesh().ufl_cell(),
                         points=to_pts,
                         point_index=point_index,
                         coefficient_mapper=builder.coefficient_mapper)
    assert len(ir) == 1

    # Deal with non-scalar expressions
    tensor_indices = ()
    if fs.shape:
        tensor_indices = tuple(gem.Index() for s in fs.shape)
        ir = [gem.Indexed(ir[0], tensor_indices)]

    # Build kernel body
    return_var = gem.Variable('A', (len(to_pts),) + fs.shape)
    return_expr = gem.Indexed(return_var, (point_index,) + tensor_indices)
    impero_c = impero_utils.compile_gem([return_expr], ir, [point_index])
    body = generate_coffee(impero_c, index_names={point_index: 'p'})

    oriented = needs_cell_orientations(ir)
    if oriented:
        args.insert(0, cell_orientations_coffee_arg)

    # Build kernel
    args.insert(0, ast.Decl("double", ast.Symbol('A', rank=(len(to_pts),) + fs.shape)))
    kernel_code = builder.construct_kernel("expression_kernel", args, body)

    return op2.Kernel(kernel_code, kernel_code.name), oriented, coefficients


class GlobalWrapper(object):
    """Wrapper object that fakes a Global to behave like a Function."""
    def __init__(self, glob):
        self.dat = glob
        self.cell_node_map = lambda *args: None


def compile_python_kernel(expression, to_pts, to_element, fs, coords):
    """Produce a :class:`PyOP2.Kernel` wrapping the eval method on the
    function provided."""

    coords_space = coords.function_space()
    coords_element = coords_space.fiat_element

    X_remap = coords_element.tabulate(0, to_pts).values()[0]

    # The par_loop will just pass us arguments, since it doesn't
    # know about keyword args at all so unpack into a dict that we
    # can pass to the user's eval method.
    def kernel(output, x, *args):
        kwargs = {}
        for (slot, _), arg in zip(expression._user_args, args):
            kwargs[slot] = arg
        X = numpy.dot(X_remap.T, x)

        for i in range(len(output)):
            # Pass a slice for the scalar case but just the
            # current vector in the VFS case. This ensures the
            # eval method has a Dolfin compatible API.
            expression.eval(output[i:i+1, ...] if numpy.rank(output) == 1 else output[i, ...],
                            X[i:i+1, ...] if numpy.rank(X) == 1 else X[i, ...], **kwargs)

    coefficients = [coords]
    for _, arg in expression._user_args:
        coefficients.append(GlobalWrapper(arg))
    return kernel, False, tuple(coefficients)


def compile_c_kernel(expression, to_pts, to_element, fs, coords):
    """Produce a :class:`PyOP2.Kernel` from the c expression provided."""

    coords_space = coords.function_space()
    coords_element = coords_space.fiat_element

    names = {v[0] for v in expression._user_args}

    X = coords_element.tabulate(0, to_pts).values()[0]

    # Produce C array notation of X.
    X_str = "{{"+"},\n{".join([",".join(map(str, x)) for x in X.T])+"}}"

    A = utils.unique_name("A", names)
    X = utils.unique_name("X", names)
    x_ = utils.unique_name("x_", names)
    k = utils.unique_name("k", names)
    d = utils.unique_name("d", names)
    i_ = utils.unique_name("i", names)
    # x is a reserved name.
    x = "x"
    if "x" in names:
        raise ValueError("cannot use 'x' as a user-defined Expression variable")
    ass_exp = [ast.Assign(ast.Symbol(A, (k,), ((len(expression.code), i),)),
                          ast.FlatBlock("%s" % code))
               for i, code in enumerate(expression.code)]
    vals = {
        "X": X,
        "x": x,
        "x_": x_,
        "k": k,
        "d": d,
        "i": i_,
        "x_array": X_str,
        "dim": coords_space.dim,
        "xndof": coords_element.space_dimension(),
        # FS will always either be a functionspace or
        # vectorfunctionspace, so just accessing dim here is safe
        # (we don't need to go through ufl_element.value_shape())
        "nfdof": to_element.space_dimension() * numpy.prod(fs.dim, dtype=int),
        "ndof": to_element.space_dimension(),
        "assign_dim": numpy.prod(expression.value_shape(), dtype=int)
    }
    init = ast.FlatBlock("""
const double %(X)s[%(ndof)d][%(xndof)d] = %(x_array)s;

double %(x)s[%(dim)d];
const double pi = 3.141592653589793;

""" % vals)
    block = ast.FlatBlock("""
for (unsigned int %(d)s=0; %(d)s < %(dim)d; %(d)s++) {
  %(x)s[%(d)s] = 0;
  for (unsigned int %(i)s=0; %(i)s < %(xndof)d; %(i)s++) {
        %(x)s[%(d)s] += %(X)s[%(k)s][%(i)s] * %(x_)s[%(i)s][%(d)s];
  };
};

""" % vals)
    loop = ast.c_for(k, "%(ndof)d" % vals, ast.Block([block] + ass_exp,
                                                     open_scope=True))
    user_args = []
    user_init = []
    for _, arg in expression._user_args:
        if arg.shape == (1, ):
            user_args.append(ast.Decl("double *", "%s_" % arg.name))
            user_init.append(ast.FlatBlock("const double %s = *%s_;" %
                                           (arg.name, arg.name)))
        else:
            user_args.append(ast.Decl("double *", arg.name))
    kernel_code = ast.FunDecl("void", "expression_kernel",
                              [ast.Decl("double", ast.Symbol(A, (int("%(nfdof)d" % vals),))),
                               ast.Decl("double**", x_)] + user_args,
                              ast.Block(user_init + [init, loop],
                                        open_scope=False))
    coefficients = [coords]
    for _, arg in expression._user_args:
        coefficients.append(GlobalWrapper(arg))
    return op2.Kernel(kernel_code, kernel_code.name), False, tuple(coefficients)
