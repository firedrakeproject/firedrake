import numpy
from functools import partial

import FIAT
import ufl

from pyop2 import op2
import loopy

from tsfc.fiatinterface import create_element
from tsfc import compile_expression_at_points as compile_ufl_kernel

import firedrake
from firedrake import utils

__all__ = ("interpolate", "Interpolator")


def interpolate(expr, V, subset=None):
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
    return Interpolator(expr, V, subset=subset).interpolate()


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
    def __init__(self, expr, V, subset=None):
        self.callable = make_interpolator(expr, V, subset)

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


def make_interpolator(expr, V, subset):
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
        loops.extend(_interpolator(V, f.dat, expr, subset))
    elif hasattr(expr, 'eval'):
        if len(V) > 1:
            raise NotImplementedError(
                "Python expressions for mixed functions are not yet supported.")
        loops.extend(_interpolator(V, f.dat, expr, subset))
    else:
        # Slice the expression and pass in the right number of values for
        # each component function space of this function
        d = 0
        for fs, dat, dim in zip(V, f.dat, dims):
            idx = d if fs.rank == 0 else slice(d, d+dim)
            loops.extend(_interpolator(fs, dat,
                                       SubExpression(expr, idx, fs.ufl_element().value_shape()),
                                       subset))
            d += dim

    def callable(loops, f):
        for l in loops:
            l()
        return f

    return partial(callable, loops, f)


def _interpolator(V, dat, expr, subset):
    to_element = create_element(V.ufl_element(), vector_is_mixed=False)
    to_pts = []

    if V.ufl_element().mapping() != "identity":
        raise NotImplementedError("Can only interpolate onto elements "
                                  "with affine mapping. Try projecting instead")

    for dual in to_element.dual_basis():
        if not isinstance(dual, FIAT.functional.PointEvaluation):
            raise NotImplementedError("Can only interpolate onto point "
                                      "evaluation operators. Try projecting instead")
        pts, = dual.pt_dict.keys()
        to_pts.append(pts)

    if len(expr.ufl_shape) != len(V.ufl_element().value_shape()):
        raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                           % (len(expr.ufl_shape), len(V.ufl_element().value_shape())))

    if expr.ufl_shape != V.ufl_element().value_shape():
        raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                           % (expr.ufl_shape, V.ufl_element().value_shape()))

    mesh = V.ufl_domain()
    coords = mesh.coordinates

    if not isinstance(expr, (firedrake.Expression, SubExpression)):
        if expr.ufl_domain() and expr.ufl_domain() != V.mesh():
            raise NotImplementedError("Interpolation onto another mesh not supported.")
        if expr.ufl_shape != V.shape:
            raise ValueError("UFL expression has incorrect shape for interpolation.")
        ast, oriented, needs_cell_sizes, coefficients = compile_ufl_kernel(expr, to_pts, coords, coffee=False)
        kernel = op2.Kernel(ast, ast.name)
        indexed = True
    elif hasattr(expr, "eval"):
        kernel, oriented, needs_cell_sizes, coefficients = compile_python_kernel(expr, to_pts, to_element, V, coords)
        indexed = False
    elif expr.code is not None:
        kernel, oriented, needs_cell_sizes, coefficients = compile_c_kernel(expr, to_pts, to_element, V, coords)
        indexed = True
    else:
        raise RuntimeError("Attempting to evaluate an Expression which has no value.")

    cell_set = coords.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset
    args = [kernel, cell_set]

    copy_back = False
    if dat in set((c.dat for c in coefficients)):
        output = dat
        dat = op2.Dat(dat.dataset)
        copy_back = True
    if indexed:
        args.append(dat(op2.WRITE, V.cell_node_map()))
    else:
        args.append(dat(op2.WRITE, V.cell_node_map()))
    if oriented:
        co = mesh.cell_orientations()
        args.append(co.dat(op2.READ, co.cell_node_map()))
    if needs_cell_sizes:
        cs = mesh.cell_sizes
        args.append(cs.dat(op2.READ, cs.cell_node_map()))
    for coefficient in coefficients:
        m_ = coefficient.cell_node_map()
        if indexed:
            args.append(coefficient.dat(op2.READ, m_))
        else:
            args.append(coefficient.dat(op2.READ, m_))

    for o in coefficients:
        domain = o.ufl_domain()
        if domain is not None and domain.topology != mesh.topology:
            raise NotImplementedError("Interpolation onto another mesh not supported.")

    if copy_back:
        return partial(op2.par_loop, *args), partial(dat.copy, output)
    else:
        return (partial(op2.par_loop, *args), )


class GlobalWrapper(object):
    """Wrapper object that fakes a Global to behave like a Function."""
    def __init__(self, glob):
        self.dat = glob
        self.cell_node_map = lambda *args: None
        self.ufl_domain = lambda: None


def compile_python_kernel(expression, to_pts, to_element, fs, coords):
    """Produce a :class:`PyOP2.Kernel` wrapping the eval method on the
    function provided."""

    coords_space = coords.function_space()
    coords_element = create_element(coords_space.ufl_element(), vector_is_mixed=False)

    X_remap = list(coords_element.tabulate(0, to_pts).values())[0]

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
            expression.eval(output[i:i+1, ...] if numpy.ndim(output) == 1 else output[i, ...],
                            X[i:i+1, ...] if numpy.ndim(X) == 1 else X[i, ...], **kwargs)

    coefficients = [coords]
    for _, arg in expression._user_args:
        coefficients.append(GlobalWrapper(arg))
    return kernel, False, False, tuple(coefficients)


def compile_c_kernel(expression, to_pts, to_element, fs, coords):
    """Produce a :class:`PyOP2.Kernel` from the c expression provided."""

    coords_space = coords.function_space()
    coords_element = create_element(coords_space.ufl_element(), vector_is_mixed=False)

    names = {v[0] for v in expression._user_args}

    X_data = list(coords_element.tabulate(0, to_pts).values())[0].T

    import pymbolic.primitives as p

    A = p.Variable(utils.unique_name("A", names))
    X = p.Variable(utils.unique_name("X", names))
    crd = p.Variable(utils.unique_name("coords", names))
    k = p.Variable(utils.unique_name("k", names))
    d = p.Variable(utils.unique_name("d", names))
    i = p.Variable(utils.unique_name("i", names))
    # x is a reserved name.
    x = p.Variable("x")
    if "x" in names:
        raise ValueError("cannot use 'x' as a user-defined Expression variable")

    dim = coords_space.value_size
    ndof = to_element.space_dimension()
    xndof = coords_element.space_dimension()

    # x[d] = 0
    insn0 = loopy.Assignment(x.index(d), 0, id="insn0", within_inames=frozenset([k.name, d.name]))
    # x[d] += X[k, i] * coord[i,d]
    insn1 = loopy.Assignment(
        x.index(d), x.index(d) + X.index((k, i)) * crd.index((i, d)), id="insn1",
        within_inames=frozenset([k.name, d.name, i.name]), depends_on=frozenset(["insn0"])
    )
    instructions = [insn0, insn1]

    from loopy.kernel.creation import parse_instructions
    for cmp, code in enumerate(expression.code):
        # A[k, c] = expression
        code = '{0}[{1}, {2}]'.format(A.name, k.name, cmp) + " = " + str(code)
        (insn, ), _, _ = parse_instructions(code, {})
        insn = insn.copy(within_inames=frozenset([k.name]), depends_on=frozenset(["insn1"]))
        instructions.append(insn)

    import islpy as isl

    inames = isl.make_zero_and_vars([k.name, d.name, i.name])
    domain = (inames[0].le_set(inames[k.name])) & (inames[k.name].lt_set(inames[0] + ndof))
    domain = domain & (inames[0].le_set(inames[d.name])) & (inames[d.name].lt_set(inames[0] + dim))
    domain = domain & (inames[0].le_set(inames[i.name])) & (inames[i.name].lt_set(inames[0] + xndof))

    data = [loopy.GlobalArg(A.name, dtype=numpy.float64, shape=(ndof, len(expression.code))),
            loopy.GlobalArg(crd.name, dtype=coords.dat.dtype, shape=(xndof, dim)),
            loopy.TemporaryVariable(X.name, initializer=X_data, dtype=X_data.dtype, shape=X_data.shape,
                                    read_only=True, scope=loopy.AddressSpace.LOCAL),
            loopy.TemporaryVariable("x", dtype=numpy.float64, shape=(dim,), scope=loopy.AddressSpace.LOCAL)]

    coefficients = [coords]
    for _i, (name, arg) in enumerate(expression._user_args):
        coefficients.append(GlobalWrapper(arg))
        if arg.shape == (1, ):
            name += "_"
            # arg = arg_[0]
            user_arg_insn = loopy.Assignment(p.Variable(arg.name), p.Variable(name).index(0), id="user_arg_{0}".format(_i))
            instructions.insert(0, user_arg_insn)
            insn0 = insn0.copy(depends_on=insn0.depends_on | frozenset([user_arg_insn.id]))
            data.append(loopy.TemporaryVariable(arg.name, dtype=arg.dtype, shape=(), scope=loopy.AddressSpace.LOCAL))
        data.append(loopy.GlobalArg(name, dtype=arg.dtype, shape=arg.shape))

    if any("pi" in str(_c) for _c in expression.code):
        data.append(loopy.TemporaryVariable("pi", dtype=numpy.float64, initializer=numpy.array(numpy.pi),
                                            read_only=True, scope=loopy.AddressSpace.LOCAL))

    knl = loopy.make_kernel([domain], instructions, data, name="expression_kernel", lang_version=(2018, 2),
                            silenced_warnings=["summing_if_branches_ops"])

    return op2.Kernel(knl, knl.name), False, False, tuple(coefficients)
