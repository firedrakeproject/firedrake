import numpy
from functools import partial

import FIAT
import ufl

from pyop2 import op2

from tsfc.fiatinterface import create_element
from tsfc import compile_expression_at_points as compile_ufl_kernel

import firedrake
from firedrake import utils

__all__ = ("interpolate", "Interpolator")


def interpolate(expr, V, subset=None, access=op2.WRITE):
    """Interpolate an expression onto a new function in V.

    :arg expr: an :class:`.Expression`.
    :arg V: the :class:`.FunctionSpace` to interpolate into (or else
        an existing :class:`.Function`).
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
        interpolation over.
    :kwarg access: The access descriptor for combining updates to shared dofs.

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
    def __init__(self, expr, V, subset=None, access=op2.WRITE):
        self.callable = make_interpolator(expr, V, subset, access)

    def interpolate(self):
        """Compute the interpolation.

        :returns: The resulting interpolated :class:`.Function`.
        """
        return self.callable()


def make_interpolator(expr, V, subset, access):
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
        loops.extend(_interpolator(V, f.dat, expr, subset, access))
    elif hasattr(expr, 'eval'):
        if len(V) > 1:
            raise NotImplementedError(
                "Python expressions for mixed functions are not yet supported.")
        loops.extend(_interpolator(V, f.dat, expr, subset, access))
    else:
        raise ValueError("Don't know how to interpolate a %r" % expr)

    def callable(loops, f):
        for l in loops:
            l()
        return f

    return partial(callable, loops, f)


@utils.known_pyop2_safe
def _interpolator(V, dat, expr, subset, access):
    to_element = create_element(V.ufl_element(), vector_is_mixed=False)
    to_pts = []

    if access is op2.READ:
        raise ValueError("Can't have READ access for output function")
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

    if not isinstance(expr, firedrake.Expression):
        if expr.ufl_domain() and expr.ufl_domain() != V.mesh():
            raise NotImplementedError("Interpolation onto another mesh not supported.")
        if expr.ufl_shape != V.shape:
            raise ValueError("UFL expression has incorrect shape for interpolation.")
        ast, oriented, needs_cell_sizes, coefficients, _ = compile_ufl_kernel(expr, to_pts, coords, coffee=False)
        kernel = op2.Kernel(ast, ast.name)
    elif hasattr(expr, "eval"):
        kernel, oriented, needs_cell_sizes, coefficients = compile_python_kernel(expr, to_pts, to_element, V, coords)
    else:
        raise RuntimeError("Attempting to evaluate an Expression which has no value.")

    cell_set = coords.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset
    args = [kernel, cell_set]

    if dat in set((c.dat for c in coefficients)):
        output = dat
        dat = op2.Dat(dat.dataset)
        if access is not op2.WRITE:
            copyin = (partial(output.copy, dat), )
        else:
            copyin = ()
        copyout = (partial(dat.copy, output), )
    else:
        copyin = ()
        copyout = ()
    args.append(dat(access, V.cell_node_map()))
    if oriented:
        co = mesh.cell_orientations()
        args.append(co.dat(op2.READ, co.cell_node_map()))
    if needs_cell_sizes:
        cs = mesh.cell_sizes
        args.append(cs.dat(op2.READ, cs.cell_node_map()))
    for coefficient in coefficients:
        m_ = coefficient.cell_node_map()
        args.append(coefficient.dat(op2.READ, m_))

    for o in coefficients:
        domain = o.ufl_domain()
        if domain is not None and domain.topology != mesh.topology:
            raise NotImplementedError("Interpolation onto another mesh not supported.")

    return copyin + (op2.ParLoop(*args).compute, ) + copyout


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
