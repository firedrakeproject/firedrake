import numpy
from functools import partial

import FIAT
import ufl
from ufl.algorithms import extract_arguments

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
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
        interpolation over.
    :kwarg freeze_expr: Set to True to prevent the expression being
        re-evaluated on each call.

    This object can be used to carry out the same interpolation
    multiple times (for example in a timestepping loop).

    .. note::

       The :class:`Interpolator` holds a reference to the provided
       arguments (such that they won't be collected until the
       :class:`Interpolator` is also collected).
    """
    def __init__(self, expr, V, subset=None, freeze_expr=False, access=op2.WRITE):
        self.callable, args = make_interpolator(expr, V, subset, access)
        self.args = args
        self.nargs = len(args)
        self.freeze_expr = freeze_expr
        self.V = V

    @utils.known_pyop2_safe
    def interpolate(self, *function, output=None, transpose=False):
        """Compute the interpolation.

        :arg function: If the expression being interpolated contains an
           :class:`ufl.Argument`, then the :class:`.Function` value to
           interpolate.
        :kwarg output: Optional. A :class:`.Function` to contain the output.
        :kwarg transpose: Set to true to apply the transpose (adjoint) of the
           interpolation operator.

        :returns: The resulting interpolated :class:`.Function`.
        """
        if transpose and not self.nargs:
            raise ValueError("Can currently only apply transpose interpolation with arguments.")
        if self.nargs != len(function):
            raise ValueError("Passed %d Functions to interpolate, expected %d"
                             % (len(function), self.nargs))
        try:
            callable = self.frozen_callable
        except AttributeError:
            callable = self.callable()
            if self.freeze_expr:
                if self.nargs:
                    self.frozen_callable = callable
                else:
                    self.frozen_callable = callable.copy()

        if self.nargs:
            callable._force_evaluation()
            function, = function
            if not transpose:
                result = output or firedrake.Function(self.V)
                if transpose:
                    mul = callable.handle.multTranspose
                    V = self.args[0].function_space()
                else:
                    mul = callable.handle.mult
                    V = self.V
                result = output or firedrake.Function(V)
                with function.dat.vec_ro as x, result.dat.vec_wo as out:
                    mul(x, out)
            else:
                result = output or firedrake.Function(self.args[0].function_space())
                if transpose:
                    mul = callable.handle.multTranspose
                    V = self.args[0].function_space()
                else:
                    mul = callable.handle.mult
                    V = self.V
                result = output or firedrake.Function(V)
                with function.dat.vec_ro as x, result.dat.vec_wo as out:
                    mul(x, out)


            return result
        else:
            if output:
                output.assign(callable)
                return output
            if isinstance(self.V, firedrake.Function):
                self.V.assign(callable)
                return self.V
            else:
                return callable.copy()


def make_interpolator(expr, V, subset, access):
    assert isinstance(expr, ufl.classes.Expr)

    arguments = extract_arguments(expr)
    if len(arguments) == 0:
        if isinstance(V, firedrake.Function):
            f = V
            V = f.function_space()
        else:
            f = firedrake.Function(V)
        tensor = f.dat
    elif len(arguments) == 1:
        if isinstance(V, firedrake.Function):
            raise ValueError("Cannot interpolate an expression with an argument into a Function")

        argfs = arguments[0].function_space()
        sparsity = op2.Sparsity((V.dof_dset, argfs.dof_dset),
                                ((V.cell_node_map(), argfs.cell_node_map()),),
                                name="%s_%s_sparsity" % (V.name, argfs.name),
                                nest=False,
                                block_sparse=True)
        tensor = op2.Mat(sparsity)
        f = tensor
    else:
        raise ValueError("Cannot interpolate an expression with %d arguments" % len(arguments))

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
        loops.extend(_interpolator(V, tensor, expr, subset, arguments, access))
    elif hasattr(expr, 'eval'):
        if len(V) > 1:
            raise NotImplementedError(
                "Python expressions for mixed functions are not yet supported.")
        loops.extend(_interpolator(V, tensor, expr, subset, arguments, access))
    else:
        raise ValueError("Don't know how to interpolate a %r" % expr)

    def callable(loops, f):
        for l in loops:
            l()
        return f

    return partial(callable, loops, f), arguments


def _interpolator(V, tensor, expr, subset, arguments, access):
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
        ast, oriented, needs_cell_sizes, coefficients, _ = compile_ufl_kernel(expr, to_pts, V.ufl_element(), coords, coffee=False)
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

    if tensor in set((c.dat for c in coefficients)):
        output = tensor
        tensor = op2.Dat(tensor.dataset)
        if access is not op2.WRITE:
            copyin = (partial(output.copy, tensor), )
        else:
            copyin = ()
        copyout = (partial(tensor.copy, output), )
    else:
        copyin = ()
        copyout = ()
    if isinstance(tensor, op2.Dat):
        args.append(tensor(access, V.cell_node_map()))
    else:
        assert access == op2.WRITE # Other access descriptors not done for Matrices.
        args.append(tensor(op2.WRITE, (V.cell_node_map(),
                                       arguments[0].function_space().cell_node_map())))
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

    if isinstance(tensor, op2.Mat):
        return partial(op2.par_loop, *args), tensor.assemble()
    else:
        return copyin + (partial(op2.par_loop, *args), ) + copyout


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
