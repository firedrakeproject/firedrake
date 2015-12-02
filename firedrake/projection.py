from __future__ import absolute_import
import ufl

from firedrake import expression
from firedrake import functionspace
from firedrake import solving
from firedrake import ufl_expr


__all__ = ['project']

# Store the solve function to use in a variable so external packages
# (dolfin-adjoint) can override it.
_solve = solving.solve


def project(v, V, bcs=None, mesh=None,
            solver_parameters=None,
            form_compiler_parameters=None,
            name=None):
    """Project an :class:`.Expression` or :class:`.Function` into a :class:`.FunctionSpace`

    :arg v: the :class:`.Expression`, :class:`ufl.Expr` or
         :class:`.Function` to project
    :arg V: the :class:`.FunctionSpace` or :class:`.Function` to project into
    :arg bcs: boundary conditions to apply in the projection
    :arg mesh: the mesh to project into
    :arg solver_parameters: parameters to pass to the solver used when
         projecting.
    :arg form_compiler_parameters: parameters to the form compiler
    :arg name: name of the resulting :class:`.Function`

    If ``V`` is a :class:`.Function` then ``v`` is projected into
    ``V`` and ``V`` is returned. If `V` is a :class:`.FunctionSpace`
    then ``v`` is projected into a new :class:`.Function` and that
    :class:`.Function` is returned.

    The ``bcs``, ``mesh`` and ``form_compiler_parameters`` are
    currently ignored."""
    from firedrake import function

    if isinstance(V, functionspace.FunctionSpaceBase):
        ret = function.Function(V, name=name)
    elif isinstance(V, function.Function):
        ret = V
        V = V.function_space()
    else:
        raise RuntimeError(
            'Can only project into functions and function spaces, not %r'
            % type(V))

    if isinstance(v, expression.Expression):
        shape = v.value_shape()
        # Build a function space that supports PointEvaluation so that
        # we can interpolate into it.
        if isinstance(V.ufl_element().degree(), tuple):
            deg = max(V.ufl_element().degree())
        else:
            deg = V.ufl_element().degree()

        if v.rank() == 0:
            fs = functionspace.FunctionSpace(V.mesh(), 'DG', deg+1)
        elif v.rank() == 1:
            fs = functionspace.VectorFunctionSpace(V.mesh(), 'DG',
                                                   deg+1,
                                                   dim=shape[0])
        else:
            fs = functionspace.TensorFunctionSpace(V.mesh(), 'DG',
                                                   deg+1,
                                                   shape=shape)
        f = function.Function(fs)
        f.interpolate(v)
        v = f
    elif isinstance(v, function.Function):
        if v.function_space().mesh() != ret.function_space().mesh():
            raise RuntimeError("Can't project between mismatching meshes")
    elif not isinstance(v, ufl.core.expr.Expr):
        raise RuntimeError("Can't only project from expressions and functions, not %r" % type(v))

    if v.ufl_shape != ret.ufl_shape:
        raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' %
                           (v.ufl_shape, ret.ufl_shape))

    p = ufl_expr.TestFunction(V)
    q = ufl_expr.TrialFunction(V)
    a = ufl.inner(p, q) * ufl.dx(domain=V.mesh())
    L = ufl.inner(p, v) * ufl.dx(domain=V.mesh())

    # Default to 1e-8 relative tolerance
    if solver_parameters is None:
        solver_parameters = {'ksp_type': 'cg', 'ksp_rtol': 1e-8}
    else:
        solver_parameters.setdefault('ksp_type', 'cg')
        solver_parameters.setdefault('ksp_rtol', 1e-8)

    _solve(a == L, ret, bcs=bcs,
           solver_parameters=solver_parameters,
           form_compiler_parameters=form_compiler_parameters)
    return ret
