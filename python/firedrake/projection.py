from core_types import *
from expression import Expression
from solving import solve
from ufl import *
from ufl_expr import TrialFunction, TestFunction


def project(v, V, bcs=None, mesh=None,
            solver_parameters={},
            form_compiler_parameters=None):
    """Project an :class:`Expression` or :class:`Function` into a :class:`FunctionSpace`

    :arg v: the :class:`Expression` or :class:`Function` to project
    :arg V: the :class:`FunctionSpace` to project into
    :arg bcs: boundary conditions to apply in the projection
    :arg mesh: the mesh to project into
    :arg solver_type: linear solver to use
    :arg preconditioner_type: preconditioner to use
    :arg form_compiler_parameters: parameters to the form compiler

    Currently, `bcs`, `mesh` and `form_compiler_parameters` are ignored."""
    if not isinstance(V, FunctionSpace):
        raise RuntimeError('Can only project into function spaces, not %r' % type(V))

    if isinstance(v, Expression):
        # It feels like there ought to be a better way to do this.
        f = Function(V)
        f.interpolate(v)
        v = f

    p = TestFunction(V)
    q = TrialFunction(V)
    a = inner(p, q) * dx
    L = inner(p, v) * dx

    ret = Function(V)
    solve(a == L, ret, bcs=bcs,
          solver_parameters=solver_parameters,
          form_compiler_parameters=form_compiler_parameters)
    return ret
