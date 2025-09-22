from ufl import inner, div, grad, curl, dx

from firedrake.assemble import assemble
from firedrake import function
from firedrake.logging import warning
from firedrake.petsc import PETSc

__all__ = ['errornorm', 'norm']


@PETSc.Log.EventDecorator()
def errornorm(u, uh, norm_type="L2", degree_rise=None, mesh=None):
    """Compute the error :math:`e = u - u_h` in the specified norm.

    :arg u: a :class:`.Function` or UFL expression containing an "exact" solution
    :arg uh: a :class:`.Function` containing the approximate solution
    :arg norm_type: the type of norm to compute, see :func:`.norm` for
         details of supported norm types.
    :arg degree_rise: ignored.
    :arg mesh: an optional mesh on which to compute the error norm
         (currently ignored).
    """
    urank = len(u.ufl_shape)
    uhrank = len(uh.ufl_shape)

    if urank != uhrank:
        raise RuntimeError("Mismatching rank between u and uh")

    if not isinstance(uh, function.Function):
        raise ValueError("uh should be a Function, is a %r", type(uh))

    if isinstance(u, function.Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            warning("Degree of exact solution less than approximation degree")

    return norm(u - uh, norm_type=norm_type, mesh=mesh)


@PETSc.Log.EventDecorator()
def norm(v, norm_type="L2", mesh=None):
    r"""Compute the norm of ``v``.

    :arg v: a ufl expression (:class:`~.ufl.classes.Expr`) to compute the norm of
    :arg norm_type: the type of norm to compute, see below for
         options.
    :arg mesh: an optional mesh on which to compute the norm
         (currently ignored).

    Available norm types are:

    - Lp :math:`||v||_{L^p} = (\int |v|^p)^{\frac{1}{p}} \mathrm{d}x`
    - H1 :math:`||v||_{H^1}^2 = \int (v, v) + (\nabla v, \nabla v) \mathrm{d}x`
    - Hdiv :math:`||v||_{H_\mathrm{div}}^2 = \int (v, v) + (\nabla\cdot v, \nabla \cdot v) \mathrm{d}x`
    - Hcurl :math:`||v||_{H_\mathrm{curl}}^2 = \int (v, v) + (\nabla \wedge v, \nabla \wedge v) \mathrm{d}x`

    """
    typ = norm_type.lower()
    p = 2
    if typ == 'l2':
        expr = inner(v, v)
    elif typ.startswith('l'):
        try:
            p = int(typ[1:])
            if p < 1:
                raise ValueError
        except ValueError:
            raise ValueError("Don't know how to interpret %s-norm" % norm_type)
        expr = inner(v, v)
    elif typ == 'h1':
        expr = inner(v, v) + inner(grad(v), grad(v))
    elif typ == "hdiv":
        expr = inner(v, v) + inner(div(v), div(v))
    elif typ == "hcurl":
        expr = inner(v, v) + inner(curl(v), curl(v))
    else:
        raise RuntimeError("Unknown norm type '%s'" % norm_type)

    return assemble((expr**(p/2))*dx)**(1/p)
