from __future__ import absolute_import
from pyop2.logger import warning
from ufl import inner, div, grad, curl, sqrt, dx

from firedrake.assemble import assemble
from firedrake import function
from firedrake import functionspace
from firedrake import projection

__all__ = ['errornorm', 'norm']


def errornorm(u, uh, norm_type="L2", degree_rise=3, mesh=None):
    """Compute the error :math:`e = u - u_h` in the specified norm.

    :arg u: a :class:`.Function` containing an "exact" solution
    :arg uh: a :class:`.Function` containing the approximate solution
    :arg norm_type: the type of norm to compute, see :func:`.norm` for
         details of supported norm types.
    :arg degree_rise: increase in polynomial degree to use as the
         approximation space for computing the error.
    :arg mesh: an optional mesh on which to compute the error norm
         (currently ignored).

    This function works by :func:`.project`\ing ``u`` and ``uh`` into
    a space of degree ``degree_rise`` higher than the degree of ``uh``
    and computing the error there.
    """
    urank = len(u.shape())
    uhrank = len(uh.shape())

    rank = urank
    if urank != uhrank:
        raise RuntimeError("Mismatching rank between u and uh")

    degree = uh.function_space().ufl_element().degree()
    if isinstance(degree, tuple):
        degree = max(degree) + degree_rise
    else:
        degree += degree_rise

    # The exact solution might be an expression, in which case this test is irrelevant.
    if isinstance(u, function.Function):
        degree_u = u.function_space().ufl_element().degree()
        if degree > degree_u:
            warning("Degree of exact solution less than approximation degree")

    mesh = uh.function_space().mesh()
    if rank == 0:
        V = functionspace.FunctionSpace(mesh, 'DG', degree)
    elif rank == 1:
        V = functionspace.VectorFunctionSpace(mesh, 'DG', degree,
                                              dim=u.shape()[0])
    else:
        raise RuntimeError("Don't know how to compute error norm for tensor valued functions")

    u_ = projection.project(u, V)
    uh_ = projection.project(uh, V)

    uh_ -= u_

    return norm(uh_, norm_type=norm_type, mesh=mesh)


def norm(v, norm_type="L2", mesh=None):
    """Compute the norm of ``v``.

    :arg v: a :class:`.Function` to compute the norm of
    :arg norm_type: the type of norm to compute, see below for
         options.
    :arg mesh: an optional mesh on which to compute the norm
         (currently ignored).

    Available norm types are:

    * L2

       .. math::

          ||v||_{L^2}^2 = \int (v, v) \mathrm{d}x

    * H1

       .. math::

          ||v||_{H^1}^2 = \int (v, v) + (\\nabla v, \\nabla v) \mathrm{d}x

    * Hdiv

       .. math::

          ||v||_{H_\mathrm{div}}^2 = \int (v, v) + (\\nabla\cdot v, \\nabla \cdot v) \mathrm{d}x

    * Hcurl

       .. math::

          ||v||_{H_\mathrm{curl}}^2 = \int (v, v) + (\\nabla \wedge v, \\nabla \wedge v) \mathrm{d}x
    """
    assert isinstance(v, function.Function)

    typ = norm_type.lower()
    if typ == 'l2':
        form = inner(v, v)*dx
    elif typ == 'h1':
        form = inner(v, v)*dx + inner(grad(v), grad(v))*dx
    elif typ == "hdiv":
        form = inner(v, v)*dx + div(v)*div(v)*dx
    elif typ == "hcurl":
        form = inner(v, v)*dx + inner(curl(v), curl(v))*dx
    else:
        raise RuntimeError("Unknown norm type '%s'" % norm_type)

    return sqrt(assemble(form))
