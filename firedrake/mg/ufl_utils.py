from __future__ import absolute_import

from operator import add

import ufl
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction

from firedrake.ffc_interface import sum_integrands
from firedrake import constant
from firedrake import function
from firedrake import ufl_expr

from . import utils


__all__ = ["coarsen_form"]


class CoarsenIntegrand(MultiFunction):

    """'Coarsen' a :class:`ufl.Expr` by replacing coefficients,
    arguments and domain data with coarse mesh equivalents."""
    def terminal(self, o):
        raise RuntimeError("Don't know how to handle %r", type(o))

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        try:
            fs = o.function_space()
            hierarchy, level = utils.get_level(fs)
            new_fs = hierarchy[level-1]
        except:
            raise RuntimeError("Don't know how to handle %r", o)

        return o.reconstruct(new_fs)

    def coefficient(self, o):
        if isinstance(o, constant.Constant):
            try:
                mesh = o.domain().data()
                hierarchy, level = utils.get_level(mesh)
                new_mesh = hierarchy[level-1]
            except:
                new_mesh = None
            return constant.Constant(value=o.dat.data,
                                     domain=new_mesh)
        elif isinstance(o, function.Function):
            hierarchy, level = utils.get_level(o)
            new_fn = hierarchy[level-1]
            return new_fn
        else:
            raise RuntimeError("Don't know how to handle %r", o)

    def circumradius(self, o):
        mesh = o.domain().data()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        return ufl_expr.Circumradius(new_mesh.ufl_domain())

    def facet_normal(self, o):
        mesh = o.domain().data()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        return ufl_expr.FacetNormal(new_mesh.ufl_domain())


def coarsen_form(form):
    """Return a coarse mesh version of a form

    :arg form: The :class:`ufl.Form` to coarsen.

    This maps over the form and replaces coefficients and arguments
    with their coarse mesh equivalents."""
    assert isinstance(form, ufl.Form), \
        "Don't know how to coarsen %r" % type(form)

    mapper = CoarsenIntegrand()
    integrals = sum_integrands(form).integrals()
    forms = []
    # Ugh, visitors can't deal with measures (they're not actual
    # Exprs) so we need to map the transformer over the integrand and
    # reconstruct the integral by building the measure by hand.
    for it in integrals:
        integrand = map_integrand_dags(mapper, it.integrand())
        mesh = it.domain().data()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]

        measure = ufl.Measure(it.integral_type(),
                              domain=new_mesh,
                              subdomain_id=it.subdomain_id(),
                              subdomain_data=it.subdomain_data(),
                              metadata=it.metadata())

        forms.append(integrand * measure)
    return reduce(add, forms)
