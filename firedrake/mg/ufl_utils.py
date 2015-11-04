from __future__ import absolute_import

from operator import add

import ufl
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.multifunction import MultiFunction

from firedrake.ffc_interface import sum_integrands
import firedrake

from . import utils


__all__ = ["coarsen_form", "coarsen_thing"]


class CoarsenIntegrand(MultiFunction):

    """'Coarsen' a :class:`ufl.Expr` by replacing coefficients,
    arguments and domain data with coarse mesh equivalents."""

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
        if isinstance(o, firedrake.Constant):
            try:
                mesh = o.domain().coordinates().as_coordinates()
                hierarchy, level = utils.get_level(mesh)
                new_mesh = hierarchy[level-1]
            except:
                new_mesh = None
            if o.rank() == 0:
                val = o.dat.data_ro[0]
            else:
                val = o.dat.data_ro.copy()
            return firedrake.Constant(value=val,
                                      domain=new_mesh)
        elif isinstance(o, firedrake.Function):
            hierarchy, level = utils.get_level(o)
            if level == -1:
                # Not found, disgusting hack, maybe it's the coords?
                if o is o.function_space().mesh().coordinates:
                    h, l = utils.get_level(o.function_space().mesh())
                    new_fn = h[l-1].coordinates
                else:
                    raise RuntimeError("Didn't find a coarse version of %r", o)
            else:
                new_fn = hierarchy[level-1]
            return new_fn
        else:
            raise RuntimeError("Don't know how to handle %r", o)

    def circumradius(self, o):
        mesh = o.domain().coordinates().as_coordinates()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        return firedrake.Circumradius(new_mesh.ufl_domain())

    def facet_normal(self, o):
        mesh = o.domain().coordinates().as_coordinates()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        return firedrake.FacetNormal(new_mesh.ufl_domain())


def coarsen_form(form):
    """Return a coarse mesh version of a form

    :arg form: The :class:`ufl.Form` to coarsen.

    This maps over the form and replaces coefficients and arguments
    with their coarse mesh equivalents."""
    if form is None:
        return None
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
        mesh = it.domain().coordinates().as_coordinates()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]

        measure = ufl.Measure(it.integral_type(),
                              domain=new_mesh,
                              subdomain_id=it.subdomain_id(),
                              subdomain_data=it.subdomain_data(),
                              metadata=it.metadata())

        forms.append(integrand * measure)
    return reduce(add, forms)


def coarsen_thing(thing):
    if thing is None:
        return None
    if isinstance(thing, firedrake.DirichletBC):
        return coarsen_bc(thing)
    if isinstance(thing, firedrake.IndexedFunctionSpace):
        idx = thing.index
        val = thing._parent
        hierarchy, level = utils.get_level(val)
        new_val = hierarchy[level-1]
        return new_val.sub(idx)
    hierarchy, level = utils.get_level(thing)
    return hierarchy[level-1]


def coarsen_bc(bc):
    new_V = coarsen_thing(bc.function_space())
    val = bc._original_val
    zeroed = bc._currently_zeroed
    subdomain = bc.sub_domain
    method = bc.method

    new_val = val

    if isinstance(val, firedrake.Expression):
        new_val = val

    if isinstance(val, (firedrake.Constant, firedrake.Function)):
        mapper = CoarsenIntegrand()
        new_val = map_integrand_dags(mapper, val)

    new_bc = firedrake.DirichletBC(new_V, new_val, subdomain,
                                   method=method)

    if zeroed:
        new_bc.homogenize()

    return new_bc
