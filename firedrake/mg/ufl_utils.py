from __future__ import absolute_import

import ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.multifunction import MultiFunction

import firedrake
from firedrake.petsc import PETSc

from . import utils


__all__ = ["coarsen_form", "coarsen_thing"]


class CoarsenIntegrand(MultiFunction):

    """'Coarsen' a :class:`ufl.Expr` by replacing coefficients,
    arguments and domain data with coarse mesh equivalents."""

    def __init__(self, coefficient_mapping):
        self.coefficient_mapping = coefficient_mapping or {}
        super(CoarsenIntegrand, self).__init__()

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
                mesh = o.ufl_domain()
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
            # find level of function space to be sure
            hierarchy, level = utils.get_level(o.function_space())
            if level == -1:
                raise RuntimeError("Didn't find a coarse version of %r", o)
            elif level != -1:
                new_fn = self.coefficient_mapping.get(o)
                if new_fn is not None:
                    return new_fn
                new_fn = firedrake.Function(hierarchy[level-1])
                # restrict state to coarse grid
                firedrake.inject(o, new_fn)
                return new_fn
            else:
                raise RuntimeError("Doesnt have level")
        else:
            raise RuntimeError("Don't know how to handle %r", o)

    def circumradius(self, o):
        mesh = o.ufl_domain()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        return firedrake.Circumradius(new_mesh.ufl_domain())

    def facet_normal(self, o):
        mesh = o.ufl_domain()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        return firedrake.FacetNormal(new_mesh.ufl_domain())


def coarsen_form(form, coefficient_mapping=None):
    """Return a coarse mesh version of a form

    :arg form: The :class:`~ufl.classes.Form` to coarsen.
    :kwarg mapping: an optional map from coefficients to their
        coarsened equivalents.

    This maps over the form and replaces coefficients and arguments
    with their coarse mesh equivalents."""
    if form is None:
        return None
    assert isinstance(form, ufl.Form), \
        "Don't know how to coarsen %r" % type(form)

    mapper = CoarsenIntegrand(coefficient_mapping)
    integrals = []
    # Ugh, visitors can't deal with measures (they're not actual
    # Exprs) so we need to map the transformer over the integrand and
    # reconstruct the integral by building the measure by hand.
    for it in form.integrals():
        integrand = map_expr_dag(mapper, it.integrand())
        mesh = it.ufl_domain()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        if isinstance(integrand, ufl.classes.Zero):
            continue
        if it.subdomain_data() is not None:
            raise ValueError("Don't know how to coarsen subdomain data")
        new_itg = it.reconstruct(integrand=integrand,
                                 domain=new_mesh)
        integrals.append(new_itg)
    return ufl.Form(integrals)


def coarsen_thing(thing):
    if thing is None:
        return None
    if isinstance(thing, firedrake.DirichletBC):
        return coarsen_bc(thing)
    if isinstance(thing, (firedrake.functionspaceimpl.FunctionSpace,
                          firedrake.functionspaceimpl.WithGeometry)) and \
       thing.index is not None:
        idx = thing.index
        val = thing.parent
        hierarchy, level = utils.get_level(val)
        new_val = hierarchy[level-1]
        return new_val.sub(idx)
    # check that we find the level of a hierarchy
    if isinstance(thing, firedrake.Function):
        hierarchy, level = utils.get_level(thing.function_space())
        new_thing = firedrake.Function(hierarchy[level-1])
        # restrict state to coarse grid
        firedrake.inject(thing, new_thing)
    else:
        hierarchy, level = utils.get_level(thing)
        new_thing = hierarchy[level-1]
    return new_thing


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
        new_val = map_expr_dag(mapper, val)

    new_bc = firedrake.DirichletBC(new_V, new_val, subdomain,
                                   method=method)

    if zeroed:
        new_bc.homogenize()

    return new_bc


def coarsen_problem(problem):
    u = problem.u
    h, lvl = utils.get_level(u.function_space())
    if lvl == -1:
        raise RuntimeError("No hierarchy to coarsen")
    if lvl == 0:
        return None

    # Build set of coefficients we need to coarsen
    coefficients = set()
    coefficients.update(problem.F.coefficients())
    coefficients.update(problem.J.coefficients())
    if problem.Jp is not None:
        coefficients.update(problem.Jp.coefficients())

    # Coarsen them, and remember where from.
    mapping = {}
    for c in coefficients:
        mapping[c] = coarsen_thing(c)

    new_u = mapping[problem.u]

    new_bcs = [coarsen_thing(bc) for bc in problem.bcs]
    new_J = coarsen_form(problem.J, coefficient_mapping=mapping)
    new_Jp = coarsen_form(problem.Jp, coefficient_mapping=mapping)
    new_F = coarsen_form(problem.F, coefficient_mapping=mapping)

    new_problem = firedrake.NonlinearVariationalProblem(new_F,
                                                        new_u,
                                                        bcs=new_bcs,
                                                        J=new_J,
                                                        Jp=new_Jp,
                                                        form_compiler_parameters=problem.form_compiler_parameters)
    return new_problem


class Interpolation(object):
    def __init__(self, cfn, ffn, cbcs=None, fbcs=None):
        self.cfn = cfn
        self.ffn = ffn
        self.cbcs = cbcs or []
        self.fbcs = fbcs or []

    def mult(self, mat, x, y, inc=False):
        with self.cfn.dat.vec as v:
            x.copy(v)
        firedrake.prolong(self.cfn, self.ffn)
        for bc in self.fbcs:
            bc.zero(self.ffn)
        with self.ffn.dat.vec_ro as v:
            if inc:
                y.axpy(1.0, v)
            else:
                v.copy(y)

    def multAdd(self, mat, x, y, w):
        if y.handle == w.handle:
            self.mult(mat, x, w, inc=True)
        else:
            self.mult(mat, x, w)
            w.axpy(1.0, y)

    def multTranspose(self, mat, x, y, inc=False):
        with self.ffn.dat.vec as v:
            x.copy(v)
        firedrake.restrict(self.ffn, self.cfn)
        for bc in self.cbcs:
            bc.zero(self.cfn)
        with self.cfn.dat.vec_ro as v:
            if inc:
                y.axpy(1.0, v)
            else:
                v.copy(y)

    def multTransposeAdd(self, mat, x, y, w):
        if y.handle == w.handle:
            self.multTranspose(mat, x, w, inc=True)
        else:
            self.multTranspose(mat, x, w)
            w.axpy(1.0, y)


class Injection(object):
    def __init__(self, cfn, ffn, cbcs=None):
        self.cfn = cfn
        self.ffn = ffn
        self.cbcs = cbcs or []

    def multTranspose(self, mat, x, y):
        with self.ffn.dat.vec as v:
            x.copy(v)
        firedrake.inject(self.ffn, self.cfn)
        for bc in self.cbcs:
            bc.apply(self.cfn)
        with self.cfn.dat.vec_ro as v:
            v.copy(y)


def create_interpolation(dmc, dmf):
    cctx = dmc.getAppCtx()
    fctx = dmf.getAppCtx()

    V_c = dmc.getAttr("__fs__")()
    V_f = dmf.getAttr("__fs__")()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs
    fbcs = fctx._problem.bcs

    ctx = Interpolation(cfn, ffn, cbcs, fbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat, None


def create_injection(dmc, dmf):
    cctx = dmc.getAppCtx()

    V_c = dmc.getAttr("__fs__")()
    V_f = dmf.getAttr("__fs__")()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs

    ctx = Injection(cfn, ffn, cbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat
