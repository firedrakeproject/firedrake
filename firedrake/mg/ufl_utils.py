
import ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.multifunction import MultiFunction

from functools import singledispatch
import firedrake
from firedrake.petsc import PETSc

from . import utils


__all__ = ["coarsen"]


class CoarseningError(Exception):
    """Exception raised when coarsening symbolic information fails."""
    pass


class CoarsenIntegrand(MultiFunction):

    """'Coarsen' a :class:`ufl.Expr` by replacing coefficients,
    arguments and domain data with coarse mesh equivalents."""

    def __init__(self, coefficient_mapping):
        self.coefficient_mapping = coefficient_mapping or {}
        super(CoarsenIntegrand, self).__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        V = coarsen(o.function_space())
        return o.reconstruct(V)

    def coefficient(self, o):
        return coarsen(o, coefficient_mapping=self.coefficient_mapping)

    def geometric_quantity(self, o):
        return type(o)(coarsen(o.ufl_domain()))

    def circumradius(self, o):
        mesh = coarsen(o.ufl_domain())
        return firedrake.Circumradius(mesh)

    def facet_normal(self, o):
        mesh = coarsen(o.ufl_domain())
        return firedrake.FacetNormal(mesh)


@singledispatch
def coarsen(expr, coefficient_mapping=None):
    # Default, just send it back
    return expr


@coarsen.register(ufl.Mesh)
def coarsen_mesh(mesh, coefficient_mapping=None):
    hierarchy, level = utils.get_level(mesh)
    if hierarchy is None:
        raise CoarseningError("No mesh hierarchy available")
    return hierarchy[level - 1]


@coarsen.register(ufl.Form)
def coarsen_form(form, coefficient_mapping=None):
    """Return a coarse mesh version of a form

    :arg form: The :class:`~ufl.classes.Form` to coarsen.
    :kwarg mapping: an optional map from coefficients to their
        coarsened equivalents.

    This maps over the form and replaces coefficients and arguments
    with their coarse mesh equivalents."""
    if form is None:
        return None

    mapper = CoarsenIntegrand(coefficient_mapping)
    integrals = []
    for it in form.integrals():
        integrand = map_expr_dag(mapper, it.integrand())
        mesh = it.ufl_domain()
        hierarchy, level = utils.get_level(mesh)
        new_mesh = hierarchy[level-1]
        if isinstance(integrand, ufl.classes.Zero):
            continue
        if it.subdomain_data() is not None:
            raise CoarseningError("Don't know how to coarsen subdomain data")
        new_itg = it.reconstruct(integrand=integrand,
                                 domain=new_mesh)
        integrals.append(new_itg)
    return ufl.Form(integrals)


@coarsen.register(firedrake.DirichletBC)
def coarsen_bc(bc, coefficient_mapping=None):
    V = coarsen(bc.function_space(), coefficient_mapping=coefficient_mapping)
    val = coarsen(bc._original_val, coefficient_mapping=coefficient_mapping)
    zeroed = bc._currently_zeroed
    subdomain = bc.sub_domain
    method = bc.method

    bc = type(bc)(V, val, subdomain, method=method)

    if zeroed:
        bc.homogenize()

    return bc


@coarsen.register(firedrake.functionspaceimpl.FunctionSpace)
@coarsen.register(firedrake.functionspaceimpl.WithGeometry)
def coarsen_function_space(V, coefficient_mapping=None):
    indices = []
    while True:
        if V.index is not None:
            indices.append(V.index)
        if V.component is not None:
            indices.append(V.component)
        if V.parent is not None:
            V = V.parent
        else:
            break

    mesh = coarsen(V.mesh())

    V = firedrake.FunctionSpace(mesh, V.ufl_element())
    for i in reversed(indices):
        V = V.sub(i)
    return V


@coarsen.register(firedrake.Function)
def coarsen_function(expr, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        V = coarsen(expr.function_space())
        new = firedrake.Function(V)
        firedrake.inject(expr, new)
    return new


@coarsen.register(firedrake.Constant)
def coarsen_constant(expr, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        mesh = coarsen(expr.ufl_domain())
        if len(expr.ufl_shape) == 0:
            val = expr.dat.data_ro[0]
        else:
            val = expr.dat.data_ro.copy()
        new = firedrake.Constant(value=val, domain=mesh)
    return new


@coarsen.register(firedrake.NonlinearVariationalProblem)
def coarsen_nlvp(problem, coefficient_mapping=None):
    # Build set of coefficients we need to coarsen
    seen = set()
    coefficients = problem.F.coefficients() + problem.J.coefficients()
    if problem.Jp is not None:
        coefficients = coefficients + problem.Jp.coefficients()

    # Coarsen them, and remember where from.
    if coefficient_mapping is None:
        coefficient_mapping = {}
    for c in coefficients:
        if c not in seen:
            coefficient_mapping[c] = coarsen(c, coefficient_mapping=coefficient_mapping)
            seen.add(c)

    u = coefficient_mapping[problem.u]

    bcs = [coarsen(bc) for bc in problem.bcs]
    J = coarsen(problem.J, coefficient_mapping=coefficient_mapping)
    Jp = coarsen(problem.Jp, coefficient_mapping=coefficient_mapping)
    F = coarsen(problem.F, coefficient_mapping=coefficient_mapping)

    problem = firedrake.NonlinearVariationalProblem(F, u, bcs=bcs, J=J, Jp=Jp,
                                                    form_compiler_parameters=problem.form_compiler_parameters)
    return problem


@coarsen.register(firedrake.solving_utils._SNESContext)
def coarsen_snescontext(context, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}

    # Have we already done this?
    coarse = context._coarse
    if coarse is not None:
        return coarse

    problem = coarsen(context._problem, coefficient_mapping=coefficient_mapping)
    appctx = context.appctx
    new_appctx = {}
    for k in sorted(appctx.keys()):
        v = appctx[k]
        if k != "state":
            # Constructor makes this one.
            try:
                new_appctx[k] = coarsen(v)
            except CoarseningError:
                # Assume not something that needs coarsening (e.g. float)
                new_appctx[k] = v
    coarse = type(context)(problem,
                           mat_type=context.mat_type,
                           pmat_type=context.pmat_type,
                           appctx=new_appctx)
    coarse._fine = context
    context._coarse = coarse
    return coarse


class Interpolation(object):
    def __init__(self, cfn, ffn, prolong, restrict, cbcs=None, fbcs=None):
        self.cfn = cfn
        self.ffn = ffn
        self.cbcs = cbcs or []
        self.fbcs = fbcs or []
        self.prolong = prolong
        self.restrict = restrict

    def mult(self, mat, x, y, inc=False):
        with self.cfn.dat.vec_wo as v:
            x.copy(v)
        self.prolong(self.cfn, self.ffn)
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
        with self.ffn.dat.vec_wo as v:
            x.copy(v)
        self.restrict(self.ffn, self.cfn)
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
    def __init__(self, cfn, ffn, inject, cbcs=None):
        self.cfn = cfn
        self.ffn = ffn
        self.cbcs = cbcs or []
        self.inject = inject

    def multTranspose(self, mat, x, y):
        with self.ffn.dat.vec_wo as v:
            x.copy(v)
        self.inject(self.ffn, self.cfn)
        for bc in self.cbcs:
            bc.apply(self.cfn)
        with self.cfn.dat.vec_ro as v:
            v.copy(y)


def create_interpolation(dmc, dmf):
    cctx = firedrake.dmhooks.get_appctx(dmc)
    fctx = firedrake.dmhooks.get_appctx(dmf)

    prolong, _, _ = firedrake.dmhooks.get_transfer_operators(dmc)
    _, restrict, _ = firedrake.dmhooks.get_transfer_operators(dmf)
    V_c = cctx._problem.u.function_space()
    V_f = fctx._problem.u.function_space()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs
    fbcs = fctx._problem.bcs

    ctx = Interpolation(cfn, ffn, prolong, restrict, cbcs, fbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat, None


def create_injection(dmc, dmf):
    cctx = firedrake.dmhooks.get_appctx(dmc)
    fctx = firedrake.dmhooks.get_appctx(dmf)

    _, _, inject = firedrake.dmhooks.get_transfer_operators(dmf)
    V_c = cctx._problem.u.function_space()
    V_f = fctx._problem.u.function_space()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs

    ctx = Injection(cfn, ffn, inject, cbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat
