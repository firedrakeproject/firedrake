import numpy
import ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.multifunction import MultiFunction

from functools import singledispatch, partial
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

    def __init__(self, coarsen, coefficient_mapping):
        self.coefficient_mapping = coefficient_mapping or {}
        self.coarsen = coarsen
        super(CoarsenIntegrand, self).__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        V = self.coarsen(o.function_space(), self.coarsen)
        return o.reconstruct(V)

    def coefficient(self, o):
        return self.coarsen(o, self.coarsen, coefficient_mapping=self.coefficient_mapping)

    def geometric_quantity(self, o):
        return type(o)(self.coarsen(o.ufl_domain(), self.coarsen))

    def circumradius(self, o):
        mesh = self.coarsen(o.ufl_domain(), self.coarsen)
        return firedrake.Circumradius(mesh)

    def facet_normal(self, o):
        mesh = self.coarsen(o.ufl_domain(), self.coarsen)
        return firedrake.FacetNormal(mesh)


@singledispatch
def coarsen(expr, self, coefficient_mapping=None):
    # Default, just send it back
    return expr


@coarsen.register(ufl.Mesh)
def coarsen_mesh(mesh, self, coefficient_mapping=None):
    hierarchy, level = utils.get_level(mesh)
    if hierarchy is None:
        raise CoarseningError("No mesh hierarchy available")
    return hierarchy[level - 1]


@coarsen.register(ufl.classes.Expr)
def coarse_expr(expr, self, coefficient_mapping=None):
    if expr is None:
        return None
    mapper = CoarsenIntegrand(self, coefficient_mapping)
    return map_expr_dag(mapper, expr)


@coarsen.register(ufl.Form)
def coarsen_form(form, self, coefficient_mapping=None):
    """Return a coarse mesh version of a form

    :arg form: The :class:`~ufl.classes.Form` to coarsen.
    :kwarg mapping: an optional map from coefficients to their
        coarsened equivalents.

    This maps over the form and replaces coefficients and arguments
    with their coarse mesh equivalents."""
    if form is None:
        return None

    mapper = CoarsenIntegrand(self, coefficient_mapping)
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
    form = ufl.Form(integrals)
    form._cache["coefficient_mapping"] = coefficient_mapping
    return form


@coarsen.register(firedrake.DirichletBC)
def coarsen_bc(bc, self, coefficient_mapping=None):
    V = self(bc.function_space(), self, coefficient_mapping=coefficient_mapping)
    val = self(bc._original_val, self, coefficient_mapping=coefficient_mapping)
    zeroed = bc._currently_zeroed
    subdomain = bc.sub_domain
    method = bc.method

    bc = type(bc)(V, val, subdomain, method=method)

    if zeroed:
        bc.homogenize()

    return bc


@coarsen.register(firedrake.functionspaceimpl.FunctionSpace)
@coarsen.register(firedrake.functionspaceimpl.WithGeometry)
def coarsen_function_space(V, self, coefficient_mapping=None):
    if hasattr(V, "_coarse"):
        return V._coarse
    from firedrake.dmhooks import get_parent, push_parent, pop_parent, add_hook
    fine = V
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

    mesh = self(V.mesh(), self)

    V = firedrake.FunctionSpace(mesh, V.ufl_element())

    for i in reversed(indices):
        V = V.sub(i)
    V._fine = fine
    fine._coarse = V

    # FIXME: This replicates some code from dmhooks.coarsen, but we
    # can't do things there because that code calls this code.

    # We need to move these operators over here because if we have
    # fieldsplits + MG with auxiliary coefficients on spaces other
    # than which we do the MG, dm.coarsen is never called, so the
    # hooks are not attached. Instead we just call (say) inject which
    # coarsens the functionspace.
    cdm = V.dm
    parent = get_parent(fine.dm)
    try:
        add_hook(parent, setup=partial(push_parent, cdm, parent), teardown=partial(pop_parent, cdm, parent),
                 call_setup=True)
    except ValueError:
        # Not in an add_hooks context
        pass

    return V


@coarsen.register(firedrake.Function)
def coarsen_function(expr, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        V = expr.function_space()
        manager = firedrake.dmhooks.get_transfer_manager(expr.function_space().dm)
        V = self(expr.function_space(), self)
        new = firedrake.Function(V, name="coarse_%s" % expr.name())
        manager.inject(expr, new)
    return new


@coarsen.register(firedrake.Constant)
def coarsen_constant(expr, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        mesh = self(expr.ufl_domain(), self)
        new = firedrake.Constant(numpy.zeros(expr.ufl_shape,
                                             dtype=expr.dat.dtype),
                                 domain=mesh)
        # Share data pointer
        new.dat = expr.dat
    return new


@coarsen.register(firedrake.NonlinearVariationalProblem)
def coarsen_nlvp(problem, self, coefficient_mapping=None):
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
            coefficient_mapping[c] = self(c, self, coefficient_mapping=coefficient_mapping)
            seen.add(c)

    u = coefficient_mapping[problem.u]

    bcs = [self(bc, self) for bc in problem.bcs]
    J = self(problem.J, self, coefficient_mapping=coefficient_mapping)
    Jp = self(problem.Jp, self, coefficient_mapping=coefficient_mapping)
    F = self(problem.F, self, coefficient_mapping=coefficient_mapping)

    problem = firedrake.NonlinearVariationalProblem(F, u, bcs=bcs, J=J, Jp=Jp,
                                                    form_compiler_parameters=problem.form_compiler_parameters)
    return problem


@coarsen.register(firedrake.VectorSpaceBasis)
def coarsen_vectorspacebasis(basis, self, coefficient_mapping=None):
    coarse_vecs = [self(vec, self, coefficient_mapping=coefficient_mapping) for vec in basis._vecs]
    vsb = firedrake.VectorSpaceBasis(coarse_vecs, constant=basis._constant)
    vsb.orthonormalize()
    return vsb


@coarsen.register(firedrake.MixedVectorSpaceBasis)
def coarsen_mixedvectorspacebasis(mspbasis, self, coefficient_mapping=None):
    coarse_V = self(mspbasis._function_space, self, coefficient_mapping=coefficient_mapping)
    coarse_bases = []

    for basis in mspbasis._bases:
        if isinstance(basis, firedrake.VectorSpaceBasis):
            coarse_bases.append(self(basis, self, coefficient_mapping=coefficient_mapping))
        elif basis.index is not None:
            coarse_bases.append(coarse_V.sub(basis.index))
        else:
            raise RuntimeError("MixedVectorSpaceBasis can only contain vector space bases or indexed function spaces")

    return firedrake.MixedVectorSpaceBasis(coarse_V, coarse_bases)


@coarsen.register(firedrake.solving_utils._SNESContext)
def coarsen_snescontext(context, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}

    # Have we already done this?
    coarse = context._coarse
    if coarse is not None:
        return coarse

    problem = self(context._problem, self, coefficient_mapping=coefficient_mapping)
    appctx = context.appctx
    new_appctx = {}
    for k in sorted(appctx.keys()):
        v = appctx[k]
        if k != "state":
            # Constructor makes this one.
            try:
                new_appctx[k] = self(v, self, coefficient_mapping=coefficient_mapping)
            except CoarseningError:
                # Assume not something that needs coarsening (e.g. float)
                new_appctx[k] = v
    coarse = type(context)(problem,
                           mat_type=context.mat_type,
                           pmat_type=context.pmat_type,
                           appctx=new_appctx,
                           transfer_manager=context.transfer_manager)
    coarse._fine = context
    context._coarse = coarse

    # Now that we have the coarse snescontext, push it to the coarsened DMs
    # Otherwise they won't have the right transfer manager when they are
    # coarsened in turn
    from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
    from firedrake.dmhooks import add_hook, get_parent
    from itertools import chain
    for val in chain(coefficient_mapping.values(), (bc._original_val for bc in problem.bcs)):
        if isinstance(val, firedrake.function.Function):
            V = val.function_space()
            coarseneddm = V.dm
            parentdm = get_parent(context._problem.u.function_space().dm)

            # Now attach the hook to the parent DM
            if get_appctx(coarseneddm) is None:
                push_appctx(coarseneddm, coarse)
                teardown = partial(pop_appctx, coarseneddm, coarse)
                add_hook(parentdm, teardown=teardown)

    ises = problem.J.arguments()[0].function_space()._ises
    coarse._nullspace = self(context._nullspace, self, coefficient_mapping=coefficient_mapping)
    coarse.set_nullspace(coarse._nullspace, ises, transpose=False, near=False)
    coarse._nullspace_T = self(context._nullspace_T, self, coefficient_mapping=coefficient_mapping)
    coarse.set_nullspace(coarse._nullspace_T, ises, transpose=True, near=False)
    coarse._near_nullspace = self(context._near_nullspace, self, coefficient_mapping=coefficient_mapping)
    coarse.set_nullspace(coarse._near_nullspace, ises, transpose=False, near=True)

    return coarse


class Interpolation(object):
    def __init__(self, cfn, ffn, manager, cbcs=None, fbcs=None):
        self.cfn = cfn
        self.ffn = ffn
        self.cbcs = cbcs or []
        self.fbcs = fbcs or []
        self.manager = manager

    def mult(self, mat, x, y, inc=False):
        with self.cfn.dat.vec_wo as v:
            x.copy(v)
        self.manager.prolong(self.cfn, self.ffn)
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
        self.manager.restrict(self.ffn, self.cfn)
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
    def __init__(self, cfn, ffn, manager, cbcs=None):
        self.cfn = cfn
        self.ffn = ffn
        self.cbcs = cbcs or []
        self.manager = manager

    def multTranspose(self, mat, x, y):
        with self.ffn.dat.vec_wo as v:
            x.copy(v)
        self.manager.inject(self.ffn, self.cfn)
        for bc in self.cbcs:
            bc.apply(self.cfn)
        with self.cfn.dat.vec_ro as v:
            v.copy(y)


def create_interpolation(dmc, dmf):

    cctx = firedrake.dmhooks.get_appctx(dmc)
    fctx = firedrake.dmhooks.get_appctx(dmf)

    manager = firedrake.dmhooks.get_transfer_manager(dmf)

    V_c = cctx._problem.u.function_space()
    V_f = fctx._problem.u.function_space()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs
    fbcs = fctx._problem.bcs

    ctx = Interpolation(cfn, ffn, manager, cbcs, fbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat, None


def create_injection(dmc, dmf):
    cctx = firedrake.dmhooks.get_appctx(dmc)
    fctx = firedrake.dmhooks.get_appctx(dmf)

    manager = firedrake.dmhooks.get_transfer_manager(dmf)

    V_c = cctx._problem.u.function_space()
    V_f = fctx._problem.u.function_space()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()

    cfn = firedrake.Function(V_c)
    ffn = firedrake.Function(V_f)
    cbcs = cctx._problem.bcs

    ctx = Injection(cfn, ffn, manager, cbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat
