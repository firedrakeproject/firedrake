import numpy
import ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.multifunction import MultiFunction

from functools import singledispatch, partial
import firedrake
from firedrake.petsc import PETSc

from . import utils


__all__ = ("coarsen", "refine")

TransferDirection = firedrake.dmhooks.TransferDirection


class TransferError(Exception):
    """Exception raised when coarsening symbolic information fails."""
    pass


class TransferIntegrand(MultiFunction):

    """'Transfer' a :class:`ufl.Expr` by replacing coefficients,
    arguments and domain data with transferred equivalents."""

    def __init__(self, transfer, coefficient_mapping):
        """
        :arg transfer: function to transfer expressions to a new mesh.
        :arg coefficient_mapping: mapping from old to new coefficients.
        """
        self.coefficient_mapping = coefficient_mapping
        self.transfer = transfer
        super().__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        V = self.transfer(o.function_space(), self.transfer)
        return o.reconstruct(V)

    def coefficient(self, o):
        return self.transfer(o, self.transfer, coefficient_mapping=self.coefficient_mapping)

    def geometric_quantity(self, o):
        return type(o)(self.transfer(o.ufl_domain(), self.transfer))

    def circumradius(self, o):
        mesh = self.transfer(o.ufl_domain(), self.transfer)
        return firedrake.Circumradius(mesh)

    def facet_normal(self, o):
        mesh = self.transfer(o.ufl_domain(), self.transfer)
        return firedrake.FacetNormal(mesh)


@singledispatch
def transfer(expr, self, direction, *, coefficient_mapping=None):
    # Default, just send it back
    return expr


def coarsen(expr, self, direction=TransferDirection.COARSEN, *,
            coefficient_mapping=None):
    return transfer(expr, self, direction=TransferDirection.COARSEN,
                    coefficient_mapping=coefficient_mapping)


def refine(expr, self, direction=TransferDirection.REFINE, *,
           coefficient_mapping=None):
    return transfer(expr, self, direction=TransferDirection.REFINE,
                    coefficient_mapping=coefficient_mapping)


@transfer.register(ufl.Mesh)
def transfer_mesh(mesh, self, direction, *, coefficient_mapping=None):
    hierarchy, level = utils.get_level(mesh)
    if hierarchy is None:
        raise TransferError("No mesh hierarchy available")
    if direction == TransferDirection.COARSEN:
        return hierarchy[level - 1]
    elif direction == TransferDirection.REFINE:
        return hierarchy[level + 1]
    else:
        raise TransferError(f"Unhandled transfer direction {direction}")


@transfer.register(ufl.classes.Expr)
def transfer_expr(expr, self, direction, *, coefficient_mapping=None):
    if expr is None:
        return None
    mapper = TransferIntegrand(self, coefficient_mapping)
    return map_expr_dag(mapper, expr)


@transfer.register(ufl.Form)
def transfer_form(form, self, direction, *, coefficient_mapping=None):
    """Return a coarse mesh version of a form

    :arg form: The :class:`~ufl.classes.Form` to coarsen.
    :kwarg mapping: an optional map from coefficients to their
        coarsened equivalents.

    This maps over the form and replaces coefficients and arguments
    with their coarse mesh equivalents."""
    if form is None:
        return None

    mapper = TransferIntegrand(self, coefficient_mapping)
    integrals = []
    for it in form.integrals():
        integrand = map_expr_dag(mapper, it.integrand())
        mesh = it.ufl_domain()
        new_mesh = self(mesh, self, direction,
                        coefficient_mapping=coefficient_mapping)
        if isinstance(integrand, ufl.classes.Zero):
            continue
        if it.subdomain_data() is not None:
            raise TransferError("Don't know how to coarsen subdomain data")
        new_itg = it.reconstruct(integrand=integrand,
                                 domain=new_mesh)
        integrals.append(new_itg)
    form = ufl.Form(integrals)
    form._cache["coefficient_mapping"] = coefficient_mapping
    return form


@transfer.register(firedrake.DirichletBC)
def transfer_bc(bc, self, direction, *, coefficient_mapping=None):
    V = self(bc.function_space(), self, direction,
             coefficient_mapping=coefficient_mapping)
    val = self(bc._original_val, self, direction,
               coefficient_mapping=coefficient_mapping)
    zeroed = bc._currently_zeroed
    subdomain = bc.sub_domain
    method = bc.method

    bc = type(bc)(V, val, subdomain, method=method)

    if zeroed:
        bc.homogenize()

    return bc


@transfer.register(firedrake.functionspaceimpl.FunctionSpace)
@transfer.register(firedrake.functionspaceimpl.WithGeometry)
def transfer_function_space(V, self, direction, *, coefficient_mapping=None):
    if direction == TransferDirection.COARSEN:
        attr = "_coarse"
    elif direction == TransferDirection.REFINE:
        attr = "_fine"
    else:
        raise TransferError(f"Unhandled transfer direction {direction}")
    if hasattr(V, attr):
        return getattr(V, attr)
    from firedrake.dmhooks import get_parent, push_parent, pop_parent, add_hook
    oldV = V
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

    mesh = self(V.mesh(), self, direction)

    newV = firedrake.FunctionSpace(mesh, V.ufl_element())

    for i in reversed(indices):
        newV = newV.sub(i)
    if direction == TransferDirection.COARSEN:
        newV._fine = oldV
        oldV._coarse = newV
    elif direction == TransferDirection.REFINE:
        newV._coarse = oldV
        oldV._fine = newV

    # FIXME: This replicates some code from dmhooks.coarsen, but we
    # can't do things there because that code calls this code.

    # We need to move these operators over here because if we have
    # fieldsplits + MG with auxiliary coefficients on spaces other
    # than which we do the MG, dm.coarsen is never called, so the
    # hooks are not attached. Instead we just call (say) inject which
    # coarsens the functionspace.
    newdm = newV.dm
    parent = get_parent(oldV.dm)
    try:
        add_hook(parent, setup=partial(push_parent, newdm, parent), teardown=partial(pop_parent, newdm, parent),
                 call_setup=True)
    except ValueError:
        # Not in an add_hooks context
        pass

    return newV


@transfer.register(firedrake.Function)
def transfer_function(expr, self, direction, *, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        V = expr.function_space()
        manager = firedrake.dmhooks.get_transfer_manager(expr.function_space().dm)
        V = self(expr.function_space(), self, direction)
        if direction == TransferDirection.COARSEN:
            name = f"coarse_{expr.name()}"
            transfer = manager.inject
        elif direction == TransferDirection.REFINE:
            name = f"fine_{expr.name()}"
            transfer = manager.prolong
        else:
            raise TransferError(f"Unhandled transfer direction {direction}")
        new = firedrake.Function(V, name=name)
        transfer(expr, new)
    return new


@transfer.register(firedrake.Constant)
def transfer_constant(expr, self, direction, *, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        mesh = self(expr.ufl_domain(), self, direction)
        new = firedrake.Constant(numpy.zeros(expr.ufl_shape,
                                             dtype=expr.dat.dtype),
                                 domain=mesh)
        # Share data pointer
        new.dat = expr.dat
    return new


@transfer.register(firedrake.NonlinearVariationalProblem)
def transfer_nlvp(problem, self, direction, *, coefficient_mapping=None):
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
            coefficient_mapping[c] = self(c, self, direction, coefficient_mapping=coefficient_mapping)
            seen.add(c)

    u = coefficient_mapping[problem.u]

    bcs = [self(bc, self, direction) for bc in problem.bcs]
    J = self(problem.J, self, direction,
             coefficient_mapping=coefficient_mapping)
    Jp = self(problem.Jp, self, direction,
              coefficient_mapping=coefficient_mapping)
    F = self(problem.F, self, direction,
             coefficient_mapping=coefficient_mapping)

    problem = firedrake.NonlinearVariationalProblem(F, u, bcs=bcs, J=J, Jp=Jp,
                                                    form_compiler_parameters=problem.form_compiler_parameters)
    return problem


@transfer.register(firedrake.VectorSpaceBasis)
def transfer_vectorspacebasis(basis, self, direction, *, coefficient_mapping=None):
    new_vecs = [self(vec, self, direction,
                     coefficient_mapping=coefficient_mapping) for vec in basis._vecs]
    vsb = firedrake.VectorSpaceBasis(new_vecs, constant=basis._constant)
    vsb.orthonormalize()
    return vsb


@transfer.register(firedrake.MixedVectorSpaceBasis)
def transfer_mixedvectorspacebasis(mspbasis, self, direction, *, coefficient_mapping=None):
    new_V = self(mspbasis._function_space, self, direction,
                 coefficient_mapping=coefficient_mapping)
    new_bases = []

    for basis in mspbasis._bases:
        if isinstance(basis, firedrake.VectorSpaceBasis):
            new_bases.append(self(basis, self, direction,
                                  coefficient_mapping=coefficient_mapping))
        elif basis.index is not None:
            new_bases.append(new_V.sub(basis.index))
        else:
            raise RuntimeError("MixedVectorSpaceBasis can only contain vector space bases or indexed function spaces")

    return firedrake.MixedVectorSpaceBasis(new_V, new_bases)


@transfer.register(firedrake.solving_utils._SNESContext)
def transfer_snescontext(context, self, direction, *, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}

    if direction == TransferDirection.COARSEN:
        # Have we already done this?
        coarse = context._coarse
        if coarse is not None:
            return coarse
    elif direction == TransferDirection.REFINE:
        fine = context._fine
        if fine is not None:
            return fine
    else:
        raise TransferError(f"Unhandled transfer direction {direction}")

    problem = self(context._problem, self, direction,
                   coefficient_mapping=coefficient_mapping)
    appctx = context.appctx
    new_appctx = {}
    for k in sorted(appctx.keys()):
        v = appctx[k]
        if k != "state":
            # Constructor makes this one.
            try:
                new_appctx[k] = self(v, self, direction,
                                     coefficient_mapping=coefficient_mapping)
            except TransferError:
                # Assume not something that needs coarsening (e.g. float)
                new_appctx[k] = v
    new_ctx = type(context)(problem,
                            mat_type=context.mat_type,
                            pmat_type=context.pmat_type,
                            appctx=new_appctx,
                            transfer_manager=context.transfer_manager)
    if direction == TransferDirection.COARSEN:
        new_ctx._fine = context
        context._coarse = new_ctx
    elif direction == TransferDirection.REFINE:
        new_ctx._coarse = context
        context._fine = new_ctx
    else:
        raise TransferError(f"Unhandled transfer direction {direction}")

    # Now that we have the transferred snescontext, push it to the
    # transferred DMs. Otherwise they won't have the right transfer
    # manager when they are transferred in turn
    from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
    from firedrake.dmhooks import add_hook, get_parent
    from itertools import chain
    for val in chain(coefficient_mapping.values(), (bc._original_val for bc in problem.bcs)):
        if isinstance(val, firedrake.function.Function):
            V = val.function_space()
            newdm = V.dm
            parentdm = get_parent(context._problem.u.function_space().dm)

            # Now attach the hook to the parent DM
            if get_appctx(newdm) is None:
                push_appctx(newdm, new_ctx)
                teardown = partial(pop_appctx, newdm, new_ctx)
                add_hook(parentdm, teardown=teardown)

    ises = problem.J.arguments()[0].function_space()._ises
    new_ctx._nullspace = self(context._nullspace, self, direction,
                              coefficient_mapping=coefficient_mapping)
    new_ctx.set_nullspace(new_ctx._nullspace, ises, transpose=False, near=False)
    new_ctx._nullspace_T = self(context._nullspace_T, self, direction,
                                coefficient_mapping=coefficient_mapping)
    new_ctx.set_nullspace(new_ctx._nullspace_T, ises, transpose=True, near=False)
    new_ctx._near_nullspace = self(context._near_nullspace, self, direction,
                                   coefficient_mapping=coefficient_mapping)
    new_ctx.set_nullspace(new_ctx._near_nullspace, ises, transpose=False, near=True)

    return new_ctx


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
