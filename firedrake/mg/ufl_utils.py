import ufl
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_unique_domain
from ufl.duals import is_dual

from functools import singledispatch, singledispatchmethod, partial
import firedrake
from firedrake.petsc import PETSc
from firedrake.solving_utils import _SNESContext
from firedrake.dmhooks import (get_transfer_manager, get_appctx, push_appctx, pop_appctx,
                               get_parent, add_hook)

from . import utils


__all__ = ["coarsen", "refine"]


class ReconstructionError(Exception):
    """Exception raised when reconstructing symbolic information fails."""
    pass


def get_cache(direction, old):
    if direction is coarsen:
        return getattr(old, "_coarse", None)
    elif direction is refine:
        return getattr(old, "_fine", None)
    return None


def set_cache(direction, old, new):
    if direction is coarsen:
        old._coarse = new
    elif direction is refine:
        old._fine = new
    return new


class ReconstructIntegrand(DAGTraverser):

    """Reconstruct a :class:`ufl.Expr` with coefficients, arguments and
    domain data on the target mesh level."""

    def __init__(self, dispatch, coefficient_mapping=None):
        super().__init__()
        if coefficient_mapping is None:
            coefficient_mapping = {}
        self.coefficient_mapping = coefficient_mapping
        self.dispatch = dispatch

    @singledispatchmethod
    def process(self, o):
        return super().process(o)

    @process.register(ufl.BaseForm)
    @process.register(ufl.classes.Expr)
    def expr(self, o):
        return self.reuse_if_untouched(o)

    @process.register(ufl.classes.Argument)
    def argument(self, o):
        V = self.dispatch(o.function_space(), self.dispatch)
        return o.reconstruct(V)

    @process.register(ufl.classes.Coefficient)
    def coefficient(self, o):
        return self.dispatch(o, self.dispatch, coefficient_mapping=self.coefficient_mapping)

    @process.register(ufl.classes.Cofunction)
    def cofunction(self, o):
        return self.dispatch(o, self.dispatch, coefficient_mapping=self.coefficient_mapping)

    @process.register(ufl.classes.GeometricQuantity)
    def geometric_quantity(self, o):
        return type(o)(self.dispatch(extract_unique_domain(o), self.dispatch))

    @process.register(ufl.classes.Circumradius)
    def circumradius(self, o):
        mesh = self.dispatch(extract_unique_domain(o), self.dispatch)
        return firedrake.Circumradius(mesh)

    @process.register(ufl.classes.FacetNormal)
    def facet_normal(self, o):
        mesh = self.dispatch(extract_unique_domain(o), self.dispatch)
        return firedrake.FacetNormal(mesh)


@singledispatch
def _reconstruct(expr, self, coefficient_mapping=None):
    return expr


@singledispatch
def coarsen(expr, self, coefficient_mapping=None):
    return _reconstruct(expr, self, coefficient_mapping=coefficient_mapping)


@singledispatch
def refine(expr, self, coefficient_mapping=None):
    return _reconstruct(expr, self, coefficient_mapping=coefficient_mapping)


@coarsen.register(ufl.Mesh)
@coarsen.register(ufl.MeshSequence)
def coarsen_mesh(mesh, self, coefficient_mapping=None):
    hierarchy, level = utils.get_level(mesh)
    if hierarchy is None:
        raise ReconstructionError("No mesh hierarchy available")
    return hierarchy[level - 1]


@refine.register(ufl.Mesh)
@refine.register(ufl.MeshSequence)
def refine_mesh(mesh, self, coefficient_mapping=None):
    hierarchy, level = utils.get_level(mesh)
    if hierarchy is None:
        raise ReconstructionError("No mesh hierarchy available")
    return hierarchy[level + 1]


@_reconstruct.register(ufl.BaseForm)
@_reconstruct.register(ufl.classes.Expr)
def reconstruct_expr(expr, self, coefficient_mapping=None):
    if expr is None:
        return None
    mapper = ReconstructIntegrand(self, coefficient_mapping)
    return mapper(expr)


@_reconstruct.register(ufl.Form)
def reconstruct_form(form, self, coefficient_mapping=None):
    """Return a coarse or fine mesh version of a form.

    :arg form: The :class:`~ufl.classes.Form` to reconstruct.
    :kwarg coefficient_mapping: an optional map from coefficients to their
        reconstructed equivalents.

    This maps over the form and replaces coefficients and arguments
    with their coarse/fine mesh equivalents."""
    if form is None:
        return None

    mapper = ReconstructIntegrand(self, coefficient_mapping)
    integrals = []
    for it in form.integrals():
        integrand = mapper(it.integrand())
        mesh = it.ufl_domain()
        new_mesh = self(mesh, self)
        if isinstance(integrand, ufl.classes.Zero):
            continue
        if it.subdomain_data() is not None:
            raise ReconstructionError("Don't know how to reconstruct subdomain data")
        # Reconstruct secondary meshes in cross-mesh integrals (e.g. intersect_measures).
        integral_type_map = {self(d, self): itype
                             for d, itype in it.extra_domain_integral_type_map().items()}
        new_itg = it.reconstruct(integrand=integrand,
                                 domain=new_mesh,
                                 extra_domain_integral_type_map=integral_type_map)
        integrals.append(new_itg)
    form = ufl.Form(integrals)
    return form


@_reconstruct.register(ufl.FormSum)
def reconstruct_formsum(form, self, coefficient_mapping=None):
    return type(form)(*[(self(ci, self, coefficient_mapping=coefficient_mapping),
                         self(wi, self, coefficient_mapping=coefficient_mapping))
                        for ci, wi in zip(form.components(), form.weights())])


@_reconstruct.register(firedrake.DirichletBC)
def reconstruct_bc(bc, self, coefficient_mapping=None):
    V = self(bc.function_space(), self, coefficient_mapping=coefficient_mapping)
    val = self(bc._original_arg, self, coefficient_mapping=coefficient_mapping)
    subdomain = bc.sub_domain

    return type(bc)(V, val, subdomain)


@_reconstruct.register(firedrake.EquationBC)
def reconstruct_equation_bc(ebc, self, coefficient_mapping=None):
    J = self(ebc._J.f, self, coefficient_mapping=coefficient_mapping)
    Jp = self(ebc._Jp.f, self, coefficient_mapping=coefficient_mapping)
    u = self(ebc._F.u, self, coefficient_mapping=coefficient_mapping)
    sub_domain = ebc._F.sub_domain
    bcs = [self(bc, self, coefficient_mapping=coefficient_mapping)
           for bc in ebc.dirichlet_bcs()]
    V = self(ebc._F.function_space(), self, coefficient_mapping=coefficient_mapping)
    lhs = self(ebc.eq.lhs, self, coefficient_mapping=coefficient_mapping)
    rhs = self(ebc.eq.rhs, self, coefficient_mapping=coefficient_mapping)

    return type(ebc)(lhs == rhs, u, sub_domain, V=V, bcs=bcs, J=J, Jp=Jp)


@coarsen.register(firedrake.functionspaceimpl.WithGeometryBase)
@refine.register(firedrake.functionspaceimpl.WithGeometryBase)
def reconstruct_function_space(V, self, coefficient_mapping=None):
    # Handle MixedFunctionSpace : V.reconstruct requires MeshSequence.
    mesh = V.mesh() if V.index is None else V.parent.mesh()
    new_mesh = self(mesh, self)
    cached = get_cache(self, V)
    if cached is not None and cached.mesh() == new_mesh:
        return cached

    reverse = coarsen if self is refine else refine
    V_parent = V
    while get_cache(reverse, V_parent) is not None:
        V_parent = get_cache(reverse, V_parent)
    name = V_parent.name
    if name is not None:
        mh, level = utils.get_level(new_mesh)
        name = f"{name}_level_{level}"
    V_new = V.reconstruct(mesh=new_mesh, name=name)
    set_cache(reverse, V_new, V)
    set_cache(self, V, V_new)
    return V_new


@coarsen.register(firedrake.Cofunction)
@coarsen.register(firedrake.Function)
def coarsen_function(expr, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        V = expr.function_space()
        Vnew = self(V, self)
        name = expr.name()
        if name is not None:
            try:
                name, prev_level = name.split("_level_")
            except ValueError:
                prev_level = 0
            level = int(prev_level) - 1
            name = f"{name}_level_{level}"

        new = firedrake.Function(Vnew, name=name)
        manager = get_transfer_manager(V.dm)
        if is_dual(expr):
            manager.restrict(expr, new)
        else:
            manager.inject(expr, new)
        coefficient_mapping[expr] = new
    return new


@refine.register(firedrake.Cofunction)
@refine.register(firedrake.Function)
def refine_function(expr, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}
    new = coefficient_mapping.get(expr)
    if new is None:
        V = expr.function_space()
        Vnew = self(V, self)
        name = expr.name()
        if name is not None:
            try:
                name, prev_level = name.split("_level_")
            except ValueError:
                prev_level = 0
            level = int(prev_level) + 1
            name = f"{name}_level_{level}"

        new = firedrake.Function(Vnew, name=name)
        new.interpolate(expr)
        coefficient_mapping[expr] = new
    return new


@_reconstruct.register(firedrake.NonlinearVariationalProblem)
def reconstruct_nlvp(problem, self, coefficient_mapping=None):
    # Have we done this already?
    mh, _ = utils.get_level(problem.u.function_space().mesh())
    cached = get_cache(self, problem)
    if cached is not None and mh is utils.get_level(cached.u.function_space().mesh())[0]:
        return cached

    def inject_on_restrict(fine, restriction, rscale, injection, coarse):
        manager = get_transfer_manager(fine)
        while coarse:
            cctx = get_appctx(coarse)
            cmapping = cctx._coefficient_mapping
            if cmapping is None:
                return
            for c, mapped in cmapping.items():
                _, clevel = utils.get_level(c.function_space().mesh())
                _, mlevel = utils.get_level(mapped.function_space().mesh())
                if is_dual(c):
                    if clevel > mlevel:
                        manager.restrict(c, mapped)
                elif clevel > mlevel:
                    manager.inject(c, mapped)
                elif clevel < mlevel:
                    manager.prolong(c, mapped)
                else:
                    mapped.assign(c)
            # Apply bcs
            if cctx.pre_apply_bcs:
                for bc in cctx._problem.dirichlet_bcs():
                    bc.apply(cctx._x)
            # When the solution is in the real space
            # PETSc fails to call this hook on coarse levels.
            # As a workaround, we inject into all levels.
            has_real_space = any(Vsub.ufl_element().family() == "Real"
                                 for Vsub in cctx._x.function_space())
            coarse = coarse.getCoarseDM() if has_real_space else None

    dm = problem.u_restrict.function_space().dm
    if not dm.getAttr("_coarsen_hook"):
        # The hook is persistent and cumulative, but also problem-independent.
        # Therefore, we are only adding it once.
        dm.addCoarsenHook(None, inject_on_restrict)
        dm.setAttr("_coarsen_hook", True)

    if coefficient_mapping is None:
        coefficient_mapping = {}

    bcs = [self(bc, self, coefficient_mapping=coefficient_mapping) for bc in problem.bcs]
    F = self(problem.F, self, coefficient_mapping=coefficient_mapping)
    J = self(problem.J, self, coefficient_mapping=coefficient_mapping)
    Jp = self(problem.Jp, self, coefficient_mapping=coefficient_mapping)
    u = coefficient_mapping[problem.u_restrict]

    new_problem = firedrake.NonlinearVariationalProblem(
        F, u, bcs=bcs, J=J, Jp=Jp, is_linear=problem.is_linear,
        form_compiler_parameters=problem.form_compiler_parameters)
    set_cache(self, problem, new_problem)
    return new_problem


@_reconstruct.register(firedrake.LinearEigenproblem)
def reconstruct_eigenproblem(problem, self, coefficient_mapping=None):
    # Have we done this already?
    mh, _ = utils.get_level(problem.output_space.mesh())
    cached = get_cache(self, problem)
    if cached is not None and mh is utils.get_level(cached.output_space.mesh())[0]:
        return cached

    if coefficient_mapping is None:
        coefficient_mapping = {}
    bcs = [self(bc, self, coefficient_mapping=coefficient_mapping)
           for bc in problem.bcs]
    A = self(problem.A, self, coefficient_mapping=coefficient_mapping)
    M = self(problem.M, self, coefficient_mapping=coefficient_mapping)
    new_problem = firedrake.LinearEigenproblem(A, M, bcs=bcs,
                                               bc_shift=problem.bc_shift, restrict=problem.restrict)
    set_cache(self, problem, new_problem)
    return new_problem


@_reconstruct.register(firedrake.VectorSpaceBasis)
def reconstruct_vectorspacebasis(basis, self, coefficient_mapping=None):
    # Do not add basis._vecs to the coefficient_mapping,
    # as they need to be normalized, and are not meant to be reinjected
    coarse_vecs = [self(vec, self) for vec in basis._vecs]
    vsb = firedrake.VectorSpaceBasis(coarse_vecs, constant=basis._constant, comm=basis.comm)
    vsb.orthonormalize()
    return vsb


@_reconstruct.register(firedrake.MixedVectorSpaceBasis)
def reconstruct_mixedvectorspacebasis(mspbasis, self, coefficient_mapping=None):
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


@_reconstruct.register(_SNESContext)
def reconstruct_snescontext(context, self, coefficient_mapping=None):
    if coefficient_mapping is None:
        coefficient_mapping = {}

    if self == refine:
        new_attr = "_fine"
        old_attr = "_coarse"
    else:
        new_attr = "_coarse"
        old_attr = "_fine"

    # Have we already done this?
    new_context = getattr(context, new_attr)
    if new_context is not None:
        return new_context

    problem = self(context._problem, self, coefficient_mapping=coefficient_mapping)
    appctx = context.appctx
    new_appctx = {}
    for k in sorted(appctx.keys()):
        v = appctx[k]
        if k != "state":
            # Constructor makes this one.
            try:
                new_appctx[k] = self(v, self, coefficient_mapping=coefficient_mapping)
            except ReconstructionError:
                # Assume not something that needs reconstruction (e.g. float)
                new_appctx[k] = v

    # Get options prefix for current level
    parent_context = context
    while getattr(parent_context, old_attr, None):
        parent_context = getattr(parent_context, old_attr, None)

    parent_prefix = parent_context.options_prefix
    opts = PETSc.Options(parent_prefix)
    if opts.getString("snes_type", "") == "fas":
        solver_prefix = "fas_"
    else:
        solver_prefix = "mg_"
    _, level = utils.get_level(problem.u_restrict.function_space().mesh())
    if level == 0:
        levels_prefix = f"{solver_prefix}coarse_"
    else:
        levels_prefix = f"{solver_prefix}levels_"
    current_level_prefix = f"{solver_prefix}levels_{level}_"
    options_prefix = f"{parent_prefix}{current_level_prefix}"

    # Use different mat_type on each level
    mat_type = None
    pmat_type = None
    sub_mat_type = None
    sub_pmat_type = None
    for prefix in (levels_prefix, current_level_prefix):
        mat_type = opts.getString(f"{prefix}mat_type", "") or mat_type
        pmat_type = opts.getString(f"{prefix}pmat_type", "") or pmat_type
        sub_mat_type = opts.getString(f"{prefix}sub_mat_type", "") or sub_mat_type
        sub_pmat_type = opts.getString(f"{prefix}sub_pmat_type", "") or sub_pmat_type

    pmat_type = pmat_type or mat_type
    sub_pmat_type = sub_pmat_type or sub_mat_type
    new_context = context.reconstruct(problem=problem,
                                      mat_type=mat_type,
                                      pmat_type=pmat_type,
                                      sub_mat_type=sub_mat_type,
                                      sub_pmat_type=sub_pmat_type,
                                      appctx=new_appctx,
                                      options_prefix=options_prefix,
                                      )
    new_context._coefficient_mapping = coefficient_mapping
    setattr(new_context, old_attr, context)
    setattr(context, new_attr, new_context)

    solutiondm = context._problem.u_restrict.function_space().dm
    parentdm = get_parent(solutiondm)
    # Now that we have the reconstructed snescontext, push it to the reconstructed DMs.
    # Otherwise they will not have the right transfer manager when they are reconstructed in turn.
    for val in coefficient_mapping.values():
        if isinstance(val, (firedrake.Function, firedrake.Cofunction)):
            V = val.function_space()
            newdm = V.dm

            # Now attach the hook to the parent DM
            if get_appctx(newdm) is None:
                push_appctx(newdm, new_context)
                if parentdm.getAttr("__setup_hooks__"):
                    add_hook(parentdm, teardown=partial(pop_appctx, newdm, new_context))

    ises = new_context._x.function_space()._ises
    new_context._nullspace = self(context._nullspace, self, coefficient_mapping=coefficient_mapping)
    new_context.set_nullspace(new_context._nullspace, ises, transpose=False, near=False)
    new_context._nullspace_T = self(context._nullspace_T, self, coefficient_mapping=coefficient_mapping)
    new_context.set_nullspace(new_context._nullspace_T, ises, transpose=True, near=False)
    new_context._near_nullspace = self(context._near_nullspace, self, coefficient_mapping=coefficient_mapping)
    new_context.set_nullspace(new_context._near_nullspace, ises, transpose=False, near=True)

    return new_context


@_reconstruct.register(firedrake.slate.AssembledVector)
def reconstruct_slate_assembled_vector(tensor, self, coefficient_mapping=None):
    form = self(tensor.form, self, coefficient_mapping=coefficient_mapping)
    return type(tensor)(form)


@_reconstruct.register(firedrake.slate.BlockAssembledVector)
def reconstruct_slate_block_assembled_vector(tensor, self, coefficient_mapping=None):
    form = self(tensor.form, self, coefficient_mapping=coefficient_mapping)
    block = self(tensor.block, self, coefficient_mapping=coefficient_mapping)
    return type(tensor)(form, *block.children, block.indices)


@_reconstruct.register(firedrake.slate.Block)
def reconstruct_slate_block(tensor, self, coefficient_mapping=None):
    children = (self(c, self, coefficient_mapping=coefficient_mapping) for c in tensor.children)
    return type(tensor)(*children, indices=tensor._indices)


@_reconstruct.register(firedrake.slate.Factorization)
def reconstruct_slate_factorization(tensor, self, coefficient_mapping=None):
    children = (self(c, self, coefficient_mapping=coefficient_mapping) for c in tensor.children)
    return type(tensor)(*children, decomposition=tensor.decomposition)


@_reconstruct.register(firedrake.slate.Tensor)
def reconstruct_slate_tensor(tensor, self, coefficient_mapping=None):
    form = self(tensor.form, self, coefficient_mapping=coefficient_mapping)
    return type(tensor)(form, diagonal=tensor.diagonal)


@_reconstruct.register(firedrake.slate.TensorOp)
def reconstruct_slate_tensor_op(tensor, self, coefficient_mapping=None):
    children = (self(c, self, coefficient_mapping=coefficient_mapping) for c in tensor.children)
    return type(tensor)(*children)


class Interpolation(object):
    def __init__(self, Vcoarse, Vfine, manager, cbcs=None, fbcs=None):
        self.cprimal = firedrake.Function(Vcoarse)
        self.fprimal = firedrake.Function(Vfine)
        self.cdual = firedrake.Cofunction(Vcoarse.dual())
        self.fdual = firedrake.Cofunction(Vfine.dual())
        self.cbcs = cbcs or []
        self.fbcs = fbcs or []
        self.manager = manager

    def mult(self, mat, x, y, inc=False):
        with self.cprimal.dat.vec_wo as v:
            x.copy(v)
        self.manager.prolong(self.cprimal, self.fprimal)
        for bc in self.fbcs:
            bc.zero(self.fprimal)
        with self.fprimal.dat.vec_ro as v:
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
        with self.fdual.dat.vec_wo as v:
            x.copy(v)
        self.manager.restrict(self.fdual, self.cdual)
        for bc in self.cbcs:
            bc.zero(self.cdual)
        with self.cdual.dat.vec_ro as v:
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
    def __init__(self, Vcoarse, Vfine, manager, cbcs=None):
        self.cfn = firedrake.Function(Vcoarse)
        self.ffn = firedrake.Function(Vfine)
        self.cbcs = cbcs or []
        self.manager = manager

    def mult(self, mat, x, y):
        with self.ffn.dat.vec_wo as v:
            x.copy(v)
        self.manager.inject(self.ffn, self.cfn)
        for bc in self.cbcs:
            bc.apply(self.cfn)
        with self.cfn.dat.vec_ro as v:
            v.copy(y)


def create_interpolation(dmc, dmf):
    cctx = get_appctx(dmc)
    fctx = get_appctx(dmf)

    V_c = cctx._problem.u_restrict.function_space()
    V_f = fctx._problem.u_restrict.function_space()

    row_size = V_f.dof_dset.layout_vec.getSizes()
    col_size = V_c.dof_dset.layout_vec.getSizes()
    cbcs = tuple(cctx._problem.dirichlet_bcs())
    fbcs = tuple(fctx._problem.dirichlet_bcs())

    manager = get_transfer_manager(dmf)
    ctx = Interpolation(V_c, V_f, manager, cbcs, fbcs)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    if row_size == col_size:
        # PETSc cannot determine the coarse space if the dimensions are equal.
        # The coarse space is identified by the dimension of rscale, so we provide one.
        rscale = mat.createVecRight()
        rscale.set(1.0)
    else:
        rscale = None
    return mat, rscale


def create_injection(dmc, dmf):
    cctx = get_appctx(dmc)
    fctx = get_appctx(dmf)

    V_c = cctx._problem.u_restrict.function_space()
    V_f = fctx._problem.u_restrict.function_space()

    row_size = V_c.dof_dset.layout_vec.getSizes()
    col_size = V_f.dof_dset.layout_vec.getSizes()

    if (V_c.ufl_element().family() == "Real"
            and V_f.ufl_element().family() == "Real"):
        assert row_size == col_size
        # If the coarse and fine spaces have equal size
        # PETSc will apply the transpose of the injection.
        # It does not make sense to implement Injection.multTranspose,
        # instead we return a concrete identity matrix.
        dvec = V_c.dof_dset.layout_vec.duplicate()
        dvec.set(1.0)
        return PETSc.Mat().createDiagonal(dvec)

    manager = get_transfer_manager(dmf)
    ctx = Injection(V_c, V_f, manager)
    mat = PETSc.Mat().create(comm=dmc.comm)
    mat.setSizes((row_size, col_size))
    mat.setType(mat.Type.PYTHON)
    mat.setPythonContext(ctx)
    mat.setUp()
    return mat
