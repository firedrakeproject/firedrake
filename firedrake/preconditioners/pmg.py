from functools import partial, lru_cache
from itertools import chain

import os
import tempfile

import ufl

from pyop2 import op2, PermutedMap
import loopy
import numpy

from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.nullspace import VectorSpaceBasis, MixedVectorSpaceBasis
from firedrake.dmhooks import attach_hooks, get_appctx, push_appctx, pop_appctx
from firedrake.dmhooks import add_hook, get_parent, push_parent, pop_parent
from firedrake.dmhooks import get_function_space, set_function_space
from firedrake.solving_utils import _SNESContext
from firedrake.utils import ScalarType_c, IntType_c
from firedrake.petsc import PETSc
import firedrake

__all__ = ("PMGPC", "PMGSNES")


class PMGBase(PCSNESBase):
    """
    A class for implementing p-multigrid
    Internally, this sets up a DM with a custom coarsen routine
    that p-coarsens the problem. This DM is passed to an internal
    PETSc PC of type MG and with options prefix ``pmg_``. The
    relaxation to apply on every p-level is described by ``pmg_mg_levels_``,
    and the coarse solve by ``pmg_mg_coarse_``. Geometric multigrid
    or any other solver in firedrake may be applied to the coarse problem.

    Other PETSc options inspected by this class are:
    - 'pmg_coarse_degree': polynomial degree of the coarse level
    - 'pmg_coarse_mat_type': can be either 'aij' or 'matfree'
    - 'pmg_coarse_form_compiler_mode': can be 'spectral' (default), 'vanilla', 'coffee', or 'tensor'
    - 'pmg_mg_levels_transfer_mat_type': can be either 'aij' or 'matfree'

    The p-coarsening is implemented in the `coarsen_element` routine.
    This takes in a :class:`ufl.FiniteElement` and either returns a
    new, coarser element, or raises a `ValueError` (if the supplied element
    should be the coarsest one of the hierarchy).

    The default coarsen_element is to perform power-of-2 reduction
    of the polynomial degree.
    It is expected that some applications of this preconditioner
    will subclass :class:`PMGBase` to override `coarsen_element`.
    """

    _prefix = "pmg_"

    def coarsen_element(self, ele):
        """
        Coarsen a given element to form the next problem down in the p-hierarchy.

        If the supplied element should form the coarsest level of the p-hierarchy,
        raise `ValueError`. Otherwise, return a new :class:`ufl.FiniteElement`.

        By default, this does power-of-2 coarsening in polynomial degree until
        we reach the coarse degree specified through PETSc options (1 by default).

        :arg ele: a :class:`ufl.FiniteElement` to coarsen.
        """
        N = PMGBase.max_degree(ele)
        if N <= self.coarse_degree:
            raise ValueError
        return PMGBase.reconstruct_degree(ele, max(N // 2, self.coarse_degree))

    @staticmethod
    def max_degree(ele):
        """
        Return the maximum degree of a :class:`ufl.FiniteElement`
        """
        if isinstance(ele, (ufl.VectorElement, ufl.TensorElement)):
            return PMGBase.max_degree(ele._sub_element)
        elif isinstance(ele, (ufl.MixedElement, ufl.TensorProductElement)):
            return max(PMGBase.max_degree(sub) for sub in ele.sub_elements())
        elif isinstance(ele, ufl.EnrichedElement):
            return max(PMGBase.max_degree(sub) for sub in ele._elements)
        elif isinstance(ele, ufl.WithMapping):
            return PMGBase.max_degree(ele.wrapee)
        elif isinstance(ele, (ufl.HDivElement, ufl.HCurlElement, ufl.BrokenElement, ufl.RestrictedElement)):
            return PMGBase.max_degree(ele._element)
        else:
            N = ele.degree()
            try:
                return max(N)
            except TypeError:
                return N

    @staticmethod
    def reconstruct_degree(ele, N):
        """
        Reconstruct an element, modifying its polynomial degree.

        By default, reconstructed EnrichedElements, TensorProductElements,
        and MixedElements will have the degree of the sub-elements shifted
        by the same amount so that the maximum degree is N.
        This is useful to coarsen spaces like NCF(N) x DQ(N-1).

        :arg ele: a :class:`ufl.FiniteElement` to reconstruct,
        :arg N: an integer degree.

        :returns: the reconstructed element
        """
        if isinstance(ele, ufl.VectorElement):
            return type(ele)(PMGBase.reconstruct_degree(ele._sub_element, N), dim=ele.num_sub_elements())
        elif isinstance(ele, ufl.TensorElement):
            return type(ele)(PMGBase.reconstruct_degree(ele._sub_element, N), shape=ele.value_shape(), symmetry=ele.symmetry())
        elif isinstance(ele, ufl.EnrichedElement):
            shift = N-PMGBase.max_degree(ele)
            return type(ele)(*(PMGBase.reconstruct_degree(e, PMGBase.max_degree(e)+shift) for e in ele._elements))
        elif isinstance(ele, ufl.TensorProductElement):
            shift = N-PMGBase.max_degree(ele)
            return type(ele)(*(PMGBase.reconstruct_degree(e, PMGBase.max_degree(e)+shift) for e in ele.sub_elements()), cell=ele.cell())
        elif isinstance(ele, ufl.MixedElement):
            shift = N-PMGBase.max_degree(ele)
            return type(ele)(*(PMGBase.reconstruct_degree(e, PMGBase.max_degree(e)+shift) for e in ele.sub_elements()))
        elif isinstance(ele, ufl.WithMapping):
            return type(ele)(PMGBase.reconstruct_degree(ele.wrapee, N), ele.mapping())
        elif isinstance(ele, (ufl.HDivElement, ufl.HCurlElement, ufl.BrokenElement, ufl.RestrictedElement)):
            return type(ele)(PMGBase.reconstruct_degree(ele._element, N))
        else:
            return ele.reconstruct(degree=N)

    def initialize(self, pc):
        # Make a new DM.
        # Hook up a (new) coarsen routine on that DM.
        # Make a new PC, of type MG.
        # Assign the DM to that PC.

        odm = pc.getDM()
        ctx = get_appctx(odm)

        test, trial = ctx.J.arguments()
        if test.function_space() != trial.function_space():
            raise NotImplementedError("test and trial spaces must be the same")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options(options_prefix)
        pdm = PETSc.DMShell().create(comm=pc.comm)
        pdm.setOptionsPrefix(options_prefix)

        # Get the coarse degree from PETSc options
        fcp = ctx._problem.form_compiler_parameters
        mode = fcp.get("mode", "spectral") if fcp is not None else "spectral"
        self.coarse_degree = opts.getInt("coarse_degree", default=1)
        self.coarse_mat_type = opts.getString("coarse_mat_type", default=ctx.mat_type)
        self.coarse_pmat_type = opts.getString("coarse_pmat_type", default=self.coarse_mat_type)
        self.coarse_form_compiler_mode = opts.getString("coarse_form_compiler_mode", default=mode)

        # Construct a list with the elements we'll be using
        V = test.function_space()
        ele = V.ufl_element()
        elements = [ele]
        while True:
            try:
                ele_ = self.coarsen_element(ele)
                assert ele_.value_shape() == ele.value_shape()
                ele = ele_
            except ValueError:
                break
            elements.append(ele)

        sf = odm.getPointSF()
        section = odm.getDefaultSection()
        attach_hooks(pdm, level=len(elements)-1, sf=sf, section=section)
        # Now overwrite some routines on the DM
        pdm.setRefine(None)
        pdm.setCoarsen(self.coarsen)
        pdm.setCreateInterpolation(self.create_interpolation)
        # We need this for p-FAS
        pdm.setCreateInjection(self.create_injection)
        pdm.setSNESJacobian(_SNESContext.form_jacobian)
        pdm.setSNESFunction(_SNESContext.form_function)
        pdm.setKSPComputeOperators(_SNESContext.compute_operators)

        set_function_space(pdm, get_function_space(odm))

        parent = get_parent(odm)
        assert parent is not None
        add_hook(parent, setup=partial(push_parent, pdm, parent), teardown=partial(pop_parent, pdm, parent), call_setup=True)
        add_hook(parent, setup=partial(push_appctx, pdm, ctx), teardown=partial(pop_appctx, pdm, ctx), call_setup=True)

        self.ppc = self.configure_pmg(pc, pdm)
        self.ppc.setFromOptions()
        self.ppc.setUp()

    def update(self, pc):
        pass

    def coarsen(self, fdm, comm):
        # Coarsen the _SNESContext of a DM fdm
        # return the coarse DM cdm of the coarse _SNESContext
        fctx = get_appctx(fdm)
        parent = get_parent(fdm)
        assert parent is not None

        test, trial = fctx.J.arguments()
        fV = test.function_space()
        cele = self.coarsen_element(fV.ufl_element())

        # Have we already done this?
        cctx = fctx._coarse
        if cctx is not None:
            cV = cctx.J.arguments()[0].function_space()
            if (cV.ufl_element() == cele) and (cV.mesh() == fV.mesh()):
                return cV.dm

        cV = firedrake.FunctionSpace(fV.mesh(), cele)
        cdm = cV.dm

        fproblem = fctx._problem
        fu = fproblem.u
        cu = firedrake.Function(cV)

        Nf = PMGBase.max_degree(fV.ufl_element())
        Nc = PMGBase.max_degree(cV.ufl_element())

        # Replace dictionary with coarse state, test and trial functions
        replace_d = {fu: cu,
                     test: test.reconstruct(function_space=cV),
                     trial: trial.reconstruct(function_space=cV)}

        coarsen_form = lambda a: self.coarsen_quadrature(self.coarsen_form(a, replace_d), Nf, Nc)
        cF = coarsen_form(fctx.F)
        cJ = coarsen_form(fctx.J)
        cJp = coarsen_form(fctx.Jp)
        fcp = self.coarsen_quadrature(fproblem.form_compiler_parameters, Nf, Nc)
        cbcs = self.coarsen_bcs(fproblem.bcs, cV)

        # Coarsen the appctx: the user might want to provide solution-dependant expressions and forms
        cappctx = dict(fctx.appctx)
        for key in cappctx:
            val = cappctx[key]
            if isinstance(val, dict):
                cappctx[key] = self.coarsen_quadrature(val, Nf, Nc)
            elif isinstance(val, ufl.Form):
                cappctx[key] = coarsen_form(val)
            elif isinstance(val, ufl.classes.Expr):
                cappctx[key] = ufl.replace(val, replace_d)

        # If we're the coarsest grid of the p-hierarchy, don't
        # overwrite the coarsen routine; this is so that you can
        # use geometric multigrid for the p-coarse problem
        try:
            self.coarsen_element(cele)
            cdm.setCoarsen(self.coarsen)
            mat_type = fctx.mat_type
            pmat_type = fctx.pmat_type
        except ValueError:
            mat_type = self.coarse_mat_type
            pmat_type = self.coarse_pmat_type
            if fcp is None:
                fcp = dict()
            elif fcp is fproblem.form_compiler_parameters:
                fcp = dict(fcp)
            fcp["mode"] = self.coarse_form_compiler_mode

        # Coarsen the problem and the _SNESContext
        cproblem = firedrake.NonlinearVariationalProblem(cF, cu, bcs=cbcs, J=cJ, Jp=cJp,
                                                         form_compiler_parameters=fcp,
                                                         is_linear=fproblem.is_linear)

        cctx = type(fctx)(cproblem, mat_type, pmat_type,
                          appctx=cappctx,
                          pre_jacobian_callback=fctx._pre_jacobian_callback,
                          pre_function_callback=fctx._pre_function_callback,
                          post_jacobian_callback=fctx._post_jacobian_callback,
                          post_function_callback=fctx._post_function_callback,
                          options_prefix=fctx.options_prefix,
                          transfer_manager=fctx.transfer_manager)

        # FIXME setting up the _fine attribute triggers gmg injection.
        # cctx._fine = fctx
        fctx._coarse = cctx

        add_hook(parent, setup=partial(push_parent, cdm, parent), teardown=partial(pop_parent, cdm, parent), call_setup=True)
        add_hook(parent, setup=partial(push_appctx, cdm, cctx), teardown=partial(pop_appctx, cdm, cctx), call_setup=True)

        cdm.setOptionsPrefix(fdm.getOptionsPrefix())
        cdm.setKSPComputeOperators(_SNESContext.compute_operators)
        cdm.setCreateInterpolation(self.create_interpolation)
        cdm.setCreateInjection(self.create_injection)

        # injection of the initial state
        def inject_state(mat):
            with cu.dat.vec_wo as xc, fu.dat.vec_ro as xf:
                mat.multTranspose(xf, xc)

        injection = self.create_injection(cdm, fdm)
        add_hook(parent, setup=partial(inject_state, injection), call_setup=True)

        # restrict the nullspace basis
        def coarsen_nullspace(coarse_V, mat, fine_nullspace):
            if isinstance(fine_nullspace, MixedVectorSpaceBasis):
                if mat.type == 'python':
                    mat = mat.getPythonContext()
                submats = [mat.getNestSubMatrix(i, i) for i in range(len(coarse_V))]
                coarse_bases = []
                for fs, submat, basis in zip(coarse_V, submats, fine_nullspace._bases):
                    if isinstance(basis, VectorSpaceBasis):
                        coarse_bases.append(coarsen_nullspace(fs, submat, basis))
                    else:
                        coarse_bases.append(coarse_V.sub(basis.index))
                return MixedVectorSpaceBasis(coarse_V, coarse_bases)
            elif isinstance(fine_nullspace, VectorSpaceBasis):
                coarse_vecs = []
                for xf in fine_nullspace._petsc_vecs:
                    wc = firedrake.Function(coarse_V)
                    with wc.dat.vec_wo as xc:
                        mat.multTranspose(xf, xc)
                    coarse_vecs.append(wc)
                vsb = VectorSpaceBasis(coarse_vecs, constant=fine_nullspace._constant)
                vsb.orthonormalize()
                return vsb
            else:
                return fine_nullspace

        I, _ = self.create_interpolation(cdm, fdm)
        ises = cV._ises
        cctx._nullspace = coarsen_nullspace(cV, I, fctx._nullspace)
        cctx.set_nullspace(cctx._nullspace, ises, transpose=False, near=False)
        cctx._nullspace_T = coarsen_nullspace(cV, I, fctx._nullspace_T)
        cctx.set_nullspace(cctx._nullspace_T, ises, transpose=True, near=False)
        cctx._near_nullspace = coarsen_nullspace(cV, I, fctx._near_nullspace)
        cctx.set_nullspace(cctx._near_nullspace, ises, transpose=False, near=True)
        return cdm

    def coarsen_quadrature(self, df, Nf, Nc):
        if isinstance(df, dict):
            # Coarsen the quadrature degree in a dictionary
            # such that the ratio of quadrature nodes to interpolation nodes (Nq+1)/(Nf+1) is preserved
            Nq = df.get("quadrature_degree", None)
            if Nq is not None:
                dc = dict(df)
                dc["quadrature_degree"] = max(2*Nc+1, ((Nq+1) * (Nc+1) + Nf) // (Nf+1) - 1)
                return dc
        elif isinstance(df, ufl.Form):
            # Coarsen a form by reconstructing each integral with a coarsened quadrature degree
            return ufl.Form([f.reconstruct(metadata=self.coarsen_quadrature(f.metadata(), Nf, Nc))
                             for f in df.integrals()])
        return df

    def coarsen_form(self, form, fine_to_coarse_map):
        """
        Coarsen a form, by replacing the solution, test and trial functions.
        Users may override this to e.g. throw away facet integrals when the coarse space is H1.
        """
        return ufl.replace(form, fine_to_coarse_map) if isinstance(form, ufl.Form) else form

    def coarsen_bcs(self, fbcs, cV):
        cbcs = []
        for bc in fbcs:
            cV_ = cV
            for index in bc._indices:
                cV_ = cV_.sub(index)
            cbc_value = self.coarsen_bc_value(bc, cV_)
            if type(bc) == firedrake.DirichletBC:
                cbcs.append(firedrake.DirichletBC(cV_, cbc_value,
                                                  bc.sub_domain))
            else:
                raise NotImplementedError("Unsupported BC type, please get in touch if you need this")
        return cbcs

    @staticmethod
    @lru_cache(maxsize=20)
    def create_transfer(cctx, fctx, mat_type, cbcs, fbcs, inject):
        cbcs = cctx._problem.bcs if cbcs else []
        fbcs = fctx._problem.bcs if fbcs else []
        if inject:
            cV = cctx._problem.u
            fV = fctx._problem.u
        else:
            cV = cctx.J.arguments()[0].function_space()
            fV = fctx.J.arguments()[0].function_space()

        if mat_type == "matfree":
            return prolongation_matrix_matfree(fV, cV, fbcs, cbcs)
        elif mat_type == "aij":
            return prolongation_matrix_aij(fV, cV, fbcs, cbcs)
        else:
            raise ValueError("Unknown matrix type")

    def create_interpolation(self, dmc, dmf):
        prefix = dmc.getOptionsPrefix()
        mat_type = PETSc.Options(prefix).getString("mg_levels_transfer_mat_type", default="matfree")
        return self.create_transfer(get_appctx(dmc), get_appctx(dmf), mat_type, True, False, False), None

    def create_injection(self, dmc, dmf):
        prefix = dmc.getOptionsPrefix()
        mat_type = PETSc.Options(prefix).getString("mg_levels_transfer_mat_type", default="matfree")
        I = self.create_transfer(get_appctx(dmf), get_appctx(dmc), mat_type, False, False, True)
        return PETSc.Mat().createTranspose(I)

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT
        viewer.printfASCII("p-multigrid PC\n")
        self.ppc.view(viewer)


class PMGPC(PCBase, PMGBase):
    _prefix = "pmg_"

    def configure_pmg(self, pc, pdm):
        odm = pc.getDM()
        ppc = PETSc.PC().create(comm=pc.comm)
        ppc.setOptionsPrefix(pc.getOptionsPrefix() + "pmg_")
        ppc.setType("mg")
        ppc.setOperators(*pc.getOperators())
        ppc.setDM(pdm)
        ppc.incrementTabLevel(1, parent=pc)

        # PETSc unfortunately requires us to make an ugly hack.
        # We would like to use GMG for the coarse solve, at least
        # sometimes. But PETSc will use this p-DM's getRefineLevels()
        # instead of the getRefineLevels() of the MeshHierarchy to
        # decide how many levels it should use for PCMG applied to
        # the p-MG's coarse problem. So we need to set an option
        # for the user, if they haven't already; I don't know any
        # other way to get PETSc to know this at the right time.
        opts = PETSc.Options(pc.getOptionsPrefix() + "pmg_")
        if "mg_coarse_pc_mg_levels" not in opts:
            opts["mg_coarse_pc_mg_levels"] = odm.getRefineLevel() + 1

        return ppc

    def apply(self, pc, x, y):
        return self.ppc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        return self.ppc.applyTranspose(x, y)

    def coarsen_bc_value(self, bc, cV):
        return firedrake.zero(cV.shape)


class PMGSNES(SNESBase, PMGBase):
    _prefix = "pfas_"

    def configure_pmg(self, snes, pdm):
        odm = snes.getDM()
        psnes = PETSc.SNES().create(comm=snes.comm)
        psnes.setOptionsPrefix(snes.getOptionsPrefix() + "pfas_")
        psnes.setType("fas")
        psnes.setDM(pdm)
        psnes.incrementTabLevel(1, parent=snes)

        (f, residual) = snes.getFunction()
        assert residual is not None
        (fun, args, kargs) = residual
        psnes.setFunction(fun, f.duplicate(), args=args, kargs=kargs)

        pdm.setGlobalVector(f.duplicate())
        self.dummy = f.duplicate()
        psnes.setSolution(f.duplicate())

        # PETSc unfortunately requires us to make an ugly hack.
        # We would like to use GMG for the coarse solve, at least
        # sometimes. But PETSc will use this p-DM's getRefineLevels()
        # instead of the getRefineLevels() of the MeshHierarchy to
        # decide how many levels it should use for PCMG applied to
        # the p-MG's coarse problem. So we need to set an option
        # for the user, if they haven't already; I don't know any
        # other way to get PETSc to know this at the right time.
        opts = PETSc.Options(snes.getOptionsPrefix() + "pfas_")
        if "fas_coarse_pc_mg_levels" not in opts:
            opts["fas_coarse_pc_mg_levels"] = odm.getRefineLevel() + 1
        if "fas_coarse_snes_fas_levels" not in opts:
            opts["fas_coarse_snes_fas_levels"] = odm.getRefineLevel() + 1

        return psnes

    def step(self, snes, x, f, y):
        ctx = get_appctx(snes.dm)
        push_appctx(self.ppc.dm, ctx)
        x.copy(y)
        self.ppc.solve(snes.vec_rhs or self.dummy, y)
        y.aypx(-1, x)
        snes.setConvergedReason(self.ppc.getConvergedReason())
        pop_appctx(self.ppc.dm)

    def coarsen_bc_value(self, bc, cV):
        if not isinstance(bc._original_arg, firedrake.Function):
            return bc._original_arg

        coarse = firedrake.Function(cV)
        coarse.interpolate(bc._original_arg)
        return coarse


def prolongation_transfer_kernel_action(Vf, expr):
    from tsfc import compile_expression_dual_evaluation
    from tsfc.finatinterface import create_element
    to_element = create_element(Vf.ufl_element())
    kernel = compile_expression_dual_evaluation(expr, to_element, Vf.ufl_element())
    coefficients = kernel.coefficients
    if kernel.first_coefficient_fake_coords:
        target_mesh = Vf.ufl_domain()
        coefficients[0] = target_mesh.coordinates

    return op2.Kernel(kernel.ast, kernel.name,
                      requires_zeroed_output_arguments=True,
                      flop_count=kernel.flop_count), coefficients


def get_sobolev_space(ele):
    if isinstance(ele, (ufl.TensorElement, ufl.VectorElement)):
        return get_sobolev_space(ele._sub_element)
    elif isinstance(ele, ufl.WithMapping):
        return get_sobolev_space(ele.wrapee)
    else:
        return ele.sobolev_space()


def expand_element(ele):
    """
    Expand a FiniteElement as an EnrichedElement of TensorProductElements, discarding modifiers.
    """
    if ele.cell() == ufl.quadrilateral:
        quadrilateral_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
        return expand_element(ele.reconstruct(cell=quadrilateral_tpc))
    elif ele.cell() == ufl.hexahedron:
        hexahedron_tpc = ufl.TensorProductCell(ufl.quadrilateral, ufl.interval)
        return expand_element(ele.reconstruct(cell=hexahedron_tpc))
    elif isinstance(ele, (ufl.TensorElement, ufl.VectorElement)):
        return expand_element(ele._sub_element)
    elif isinstance(ele, (ufl.HDivElement, ufl.HCurlElement, ufl.BrokenElement, ufl.RestrictedElement)):
        return expand_element(ele._element)
    elif isinstance(ele, ufl.WithMapping):
        return expand_element(ele.wrapee)
    elif isinstance(ele, ufl.MixedElement):
        return ufl.MixedElement(*[expand_element(e) for e in ele.sub_elements()])
    elif isinstance(ele, ufl.EnrichedElement):
        terms = []
        for e in ele._elements:
            ee = expand_element(e)
            if isinstance(ee, ufl.EnrichedElement):
                terms.extend(ee._elements)
            else:
                terms.append(ee)
        return ufl.EnrichedElement(*terms)
    elif isinstance(ele, ufl.TensorProductElement):
        factors = [expand_element(e) for e in ele.sub_elements()]
        terms = [tuple()]
        for e in factors:
            new_terms = []
            for f in e._elements if isinstance(e, ufl.EnrichedElement) else [e]:
                f_factors = tuple(f.sub_elements()) if isinstance(f, ufl.TensorProductElement) else (f,)
                new_terms.extend([t_factors + f_factors for t_factors in terms])
            terms = new_terms
        if len(terms) == 1:
            return ufl.TensorProductElement(*terms[0])
        else:
            return ufl.EnrichedElement(*[ufl.TensorProductElement(*k) for k in terms])
    else:
        return ele


def get_line_elements(ele):
    from FIAT import ufc_cell, gauss_legendre, gauss_lobatto_legendre, lagrange, discontinuous_lagrange, fdm_element
    if isinstance(ele, ufl.MixedElement) and not isinstance(ele, (ufl.TensorElement, ufl.VectorElement)):
        raise ValueError("MixedElements are not decomposed into tensor products")

    sobolev = get_sobolev_space(ele)
    ele = expand_element(ele)
    if isinstance(ele, ufl.EnrichedElement):
        # TODO assert that all components are permutations of each other
        ele = ele._elements[-1 if sobolev == ufl.HCurl else 0]

    factors = ele.sub_elements() if isinstance(ele, ufl.TensorProductElement) else [ele]
    elements = []
    for e in factors:
        if e.cell() != ufl.interval:
            raise ValueError("Expecting %s to be on the interval" % e)

        degree = e.degree()
        variant = e.variant()
        ref_el = ufc_cell(e.cell())
        formdegree = 0 if e.sobolev_space() == ufl.H1 else ref_el.get_spatial_dimension()
        if variant == "equispaced":
            if formdegree == 0:
                elements.append(lagrange.Lagrange(ref_el, degree))
            else:
                elements.append(discontinuous_lagrange.DiscontinuousLagrange(ref_el, degree))
        elif variant == "fdm":
            elements.append(fdm_element.FDMElement(ref_el, degree, formdegree=formdegree))
        elif (variant == "spectral") or (variant is None):
            if formdegree == 0:
                elements.append(gauss_lobatto_legendre.GaussLobattoLegendre(ref_el, degree))
            else:
                elements.append(gauss_legendre.GaussLegendre(ref_el, degree))
        else:
            raise ValueError("Variant %s is not supported" % variant)
    return elements


def get_line_nodes(element):
    # Return the Line nodes for 1D elements
    from FIAT import quadrature, Lagrange, P0
    from FIAT.discontinuous_lagrange import HigherOrderDiscontinuousLagrange
    cell = element.ref_el
    degree = element.degree()
    equispaced = isinstance(element, (Lagrange, P0, HigherOrderDiscontinuousLagrange))
    if equispaced:
        return cell.make_points(1, 0, degree+1)
    elif element.formdegree == 0:
        rule = quadrature.GaussLobattoLegendreQuadratureLineRule(cell, degree+1)
        return rule.get_points()
    elif element.formdegree == 1:
        rule = quadrature.GaussLegendreQuadratureLineRule(cell, degree+1)
        return rule.get_points()
    else:
        raise ValueError("Don't know how to get line nodes for %s" % element)


# Common kernel to compute y = kron(A3, kron(A2, A1)) * x
# Vector and tensor field generalization from Deville, Fischer, and Mund section 8.3.1.
kronmxv_code = """
#include <petscsys.h>
#include <petscblaslapack.h>

static void kronmxv(int tflag,
    PetscBLASInt mx, PetscBLASInt my, PetscBLASInt mz,
    PetscBLASInt nx, PetscBLASInt ny, PetscBLASInt nz, PetscBLASInt nel,
    PetscScalar *A1, PetscScalar *A2, PetscScalar *A3,
    PetscScalar *x , PetscScalar *y){

/*
Kronecker matrix-vector product

y = op(A) * x,  A = kron(A3, kron(A2, A1))

where:
op(A) = transpose(A) if tflag>0 else A
op(A1) is mx-by-nx,
op(A2) is my-by-ny,
op(A3) is mz-by-nz,
x is (nx*ny*nz)-by-nel,
y is (mx*my*mz)-by-nel.

Important notes:
The input data in x is destroyed in the process.
Need to allocate nel*max(mx, nx)*max(my, ny)*max(mz, nz) memory for both x and y.
*/

PetscBLASInt m, n, k, s, p, lda;
char TA1, TA2, TA3;
char tran='T', notr='N';
PetscScalar zero=0.0E0, one=1.0E0;

if(tflag>0){
   TA1 = tran;
   TA2 = notr;
}else{
   TA1 = notr;
   TA2 = tran;
}
TA3 = TA2;

m = mx;  k = nx;  n = ny*nz*nel;
lda = (tflag>0)? nx : mx;

BLASgemm_(&TA1, &notr, &m, &n, &k, &one, A1, &lda, x, &k, &zero, y, &m);

p = 0;  s = 0;
m = mx;  k = ny;  n = my;
lda = (tflag>0)? ny : my;
for(PetscBLASInt i=0; i<nz*nel; i++){
   BLASgemm_(&notr, &TA2, &m, &n, &k, &one, y+p, &m, A2, &lda, &zero, x+s, &m);
   p += m*k;
   s += m*n;
}

p = 0;  s = 0;
m = mx*my;  k = nz;  n = mz;
lda = (tflag>0)? nz : mz;
for(PetscBLASInt i=0; i<nel; i++){
   BLASgemm_(&notr, &TA3, &m, &n, &k, &one, x+p, &m, A3, &lda, &zero, y+s, &m);
   p += m*k;
   s += m*n;
}
return;
}
"""


def make_kron_code(Vf, Vc, t_in, t_out, mat):
    from FIAT import functional
    nscal = Vf.ufl_element().value_size()
    celem = get_line_elements(Vc.ufl_element())
    felem = get_line_elements(Vf.ufl_element())
    nodes = [get_line_nodes(e) for e in felem]
    Jhat = [e.tabulate(0, z)[(0,)] for e, z in zip(celem, nodes)]
    ndim = len(Jhat)

    for k, e in enumerate(felem):
        if not all([isinstance(phi, functional.PointEvaluation) for phi in e.dual_basis()]):
            Jhat[k] = numpy.dot(Jhat[k], numpy.linalg.inv(e.tabulate(0, nodes[k])[(0,)]))

    # Declare array shapes to be used as literals inside the kernels
    # I follow to the m-by-n convention with the FORTRAN ordering (so I have to do n-by-m in python)
    shapes = [[Jk.shape[j] for Jk in Jhat] for j in range(2)]
    n, m = shapes.copy()
    m += [1]*(3-ndim)
    n += [1]*(3-ndim)
    shapes[0].append(nscal)
    shapes[1].append(nscal)

    # Pass the 1D tabulation as hexadecimal string
    # The Kronecker product routines assume 3D shapes, so in 1D and 2D we pass one instead of Jhat
    JX = ', '.join(map(float.hex, numpy.concatenate([numpy.asarray(Jk).flatten() for Jk in Jhat])))
    JY = "&one" if ndim < 2 else f"{mat}+{Jhat[0].size}"
    JZ = "&one" if ndim < 3 else f"{mat}+{Jhat[0].size+Jhat[1].size}"
    Jlen = sum([Jk.size for Jk in Jhat])

    operator_decl = f"""
            PetscScalar {mat}[{Jlen}] = {{ {JX} }};
    """
    prolong_code = f"""
            kronmxv(0, {m[0]}, {m[1]}, {m[2]}, {n[0]}, {n[1]}, {n[2]}, {nscal}, {mat}, {JY}, {JZ}, {t_in}, {t_out});
    """
    restrict_code = f"""
            kronmxv(1, {n[0]}, {n[1]}, {n[2]}, {m[0]}, {m[1]}, {m[2]}, {nscal}, {mat}, {JY}, {JZ}, {t_out}, {t_in});
    """
    return operator_decl, prolong_code, restrict_code, shapes


def get_piola_tensor(elem, domain, inverse=False):
    emap = elem.mapping().lower()
    if emap == "identity":
        return None
    elif emap == "contravariant piola":
        if inverse:
            return ufl.JacobianInverse(domain) * ufl.JacobianDeterminant(domain)
        else:
            return ufl.Jacobian(domain) / ufl.JacobianDeterminant(domain)
    elif emap == "covariant piola":
        if inverse:
            return ufl.Jacobian(domain).T
        else:
            return ufl.JacobianInverse(domain).T
    else:
        raise ValueError("Unsupported mapping")


def cache_generate_code(kernel, comm):
    _cachedir = os.environ.get('PYOP2_CACHE_DIR',
                               os.path.join(tempfile.gettempdir(),
                                            'pyop2-cache-uid%d' % os.getuid()))

    key = kernel.cache_key
    shard, disk_key = key[:2], key[2:]
    filepath = os.path.join(_cachedir, shard, disk_key)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            code = f.read()
    else:
        code = loopy.generate_code_v2(kernel.code).device_code()
        if comm.rank == 0:
            os.makedirs(os.path.join(_cachedir, shard), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(code)
        comm.barrier()
    return code


def make_mapping_code(Q, felem, celem, t_in, t_out):
    domain = Q.ufl_domain()
    A = get_piola_tensor(celem, domain, inverse=False)
    B = get_piola_tensor(felem, domain, inverse=True)
    tensor = A
    if B:
        tensor = firedrake.dot(B, tensor) if tensor else B
    if tensor is None:
        tensor = firedrake.Identity(Q.ufl_element().value_shape()[0])

    u = firedrake.Coefficient(Q)
    expr = ufl.dot(tensor, u)
    prolong_map_kernel, coefficients = prolongation_transfer_kernel_action(Q, expr)
    prolong_map_code = cache_generate_code(prolong_map_kernel, Q.comm)
    prolong_map_code = prolong_map_code.replace("void expression_kernel", "static void prolongation_mapping")
    coefficients.remove(u)

    expr = ufl.dot(u, tensor)
    restrict_map_kernel, coefficients = prolongation_transfer_kernel_action(Q, expr)
    restrict_map_code = cache_generate_code(restrict_map_kernel, Q.comm)
    restrict_map_code = restrict_map_code.replace("void expression_kernel", "static void restriction_mapping")
    restrict_map_code = restrict_map_code.replace("#include <stdint.h>", "")
    coefficients.remove(u)

    coef_args = "".join([", c%d" % i for i in range(len(coefficients))])
    coef_decl = "".join([", PetscScalar const *restrict c%d" % i for i in range(len(coefficients))])
    qlen = Q.value_size * Q.finat_element.space_dimension()
    prolong_code = f"""
            for({IntType_c} i=0; i<{qlen}; i++) {t_out}[i] = 0.0E0;

            prolongation_mapping({t_out}{coef_args}, {t_in});
    """
    restrict_code = f"""
            for({IntType_c} i=0; i<{qlen}; i++) {t_in}[i] = 0.0E0;

            restriction_mapping({t_in}{coef_args}, {t_out});
    """
    mapping_code = prolong_map_code + restrict_map_code
    return coef_decl, prolong_code, restrict_code, mapping_code, coefficients


def make_permutation_code(elem, vshape, shapes, t_in, t_out, array_name):
    sobolev = get_sobolev_space(elem)
    if sobolev in [ufl.HDiv, ufl.HCurl]:
        ndim = elem.cell().topological_dimension()
        pshape = shapes.copy()
        pshape = [-1] + pshape[:ndim]

        ndof = numpy.prod(vshape)
        shift = int(sobolev == ufl.HDiv)

        # compose with the inverse H(div)/H(curl) permutation
        permutation = numpy.reshape(numpy.arange(ndof), pshape)
        for k in range(permutation.shape[0]):
            permutation[k] = numpy.reshape(numpy.transpose(permutation[k], axes=(numpy.arange(ndim)-((2*shift-1)*k+shift)) % ndim), pshape[1:])

        if sobolev == ufl.HCurl:
            permutation = numpy.flip(permutation, axis=0)

        permutation = numpy.transpose(numpy.reshape(permutation, vshape))
        permutation = numpy.reshape(permutation, (-1,))
        perm = ", ".join(map(str, permutation))
        decl = f"""
            PetscInt {array_name}[{ndof}] = {{ {perm} }};
        """

        nflip = 0
        if sobolev == ufl.HDiv:
            # flip the sign of the first component
            nflip = ndof // elem.value_shape()[0]

        prolong = f"""
            for({IntType_c} i=0; i<{ndof}; i++) {t_out}[{array_name}[i]] = {t_in}[i];
            for({IntType_c} i=0; i<{nflip}; i++) {t_out}[i] = -{t_out}[i];
        """
        restrict = f"""
            for({IntType_c} i=0; i<{nflip}; i++) {t_out}[i] = -{t_out}[i];
            for({IntType_c} i=0; i<{ndof}; i++) {t_in}[i] = {t_out}[{array_name}[i]];
        """
    else:
        decl = ""
        prolong = f"""
            for({IntType_c} j=0; j<{vshape[1]}; j++)
                for({IntType_c} i=0; i<{vshape[0]}; i++)
                    {t_out}[j + {vshape[1]}*i] = {t_in}[i + {vshape[0]}*j];
        """
        restrict = f"""
            for({IntType_c} j=0; j<{vshape[1]}; j++)
                for({IntType_c} i=0; i<{vshape[0]}; i++)
                    {t_in}[i + {vshape[0]}*j] = {t_out}[j + {vshape[1]}*i];
        """
    return decl, prolong, restrict


def get_permuted_map(V):
    """
    Return a PermutedMap with the same tensor product shape for every component of H(div) or H(curl) tensor product elements
    """
    e = V.ufl_element()
    sobolev = get_sobolev_space(e)
    if sobolev == ufl.HDiv:
        shift = 1
    elif sobolev == ufl.HCurl:
        shift = 0
    else:
        return V.cell_node_map()

    elements = get_line_elements(e)
    ndim = len(elements)
    pshape = [-1] + [e.space_dimension() for e in elements]
    ndof = V.value_size * V.finat_element.space_dimension()
    permutation = numpy.reshape(numpy.arange(ndof), pshape)
    for k in range(permutation.shape[0]):
        permutation[k] = numpy.reshape(numpy.transpose(permutation[k], axes=(numpy.arange(ndim)+((2*shift-1)*k+shift)) % ndim), pshape[1:])

    permutation = numpy.reshape(permutation, (-1,))
    return PermutedMap(V.cell_node_map(), permutation)


class StandaloneInterpolationMatrix(object):
    """
    Interpolation matrix for a single standalone space.
    """
    def __init__(self, Vf, Vc, Vf_bcs, Vc_bcs):
        self.Vf_bcs = Vf_bcs
        self.Vc_bcs = Vc_bcs
        if isinstance(Vf, firedrake.Function):
            self.uf = Vf
            Vf = Vf.function_space()
        else:
            self.uf = firedrake.Function(Vf)
        if isinstance(Vc, firedrake.Function):
            self.uc = Vc
            Vc = Vc.function_space()
        else:
            self.uc = firedrake.Function(Vc)

        self.weight = self.multiplicity(Vf)
        with self.weight.dat.vec as w:
            w.reciprocal()

        try:
            uf_map = get_permuted_map(Vf)
            uc_map = get_permuted_map(Vc)
            prolong_kernel, restrict_kernel, coefficients = self.make_blas_kernels(Vf, Vc)
            prolong_args = [prolong_kernel, self.uf.cell_set,
                            self.uf.dat(op2.INC, uf_map),
                            self.uc.dat(op2.READ, uc_map),
                            self.weight.dat(op2.READ, uf_map)]
        except ValueError:
            uf_map = Vf.cell_node_map()
            uc_map = Vc.cell_node_map()
            prolong_kernel, restrict_kernel, coefficients = self.make_kernels(Vf, Vc)
            prolong_args = [prolong_kernel, self.uf.cell_set,
                            self.uf.dat(op2.WRITE, uf_map),
                            self.uc.dat(op2.READ, uc_map)]

        restrict_args = [restrict_kernel, self.uf.cell_set,
                         self.uc.dat(op2.INC, uc_map),
                         self.uf.dat(op2.READ, uf_map),
                         self.weight.dat(op2.READ, uf_map)]
        coefficient_args = [c.dat(op2.READ, c.cell_node_map()) for c in coefficients]
        self._prolong = partial(op2.par_loop, *prolong_args, *coefficient_args)
        self._restrict = partial(op2.par_loop, *restrict_args, *coefficient_args)

    @staticmethod
    def make_blas_kernels(Vf, Vc):
        """
        Interpolation and restriction kernels between CG / DG
        tensor product spaces on quads and hexes.

        Works by tabulating the coarse 1D Lagrange basis
        functions as the (Nf+1)-by-(Nc+1) matrix Jhat,
        and using the fact that the 2D / 3D tabulation is the
        tensor product J = kron(Jhat, kron(Jhat, Jhat))
        """
        Vf_bsize = Vf.value_size
        Vc_bsize = Vc.value_size
        Vf_sdim = Vf.finat_element.space_dimension()
        Vc_sdim = Vc.finat_element.space_dimension()

        felem = Vf.ufl_element()
        celem = Vc.ufl_element()
        Vf_mapping = felem.mapping().lower()
        Vc_mapping = celem.mapping().lower()

        fine_is_ordered = False
        coefficients = []
        mapping_code = ""
        coef_decl = ""
        if Vf_mapping == Vc_mapping:
            # interpolate on each direction via Kroncker product
            operator_decl, prolong_code, restrict_code, shapes = make_kron_code(Vf, Vc, "t0", "t1", "J0")
        else:
            decl = [""]*4
            prolong = [""]*5
            restrict = [""]*5
            # get embedding element with identity mapping and collocated vector component DOFs
            if Vf_bsize != felem.value_size():
                qdegree = PMGBase.max_degree(felem)
                Qe = ufl.TensorElement("DQ", cell=felem.cell(), degree=qdegree, shape=felem.value_shape(), symmetry=felem.symmetry())
                Q = firedrake.FunctionSpace(Vf.ufl_domain(), Qe)
            else:
                fine_is_ordered = True
                Q = Vf if Vf_mapping == "identity" else firedrake.FunctionSpace(Vf.ufl_domain(), felem.reconstruct(mapping="identity"))

            qshape = (Q.value_size, Q.finat_element.space_dimension())
            # interpolate to embedding fine space, permute to firedrake ordering, and apply the mapping
            decl[0], prolong[0], restrict[0], shapes = make_kron_code(Q, Vc, "t0", "t1", "J0")
            decl[1], restrict[1], prolong[1] = make_permutation_code(celem, qshape, shapes[1], "t0", "t1", "perm0")
            coef_decl, prolong[2], restrict[2], mapping_code, coefficients = make_mapping_code(Q, felem, celem, "t0", "t1")

            if Vf_bsize != felem.value_size():
                # permute to tensor-friendly ordering and interpolate to fine space
                decl[2], prolong[3], restrict[3] = make_permutation_code(felem, qshape, shapes[1], "t1", "t0", "perm1")
                decl[3], prolong[4], restrict[4], _shapes = make_kron_code(Vf, Q, "t0", "t1", "J1")
                shapes.extend(_shapes)

            operator_decl = "".join(decl)
            prolong_code = "".join(prolong)
            restrict_code = "".join(reversed(restrict))

        lwork = numpy.prod([max(*dims) for dims in zip(*shapes)])
        # Firedrake elements order the component DOFs related to the same node contiguously.
        # We transpose before and after the multiplication times J to have each component
        # stored contiguously as a scalar field, thus reducing the number of dgemm calls.

        # We could benefit from loop tiling for the transpose, but that makes the code
        # more complicated.

        if Vc_bsize == 1:
            coarse_read = f"""for({IntType_c} i=0; i<{Vc_sdim*Vc_bsize}; i++) t0[i] = x[i];"""
            coarse_write = f"""for({IntType_c} i=0; i<{Vc_sdim*Vc_bsize}; i++) x[i] += t0[i];"""
        else:
            coarse_read = f"""
            for({IntType_c} j=0; j<{Vc_sdim}; j++)
                for({IntType_c} i=0; i<{Vc_bsize}; i++)
                    t0[j + {Vc_sdim}*i] = x[i + {Vc_bsize}*j];
            """
            coarse_write = f"""
            for({IntType_c} j=0; j<{Vc_sdim}; j++)
                for({IntType_c} i=0; i<{Vc_bsize}; i++)
                    x[i + {Vc_bsize}*j] += t0[j + {Vc_sdim}*i];
            """
        if (Vf_bsize == 1) or fine_is_ordered:
            fine_read = f"""for({IntType_c} i=0; i<{Vf_sdim*Vf_bsize}; i++) t1[i] = y[i] * w[i];"""
            fine_write = f"""for({IntType_c} i=0; i<{Vf_sdim*Vf_bsize}; i++) y[i] += t1[i] * w[i];"""
        else:
            fine_read = f"""
            for({IntType_c} j=0; j<{Vf_sdim}; j++)
                for({IntType_c} i=0; i<{Vf_bsize}; i++)
                    t1[j + {Vf_sdim}*i] = y[i + {Vf_bsize}*j] * w[i + {Vf_bsize}*j];
            """
            fine_write = f"""
            for({IntType_c} j=0; j<{Vf_sdim}; j++)
                for({IntType_c} i=0; i<{Vf_bsize}; i++)
                   y[i + {Vf_bsize}*j] += t1[j + {Vf_sdim}*i] * w[i + {Vf_bsize}*j];
            """
        kernel_code = f"""
        {mapping_code}

        {kronmxv_code}

        void prolongation(PetscScalar *restrict y, const PetscScalar *restrict x,
                          const PetscScalar *restrict w{coef_decl}){{
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;
            {operator_decl}
            {coarse_read}
            {prolong_code}
            {fine_write}
            return;
        }}

        void restriction(PetscScalar *restrict x, const PetscScalar *restrict y,
                         const PetscScalar *restrict w{coef_decl}){{
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;
            {operator_decl}
            {fine_read}
            {restrict_code}
            {coarse_write}
            return;
        }}
        """
        from firedrake.slate.slac.compiler import BLASLAPACK_LIB, BLASLAPACK_INCLUDE
        prolong_kernel = op2.Kernel(kernel_code, "prolongation", include_dirs=BLASLAPACK_INCLUDE.split(),
                                    ldargs=BLASLAPACK_LIB.split(), requires_zeroed_output_arguments=True)
        restrict_kernel = op2.Kernel(kernel_code, "restriction", include_dirs=BLASLAPACK_INCLUDE.split(),
                                     ldargs=BLASLAPACK_LIB.split(), requires_zeroed_output_arguments=True)
        return prolong_kernel, restrict_kernel, coefficients

    def make_kernels(self, Vf, Vc):
        """
        Interpolation and restriction kernels between arbitrary elements.

        This is temporary while we wait for dual evaluation in FInAT.
        """
        prolong_kernel, _ = prolongation_transfer_kernel_action(Vf, self.uc)
        matrix_kernel, coefficients = prolongation_transfer_kernel_action(Vf, firedrake.TestFunction(Vc))
        # The way we transpose the prolongation kernel is suboptimal.
        # A local matrix is generated each time the kernel is executed.
        element_kernel = loopy.generate_code_v2(matrix_kernel.code).device_code()
        element_kernel = element_kernel.replace("void expression_kernel", "static void expression_kernel")
        dimc = Vc.finat_element.space_dimension() * Vc.value_size
        dimf = Vf.finat_element.space_dimension() * Vf.value_size

        coef_args = "".join([", c%d" % i for i in range(len(coefficients))])
        coef_decl = "".join([", const %s *restrict c%d" % (ScalarType_c, i) for i in range(len(coefficients))])
        restrict_code = f"""
        {element_kernel}

        void restriction({ScalarType_c} *restrict Rc, const {ScalarType_c} *restrict Rf, const {ScalarType_c} *restrict w{coef_decl})
        {{
            {ScalarType_c} Afc[{dimf}*{dimc}] = {{0}};
            expression_kernel(Afc{coef_args});
            for ({IntType_c} i = 0; i < {dimf}; i++)
               for ({IntType_c} j = 0; j < {dimc}; j++)
                   Rc[j] += Afc[i*{dimc} + j] * Rf[i] * w[i];
        }}
        """
        restrict_kernel = op2.Kernel(restrict_code, "restriction", requires_zeroed_output_arguments=True)
        return prolong_kernel, restrict_kernel, coefficients

    @staticmethod
    def multiplicity(V):
        # Lawrence's magic code for calculating dof multiplicities
        shapes = (V.finat_element.space_dimension(),
                  numpy.prod(V.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        weight = firedrake.Function(V)
        firedrake.par_loop((domain, instructions), firedrake.dx,
                           {"w": (weight, op2.INC)}, is_loopy_kernel=True)
        return weight

    def multTranspose(self, mat, resf, resc):
        """
        Implement restriction: restrict residual on fine grid resf to coarse grid resc.
        """
        with self.uf.dat.vec_wo as xf:
            resf.copy(xf)

        with self.uc.dat.vec_wo as xc:
            xc.set(0.0E0)

        for bc in self.Vf_bcs:
            bc.zero(self.uf)

        self._restrict()

        for bc in self.Vc_bcs:
            bc.zero(self.uc)

        with self.uc.dat.vec_ro as xc:
            xc.copy(resc)

    def mult(self, mat, xc, xf, inc=False):
        """
        Implement prolongation: prolong correction on coarse grid xc to fine grid xf.
        """
        with self.uc.dat.vec_wo as xc_:
            xc.copy(xc_)

        with self.uf.dat.vec_wo as xf_:
            xf_.set(0.0E0)

        for bc in self.Vc_bcs:
            bc.zero(self.uc)

        self._prolong()

        for bc in self.Vf_bcs:
            bc.zero(self.uf)

        if inc:
            with self.uf.dat.vec_ro as xf_:
                xf.axpy(1.0, xf_)
        else:
            with self.uf.dat.vec_ro as xf_:
                xf_.copy(xf)

    def multAdd(self, mat, x, y, w):
        if y.handle == w.handle:
            self.mult(mat, x, w, inc=True)
        else:
            self.mult(mat, x, w)
            w.axpy(1.0, y)


class MixedInterpolationMatrix(StandaloneInterpolationMatrix):
    """
    Interpolation matrix for a mixed finite element space.
    """
    def __init__(self, Vf, Vc, Vf_bcs, Vc_bcs):
        self.Vf_bcs = Vf_bcs
        self.Vc_bcs = Vc_bcs
        self.uf = Vf if isinstance(Vf, firedrake.Function) else firedrake.Function(Vf)
        self.uc = Vc if isinstance(Vc, firedrake.Function) else firedrake.Function(Vc)

        self.standalones = []
        for (i, (uf_sub, uc_sub)) in enumerate(zip(self.uf.split(), self.uc.split())):
            Vf_sub_bcs = [bc for bc in Vf_bcs if bc.function_space().index == i]
            Vc_sub_bcs = [bc for bc in Vc_bcs if bc.function_space().index == i]
            standalone = StandaloneInterpolationMatrix(uf_sub, uc_sub, Vf_sub_bcs, Vc_sub_bcs)
            self.standalones.append(standalone)

        self._prolong = lambda: [standalone._prolong() for standalone in self.standalones]
        self._restrict = lambda: [standalone._restrict() for standalone in self.standalones]

    def getNestSubMatrix(self, i, j):
        if i == j:
            s = self.standalones[i]
            sizes = (s.uf.dof_dset.layout_vec.getSizes(), s.uc.dof_dset.layout_vec.getSizes())
            M_shll = PETSc.Mat().createPython(sizes, s, comm=s.uf.comm)
            M_shll.setUp()
            return M_shll
        else:
            return None


def prolongation_matrix_aij(Pk, P1, Pk_bcs=[], P1_bcs=[]):
    if isinstance(Pk, firedrake.Function):
        Pk = Pk.function_space()
    if isinstance(P1, firedrake.Function):
        P1 = P1.function_space()
    sp = op2.Sparsity((Pk.dof_dset,
                       P1.dof_dset),
                      (Pk.cell_node_map(),
                       P1.cell_node_map()))
    mat = op2.Mat(sp, PETSc.ScalarType)
    mesh = Pk.ufl_domain()

    fele = Pk.ufl_element()
    if isinstance(fele, ufl.MixedElement) and not isinstance(fele, (ufl.VectorElement, ufl.TensorElement)):
        for i in range(fele.num_sub_elements()):
            Pk_bcs_i = [bc for bc in Pk_bcs if bc.function_space().index == i]
            P1_bcs_i = [bc for bc in P1_bcs if bc.function_space().index == i]

            rlgmap, clgmap = mat[i, i].local_to_global_maps
            rlgmap = Pk.sub(i).local_to_global_map(Pk_bcs_i, lgmap=rlgmap)
            clgmap = P1.sub(i).local_to_global_map(P1_bcs_i, lgmap=clgmap)
            unroll = any(bc.function_space().component is not None
                         for bc in chain(Pk_bcs_i, P1_bcs_i) if bc is not None)
            matarg = mat[i, i](op2.WRITE, (Pk.sub(i).cell_node_map(), P1.sub(i).cell_node_map()),
                               lgmaps=((rlgmap, clgmap), ), unroll_map=unroll)
            expr = firedrake.TestFunction(P1.sub(i))
            kernel, coefficients = prolongation_transfer_kernel_action(Pk.sub(i), expr)
            parloop_args = [kernel, mesh.cell_set, matarg]
            for coefficient in coefficients:
                m_ = coefficient.cell_node_map()
                parloop_args.append(coefficient.dat(op2.READ, m_))

            op2.par_loop(*parloop_args)

    else:
        rlgmap, clgmap = mat.local_to_global_maps
        rlgmap = Pk.local_to_global_map(Pk_bcs, lgmap=rlgmap)
        clgmap = P1.local_to_global_map(P1_bcs, lgmap=clgmap)
        unroll = any(bc.function_space().component is not None
                     for bc in chain(Pk_bcs, P1_bcs) if bc is not None)
        matarg = mat(op2.WRITE, (Pk.cell_node_map(), P1.cell_node_map()),
                     lgmaps=((rlgmap, clgmap), ), unroll_map=unroll)
        expr = firedrake.TestFunction(P1)
        kernel, coefficients = prolongation_transfer_kernel_action(Pk, expr)
        parloop_args = [kernel, mesh.cell_set, matarg]
        for coefficient in coefficients:
            m_ = coefficient.cell_node_map()
            parloop_args.append(coefficient.dat(op2.READ, m_))

        op2.par_loop(*parloop_args)

    mat.assemble()
    return mat.handle


def prolongation_matrix_matfree(Vf, Vc, Vf_bcs=[], Vc_bcs=[]):
    fele = Vf.ufl_element()
    if isinstance(fele, ufl.MixedElement) and not isinstance(fele, (ufl.VectorElement, ufl.TensorElement)):
        ctx = MixedInterpolationMatrix(Vf, Vc, Vf_bcs, Vc_bcs)
    else:
        ctx = StandaloneInterpolationMatrix(Vf, Vc, Vf_bcs, Vc_bcs)

    sizes = (Vf.dof_dset.layout_vec.getSizes(), Vc.dof_dset.layout_vec.getSizes())
    M_shll = PETSc.Mat().createPython(sizes, ctx, comm=Vf.comm)
    M_shll.setUp()

    return M_shll
