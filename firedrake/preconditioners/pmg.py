from functools import partial
from firedrake.dmhooks import (attach_hooks, get_appctx, push_appctx, pop_appctx,
                               add_hook, get_parent, push_parent, pop_parent,
                               get_function_space, set_function_space)
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.solving_utils import _SNESContext
from pyop2.utils import as_tuple

import firedrake
import finat
import ufl
import finat.ufl
import weakref

__all__ = ("PMGPC", "PMGSNES")


class PMGBase(PCSNESBase):
    """A class for implementing p-multigrid.

    Internally, this sets up a DM with a custom coarsen routine
    that p-coarsens the problem. This DM is passed to an internal
    PETSc PC of type MG and with options prefix ``pmg_``. The
    relaxation to apply on every p-level is described by ``pmg_mg_levels_``,
    and the coarse solve by ``pmg_mg_coarse_``. Geometric multigrid
    or any other solver in firedrake may be applied to the coarse problem.

    Other PETSc options inspected by this class are:
    - 'pmg_mg_coarse_degree': polynomial degree of the coarse level
    - 'pmg_mg_coarse_mat_type': can be either a `PETSc.Mat.Type`, or 'matfree'
    - 'pmg_mg_coarse_pmat_type': can be either a `PETSc.Mat.Type`, or 'matfree'
    - 'pmg_mg_coarse_form_compiler_mode': can be 'spectral' (default), 'vanilla', 'coffee', or 'tensor'
    - 'pmg_mg_levels_transfer_mat_type': can be either 'aij' or 'matfree'

    The p-coarsening is implemented in the `coarsen_element` routine.
    This takes in a :class:`finat.ufl.finiteelement.FiniteElement` and either returns a
    new, coarser element, or raises a `ValueError` (if the supplied element
    should be the coarsest one of the hierarchy).

    The default coarsen_element is to perform power-of-2 reduction
    of the polynomial degree.
    It is expected that some applications of this preconditioner
    will subclass :class:`PMGBase` to override `coarsen_element` and
    `coarsen_form`.
    """

    _prefix = "pmg_"
    # This is parallel-safe because the keys are ids of a collective objects
    _transfer_cache = weakref.WeakKeyDictionary()

    def coarsen_element(self, ele: finat.ufl.FiniteElementBase) -> finat.ufl.FiniteElementBase:
        """Coarsen a given element to form the next problem down in the p-hierarchy.

        If the supplied element should form the coarsest level of the p-hierarchy,
        raise `ValueError`. Otherwise, return a new :class:`finat.ufl.finiteelement.FiniteElement`.

        By default, this does power-of-2 coarsening in polynomial degree until
        we reach the coarse degree specified through PETSc options (1 by default).

        Parameters
        ----------
        ele
            A :class:`finat.ufl.finiteelement.FiniteElement` to coarsen.

        Returns
        -------
        finat.ufl.FiniteElementBase
            The coarsened element.
        """
        degree = PMGBase.max_degree(ele)
        if degree <= self.coarse_degree:
            raise ValueError
        return ele.reconstruct(degree=max(degree//2, self.coarse_degree))

    def coarsen_form(self, form, fine_to_coarse_map):
        """Coarsen a form, by replacing the solution, test and trial functions.
        """
        return ufl.replace(form, fine_to_coarse_map)

    def initialize(self, obj):
        # Make a new DM.
        # Hook up a (new) coarsen routine on that DM.
        # Make a new PC, of type MG (or SNES of type FAS).
        # Assign the DM to that PC (or SNES).

        odm = obj.getDM()
        ctx = get_appctx(odm)
        if ctx is None:
            raise ValueError("No context found.")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r" % ctx)
        fcp = ctx._problem.form_compiler_parameters
        mode = fcp.get("mode", "spectral") if fcp is not None else "spectral"

        test, trial = ctx.J.arguments()
        if test.function_space() != trial.function_space():
            raise NotImplementedError("test and trial spaces must be the same")

        prefix = obj.getOptionsPrefix() or ""
        options_prefix = prefix + self._prefix
        pdm = PETSc.DMShell().create(comm=obj.comm)
        pdm.setOptionsPrefix(options_prefix)

        ppc = self.configure_pmg(obj, pdm)
        self.is_snes = isinstance(obj, PETSc.SNES)

        default_mat_type = ctx.mat_type
        if default_mat_type == "submatrix":
            default_mat_type = "matfree"

        # Get the coarse degree from PETSc options
        copts = PETSc.Options((ppc.getOptionsPrefix() or "") + ppc.getType() + "_coarse_")
        self.coarse_degree = copts.getInt("degree", default=1)
        self.coarse_mat_type = copts.getString("mat_type", default=default_mat_type)
        self.coarse_pmat_type = copts.getString("pmat_type", default=self.coarse_mat_type)
        self.coarse_form_compiler_mode = copts.getString("form_compiler_mode", default=mode)

        # Construct a list with the elements we'll be using
        V = test.function_space()
        ele = V.ufl_element()
        elements = [ele]
        while True:
            try:
                ele = self.coarsen_element(ele)
            except ValueError:
                break
            elements.append(ele)

        sf = odm.getPointSF()
        section = odm.getDefaultSection()
        attach_hooks(pdm, level=len(elements)-1, sf=sf, section=section)
        # Now overwrite some routines on the DM
        pdm.setRefine(None)
        pdm.setCoarsen(self.coarsen)
        if self.is_snes:
            pdm.setSNESFunction(_SNESContext.form_function)
            pdm.setSNESJacobian(_SNESContext.form_jacobian)
            pdm.setKSPCreateOperators(_SNESContext.create_operators)
            pdm.setKSPComputeOperators(_SNESContext.compute_operators)

        set_function_space(pdm, get_function_space(odm))

        parent = get_parent(odm)
        assert parent is not None
        add_hook(parent, setup=partial(push_parent, pdm, parent), teardown=partial(pop_parent, pdm, parent), call_setup=True)
        add_hook(parent, setup=partial(push_appctx, pdm, ctx), teardown=partial(pop_appctx, pdm, ctx), call_setup=True)

        ppc.incrementTabLevel(1, parent=obj)
        ppc.setFromOptions()
        ppc.setUp()
        self.ppc = ppc

    def update(self, obj):
        self.ppc.setUp()

    def view(self, obj, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT
        viewer.printfASCII("p-multigrid PC\n")
        if hasattr(self, "ppc"):
            self.ppc.view(viewer=viewer)

    def destroy(self, obj):
        if hasattr(self, "ppc"):
            self.ppc.destroy()

    def coarsen(self, fdm: PETSc.DM, comm: PETSc.Comm) -> PETSc.DM:
        """Coarsen the _SNESContext of a DM.

        Parameters
        ----------
        fdm
            The fine-level DM, holding the fine :class:`~.solving_utils._SNESContext`
            as its application context.
        comm
            The communicator for the coarse DM (ignored, PETSc callback signature).

        Returns
        -------
        PETSc.DM
            The coarse DM, holding the coarsened :class:`~.solving_utils._SNESContext`
            as its application context.
        """
        fctx = get_appctx(fdm)
        parent = get_parent(fdm)
        assert parent is not None

        test, trial = fctx.J.arguments()
        fV = trial.function_space()
        cele = self.coarsen_element(fV.ufl_element())

        # Have we already done this?
        cctx = fctx._coarse
        if cctx is not None:
            cV = cctx.J.arguments()[1].function_space()
            if (cV.ufl_element() == cele) and (cV.mesh() == fV.mesh()) and all(cV_.boundary_set == fV_.boundary_set for cV_, fV_ in zip(cV, fV)):
                return cV.dm

        cV = fV.reconstruct(element=cele)
        cdm = cV.dm

        fproblem = fctx._problem
        fdeg = PMGBase.max_degree(fV.ufl_element())
        cdeg = PMGBase.max_degree(cV.ufl_element())

        def _coarsen_form(a, coefficient_mapping):
            if isinstance(a, ufl.Form):
                a = self.coarsen_form(a, coefficient_mapping)
                a = ufl.Form([f.reconstruct(metadata=self.coarsen_quadrature(f.metadata(), fdeg, cdeg))
                              for f in a.integrals()])
            elif isinstance(a, ufl.BaseForm):
                ctest = coefficient_mapping[test]
                coeffs = {k: v for k, v in coefficient_mapping.items() if k not in (test, trial)}
                a = firedrake.interpolate(ctest, ufl.replace(a, coeffs))
            return a

        # Inherit mat_type from the fine _SNESContext
        mat_type = None
        pmat_type = None
        fcp = self.coarsen_quadrature(fproblem.form_compiler_parameters, fdeg, cdeg)
        # If we're the coarsest grid of the p-hierarchy, don't
        # overwrite the coarsen routine; this is so that you can
        # use geometric multigrid for the p-coarse problem
        try:
            self.coarsen_element(cele)
            cdm.setCoarsen(self.coarsen)
        except ValueError:
            mat_type = self.coarse_mat_type
            pmat_type = self.coarse_pmat_type
            fcp = dict(fcp or {}, mode=self.coarse_form_compiler_mode)

        if self.is_snes:
            cF = None
            homogenize_bcs = False
        else:
            # PC-only levels don't need to assemble source terms
            cF = ufl.ZeroBaseForm((test.reconstruct(function_space=cV),))
            homogenize_bcs = True

        # Coarsen the problem
        cu = firedrake.Function(cV)
        cproblem = fproblem.rediscretise(F=cF, u=cu,
                                         form_compiler_parameters=fcp,
                                         form_transform=_coarsen_form,
                                         homogenize_bcs=homogenize_bcs)
        fu = fproblem.u_restrict
        fine_to_coarse_map = dict(zip(fproblem.J.arguments(), cproblem.J.arguments()))
        fine_to_coarse_map[fu] = cu

        # Coarsen the appctx: the user might want to provide solution-dependent expressions and forms
        cappctx = dict(fctx.appctx)
        for key in cappctx:
            val = cappctx[key]
            if isinstance(val, dict):
                cappctx[key] = self.coarsen_quadrature(val, fdeg, cdeg)
            elif isinstance(val, ufl.Form):
                cappctx[key] = _coarsen_form(val, fine_to_coarse_map)
            elif isinstance(val, ufl.classes.Expr):
                cappctx[key] = ufl.replace(val, fine_to_coarse_map)

        # Coarsen the _SNESContext
        cctx = fctx.reconstruct(cproblem, mat_type, pmat_type,
                                appctx=cappctx)

        # FIXME setting up the _fine attribute triggers gmg injection.
        # cctx._fine = fctx
        fctx._coarse = cctx

        add_hook(parent, setup=partial(push_parent, cdm, parent), teardown=partial(pop_parent, cdm, parent), call_setup=True)
        add_hook(parent, setup=partial(push_appctx, cdm, cctx), teardown=partial(pop_appctx, cdm, cctx), call_setup=True)

        cdm.setOptionsPrefix(fdm.getOptionsPrefix())
        cdm.setKSPCreateOperators(_SNESContext.create_operators)
        cdm.setKSPComputeOperators(_SNESContext.compute_operators)
        cdm.setCreateInterpolation(self.create_interpolation)
        cdm.setCreateInjection(self.create_injection)

        if cu in cproblem.J.coefficients():
            # Only inject state if the coarse state is a dependency of the coarse Jacobian.
            inject = cdm.createInjection(fdm)

            def inject_state():
                with cu.dat.vec_wo as xc, fu.dat.vec_ro as xf:
                    inject.mult(xf, xc)

            add_hook(parent, setup=inject_state, call_setup=True)

        def _coarsen_nullspace(fine_nullspace):
            return None if fine_nullspace is None else fine_nullspace.rediscretise(cV)

        cctx._nullspace = _coarsen_nullspace(fctx._nullspace)
        cctx._nullspace_T = _coarsen_nullspace(fctx._nullspace_T)
        cctx._near_nullspace = _coarsen_nullspace(fctx._near_nullspace)
        cctx.set_nullspace(cctx._nullspace, cV._ises, transpose=False, near=False)
        cctx.set_nullspace(cctx._nullspace_T, cV._ises, transpose=True, near=False)
        cctx.set_nullspace(cctx._near_nullspace, cV._ises, transpose=False, near=True)
        return cdm

    @staticmethod
    def coarsen_quadrature(metadata: dict | None, fdeg: int, cdeg: int) -> dict | None:
        """Coarsen the quadrature degree in a dictionary preserving the ratio of
        quadrature nodes to interpolation nodes (qdeg+1)//(fdeg+1).

        Parameters
        ----------
        metadata
            The fine-level form metadata, possibly containing a ``quadrature_degree`` entry.
        fdeg
            The fine-level polynomial degree.
        cdeg
            The coarse-level polynomial degree.

        Returns
        -------
        dict or None
            A copy of `metadata` with the coarsened ``quadrature_degree``, or `metadata`
            unchanged if it does not specify a quadrature degree.
        """
        try:
            qdeg = metadata["quadrature_degree"]
            coarse_qdeg = max(2*cdeg+1, ((qdeg+1)*(cdeg+1)+fdeg)//(fdeg+1)-1)
            return dict(metadata, quadrature_degree=coarse_qdeg)
        except (KeyError, TypeError):
            return metadata

    def create_transfer(self, mat_type, cctx, fctx, cbcs, fbcs):
        """Create a transfer operator"""
        cache = self._transfer_cache.setdefault(fctx, {})
        key = (mat_type, cctx, cbcs, fbcs)
        try:
            return cache[key]
        except KeyError:
            cV = cctx._problem.u_restrict.function_space()
            fV = fctx._problem.u_restrict.function_space()
            cbcs = tuple(cctx._problem.bcs) if cbcs else tuple()
            fbcs = tuple(fctx._problem.bcs) if fbcs else tuple()
            bcs = cbcs + fbcs
            interp = firedrake.interpolate(firedrake.TrialFunction(cV), fV)
            Pmat = firedrake.assemble(interp, bcs=bcs, mat_type=mat_type).petscmat
            return cache.setdefault(key, Pmat)

    def create_interpolation(self, dmc, dmf):
        prefix = dmc.getOptionsPrefix()
        mat_type = PETSc.Options(prefix).getString("mg_levels_transfer_mat_type", default="matfree")
        interpolation = self.create_transfer(mat_type, get_appctx(dmc), get_appctx(dmf), True, False)
        rscale = interpolation.createVecRight()
        return interpolation, rscale

    def create_injection(self, dmc, dmf):
        prefix = dmc.getOptionsPrefix()
        mat_type = PETSc.Options(prefix).getString("mg_levels_transfer_mat_type", default="matfree")
        return self.create_transfer(mat_type, get_appctx(dmf), get_appctx(dmc), False, False)

    @staticmethod
    def max_degree(ele):
        """Return the maximum degree of a :class:`finat.ufl.finiteelement.FiniteElement`"""
        return max(as_tuple(ele.degree()))


class PMGPC(PCBase, PMGBase):
    _prefix = "pmg_"

    def configure_pmg(self, pc, pdm):
        odm = pc.getDM()
        ppc = PETSc.PC().create(comm=pc.comm)
        ppc.setOptionsPrefix((pc.getOptionsPrefix() or "") + self._prefix)
        ppc.setType("mg")
        ppc.setOperators(*pc.getOperators())
        ppc.setDM(pdm)

        # PETSc unfortunately requires us to make an ugly hack.
        # We would like to use GMG for the coarse solve, at least
        # sometimes. But PETSc will use this p-DM's getRefineLevels()
        # instead of the getRefineLevels() of the MeshHierarchy to
        # decide how many levels it should use for PCMG applied to
        # the p-MG's coarse problem. So we need to set an option
        # for the user, if they haven't already; I don't know any
        # other way to get PETSc to know this at the right time.
        max_levels = odm.getRefineLevel() + 1
        if max_levels > 1:
            opts = PETSc.Options((pc.getOptionsPrefix() or "") + "pmg_")
            if opts.getString("mg_coarse_pc_type") == "mg":
                opts["mg_coarse_pc_mg_levels"] = min(opts.getInt("mg_coarse_pc_mg_levels", max_levels), max_levels)
        return ppc

    def apply(self, pc, x, y):
        return self.ppc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        return self.ppc.applyTranspose(x, y)


class PMGSNES(SNESBase, PMGBase):
    _prefix = "pfas_"

    def configure_pmg(self, snes, pdm):
        odm = snes.getDM()
        psnes = PETSc.SNES().create(comm=snes.comm)
        psnes.setOptionsPrefix((snes.getOptionsPrefix() or "") + self._prefix)
        psnes.setType("fas")
        psnes.setDM(pdm)
        psnes.setTolerances(max_it=1)
        psnes.setConvergenceTest("skip")

        (f, residual) = snes.getFunction()
        assert residual is not None
        (fun, args, kargs) = residual
        psnes.setFunction(fun, f.duplicate(), args=args, kargs=kargs)

        pdm.setGlobalVector(f.duplicate())
        psnes.setSolution(snes.getSolution())

        # PETSc unfortunately requires us to make an ugly hack.
        # We would like to use GMG for the coarse solve, at least
        # sometimes. But PETSc will use this p-DM's getRefineLevels()
        # instead of the getRefineLevels() of the MeshHierarchy to
        # decide how many levels it should use for PCMG applied to
        # the p-MG's coarse problem. So we need to set an option
        # for the user, if they haven't already; I don't know any
        # other way to get PETSc to know this at the right time.
        max_levels = odm.getRefineLevel() + 1
        if max_levels > 1:
            opts = PETSc.Options((snes.getOptionsPrefix() or "") + "pfas_")
            if opts.getString("fas_coarse_pc_type") == "mg":
                opts["fas_coarse_pc_mg_levels"] = min(opts.getInt("fas_coarse_pc_mg_levels", max_levels), max_levels)
            if opts.getString("fas_coarse_snes_type") == "fas":
                opts["fas_coarse_snes_fas_levels"] = min(opts.getInt("fas_coarse_snes_fas_levels", max_levels), max_levels)
        return psnes

    def step(self, snes, x, f, y):
        ctx = get_appctx(snes.dm)
        push_appctx(self.ppc.dm, ctx)
        x.copy(y)
        self.ppc.solve(snes.vec_rhs or None, y)
        y.aypx(-1, x)
        snes.setConvergedReason(self.ppc.getConvergedReason())
        pop_appctx(self.ppc.dm)
