from functools import partial

from ufl import MixedElement, FiniteElement, VectorElement, TensorElement, replace
from ufl.algorithms import map_integrands

from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.low_order import ArgumentReplacer, restriction_matrix
from firedrake.dmhooks import attach_hooks, get_appctx, push_appctx, pop_appctx
from firedrake.dmhooks import add_hook, get_parent, push_parent, pop_parent
from firedrake.solving_utils import _SNESContext
import firedrake


class PMGPC(PCBase):
    @staticmethod
    def coarsen_element(ele):
        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        degree = ele.degree()
        family = ele.family()

        if family == "Discontinuous Galerkin" and degree == 0:
            raise ValueError
        elif degree == 1:
            raise ValueError

        shape = ele.value_shape()

        if len(shape) == 0:
            new_ele = FiniteElement(family, ele.cell(), degree // 2)
        elif len(shape) == 1:
            new_ele = VectorElement(family, ele.cell(), degree // 2, dim=shape[0])
        else:
            new_ele = TensorElement(family, ele.cell(), degree // 2, shape=shape,
                                    symmetry=ele.symmetry())
        return new_ele

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

        pdm = PETSc.DMShell().create(comm=pc.comm)
        sf = odm.getPointSF()
        section = odm.getDefaultSection()
        attach_hooks(pdm, level=len(elements)-1, sf=sf, section=section)
        # Now overwrite some routines on the DM
        pdm.setRefine(None)
        pdm.setCoarsen(self.coarsen)
        pdm.setCreateInterpolation(self.create_interpolation)

        parent = get_parent(odm)
        add_hook(parent, setup=partial(push_parent, pdm, parent), teardown=partial(pop_parent, pdm, parent),
                 call_setup=True)
        add_hook(parent, setup=partial(push_appctx, pdm, ctx), teardown=partial(pop_appctx, pdm, ctx),
                 call_setup=True)

        ppc = PETSc.PC().create()
        ppc.setOptionsPrefix(pc.getOptionsPrefix() + "pmg_")
        ppc.setType("mg")
        ppc.setOperators(*pc.getOperators())
        ppc.setDM(pdm)
        ppc.incrementTabLevel(1, parent=pc)
        ppc.setFromOptions()
        ppc.setUp()
        self.ppc = ppc

    def apply(self, pc, x, y):
        return self.ppc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        return self.ppc.applyTranspose(x, y)

    def update(self, pc):
        pass

    def coarsen(self, fdm, comm):
        fctx = get_appctx(fdm)
        test, trial = fctx.J.arguments()
        fV = test.function_space()
        fu = fctx._problem.u

        cele = self.coarsen_element(fV.ufl_element())
        cV = firedrake.FunctionSpace(fV.mesh(), cele)
        cdm = cV.dm
        cu = firedrake.Function(cV)

        parent = get_parent(fdm)
        add_hook(parent, setup=partial(push_parent, cdm, parent), teardown=partial(pop_parent, cdm, parent),
                 call_setup=True)

        mapper = ArgumentReplacer({test: firedrake.TestFunction(cV),
                                   trial: firedrake.TrialFunction(cV)})
        cJ = map_integrands.map_integrand_dags(mapper, fctx.J)
        cF = map_integrands.map_integrand_dags(mapper, fctx.F)
        cF = replace(cF, {fu: cu})
        if fctx.Jp is not None:
            cJp = map_integrands.map_integrand_dags(mapper, fctx.Jp)
        else:
            cJp = None

        cbcs = []
        for bc in fctx._problem.bcs:
            # Don't actually need the value, since it's only used for
            # killing parts of the matrix. This should be generalised
            # for p-FAS, if anyone ever wants to do that
            cbcs.append(firedrake.DirichletBC(cV, firedrake.zero(cV.shape),
                                              bc.sub_domain,
                                              method=bc.method))

        fcp = fctx._problem.form_compiler_parameters
        cproblem = firedrake.NonlinearVariationalProblem(cF, cu, cbcs, cJ,
                                                         Jp=cJp,
                                                         form_compiler_parameters=fcp,
                                                         is_linear=fctx._problem.is_linear)

        cctx = _SNESContext(cproblem, fctx.mat_type, fctx.pmat_type,
                            appctx=fctx.appctx,
                            pre_jacobian_callback=fctx._pre_jacobian_callback,
                            pre_function_callback=fctx._pre_function_callback,
                            post_jacobian_callback=fctx._post_jacobian_callback,
                            post_function_callback=fctx._post_function_callback,
                            options_prefix=fctx.options_prefix,
                            transfer_manager=fctx.transfer_manager)

        add_hook(parent, setup=partial(push_appctx, cdm, cctx), teardown=partial(pop_appctx, cdm, cctx),
                 call_setup=True)

        cdm.setKSPComputeOperators(_SNESContext.compute_operators)
        cdm.setCreateInterpolation(self.create_interpolation)

        # If we're the coarsest grid of the p-hierarchy, don't
        # overwrite the coarsen routine; this is so that you can
        # use geometric multigrid for the p-coarse problem
        try:
            self.coarsen_element(cele)
            cdm.setCoarsen(self.coarsen)
        except ValueError:
            pass

        return cdm

    def create_interpolation(self, dmc, dmf):
        # This should be generalised to work for arbitrary function
        # spaces. Currently I think it only works for CG/DG on simplices.
        # I used the same code as firedrake.P1PC.
        cctx = get_appctx(dmc)
        fctx = get_appctx(dmf)

        cV = cctx.J.arguments()[0].function_space()
        fV = fctx.J.arguments()[0].function_space()

        assert cV.ufl_element().family() in ["Lagrange", "Discontinous Lagrange"]
        assert fV.ufl_element().family() in ["Lagrange", "Discontinous Lagrange"]

        cbcs = cctx._problem.bcs
        fbcs = fctx._problem.bcs

        R = restriction_matrix(fV, cV, fbcs, cbcs)
        return R, None

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT
        viewer.printfASCII("p-multigrid PC\n")
        self.ppc.view(viewer)
