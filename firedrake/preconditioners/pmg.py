from functools import partial
from itertools import chain

from ufl import MixedElement, VectorElement, TensorElement, replace

from pyop2 import op2

from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.dmhooks import attach_hooks, get_appctx, push_appctx, pop_appctx
from firedrake.dmhooks import add_hook, get_parent, push_parent, pop_parent
from firedrake.dmhooks import get_function_space, set_function_space
from firedrake.solving_utils import _SNESContext
import firedrake


class PMGPC(PCBase):
    """
    A class for implementing p-multigrid.

    Internally, this sets up a DM with a custom coarsen routine
    that p-coarsens the problem. This DM is passed to an internal
    PETSc PC of type MG and with options prefix 'pmg_'. The
    relaxation to apply on every p-level is described by 'pmg_mg_levels_',
    and the coarse solve by 'pmg_mg_coarse_'. Geometric multigrid
    or any other solver in firedrake may be applied to the coarse problem.
    An example chaining p-MG, GMG and AMG is given in the tests.

    The p-coarsening is implemented in the `coarsen_element` routine.
    This takes in a :class:`ufl.FiniteElement` and either returns a
    new, coarser element, or raises a `ValueError` (if the supplied element
    should be the coarsest one of the hierarchy).

    The default coarsen_element is to perform power-of-2 reduction
    of the polynomial degree. For mixed systems a `NotImplementedError`
    is raised, as I don't know how to make a sensible default for this.
    It is expected that many (most?) applications of this preconditioner
    will subclass :class:`PMGPC` to override `coarsen_element`.
    """
    @staticmethod
    def coarsen_element(ele):
        """
        Coarsen a given element to form the next problem down in the p-hierarchy.

        If the supplied element should form the coarsest level of the p-hierarchy,
        raise `ValueError`. Otherwise, return a new :class:`ufl.FiniteElement`.

        By default, this does power-of-2 coarsening in polynomial degree.
        It raises a `NotImplementedError` for :class:`ufl.MixedElement`s, as
        I don't know if there's a sensible default strategy to implement here.
        It is intended that the user subclass `PMGPC` to override this method
        for their problem.

        :arg ele: a :class:`ufl.FiniteElement` to coarsen.
        """
        if isinstance(ele, MixedElement) and not isinstance(ele, (VectorElement, TensorElement)):
            raise NotImplementedError("Implement this method yourself")

        degree = ele.degree()
        family = ele.family()

        if family == "Discontinuous Galerkin" and degree == 0:
            raise ValueError
        elif degree == 1:
            raise ValueError

        return ele.reconstruct(degree=degree // 2)

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
        set_function_space(pdm, get_function_space(odm))

        parent = get_parent(odm)
        assert parent is not None
        add_hook(parent, setup=partial(push_parent, pdm, parent), teardown=partial(pop_parent, pdm, parent),
                 call_setup=True)
        add_hook(parent, setup=partial(push_appctx, pdm, ctx), teardown=partial(pop_appctx, pdm, ctx),
                 call_setup=True)

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
        assert parent is not None
        add_hook(parent, setup=partial(push_parent, cdm, parent), teardown=partial(pop_parent, cdm, parent),
                 call_setup=True)

        replace_d = {fu: cu,
                     test: firedrake.TestFunction(cV),
                     trial: firedrake.TrialFunction(cV)}
        cJ = replace(fctx.J, replace_d)
        cF = replace(fctx.F, replace_d)
        if fctx.Jp is not None:
            cJp = replace(fctx.Jp, replace_d)
        else:
            cJp = None

        cbcs = []
        for bc in fctx._problem.bcs:
            # Don't actually need the value, since it's only used for
            # killing parts of the matrix. This should be generalised
            # for p-FAS, if anyone ever wants to do that

            cV_ = cV
            for index in bc._indices:
                cV_ = cV_.sub(index)

            cbcs.append(firedrake.DirichletBC(cV_, firedrake.zero(cV_.shape),
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

        cbcs = cctx._problem.bcs
        fbcs = fctx._problem.bcs

        I = prolongation_matrix(fV, cV, fbcs, cbcs)
        R = PETSc.Mat().createTranspose(I)
        return R, None

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT
        viewer.printfASCII("p-multigrid PC\n")
        self.ppc.view(viewer)


def prolongation_transfer_kernel(Pk, P1):
    # Works for Pk, Pm; I just retain the notation
    # P1 to remind you that P1 is of lower degree
    # than Pk
    from tsfc import compile_expression_dual_evaluation
    from tsfc.fiatinterface import create_element
    from firedrake import TestFunction

    expr = TestFunction(P1)
    coords = Pk.ufl_domain().coordinates
    to_element = create_element(Pk.ufl_element(), vector_is_mixed=False)

    ast, oriented, needs_cell_sizes, coefficients, _ = compile_expression_dual_evaluation(expr, to_element, coords, coffee=False)
    kernel = op2.Kernel(ast, ast.name)
    return kernel


def prolongation_matrix(Pk, P1, Pk_bcs, P1_bcs):
    sp = op2.Sparsity((Pk.dof_dset,
                       P1.dof_dset),
                      (Pk.cell_node_map(),
                       P1.cell_node_map()))
    mat = op2.Mat(sp, PETSc.ScalarType)
    mesh = Pk.ufl_domain()

    fele = Pk.ufl_element()
    if isinstance(fele, MixedElement) and not isinstance(fele, (VectorElement, TensorElement)):
        for i in range(fele.num_sub_elements()):
            Pk_bcs_i = [bc for bc in Pk_bcs if bc.function_space().index == i]
            P1_bcs_i = [bc for bc in P1_bcs if bc.function_space().index == i]

            rlgmap, clgmap = mat[i, i].local_to_global_maps
            rlgmap = Pk.sub(i).local_to_global_map(Pk_bcs_i, lgmap=rlgmap)
            clgmap = P1.sub(i).local_to_global_map(P1_bcs_i, lgmap=clgmap)
            unroll = any(bc.function_space().component is not None
                         for bc in chain(Pk_bcs_i, P1_bcs_i) if bc is not None)
            matarg = mat[i, i](op2.WRITE, (Pk.sub(i).cell_node_map(), P1.sub(i).cell_node_map()),
                               lgmaps=(rlgmap, clgmap), unroll_map=unroll)
            op2.par_loop(prolongation_transfer_kernel(Pk.sub(i), P1.sub(i)), mesh.cell_set,
                         matarg)

    else:
        rlgmap, clgmap = mat.local_to_global_maps
        rlgmap = Pk.local_to_global_map(Pk_bcs, lgmap=rlgmap)
        clgmap = P1.local_to_global_map(P1_bcs, lgmap=clgmap)
        unroll = any(bc.function_space().component is not None
                     for bc in chain(Pk_bcs, P1_bcs) if bc is not None)
        matarg = mat(op2.WRITE, (Pk.cell_node_map(), P1.cell_node_map()),
                     lgmaps=(rlgmap, clgmap), unroll_map=unroll)
        op2.par_loop(prolongation_transfer_kernel(Pk, P1), mesh.cell_set,
                     matarg)

    mat.assemble()
    return mat.handle
