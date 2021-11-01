from functools import partial, lru_cache
from itertools import chain

from ufl import MixedElement, VectorElement, TensorElement, TensorProductElement
from ufl import EnrichedElement, HDivElement, HCurlElement, WithMapping
from ufl import Form, replace
from ufl.classes import Expr

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
        if isinstance(ele, (VectorElement, TensorElement)):
            return PMGBase.max_degree(ele._sub_element)
        elif isinstance(ele, (MixedElement, TensorProductElement)):
            return max(PMGBase.max_degree(sub) for sub in ele.sub_elements())
        elif isinstance(ele, EnrichedElement):
            return max(PMGBase.max_degree(sub) for sub in ele._elements)
        elif isinstance(ele, WithMapping):
            return PMGBase.max_degree(ele.wrapee)
        else:
            try:
                return PMGBase.max_degree(ele._element)
            except AttributeError:
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
        if isinstance(ele, VectorElement):
            return VectorElement(PMGBase.reconstruct_degree(ele._sub_element, N), dim=ele.num_sub_elements())
        elif isinstance(ele, TensorElement):
            return TensorElement(PMGBase.reconstruct_degree(ele._sub_element, N), shape=ele.value_shape(), symmetry=ele.symmetry())
        elif isinstance(ele, EnrichedElement):
            shift = N-PMGBase.max_degree(ele)
            return EnrichedElement(*(PMGBase.reconstruct_degree(e, PMGBase.max_degree(e)+shift) for e in ele._elements))
        elif isinstance(ele, TensorProductElement):
            shift = N-PMGBase.max_degree(ele)
            return TensorProductElement(*(PMGBase.reconstruct_degree(e, PMGBase.max_degree(e)+shift) for e in ele.sub_elements()), cell=ele.cell())
        elif isinstance(ele, MixedElement):
            shift = N-PMGBase.max_degree(ele)
            return MixedElement(*(PMGBase.reconstruct_degree(e, PMGBase.max_degree(e)+shift) for e in ele.sub_elements()))
        elif isinstance(ele, WithMapping):
            return WithMapping(PMGBase.reconstruct_degree(ele.wrapee, N), ele.mapping())
        else:
            try:
                return type(ele)(PMGBase.reconstruct_degree(ele._element, N))
            except AttributeError:
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
        pdm = PETSc.DMShell().create(comm=pc.comm)
        pdm.setOptionsPrefix(options_prefix)

        # Get the coarse degree from PETSc options
        fcp = ctx._problem.form_compiler_parameters
        mode = fcp.get("mode", "spectral") if fcp is not None else "spectral"
        self.coarse_degree = PETSc.Options(options_prefix).getInt("coarse_degree", default=1)
        self.coarse_mat_type = PETSc.Options(options_prefix).getString("coarse_mat_type", default=ctx.mat_type)
        self.coarse_form_compiler_mode = PETSc.Options(options_prefix).getString("coarse_form_compiler_mode", default=mode)

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

        def coarsen_quadrature(df, Nf, Nc):
            # Coarsen the quadrature degree in a dictionary
            # such that the ratio of quadrature nodes to interpolation nodes (Nq+1)/(Nf+1) is preserved
            if isinstance(df, dict):
                Nq = df.get("quadrature_degree", None)
                if Nq is not None:
                    dc = dict(df)
                    dc["quadrature_degree"] = max(2*Nc+1, ((Nq+1) * (Nc+1) + Nf) // (Nf+1) - 1)
                    return dc
            return df

        def coarsen_form(form, Nf, Nc, replace_d):
            # Coarsen a form, by replacing the solution, test and trial functions, and
            # reconstructing each integral with a coarsened quadrature degree.
            # If form is not a Form, then return form.
            return Form([f.reconstruct(metadata=coarsen_quadrature(f.metadata(), Nf, Nc))
                         for f in replace(form, replace_d).integrals()]) if isinstance(form, Form) else form

        def coarsen_bcs(fbcs):
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

        Nf = PMGBase.max_degree(fV.ufl_element())
        Nc = PMGBase.max_degree(cV.ufl_element())

        # Replace dictionary with coarse state, test and trial functions
        replace_d = {fu: cu,
                     test: firedrake.TestFunction(cV),
                     trial: firedrake.TrialFunction(cV)}

        cF = coarsen_form(fctx.F, Nf, Nc, replace_d)
        cJ = coarsen_form(fctx.J, Nf, Nc, replace_d)
        cJp = coarsen_form(fctx.Jp, Nf, Nc, replace_d)
        fcp = coarsen_quadrature(fproblem.form_compiler_parameters, Nf, Nc)
        cbcs = coarsen_bcs(fproblem.bcs)

        # Coarsen the appctx: the user might want to provide solution-dependant expressions and forms
        cappctx = dict(fctx.appctx)
        for key in cappctx:
            val = cappctx[key]
            if isinstance(val, dict):
                cappctx[key] = coarsen_quadrature(val, Nf, Nc)
            elif isinstance(val, Expr):
                cappctx[key] = replace(val, replace_d)
            elif isinstance(val, Form):
                cappctx[key] = coarsen_form(val, Nf, Nc, replace_d)

        cmat_type = fctx.mat_type
        cpmat_type = fctx.pmat_type
        if Nc == self.coarse_degree:
            cmat_type = self.coarse_mat_type
            cpmat_type = self.coarse_mat_type
            if fcp is None:
                fcp = dict()
            fcp["mode"] = self.coarse_form_compiler_mode

        # Coarsen the problem and the _SNESContext
        cproblem = firedrake.NonlinearVariationalProblem(cF, cu, bcs=cbcs, J=cJ, Jp=cJp,
                                                         form_compiler_parameters=fcp,
                                                         is_linear=fproblem.is_linear)

        cctx = type(fctx)(cproblem, cmat_type, cpmat_type,
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

        # If we're the coarsest grid of the p-hierarchy, don't
        # overwrite the coarsen routine; this is so that you can
        # use geometric multigrid for the p-coarse problem
        try:
            self.coarsen_element(cele)
            cdm.setCoarsen(self.coarsen)
        except ValueError:
            pass

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


def tensor_product_space_query(V):
    """
    Checks whether the custom transfer kernels support the FunctionSpace V.

    V must be either Q(N) or DQ(N) (same N along every direction),
    RTCF(N), RTCE(N), NCF(N), or NCE(N) on quads or hexes.

    :arg V: FunctionSpace
    :returns: 4-tuple of (use_tensorproduct, degree, topological_dimension, sub_families, variant)
    """
    from FIAT import reference_element
    ndim = V.ufl_domain().topological_dimension()
    iscube = (ndim == 2 or ndim == 3) and reference_element.is_hypercube(V.finat_element.cell)

    ele = V.ufl_element()
    if isinstance(ele, firedrake.WithMapping):
        ele = ele.wrapee
    if isinstance(ele, (firedrake.VectorElement, firedrake.TensorElement)):
        subel = ele.sub_elements()
        ele = subel[0]

    N = ele.degree()
    use_tensorproduct = True
    try:
        N, = set(N)
    except ValueError:
        # Tuple with different extents
        use_tensorproduct = False
    except TypeError:
        # Just a single int
        pass
    if isinstance(ele, TensorProductElement):
        sub_families = set(e.family() for e in ele.sub_elements())
        try:
            # variant = None defaults to spectral
            # We must allow tensor products between None and spectral
            variant, = set(e.variant() or "spectral" for e in ele.sub_elements())
        except ValueError:
            # Multiple variants
            variant = "unsupported"
            use_tensorproduct = False
    elif isinstance(ele, EnrichedElement):
        if all(isinstance(sub, HDivElement) for sub in ele._elements):
            sub_families = {"NCF"}
        elif all(isinstance(sub, HCurlElement) for sub in ele._elements):
            sub_families = {"NCE"}
        else:
            sub_families = {"unknown"}
        variant = None
    else:
        sub_families = {ele.family()}
        variant = ele.variant()

    isCG = sub_families <= {"Q", "Lagrange"}
    isDG = sub_families <= {"DQ", "Discontinuous Lagrange"}
    isHdiv = sub_families < {"RTCF", "NCF"}
    isHcurl = sub_families < {"RTCE", "NCE"}
    isspectral = variant is None or variant == "spectral"
    use_tensorproduct = use_tensorproduct and iscube and isspectral and (isCG or isDG or isHdiv or isHcurl)

    return use_tensorproduct, N, ndim, sub_families, variant


def get_permuted_map(V):
    # Return a PermutedMap with the same tensor product shape for every component of H(div) or H(curl) tensor product elements
    use_tensorproduct, N, ndim, sub_families, _ = tensor_product_space_query(V)
    if use_tensorproduct and sub_families < {"RTCF", "NCF"}:
        pshape = [N]*ndim
        pshape[0] = -1
    elif use_tensorproduct and sub_families < {"RTCE", "NCE"}:
        pshape = [N+1]*ndim
        pshape[0] = -1
    else:
        return V.cell_node_map()

    ncomp, = V.finat_element.value_shape
    permutation = numpy.reshape(numpy.arange(V.finat_element.space_dimension()), (ncomp, -1))
    for k in range(ncomp):
        permutation[k] = numpy.reshape(numpy.transpose(numpy.reshape(permutation[k], pshape), axes=(1+k+numpy.arange(ncomp)) % ncomp), (-1,))
    permutation = numpy.reshape(permutation, (-1,))
    return PermutedMap(V.cell_node_map(), permutation)


def get_line_element(V):
    # Return the Line elements for Q, DQ, RTCF/E, NCF/E
    from FIAT.reference_element import UFCInterval
    from FIAT import gauss_legendre, gauss_lobatto_legendre, lagrange, discontinuous_lagrange
    use_tensorproduct, N, ndim, sub_families, variant = tensor_product_space_query(V)
    assert use_tensorproduct
    cell = UFCInterval()
    if sub_families <= {"Q", "Lagrange"}:
        if variant == "equispaced":
            element = lagrange.Lagrange(cell, N)
        else:
            element = gauss_lobatto_legendre.GaussLobattoLegendre(cell, N)
        element = [element]*ndim
    elif sub_families <= {"DQ", "Discontinuous Lagrange"}:
        if variant == "equispaced":
            element = discontinuous_lagrange.DiscontinuousLagrange(cell, N)
        else:
            element = gauss_legendre.GaussLegendre(cell, N)
        element = [element]*ndim
    elif sub_families < {"RTCF", "NCF"}:
        cg = gauss_lobatto_legendre.GaussLobattoLegendre(cell, N)
        dg = gauss_legendre.GaussLegendre(cell, N-1)
        element = [cg] + [dg]*(ndim-1)
    elif sub_families < {"RTCE", "NCE"}:
        cg = gauss_lobatto_legendre.GaussLobattoLegendre(cell, N)
        dg = gauss_legendre.GaussLegendre(cell, N-1)
        element = [dg] + [cg]*(ndim-1)
    else:
        raise ValueError("Don't know how to get line element for %s" % V.ufl_element())
    return element


def get_line_nodes(V):
    # Return the Line nodes for Q, DQ, RTCF/E, NCF/E
    from FIAT.reference_element import UFCInterval
    from FIAT import quadrature
    use_tensorproduct, N, ndim, sub_families, variant = tensor_product_space_query(V)
    assert use_tensorproduct
    cell = UFCInterval()
    if variant == "equispaced" and sub_families <= {"Q", "DQ", "Lagrange", "Discontinuous Lagrange"}:
        return [cell.make_points(1, 0, N+1)]*ndim
    elif sub_families <= {"Q", "Lagrange"}:
        rule = quadrature.GaussLobattoLegendreQuadratureLineRule(cell, N+1)
        return [rule.get_points()]*ndim
    elif sub_families <= {"DQ", "Discontinuous Lagrange"}:
        rule = quadrature.GaussLegendreQuadratureLineRule(cell, N+1)
        return [rule.get_points()]*ndim
    elif sub_families < {"RTCF", "NCF"}:
        cg = quadrature.GaussLobattoLegendreQuadratureLineRule(cell, N+1)
        dg = quadrature.GaussLegendreQuadratureLineRule(cell, N)
        return [cg.get_points()] + [dg.get_points()]*(ndim-1)
    elif sub_families < {"RTCE", "NCE"}:
        cg = quadrature.GaussLobattoLegendreQuadratureLineRule(cell, N+1)
        dg = quadrature.GaussLegendreQuadratureLineRule(cell, N)
        return [dg.get_points()] + [cg.get_points()]*(ndim-1)
    else:
        raise ValueError("Don't know how to get line nodes for %s" % V.ufl_element())


# Common kernel to compute y = kron(A3, kron(A2, A1)) * x
# Vector and tensor field generalization from Deville, Fischer, and Mund section 8.3.1.
kronmxv_code = """
#include <petscsys.h>
#include <petscblaslapack.h>

static void kronmxv(int tflag,
    PetscBLASInt mx, PetscBLASInt my, PetscBLASInt mz,
    PetscBLASInt nx, PetscBLASInt ny, PetscBLASInt nz, PetscBLASInt nel,
    PetscScalar  *A1, PetscScalar *A2, PetscScalar *A3,
    PetscScalar  *x , PetscScalar *y){

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

PetscBLASInt m,n,k,s,p,lda;
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
    nscal = Vf.ufl_element().value_size()
    celem = get_line_element(Vc)
    nodes = get_line_nodes(Vf)
    Jhat = [e.tabulate(0, z)[(0,)] for e, z in zip(celem, nodes)]
    ndim = len(Jhat)

    # Declare array shapes to be used as literals inside the kernels
    # I follow to the m-by-n convention with the FORTRAN ordering (so I have to do n-by-m in python)
    shapes = [[Jk.shape[j] for Jk in Jhat] for j in range(2)]
    n, m = shapes.copy()
    m += [1]*(3-ndim)
    n += [1]*(3-ndim)
    shapes[0].append(nscal)
    shapes[1].append(nscal)

    # Pass the 1D tabulation as hexadecimal string
    # The Kronecker product routines assume 3D shapes, so in 2D we pass one instead of Jhat
    JX = ', '.join(map(float.hex, numpy.concatenate([numpy.asarray(Jk).flatten() for Jk in Jhat])))
    JY = "&one" if ndim < 2 else f"{mat}+{Jhat[0].size}"
    JZ = "&one" if ndim < 3 else f"{mat}+{Jhat[0].size+Jhat[1].size}"
    Jlen = sum([Jk.size for Jk in Jhat])

    operator_decl = f"""const PetscScalar {mat}[{Jlen}] = {{ {JX} }};"""
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
            return firedrake.JacobianInverse(domain) * firedrake.JacobianDeterminant(domain)
        else:
            return firedrake.Jacobian(domain) / firedrake.JacobianDeterminant(domain)
    elif emap == "covariant piola":
        if inverse:
            return firedrake.Jacobian(domain).T
        else:
            return firedrake.JacobianInverse(domain).T
    else:
        raise ValueError("Unsupported mapping")


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
    expr = firedrake.dot(tensor, u)
    prolong_map_kernel, coefficients = prolongation_transfer_kernel_action(Q, expr)
    prolong_map_code = loopy.generate_code_v2(prolong_map_kernel.code).device_code()
    prolong_map_code = prolong_map_code.replace("void expression_kernel", "static void prolongation_mapping")
    coefficients.remove(u)

    expr = firedrake.dot(u, tensor)
    restrict_map_kernel, coefficients = prolongation_transfer_kernel_action(Q, expr)
    restrict_map_code = loopy.generate_code_v2(restrict_map_kernel.code).device_code()
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


def make_permutation_code(elem, vshape, shapes, t_in, t_out):
    ndim = elem.cell().topological_dimension()
    if isinstance(elem, WithMapping):
        sobolev = ""
    else:
        sobolev = elem.sobolev_space()

    pshape = shapes.copy()
    pshape = pshape[:ndim]
    pshape.append(-1)
    ndof = numpy.prod(vshape)
    permutation = numpy.arange(ndof)
    permutation = numpy.transpose(numpy.reshape(permutation, vshape))
    if sobolev in [firedrake.HDiv, firedrake.HCurl]:
        # compose with the inverse H(div)/H(curl) permutation
        permutation = numpy.reshape(permutation, pshape)
        for k in range(permutation.shape[-1]):
            permutation[..., k] = numpy.transpose(permutation[..., k], axes=(numpy.arange(ndim)-k-1) % ndim)

    permutation = numpy.reshape(permutation, (-1,))
    perm = ", ".join(map(str, permutation))
    decl = f"""const PetscInt perm[{ndof}] = {{ {perm} }};"""

    nflip = 0
    if sobolev == firedrake.HDiv:
        # flip the sign of the first component
        nflip = ndof // elem.value_shape()[0]

    prolong = f"""
            for({IntType_c} i=0; i<{ndof}; i++) {t_out}[perm[i]] = {t_in}[i];
            for({IntType_c} i=0; i<{nflip}; i++) {t_out}[i] = -{t_out}[i];
    """
    restrict = f"""
            for({IntType_c} i=0; i<{nflip}; i++) {t_out}[i] = -{t_out}[i];
            for({IntType_c} i=0; i<{ndof}; i++) {t_in}[i] = {t_out}[perm[i]];
    """
    return decl, prolong, restrict


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

        tf, _, _, _, _ = tensor_product_space_query(Vf)
        tc, _, _, _, _ = tensor_product_space_query(Vc)
        mf = Vf.ufl_element().mapping().lower()
        mc = Vc.ufl_element().mapping().lower()

        if (tf and tc) and ((mf == mc) or (mc == "identity") or (mf == "identity")):
            uf_map = get_permuted_map(Vf)
            uc_map = get_permuted_map(Vc)
            prolong_kernel, restrict_kernel, coefficients = self.make_blas_kernels(Vf, Vc)
        else:
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
    @lru_cache(maxsize=20)
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
        Vf_mapping = Vf.ufl_element().mapping().lower()
        Vc_mapping = Vc.ufl_element().mapping().lower()

        # FInAT elements order the component DoFs related to the same node contiguously.
        # We transpose before and after the multiplcation times J to have each component
        # stored contiguously as a scalar field, thus reducing the number of dgemm calls.

        # We could benefit from loop tiling for the transpose, but that makes the code
        # more complicated.

        if (Vc_bsize == 1):
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
        if (Vf_bsize == 1):
            fine_read = f"""for({IntType_c} i=0; i<{Vf_sdim*Vf_bsize}; i++) t1[i] = y[i] * w[i];"""
            fine_write = f"""for({IntType_c} i=0; i<{Vf_sdim*Vf_bsize}; i++) y[i] = t1[i];"""
        else:
            fine_read = f"""
            for({IntType_c} j=0; j<{Vf_sdim}; j++)
                for({IntType_c} i=0; i<{Vf_bsize}; i++)
                    t1[j + {Vf_sdim}*i] = y[i + {Vf_bsize}*j] * w[i + {Vf_bsize}*j];
            """
            fine_write = f"""
            for({IntType_c} j=0; j<{Vf_sdim}; j++)
                for({IntType_c} i=0; i<{Vf_bsize}; i++)
                   y[i + {Vf_bsize}*j] = t1[j + {Vf_sdim}*i];
            """

        coefficients = []
        mapping_code = ""
        coef_decl = ""
        if Vf_mapping == Vc_mapping:
            operator_decl, prolong_code, restrict_code, shapes = make_kron_code(Vf, Vc, "t0", "t1", "J0")
            lwork = numpy.prod([max(m, n) for m, n in zip(*shapes)])
        else:
            felem = Vf.ufl_element()
            celem = Vc.ufl_element()

            decl = [""]*3
            prolong = [""]*5
            restrict = [""]*5
            if Vc_mapping == "identity":
                Qe = PMGBase.reconstruct_degree(celem, PMGBase.max_degree(felem))
                Q = firedrake.FunctionSpace(Vf.ufl_domain(), Qe)
                vshape = (Q.value_size, Q.finat_element.space_dimension())
                decl[0], prolong[0], restrict[0], shapes = make_kron_code(Q, Vc, "t0", "t1", "J0")
                prolong[1] = f"""
            for({IntType_c} j=0; j<{vshape[1]}; j++)
                for({IntType_c} i=0; i<{vshape[0]}; i++)
                    t0[i + {vshape[0]}*j] = t1[j + {vshape[1]}*i];
                """
                restrict[1] = f"""
            for({IntType_c} j=0; j<{vshape[1]}; j++)
                for({IntType_c} i=0; i<{vshape[0]}; i++)
                    t1[j + {vshape[1]}*i] = t0[i + {vshape[0]}*j];
                """
                declc, prolong[2], restrict[2], mapping_code, coefficients = make_mapping_code(Q, felem, celem, "t0", "t1")
                decl[1], prolong[3], restrict[3] = make_permutation_code(felem, vshape, shapes[1], "t1", "t0")
                decl[2], prolong[4], restrict[4], shapes1 = make_kron_code(Vf, Q, "t0", "t1", "J1")
                lwork = numpy.prod([max(*dims) for dims in zip(*shapes, *shapes1)])
            elif Vf_mapping == "identity":
                vshape = (Vf_bsize, Vf_sdim)
                decl[0], prolong[0], restrict[0], shapes = make_kron_code(Vf, Vc, "t0", "t1", "J0")
                decl[1], restrict[1], prolong[1] = make_permutation_code(celem, vshape, shapes[1], "t0", "t1")
                declc, prolong[2], restrict[2], mapping_code, coefficients = make_mapping_code(Vf, felem, celem, "t0", "t1")
                fine_write = f"""for({IntType_c} i=0; i<{Vf_bsize*Vf_sdim}; i++) y[i] = t1[i];"""
                fine_read = f"""for({IntType_c} i=0; i<{Vf_bsize*Vf_sdim}; i++) t1[i] = y[i] * w[i];"""
                lwork = numpy.prod([max(*dims) for dims in zip(*shapes)])
            else:
                raise NotImplementedError("Need at least one space with identity mapping")

            coef_decl = declc
            operator_decl = "".join(decl)
            prolong_code = "".join(prolong)
            restrict_code = "".join(reversed(restrict))

        kernel_code = f"""
        {mapping_code}

        {kronmxv_code}

        void prolongation(PetscScalar *restrict y, const PetscScalar *restrict x{coef_decl}){{
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

        if self.uc.dat.vec != xc:
            with self.uc.dat.vec_wo as xc_:
                xc.copy(xc_)

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
    if isinstance(fele, MixedElement) and not isinstance(fele, (VectorElement, TensorElement)):
        ctx = MixedInterpolationMatrix(Vf, Vc, Vf_bcs, Vc_bcs)
    else:
        ctx = StandaloneInterpolationMatrix(Vf, Vc, Vf_bcs, Vc_bcs)

    sizes = (Vf.dof_dset.layout_vec.getSizes(), Vc.dof_dset.layout_vec.getSizes())
    M_shll = PETSc.Mat().createPython(sizes, ctx, comm=Vf.comm)
    M_shll.setUp()

    return M_shll
