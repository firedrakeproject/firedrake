import abc

from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks


__all__ = ("HiptmairPC",)


class TwoLevelPC(PCBase):

    needs_python_pmat = False

    @abc.abstractmethod
    def coarsen(self, pc):
        raise NotImplementedError

    def initialize(self, pc):
        from firedrake import parameters
        from firedrake.assemble import allocate_matrix, TwoFormAssembler

        A, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()

        coarse_operator, coarse_space_bcs, interp_petscmat = self.coarsen(pc)

        # Handle the coarse operator
        coarse_options_prefix = options_prefix + "mg_coarse_"
        coarse_mat_type = opts.getString(coarse_options_prefix + "mat_type",
                                         parameters["default_matrix_type"])

        self.coarse_op = allocate_matrix(coarse_operator,
                                         bcs=coarse_space_bcs,
                                         form_compiler_parameters=fcp,
                                         mat_type=coarse_mat_type,
                                         options_prefix=coarse_options_prefix)
        self._assemble_coarse_op = TwoFormAssembler(coarse_operator, tensor=self.coarse_op,
                                                    form_compiler_parameters=fcp,
                                                    bcs=coarse_space_bcs).assemble
        self._assemble_coarse_op()
        coarse_opmat = self.coarse_op.petscmat

        # We set up a PCMG object that uses the constructed interpolation
        # matrix to generate the restriction/prolongation operators.
        # This is a two-level multigrid preconditioner.
        pcmg = PETSc.PC().create(comm=pc.comm)
        pcmg.incrementTabLevel(1, parent=pc)

        pcmg.setType(pc.Type.MG)
        pcmg.setOptionsPrefix(options_prefix)
        pcmg.setMGLevels(2)
        pcmg.setMGType(pc.MGType.ADDITIVE)
        pcmg.setMGCycleType(pc.MGCycleType.V)
        pcmg.setMGInterpolation(1, interp_petscmat)
        # FIXME the default for MGRScale is created with the wrong shape when dim(coarse) > dim(fine)
        # FIXME there is no need for injection in a KSP context, probably this comes from the snes_ctx below
        # as workaround define injection as the restriction of the solution times a zero vector
        pcmg.setMGRScale(1, interp_petscmat.createVecRight())
        pcmg.setOperators(A=A, P=P)

        coarse_solver = pcmg.getMGCoarseSolve()
        coarse_solver.setOperators(A=coarse_opmat, P=coarse_opmat)
        # coarse space dm
        coarse_space = coarse_operator.arguments()[-1].function_space()
        coarse_dm = coarse_space.dm
        coarse_solver.setDM(coarse_dm)
        coarse_solver.setDMActive(False)
        pcmg.setDM(pc.getDM())
        pcmg.setFromOptions()
        self.pc = pcmg
        self._dm = coarse_dm

        prefix = coarse_solver.getOptionsPrefix()
        # Create new appctx
        self._ctx_ref = self.new_snes_ctx(pc,
                                          coarse_operator,
                                          coarse_space_bcs,
                                          coarse_mat_type,
                                          fcp,
                                          options_prefix=prefix)

        with dmhooks.add_hooks(coarse_dm, self,
                               appctx=self._ctx_ref,
                               save=False):
            coarse_solver.setFromOptions()

    def update(self, pc):
        self._assemble_coarse_op()
        self.pc.setUp()

    def apply(self, pc, X, Y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(X, Y)

    def view(self, pc, viewer=None):
        super(TwoLevelPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("Two level PC\n")
            self.pc.view(viewer)


class HiptmairPC(TwoLevelPC):

    _prefix = "hiptmair_"

    def coarsen(self, pc):
        import numpy
        from firedrake_citations import Citations
        from firedrake import FunctionSpace, TestFunction, TrialFunction
        from firedrake.interpolation import Interpolator
        from ufl.algorithms.ad import expand_derivatives
        from ufl import (FiniteElement, TensorElement, FacetElement,
                         replace, zero, grad, curl, as_vector)

        Citations().register("Hiptmair1998")
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        ctx = dmhooks.get_appctx(pc.getDM())
        problem = ctx._problem
        a = problem.Jp or problem.J
        bcs = problem.bcs

        mesh = V.mesh()
        element = V.ufl_element()
        formdegree = V.finat_element.formdegree
        if formdegree == 1:
            dminus = grad
            cfamily = "Lagrange"
            G_callback = appctx.get("get_gradient", None)
        elif formdegree == 2:
            dminus = curl
            if V.shape:
                dminus = lambda u: as_vector([curl(u[k, ...]) for k in range(u.ufl_shape[0])])
            cfamily = "N1curl" if mesh.ufl_cell().is_simplex() else "NCE"
            G_callback = appctx.get("get_curl", None)
        else:
            raise ValueError("Hiptmair decomposition not available for", element)

        variant = element.variant()
        degree = element.degree()
        try:
            degree = max(degree)
        except TypeError:
            pass

        celement = FiniteElement(cfamily, cell=mesh.ufl_cell(), degree=degree, variant=variant)
        if degree > 1:
            if not V.finat_element.entity_dofs()[V.finat_element.cell.get_dimension()][0]:
                celement = FacetElement(celement)
                # TODO provide statically-condensed form with SLATE
        if V.shape:
            celement = TensorElement(celement, shape=V.shape)

        coarse_space = FunctionSpace(mesh, celement)
        coarse_space_bcs = [bc.reconstruct(V=coarse_space, g=0) for bc in bcs]

        # Get only the zero-th order term of the form
        beta = replace(expand_derivatives(a), {grad(t): zero(grad(t).ufl_shape) for t in a.arguments()})

        test = TestFunction(coarse_space)
        trial = TrialFunction(coarse_space)
        coarse_operator = beta(dminus(test), dminus(trial), coefficients={})
        if formdegree > 1 and degree > 1:
            shift = appctx.get("hiptmair_shift", None)
            if shift is not None:
                coarse_operator += beta(test, shift*trial, coefficients={})

        if G_callback is None:
            from firedrake.preconditioners.hypre_ams import chop
            interp_petscmat = chop(Interpolator(dminus(test), V).callable().handle)
        else:
            interp_petscmat = G_callback(V, coarse_space, bcs, coarse_space_bcs)

        return coarse_operator, coarse_space_bcs, interp_petscmat
