from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks
from firedrake_citations import Citations
import numpy as np


__all__ = ("HiptmairPC",)


class HiptmairPC(PCBase):

    needs_python_pmat = False
    _prefix = "hiptmair_"

    def initialize(self, pc):

        from firedrake import TestFunction, TrialFunction, FunctionSpace, parameters
        from firedrake.assemble import allocate_matrix, TwoFormAssembler
        from firedrake.interpolation import Interpolator
        from ufl.algorithms.ad import expand_derivatives
        from ufl import (grad, curl, zero, inner, dx, replace,
                         FiniteElement, TensorElement, FacetElement)

        Citations().register("Hiptmair1998")
        A, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()

        V = dmhooks.get_function_space(pc.getDM())
        mesh = V.mesh()

        element = V.ufl_element()
        variant = element.variant()
        degree = element.degree()
        try:
            degree = max(degree)
        except TypeError:
            pass

        formdegree = V.finat_element.formdegree
        if formdegree == 1:
            dminus = grad
            cfamily = "Lagrange"
            G_callback = appctx.get("get_gradient", None)
        elif formdegree == 2:
            dminus = curl
            cfamily = "N1curl" if mesh.ufl_cell().is_simplex() else "NCE"
            G_callback = appctx.get("get_curl", None)
        else:
            raise ValueError("Hiptmair decomposition not available for", element)

        celement = FiniteElement(cfamily, cell=mesh.ufl_cell(), degree=degree, variant=variant)
        if (not list(list(V.finat_element.entity_dofs().values())[-1].values())[0]) and (degree > 1):
            celement = FacetElement(celement)
        if V.shape:
            celement = TensorElement(element, shape=V.shape)

        Vc = FunctionSpace(mesh, celement)
        ctx = dmhooks.get_appctx(pc.getDM())
        bcs = ctx._problem.bcs
        J = ctx._problem.J

        coarse_space = Vc
        coarse_space_bcs = [bc.reconstruct(V=Vc, g=0) for bc in bcs]
        test = TestFunction(Vc)
        trial = TrialFunction(Vc)

        # Get only the zero-th order term of the form
        beta = replace(expand_derivatives(J), {grad(t): zero(grad(t).ufl_shape) for t in J.arguments()})
        coarse_operator = beta(dminus(test), dminus(trial), coefficients={})
        if formdegree > 1 and degree > 1:
            shift = appctx.get("hiptmair_shift", None)
            if shift is not None:
                coarse_operator += beta(test, shift*trial, coefficients={})

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

        if G_callback is None:
            G = Interpolator(dminus(test), V).callable().handle

            # remove (near) zeros from sparsity pattern
            ai, aj, a = G.getValuesCSR()
            a[np.abs(a) < 1e-10] = 0
            G2 = PETSc.Mat().create()
            G2.setType(PETSc.Mat.Type.AIJ)
            G2.setSizes(G.sizes)
            G2.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
            G2.setPreallocationCSR((ai, aj, a))
            G2.assemble()
        else:
            G2 = G_callback(V, Vc, bcs, coarse_space_bcs)
        interp_petscmat = G2

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
        pcmg.setOperators(A=A, P=P)

        coarse_solver = pcmg.getMGCoarseSolve()
        coarse_solver.setOperators(A=coarse_opmat, P=coarse_opmat)
        # coarse space dm
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
        super(HiptmairPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC using Hiptmair decomposition\n")
            self.pc.view(viewer)
