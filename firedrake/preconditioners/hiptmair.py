from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.functionspace import FunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.interpolation import Interpolator
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake_citations import Citations
from ufl import grad, curl, HCurl, HDiv, FiniteElement, TensorElement
import firedrake
import numpy as np

__all__ = ("HiptmairPC",)


class HiptmairPC(PCBase):
    def initialize(self, obj):

        Citations().register("Hiptmair1998")
        A, P = obj.getOperators()
        appctx = self.get_appctx(obj)
        # prefix = obj.getOptionsPrefix()
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        sobolev_space = V.ufl_element().sobolev_space()
        degree = V.ufl_element().degree()
        try:
            degree = max(degree)
        except TypeError:
            pass

        if sobolev_space == HCurl:
            dminus = grad
            cfamily = "Lagrange"
            G_callback = appctx.get("get_gradient", None)
        elif sobolev_space == HDiv:
            dminus = curl
            cfamily = "N1curl" if mesh.ufl_cell().is_simplex() else "NCE"
            G_callback = appctx.get("get_curl", None)
        else:
            raise ValueError("Hiptmair decomposition not available in", sobolev_space)

        celement = FiniteElement(cfamily, cell=mesh.ufl_cell(), degree=degree)
        if V.shape:
            celement = TensorElement(element, shape=V.shape)

        Vc = FunctionSpace(mesh, celement)

        ctx = get_appctx(obj.getDM())
        a = ctx.J
        bcs = ctx._problem.bcs
        ac = a(dminus(TestFunction(Vc)), dminus(TrialFunction(Vc)), coefficients={})
        cbcs = [bc.reconstruct(V=Vc, g=0) for bc in bcs]

        if G_callback is None:
            G = Interpolator(dminus(TestFunction(Vc)), V).callable().handle

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
            G2 = G_callback(V, Vc, bcs, cbcs)
        self.interp = G2
        self.xc = G2.createVecRight()
        self.yc = G2.createVecRight()

        # TODO create PC objects to replace Jacobi and give them options
        def get_jacobi_smoother(a, bcs, tensor=None):
            tensor = firedrake.assemble(a, bcs=bcs, diagonal=True, tensor=tensor)
            with tensor.dat.vec as diag:
                diag.reciprocal()
            return tensor

        def apply_jacobi(diag, x, y):
            with diag.dat.vec as d:
                y.pointwiseMult(x, d)

        self.fdiag = None
        self.cdiag = None
        self._update_fdiag = lambda: get_jacobi_smoother(a, bcs, tensor=self.fdiag)
        self._update_cdiag = lambda: get_jacobi_smoother(ac, cbcs, tensor=self.cdiag)
        self.fdiag = self._update_fdiag()
        self.cdiag = self._update_cdiag()

        self.fine_apply = lambda x, y: apply_jacobi(self.fdiag, x, y)
        self.coarse_apply = lambda x, y: apply_jacobi(self.cdiag, x, y)

    def update(self, pc):
        self._update_fdiag()
        self._update_cdiag()

    def apply(self, pc, x, y):
        self.fine_apply(x, y)

        self.interp.multTranspose(x, self.xc)
        self.coarse_apply(self.xc, self.yc)
        self.interp.multAdd(self.yc, y, y)

    def applyTranspose(self, pc, x, y):
        self.apply(pc, x, y)

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
