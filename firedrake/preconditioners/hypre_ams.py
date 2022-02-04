from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.ufl_expr import TestFunction
from firedrake.interpolation import Interpolator, interpolate
from firedrake.dmhooks import get_function_space
from firedrake.utils import complex_mode
from firedrake_citations import Citations
from firedrake import SpatialCoordinate
from ufl import grad
import numpy as np

__all__ = ("HypreAMS",)


class HypreAMS(PCBase):
    def initialize(self, obj):
        if complex_mode:
            raise NotImplementedError("HypreAMS preconditioner not yet implemented in complex mode")

        Citations().register("Kolev2009")
        A, P = obj.getOperators()
        prefix = obj.getOptionsPrefix()
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        family = str(V.ufl_element().family())
        degree = V.ufl_element().degree()
        if family != 'Nedelec 1st kind H(curl)' or degree != 1:
            raise ValueError("Hypre AMS requires lowest order Nedelec elements! (not %s of degree %d)" % (family, degree))

        P1 = FunctionSpace(mesh, "Lagrange", 1)
        G = Interpolator(grad(TestFunction(P1)), V).callable().handle

        # remove (near) zeros from sparsity pattern
        ai, aj, a = G.getValuesCSR()
        a[np.abs(a) < 1e-10] = 0
        G2 = PETSc.Mat().create()
        G2.setType(PETSc.Mat.Type.AIJ)
        G2.setSizes(G.sizes)
        G2.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        G2.setPreallocationCSR((ai, aj, a))
        G2.assemble()

        pc = PETSc.PC().create(comm=obj.comm)
        pc.incrementTabLevel(1, parent=obj)
        pc.setOptionsPrefix(prefix + "hypre_ams_")
        pc.setOperators(A, P)

        pc.setType('hypre')
        pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(G2)

        zero_beta = PETSc.Options(prefix).getBool("pc_hypre_ams_zero_beta_poisson", default=False)
        if zero_beta:
            pc.setHYPRESetBetaPoissonMatrix(None)

        VectorP1 = VectorFunctionSpace(mesh, "Lagrange", 1)
        pc.setCoordinates(interpolate(SpatialCoordinate(mesh), VectorP1).dat.data_ro.copy())
        pc.setUp()

        self.pc = pc

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    def update(self, pc):
        self.pc.setUp()
