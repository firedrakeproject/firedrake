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

__all__ = ("HypreAMS",)


def chop(A, tol=1E-10):
    # remove (near) zeros from sparsity pattern
    A.chop(tol)
    B = PETSc.Mat().create(comm=A.comm)
    B.setType(A.getType())
    B.setSizes(A.getSizes())
    B.setBlockSize(A.getBlockSize())
    B.setUp()
    B.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    B.setPreallocationCSR(A.getValuesCSR())
    B.assemble()
    A.destroy()
    return B


class HypreAMS(PCBase):
    def initialize(self, obj):
        if complex_mode:
            raise NotImplementedError("HypreAMS preconditioner not yet implemented in complex mode")

        Citations().register("Kolev2009")
        A, P = obj.getOperators()
        appctx = self.get_appctx(obj)
        prefix = obj.getOptionsPrefix()
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        family = str(V.ufl_element().family())
        formdegree = V.finat_element.formdegree
        degree = V.ufl_element().degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        if formdegree != 1 or degree != 1:
            raise ValueError("Hypre AMS requires lowest order Nedelec elements! (not %s of degree %d)" % (family, degree))

        P1 = FunctionSpace(mesh, "Lagrange", 1)
        G_callback = appctx.get("get_gradient", None)
        if G_callback is None:
            G = chop(Interpolator(grad(TestFunction(P1)), V).callable().handle)
        else:
            G = G_callback(V, P1)

        pc = PETSc.PC().create(comm=obj.comm)
        pc.incrementTabLevel(1, parent=obj)
        pc.setOptionsPrefix(prefix + "hypre_ams_")
        pc.setOperators(A, P)

        pc.setType('hypre')
        pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(G)

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
