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
        appctx = self.get_appctx(obj)
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        formdegree = V.finat_element.formdegree
        degree = V.ufl_element().degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        if formdegree != 1 or degree != 1:
            family = str(V.ufl_element().family())
            raise ValueError("Hypre AMS requires lowest order Nedelec elements! (not %s of degree %d)" % (family, degree))

        P1 = FunctionSpace(mesh, "Lagrange", 1)
        G_callback = appctx.get("get_gradient", None)
        if G_callback is None:
            self.G = chop(Interpolator(grad(TestFunction(P1)), V).callable().handle)
        else:
            self.G = G_callback(P1, V)

        VectorP1 = VectorFunctionSpace(mesh, "Lagrange", 1)
        self.coordinates = interpolate(SpatialCoordinate(mesh), VectorP1)

        self.pc = PETSc.PC()
        self.build_hypre(obj, self.pc)

    def build_hypre(self, obj, pc):
        A, P = obj.getOperators()
        prefix = obj.getOptionsPrefix()

        pc.create(comm=obj.comm)
        pc.incrementTabLevel(1, parent=obj)
        pc.setOptionsPrefix(prefix + "hypre_ams_")
        pc.setOperators(A=A, P=P)

        pc.setType('hypre')
        pc.setHYPREType('ams')
        pc.setHYPREDiscreteGradient(self.G)
        pc.setCoordinates(self.coordinates.dat.data_ro)

        zero_beta = PETSc.Options(prefix).getBool("pc_hypre_ams_zero_beta_poisson", default=False)
        if zero_beta:
            pc.setHYPRESetBetaPoissonMatrix(None)
        pc.setCoordinates(self.coordinates.dat.data_ro)
        pc.setFromOptions()
        self.pc = pc

    def apply(self, obj, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, obj, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, obj, viewer=None):
        super().view(obj, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    def update(self, obj):
        self.pc.destroy()
        self.build_hypre(obj, self.pc)

    def destroy(self, obj):
        if hasattr(self, "G"):
            self.G.destroy()
        if hasattr(self, "pc"):
            self.pc.destroy()
