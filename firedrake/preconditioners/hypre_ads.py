from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace
from firedrake.ufl_expr import TestFunction
from firedrake.interpolation import Interpolator, interpolate
from firedrake.dmhooks import get_function_space
from firedrake.preconditioners.hypre_ams import chop
from ufl import grad, curl, SpatialCoordinate

__all__ = ("HypreADS",)


class HypreADS(PCBase):
    def initialize(self, obj):
        appctx = self.get_appctx(obj)
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        formdegree = V.finat_element.formdegree
        degree = V.ufl_element().degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        if formdegree != 2 or degree != 1:
            family = str(V.ufl_element().family())
            raise ValueError("Hypre ADS requires lowest order RT elements! (not %s of degree %d)" % (family, degree))

        P1 = FunctionSpace(mesh, "Lagrange", 1)
        NC1 = FunctionSpace(mesh, "N1curl" if mesh.ufl_cell().is_simplex() else "NCE", 1)
        G_callback = appctx.get("get_gradient", None)
        if G_callback is None:
            self.G = chop(Interpolator(grad(TestFunction(P1)), NC1).callable().handle)
        else:
            self.G = G_callback(P1, NC1)
        C_callback = appctx.get("get_curl", None)
        if C_callback is None:
            self.C = chop(Interpolator(curl(TestFunction(NC1)), V).callable().handle)
        else:
            self.C = C_callback(NC1, V)

        VectorP1 = VectorFunctionSpace(mesh, "Lagrange", 1)
        self.coordinates = interpolate(SpatialCoordinate(mesh), VectorP1)

        self.pc = PETSc.PC()
        self.build_hypre(obj, self.pc)

    def build_hypre(self, obj, pc):
        A, P = obj.getOperators()
        prefix = obj.getOptionsPrefix()

        pc.create(comm=obj.comm)
        pc.incrementTabLevel(1, parent=obj)
        pc.setOptionsPrefix(prefix + "hypre_ads_")
        pc.setOperators(A=A, P=P)

        pc.setType('hypre')
        pc.setHYPREType('ads')
        pc.setHYPREDiscreteGradient(self.G)
        pc.setHYPREDiscreteCurl(self.C)
        pc.setCoordinates(self.coordinates.dat.data_ro)
        pc.setFromOptions()
        self.pc = pc

    def apply(self, obj, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, obj, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, obj, viewer=None):
        super(HypreADS, self).view(obj, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    def update(self, obj):
        self.pc.destroy()
        self.build_hypre(obj, self.pc)

    def destroy(self, obj):
        if hasattr(self, "G"):
            self.G.destroy()
        if hasattr(self, "C"):
            self.C.destroy()
        if hasattr(self, "pc"):
            self.pc.destroy()
