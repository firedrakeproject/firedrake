from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.ufl_expr import TrialFunction
from firedrake.dmhooks import get_function_space
from firedrake.preconditioners.hypre_ams import chop
from firedrake.interpolation import interpolate
from finat.ufl import VectorElement
from ufl import grad, curl, SpatialCoordinate
from pyop2.utils import as_tuple

__all__ = ("HypreADS",)


class HypreADS(PCBase):
    def initialize(self, obj):
        from firedrake.assemble import assemble
        A, P = obj.getOperators()
        appctx = self.get_appctx(obj)
        prefix = obj.getOptionsPrefix() or ""
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        family = str(V.ufl_element().family())
        formdegree = V.finat_element.formdegree
        degree = max(as_tuple(V.ufl_element().degree()))
        if formdegree != 2 or degree != 1:
            raise ValueError("Hypre ADS requires lowest order RT elements! (not %s of degree %d)" % (family, degree))

        P1 = V.reconstruct(family="Lagrange", degree=1)
        NC1 = V.reconstruct(family="N1curl" if mesh.ufl_cell().is_simplex() else "NCE", degree=1)
        G_callback = appctx.get("get_gradient", None)
        if G_callback is None:
            G = chop(assemble(interpolate(grad(TrialFunction(P1)), NC1)).petscmat)
        else:
            G = G_callback(P1, NC1)
        C_callback = appctx.get("get_curl", None)
        if C_callback is None:
            C = chop(assemble(interpolate(curl(TrialFunction(NC1)), V)).petscmat)
        else:
            C = C_callback(NC1, V)

        pc = PETSc.PC().create(comm=obj.comm)
        pc.incrementTabLevel(1, parent=obj)
        pc.setOptionsPrefix(prefix + "hypre_ads_")
        pc.setOperators(A, P)

        pc.setType('hypre')
        pc.setHYPREType('ads')
        pc.setHYPREDiscreteGradient(G)
        pc.setHYPREDiscreteCurl(C)

        VectorP1 = P1.reconstruct(element=VectorElement(P1.ufl_element()))
        coords = Function(VectorP1).interpolate(SpatialCoordinate(mesh))
        pc.setCoordinates(coords.dat.data_ro.copy())

        pc.setFromOptions()
        self.pc = pc

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super(HypreADS, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    def update(self, pc):
        self.pc.setUp()
