from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.ufl_expr import TestFunction
from firedrake.dmhooks import get_function_space
from firedrake.utils import complex_mode
from firedrake.interpolation import interpolate
from ufl import grad, SpatialCoordinate
from firedrake_citations import Citations
from finat.ufl import VectorElement
from pyop2.utils import as_tuple

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
        from firedrake.assemble import assemble

        if complex_mode:
            raise NotImplementedError("HypreAMS preconditioner not yet implemented in complex mode")

        Citations().register("Kolev2009")
        A, P = obj.getOperators()
        appctx = self.get_appctx(obj)
        prefix = obj.getOptionsPrefix() or ""
        V = get_function_space(obj.getDM())
        mesh = V.mesh()

        family = str(V.ufl_element().family())
        formdegree = V.finat_element.formdegree
        degree = max(as_tuple(V.ufl_element().degree()))
        if formdegree != 1 or degree != 1:
            raise ValueError("Hypre AMS requires lowest order Nedelec elements! (not %s of degree %d)" % (family, degree))

        P1 = V.reconstruct(family="Lagrange", degree=1)
        G_callback = appctx.get("get_gradient", None)
        if G_callback is None:
            G = chop(assemble(interpolate(grad(TestFunction(P1)), V)).petscmat)
        else:
            G = G_callback(P1, V)

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
        super().view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    def update(self, pc):
        self.pc.setUp()
