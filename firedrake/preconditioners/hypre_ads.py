from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.fdm import tabulate_exterior_derivative
from firedrake.petsc import PETSc
from firedrake.function import Function
from firedrake.ufl_expr import TrialFunction
from firedrake.dmhooks import get_function_space
from firedrake.preconditioners.hypre_ams import chop
from firedrake.interpolation import interpolate
from finat.ufl import FiniteElement, TensorElement, VectorElement
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

        # Get the auxiliary Nedelec and Lagrange spaces and the coordinate space
        cell = V.ufl_element().cell
        NC1_element = FiniteElement("N1curl" if cell.is_simplex() else "NCE", cell=cell, degree=1)
        P1_element = FiniteElement("Lagrange", cell=cell, degree=1)
        coords_element = VectorElement(P1_element, dim=mesh.geometric_dimension())
        if V.shape:
            NC1_element = TensorElement(NC1_element, shape=V.shape)
            P1_element = TensorElement(P1_element, shape=V.shape)

        NC1 = V.reconstruct(element=NC1_element)
        P1 = V.reconstruct(element=P1_element)
        VectorP1 = V.reconstruct(element=coords_element)

        G_callback = appctx.get("get_gradient", None)
        if G_callback is None:
            try:
                G = chop(assemble(interpolate(grad(TrialFunction(P1)), NC1)).petscmat)
            except NotImplementedError:
                G = tabulate_exterior_derivative(P1, NC1)
        else:
            G = G_callback(P1, NC1)
        C_callback = appctx.get("get_curl", None)
        if C_callback is None:
            try:
                C = chop(assemble(interpolate(curl(TrialFunction(NC1)), V)).petscmat)
            except NotImplementedError:
                C = tabulate_exterior_derivative(NC1, V)
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
