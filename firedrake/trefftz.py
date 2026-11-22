"""
Provides a class to compute the Trefftz embedding of a given function space.
It is also used to compute aggregation embedding of a given function space.
"""
from typing import Optional
from firedrake.petsc import PETSc
from firedrake.cython.dmcommon import FACE_SETS_LABEL, CELL_SETS_LABEL
from firedrake.assemble import assemble
from firedrake.mesh import Mesh
from firedrake.functionspace import FunctionSpace
from firedrake.function import Function
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.constant import Constant
from ufl import dx, dS, inner, jump, grad, dot, CellDiameter, FacetNormal, Form
import scipy.sparse as sp
import numpy as np

__all__ = ["TrefftzEmbedding", "TrefftzKSP", "AggregationEmbedding", "dumb_aggregation"]

class TrefftzEmbedding:
    """
    Computes the Trefftz embedding of a given function space
    Parameters
    ----------
    V : Ambient function space.
    b : Bilinear form defining the Trefftz operator.
    dim : Dimension of the embedding.
        Default is the dimension of the function space.
    tol : Tolerance for the singular values cutoff.
        Default is 1e-12.
    backend : Backend to use for the computation of the SVD.
        Default is "scipy".
    """
    def __init__(self, V: FunctionSpace, b: Form, dim: Optional[int] = None,
                 tol: Optional[float]=1e-12):
        self.V = V
        self.b = b
        self.dim = V.dim() if not dim else dim + 1
        self.tol = tol
        self.svdsolver = "scipy"

    def assemble(self) -> tuple[PETSc.Mat, np.array]:
        """
        Assemble the embedding, compute the SVD and return the embedding matrix
        """
        self.B = assemble(self.b).M.handle
        if self.svdsolver == "scipy":
            indptr, indices, data = self.B.getValuesCSR()
            Bsp = sp.csr_matrix((data, indices, indptr), shape=self.B.getSize())
            _, sig, VT = sp.linalg.svds(Bsp, k=self.dim-1, which="SM")
            QT = sp.csr_matrix(VT[0:sum(sig < self.tol), :])
            QTpsc = PETSc.Mat().createAIJ(size=QT.shape, csr=(QT.indptr, QT.indices, QT.data))
            self.dimT = QT.shape[0]
            self.sig = sig
        else:
            raise NotImplementedError("Only scipy backend is supported")
        return QTpsc, sig


class TrefftzKSP:
    """
    Wraps a PETSc KSP object to solve the reduced
    system obtained by the Trefftz embedding.
    
    There will bne no type hinting following petsc4py's style.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_appctx(ksp: PETSc.KSP):
        """
        Get the application context from the KSP
        Parameters
        ----------
        ksp : The KSP object
        """
        from firedrake.dmhooks import get_appctx
        return get_appctx(ksp.getDM()).appctx

    def setUp(self, ksp: PETSc.KSP):
        """
        Set up the Trefftz KSP
        Parameters
        ----------
        ksp : The KSP object
        """
        appctx = self.get_appctx(ksp)
        self.QT, _ = appctx["trefftz_embedding"].assemble()

    def solve(self, ksp: PETSc.KSP, b: PETSc.Vec, x:PETSc.Vec):
        """
        Solve the Trefftz KSP
        Parameters
        ----------
        ksp : The KSP object
        b : The right-hand side
        x : The solution
        """
        A, P = ksp.getOperators()
        self.Q = PETSc.Mat().createTranspose(self.QT)
        ATF = self.QT @ A @ self.Q
        PTF = self.QT @ P @ self.Q
        bTF = self.QT.createVecLeft()
        self.QT.mult(b, bTF)

        tiny_ksp = PETSc.KSP().create()
        tiny_ksp.setOperators(ATF, PTF)
        tiny_ksp.setOptionsPrefix("trefftz_")
        tiny_ksp.setFromOptions()
        xTF = ATF.createVecRight()
        tiny_ksp.solve(bTF, xTF)
        self.QT.multTranspose(xTF, x)
        ksp.setConvergedReason(tiny_ksp.getConvergedReason())


class AggregationEmbedding(TrefftzEmbedding):
    """
    This class computes the aggregation embedding of a given function space.
    Parameters
    ----------
    V : Ambient function space.
    mesh : The mesh on which the aggregation is defined.
    polyMesh : The function defining the aggregation.
    dim : Dimension of the embedding.
        Default is the dimension of the function space.
    tol : Tolerance for the singular values cutoff.
        Default is 1e-12.
    """
    def __init__(self, V: FunctionSpace, mesh: Mesh, polyMesh: Function,
                 dim: Optional[int] = None, tol: Optional[float]=1e-12):
        # Relabel facets that are inside an aggregated region
        offset = 1 + mesh.topology_dm.getLabelSize(FACE_SETS_LABEL)
        offset += mesh.topology_dm.getLabelSize(CELL_SETS_LABEL)
        nPoly = int(max(polyMesh.dat.data[:]))  # Number of aggregates
        getIdx = mesh._cell_numbering.getOffset
        plex = mesh.topology_dm
        pStart, pEnd = plex.getDepthStratum(2)
        self.facet_index = []
        for poly in range(nPoly+1):
            facets = []
            for i in range(pStart, pEnd):
                if polyMesh.dat.data[getIdx(i)] == poly:
                    for f in plex.getCone(i):
                        if f in facets:
                            plex.setLabelValue(FACE_SETS_LABEL, f, offset+poly)
                            if offset+poly not in self.facet_index:
                                self.facet_index = self.facet_index + [offset+poly]
                    facets = facets + list(plex.getCone(i))
        self.mesh = Mesh(plex)
        h = CellDiameter(self.mesh)
        n = FacetNormal(self.mesh)
        W = FunctionSpace(self.mesh, V.ufl_element())
        u = TrialFunction(W)
        v = TestFunction(W)
        self.b = Constant(0)*inner(u, v)*dx
        for i in self.facet_index:
            self.b += inner(jump(u), jump(v))*dS(i)
        for k in range(1, V.ufl_element().degree()+1):
            for i in self.facet_index:
                self.b += ((0.5 * h("+") + 0.5 * h("-"))**(2*k)) *\
                    inner(jump_normal(u, n("+"), k), jump_normal(v, n("+"), k))*dS(i)
        super().__init__(W, self.b, dim, tol)


def jump_normal(u: Function, n: FacetNormal, k: int):
    """
    Compute the jump of the normal derivative of a function u
    Parameters
    ----------
    u : The function.
    n : The normal vector.
    k : The order of the normal derivative we aim to compute.
    """
    j = 0.5*dot(n, (grad(u)("+")-grad(u)("-")))
    for _ in range(1, k):
        j = 0.5*dot(n, (grad(j)-grad(j)))
    return j


def dumb_aggregation(mesh: Mesh) -> Function:
    """
    Compute a dumb aggregation of the mesh
    Parameters
    ----------
    mesh : The mesh we aim to aggregate.
    """
    if mesh.comm.size > 1:
        raise NotImplementedError("Parallel mesh aggregation not supported")
    plex = mesh.topology_dm
    pStart, pEnd = plex.getDepthStratum(2)
    _, eEnd = plex.getDepthStratum(1)
    adjacency = []
    for i in range(pStart, pEnd):
        ad = plex.getAdjacency(i)
        local = []
        for a in ad:
            supp = plex.getSupport(a)
            supp = supp[supp < eEnd]
            for s in supp:
                if s < pEnd and s != ad[0]:
                    local = local + [s]
        adjacency = adjacency + [(i, local)]
    adjacency = sorted(adjacency, key=lambda x: len(x[1]))[::-1]
    u = Function(FunctionSpace(mesh, "DG", 0))

    getIdx = mesh._cell_numbering.getOffset
    av = list(range(pStart, pEnd))
    col = 0
    for a in adjacency:
        if a[0] in av:
            for k in a[1]:
                if k in av:
                    av.remove(k)
                    u.dat.data[getIdx(k)] = col
            av.remove(a[0])
            u.dat.data[getIdx(a[0])] = col
            col = col + 1
    return u
