"""
This module provides a class to compute the Trefftz embedding of a given function space.
It is also used to compute aggregation embedding of a given function space.
"""
from firedrake.petsc import PETSc
from firedrake.cython.dmcommon import FACE_SETS_LABEL, CELL_SETS_LABEL
from firedrake.assemble import assemble
from firedrake.mesh import Mesh
from firedrake.functionspace import FunctionSpace
from firedrake.function import Function
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.constant import Constant
from ufl import dx, dS, inner, jump, grad, dot, CellDiameter, FacetNormal
import scipy.sparse as sp

class TrefftzEmbedding(object):
    """
    This class computes the Trefftz embedding of a given function space
    :arg V: the function space
    :arg b: the bilinear form defining the embedding
    :arg dim: the dimension of the embedding
    :arg tol: the tolerance for the singular values
    :arg backend: the backend to use for the computation of the SVD
    """
    def __init__(self, V, b, dim=None, tol=1e-12, backend="scipy"):
        self.V = V
        self.b = b
        self.dim = V.dim() if not dim else dim + 1
        self.tol = tol
        self.backend = backend

    def assemble(self):
        """
        Assemble the embedding, compute the SVD and return the embedding matrix
        """
        self.B = assemble(self.b).M.handle
        if self.backend == "scipy":
            indptr, indices, data = self.B.getValuesCSR()
            Bsp = sp.csr_matrix((data, indices, indptr), shape=self.B.getSize())
            _, sig, VT = sp.linalg.svds(Bsp, k=self.dim-1, which="SM")
            QT = sp.csr_matrix(VT[0:sum(sig<self.tol), :])
            QTpsc = PETSc.Mat().createAIJ(size=QT.shape, csr=(QT.indptr, QT.indices, QT.data))
            self.dimT = QT.shape[0]
            self.sig = sig
        else:
            raise NotImplementedError("Only scipy backend is supported")
        return QTpsc, sig

class trefftz_ksp(object):
    """
    This class wraps a PETSc KSP object to solve the reduced
    system obtained by the Trefftz embedding.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_appctx(ksp):
        """
        Get the application context from the KSP
        """
        from firedrake.dmhooks import get_appctx
        return get_appctx(ksp.getDM()).appctx

    def setUp(self, ksp):
        """
        Set up the Trefftz KSP
        """
        appctx = self.get_appctx(ksp)
        self.QT, _ = appctx["trefftz_embedding"].assemble()

    def solve(self, ksp, b, x):
        """
        Solve the Trefftz KSP
        """
        A, P = ksp.getOperators()
        self.Q = PETSc.Mat().createTranspose(self.QT)
        ATF = self.QT @ A @ self.Q
        PTF = self.QT @ P @ self.Q
        bTF =  self.QT.createVecLeft()
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
    :arg V: the function space
    :arg mesh: the mesh
    :arg polyMesh: the aggregation mesh
    :arg dim: the dimension of the embedding
    :arg tol: the tolerance for the singular values
    """
    def __init__(self, V, mesh, polyMesh, dim=None, tol=1e-12):
        # Relabel facets that are inside an aggregated region
        offset = 1+mesh.topology_dm.getLabelSize(FACE_SETS_LABEL)
        offset += mesh.topology_dm.getLabelSize(CELL_SETS_LABEL)
        nPoly = int(max(polyMesh.dat.data[:])) # Number of aggregates
        getIdx = mesh._cell_numbering.getOffset
        plex = mesh.topology_dm
        pStart,pEnd = plex.getDepthStratum(2)
        self.facet_index = []
        for poly in range(nPoly+1):
            facets = []
            for i in range(pStart,pEnd):
                if polyMesh.dat.data[getIdx(i)] == poly:
                    for f in plex.getCone(i):
                        if f in facets:
                            plex.setLabelValue(FACE_SETS_LABEL,f,offset+poly)
                            if offset+poly not in self.facet_index:
                                self.facet_index = self.facet_index + [offset+poly]
                    facets = facets + list(plex.getCone(i))
        self.mesh = Mesh(plex)
        h = CellDiameter(self.mesh)
        n = FacetNormal(self.mesh)
        W = FunctionSpace(self.mesh, V.ufl_element())
        u = TrialFunction(W)
        v = TestFunction(W)
        self.b = Constant(0)*inner(u,v)*dx
        for i in self.facet_index:
            self.b += inner(jump(u),jump(v))*dS(i)
        for k in range(1,V.ufl_element().degree()+1):
            for i in self.facet_index:
                self.b += ((0.5*h("+")+0.5*h("-"))**(2*k))*\
                inner(jump_normal(u,n("+"),k),jump_normal(v, n("+"),k))*dS(i)
        super().__init__(W, self.b, dim, tol)

def jump_normal(u,n,k):
    """
    Compute the jump of the normal derivative of a function u
    :arg u: the function
    :arg n: the normal vector
    :arg k: the degree of the normal derivative
    """
    j = 0.5*dot(n, (grad(u)("+")-grad(u)("-")))
    for _ in range(1,k):
        j = 0.5*dot(n, (grad(j)-grad(j)))
    return j

def dumb_aggregation(mesh):
    """
    Compute a dumb aggregation of the mesh
    :arg mesh: the mesh
    """
    if mesh.comm.size > 1:
        raise NotImplementedError("Parallel mesh aggregation not supported")
    plex = mesh.topology_dm
    pStart,pEnd = plex.getDepthStratum(2)
    _,eEnd = plex.getDepthStratum(1)
    adjacency = []
    for i in range(pStart,pEnd):
        ad = plex.getAdjacency(i)
        local = []
        for a in ad:
            supp = plex.getSupport(a)
            supp = supp[supp<eEnd]
            for s in supp:
                if s < pEnd and s != ad[0]:
                    local = local + [s]
        adjacency = adjacency + [(i, local)]
    adjacency = sorted(adjacency, key=lambda x: len(x[1]))[::-1]
    u = Function(FunctionSpace(mesh,"DG",0))

    getIdx = mesh._cell_numbering.getOffset
    av = list(range(pStart,pEnd))
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
