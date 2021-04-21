from functools import lru_cache
import numpy as np

from pyop2 import op2
from pyop2.sparsity import get_preallocation

from ufl import FiniteElement, TensorElement, Jacobian, JacobianInverse, dx, inner, dot, inv
from firedrake.petsc import PETSc
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.base import PCBase
from firedrake.utils import IntType_c
from firedrake.dmhooks import get_function_space, get_appctx
import firedrake


class FDMPC(PCBase):

    _prefix = "fdm_"

    def initialize(self, pc):
        A, P = pc.getOperators()

        # Read options
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options(options_prefix)
        fdm_type = opts.getString("type", default="affine")

        dm = pc.getDM()
        V = get_function_space(dm)

        ele = V.ufl_element()
        if isinstance(ele, firedrake.TensorProductElement):
            family = set(e.family() for e in ele.sub_elements())
        else:
            family = {ele.family()}
        needs_interior_facet = family <= {"DQ", "Discontinuous Lagrange"}

        self.mesh = V.mesh()
        self.uf = firedrake.Function(V)
        self.uc = firedrake.Function(V)

        ndim = self.mesh.topological_dimension()
        nscal = V.value_size
        N = V.ufl_element().degree()
        try:
            N, = set(N)
        except TypeError:
            pass
        Nq = 2 * N + 1

        # Get problem solution and bcs
        solverctx = get_appctx(dm)
        self.u = solverctx._problem.u
        self.bcs = solverctx.bcs_F

        if len(self.bcs) > 0:
            self.bc_nodes = np.unique(np.concatenate([bcdofs(bc, ghost=False)
                                                      for bc in self.bcs]))
        else:
            self.bc_nodes = np.empty(0, dtype=PETSc.IntType)

        bcflags, cell2cell = self.get_bc_flags(self.mesh, self.bcs, solverctx._problem.J)

        # Encode bcflags into a VectorFunction that can be used in the pyOp2 kernels
        # FIXME very ugly interface
        VDG = firedrake.VectorFunctionSpace(self.mesh, 'DG', 0, dim=ndim)
        self.fbc = firedrake.Function(VDG, name="bcflags")
        extra_dim = (-1,) if ndim > 1 else ()
        self.fbc.dat.data[cell2cell] = np.reshape(bcflags @ np.kron(np.eye(ndim), [[1], [3]]), cell2cell.shape + extra_dim)

        self.weight = self.multiplicity(V)
        with self.weight.dat.vec as w:
            w.reciprocal()

        # Get problem coefficients
        appctx = self.get_appctx(pc)
        mu = appctx.get("mu", None)  # sets the viscosity
        helm = appctx.get("helm", None)  # sets the potential
        hflag = helm is not None

        self.restrict_kernel, self.prolong_kernel, self.stencil_kernel, Afdm, Dfdm = self.assemble_matfree(ndim, nscal, N, Nq, needs_interior_facet, hflag)

        self.stencil = None
        if fdm_type == "stencil":
            # Compute high-order PDE coefficients and only extract
            # nonzeros from the diagonal and interface neighbors
            # Vertex-vertex couplings are ignored here,
            # so this should work as direct solver only on star patches
            W = firedrake.VectorFunctionSpace(self.mesh, "DQ", N, dim=2*ndim+1)
            self.stencil = firedrake.Function(W)
            Gq, Bq = self.assemble_coef(mu, helm, Nq)
            Pmat = self.assemble_stencil(P, V, Gq, Bq, N, bcflags)
        elif fdm_type == "affine":
            # Compute low-order PDE coefficients, such that the FDM
            # sparsifies the assembled matrix
            Gq, Bq = self.assemble_coef(mu, helm, Nq)
            Pmat = self.assemble_affine(P, V, Gq, Bq, Afdm, Dfdm, bcflags)
        else:
            raise ValueError("Unknown fdm_type")
        Pmat.zeroRowsColumnsLocal(self.bc_nodes)

        # Monkey see, monkey do
        opc = pc

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -fdm_pc_type ksp.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        dm = opc.getDM()
        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(Pmat, Pmat)
        self.pc = pc
        pc.setFromOptions()

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        pass

    def apply(self, pc, x, y):
        self.uc.assign(firedrake.zero())

        with self.uf.dat.vec_wo as xf:
            x.copy(xf)

        op2.par_loop(self.restrict_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.INC, self.uc.cell_node_map()),
                     self.uf.dat(op2.READ, self.uf.cell_node_map()),
                     self.fbc.dat(op2.READ, self.fbc.cell_node_map()),
                     self.weight.dat(op2.READ, self.weight.cell_node_map()))

        with self.uc.dat.vec as x_, self.uf.dat.vec as y_:
            self.pc.apply(x_, y_)

        op2.par_loop(self.prolong_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.WRITE, self.uc.cell_node_map()),
                     self.uf.dat(op2.READ, self.uf.cell_node_map()),
                     self.fbc.dat(op2.READ, self.fbc.cell_node_map()))

        with self.uc.dat.vec_ro as xc:
            xc.copy(y)

        y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]

    def view(self, pc, viewer=None):
        super(FDMPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    @staticmethod
    def pull_facet(x, pshape, fdata):
        return np.reshape(np.moveaxis(np.reshape(x.copy(), pshape), fdata // 2, 0), x.shape)

    @staticmethod
    def index_bcs(x, pshape, bc, val):
        xshape = x.shape
        x.shape = pshape
        if bc[0]:
            x[0, ...] = val
        if bc[1]:
            x[-1, ...] = val
        if len(pshape) >= 2:
            if bc[2]:
                x[:, 0, ...] = val
            if bc[3]:
                x[:, -1, ...] = val
        if len(pshape) >= 3:
            if bc[4]:
                x[:, :, 0, ...] = val
            if bc[5]:
                x[:, :, -1, ...] = val
        x.shape = xshape
        return

    def assemble_stencil(self, A, V, Gq, Bq, N, bcflags):
        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map([])
        # TODO implement stencil for IPDG
        strong = 1

        lexico_cg, nel = self.glonum_fun(V)
        lexico_dg, _ = self.glonum_fun(self.stencil)

        ndim = V.mesh().topological_dimension()
        ndof_cell = V.cell_node_list.shape[1]
        nx1 = N + 1
        pshape = (nx1,)*ndim

        self.stencil.assign(firedrake.zero())
        # FIXME I don't know how to use optional arguments here, maybe a MixedFunctionSpace
        if Bq is not None:
            op2.par_loop(self.stencil_kernel, self.mesh.cell_set,
                         self.stencil.dat(op2.WRITE, self.stencil.cell_node_map()),
                         Gq.dat(op2.READ, Gq.cell_node_map()),
                         Bq.dat(op2.READ, Bq.cell_node_map()),
                         self.fbc.dat(op2.READ, self.fbc.cell_node_map()))
        else:
            op2.par_loop(self.stencil_kernel, self.mesh.cell_set,
                         self.stencil.dat(op2.WRITE, self.stencil.cell_node_map()),
                         Gq.dat(op2.READ, Gq.cell_node_map()),
                         self.fbc.dat(op2.READ, self.fbc.cell_node_map()))

        # Connectivity graph between the nodes within a cell
        i = np.arange(ndof_cell, dtype=PETSc.IntType)
        sx = i - (i % nx1)
        sy = i - ((i // nx1) % nx1) * nx1
        if ndim == 2:
            graph = np.array([sy, sy+(nx1-1)*nx1, sx, sx+(nx1-1)])
        else:
            sz = i - (((i // nx1) // nx1) % nx1) * nx1 * nx1
            graph = np.array([sz, sz+(nx1-1)*nx1*nx1, sy, sy+(nx1-1)*nx1, sx, sx+(nx1-1)])

        ondiag = (graph == i).T
        graph = graph.T

        prealloc = PETSc.Mat().create(comm=A.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(A.getSizes())
        prealloc.setUp()

        aones = np.ones(graph.shape[1], dtype=PETSc.RealType)
        for e in range(nel):
            ie = lgmap.apply(lexico_cg(e))

            # Preallocate diagonal
            for row in ie:
                prealloc.setValue(row, row, 1.0E0)

            # Preallocate off-diagonal
            self.index_bcs(ie, pshape, bcflags[e] == strong, -1)
            je = ie[graph]
            je[ondiag] = -1
            for row, cols in zip(ie, je):
                prealloc.setValues(row, cols, aones)
                prealloc.setValues(cols, row, aones)

        prealloc.assemble()
        nnz = get_preallocation(prealloc, V.dof_dset.set.size)
        Pmat = PETSc.Mat().createAIJ(A.getSizes(), nnz=nnz, comm=A.comm)
        Pmat.setLGMap(lgmap, lgmap)
        Pmat.zeroEntries()
        for e in range(nel):
            ie = lgmap.apply(lexico_cg(e))
            vals = self.stencil.dat.data_ro[lexico_dg(e)]

            # Assemble diagonal
            for row, aij in zip(ie, vals):
                Pmat.setValue(row, row, aij[0], imode)

            # Assemble off-diagonal
            self.index_bcs(ie, pshape, bcflags[e] == strong, -1)
            je = ie[graph]
            je[ondiag] = -1
            for row, cols, aij in zip(ie, je, vals):
                Pmat.setValues(row, cols, aij[1:], imode)
                Pmat.setValues(cols, row, aij[1:], imode)

        Pmat.assemble()
        return Pmat

    def assemble_affine(self, A, V, Gq, Bq, Afdm, Dfdm, bcflags):
        from scipy.sparse import kron, csr_matrix

        needs_interior_facet = Dfdm is not None
        strong = -1 if needs_interior_facet else 1

        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map([])

        lexico_cell, nel = self.glonum_fun(V)
        gid, _ = self.glonum_fun(Gq)
        bid, _ = self.glonum_fun(Bq) if Bq is not None else (None, nel)

        ndim = V.mesh().topological_dimension()
        idsym = [0, 3, 5] if ndim == 3 else [0, 2]
        idsym = idsym[:ndim]

        nx1 = Afdm[0][0].shape[0]
        pshape = (nx1,)*ndim

        prealloc = PETSc.Mat().create(comm=A.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(A.getSizes())
        prealloc.setUp()

        # Build elemental sparse matrices and preallocate matrix
        cell_csr = []
        facet_csr = []
        flag2id = np.kron(np.eye(ndim, ndim, dtype=PETSc.IntType), [[1], [3]])
        for e in range(nel):
            fbc = bcflags[e] @ flag2id
            mue = np.atleast_1d(np.sum(Gq.dat.data_ro[gid(e)], axis=0))
            mue = mue[idsym]

            be = Afdm[fbc[0]][1]
            ae = Afdm[fbc[0]][0] * mue[0]
            if Bq is not None:
                ae += be * sum(Bq.dat.data_ro[bid(e)])

            if ndim > 1:
                ae = kron(ae, Afdm[fbc[1]][1], format="csr")
                ae += kron(be, Afdm[fbc[1]][0] * mue[1], format="csr")
                if ndim > 2:
                    be = kron(be, Afdm[fbc[1]][1], format="csr")
                    ae = kron(ae, Afdm[fbc[2]][1], format="csr")
                    ae += kron(be, Afdm[fbc[2]][0] * mue[2], format="csr")

            cell_csr.append(ae)
            rows = lgmap.apply(lexico_cell(e))
            self.index_bcs(rows, pshape, bcflags[e] == strong, -1)
            cols = rows[ae.indices]
            for i, row in enumerate(rows):
                i0 = ae.indptr[i]
                i1 = ae.indptr[i+1]
                prealloc.setValues(row, cols[i0:i1], ae.data[i0:i1])

        if needs_interior_facet:
            eta = nx1*(nx1+1)
            adense = np.zeros((nx1, nx1))

            # TODO extrude these arrays
            lexico_facet, nfacet = self.glonum_fun(V, interior_facet=True)
            facet_cells = self.mesh.interior_facets.facet_cell_map.values
            facet_data = self.mesh.interior_facets.local_facet_dat.data

            for f in range(nfacet):
                e0, e1 = facet_cells[f]
                idir = facet_data[f] // 2
                fbc0 = bcflags[e0] @ flag2id
                fbc1 = bcflags[e1] @ flag2id
                bc0 = fbc0[idir[0]]
                bc1 = fbc1[idir[1]]
                fbc = np.delete(fbc0, idir[0])

                mu0 = np.atleast_1d(np.sum(Gq.dat.data_ro[gid(e0)], axis=0))
                mu1 = np.atleast_1d(np.sum(Gq.dat.data_ro[gid(e1)], axis=0))
                mu0 = mu0[idsym[idir[0]]]
                mu1 = mu1[idsym[idir[1]]]

                i0 = -(facet_data[f, 0] % 2)
                i1 = -(facet_data[f, 1] % 2)
                adense.fill(0.0E0)
                adense[:, i1] += (0.5E0*mu0) * Dfdm[bc0][:, i0]
                adense[i0, :] += (0.5E0*mu1) * Dfdm[bc1][:, i1]
                adense[i0, i1] -= eta * (mu0 + mu1)
                ae = csr_matrix(adense)
                if ndim > 1:
                    ae = kron(ae, Afdm[fbc[0]][1], format="csr")
                    if ndim > 2:
                        ae = kron(ae, Afdm[fbc[1]][1], format="csr")

                facet_csr.append(ae)
                icell = np.reshape(lgmap.apply(lexico_facet(f)), (2, -1))
                rows = self.pull_facet(icell[0], pshape, facet_data[f, 0])
                cols = self.pull_facet(icell[1], pshape, facet_data[f, 1])
                cols = cols[ae.indices]
                for i, row in enumerate(rows):
                    i0 = ae.indptr[i]
                    i1 = ae.indptr[i+1]
                    prealloc.setValues(row, cols[i0:i1], ae.data[i0:i1])
                    prealloc.setValues(cols[i0:i1], row, ae.data[i0:i1])

        for row in range(V.dof_dset.set.size):
            prealloc.setValue(row, row, 1.0E0)

        prealloc.assemble()
        nnz = get_preallocation(prealloc, V.dof_dset.set.size)
        Pmat = PETSc.Mat().createAIJ(A.getSizes(), nnz=nnz, comm=A.comm)
        Pmat.setLGMap(lgmap, lgmap)
        Pmat.zeroEntries()

        # Assemble global matrix
        for e, ae in enumerate(cell_csr):
            ie = lgmap.apply(lexico_cell(e))
            rows = ie.copy()
            self.index_bcs(rows, pshape, bcflags[e] == strong, -1)
            adiag = ae.diagonal()
            adiag = adiag[rows == -1]
            ibc = ie[rows == -1]
            for row, val in zip(ibc, adiag):
                Pmat.setValue(row, row, val, imode)

            cols = rows[ae.indices]
            for i, row in enumerate(rows):
                i0 = ae.indptr[i]
                i1 = ae.indptr[i+1]
                Pmat.setValues(row, cols[i0:i1], ae.data[i0:i1], imode)

        if needs_interior_facet:
            for f, ae in enumerate(facet_csr):
                icell = np.reshape(lgmap.apply(lexico_facet(f)), (2, -1))
                rows = self.pull_facet(icell[0], pshape, facet_data[f, 0])
                cols = self.pull_facet(icell[1], pshape, facet_data[f, 1])
                cols = cols[ae.indices]
                for i, row in enumerate(rows):
                    i0 = ae.indptr[i]
                    i1 = ae.indptr[i+1]
                    Pmat.setValues(row, cols[i0:i1], ae.data[i0:i1], imode)
                    Pmat.setValues(cols[i0:i1], row, ae.data[i0:i1], imode)

        Pmat.assemble()
        return Pmat

    def assemble_coef(self, mu, helm, Nq=0):
        ndim = self.mesh.topological_dimension()
        gdim = self.mesh.geometric_dimension()
        gshape = (ndim, ndim)

        if gdim == ndim:
            Finv = JacobianInverse(self.mesh)
            if mu is None:
                G = dot(Finv, Finv.T)
            elif mu.ufl_shape == gshape:
                G = dot(dot(Finv, mu), Finv.T)
            else:  # treat mu as scalar
                G = mu * dot(Finv, Finv.T)
        else:
            F = Jacobian(self.mesh)
            G = inv(dot(F.T, F))
            if mu:
                G = mu * G
            # I don't know how to use tensor viscosity on embedded manifolds

        Qe = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                           quad_scheme="default", shape=gshape, symmetry=True)
        Q = firedrake.FunctionSpace(self.mesh, Qe)
        q = firedrake.TestFunction(Q)
        Gq = firedrake.assemble(inner(G, q)*dx(degree=Nq))

        if helm is None:
            Bq = None
        else:
            Qe = FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                               quad_scheme="default")
            Q = firedrake.FunctionSpace(self.mesh, Qe)
            q = firedrake.TestFunction(Q)
            Bq = firedrake.assemble(inner(helm, q)*dx(degree=Nq))

        return Gq, Bq

    @staticmethod
    @lru_cache(maxsize=10)
    def semhat(N, Nq):
        from FIAT.reference_element import UFCInterval
        from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
        from FIAT.quadrature import GaussLegendreQuadratureLineRule
        cell = UFCInterval()
        elem = GaussLobattoLegendre(cell, N)
        rule = GaussLegendreQuadratureLineRule(cell, (Nq + 2) // 2)
        basis = elem.tabulate(1, rule.get_points())
        Jhat = basis[(0,)]
        Dhat = basis[(1,)]
        what = rule.get_weights()
        Ahat = Dhat @ np.diag(what) @ Dhat.T
        Bhat = Jhat @ np.diag(what) @ Jhat.T
        return Ahat, Bhat, Jhat, Dhat, what

    @staticmethod
    def fdm_cg(Ahat, Bhat):
        from scipy.linalg import eigh
        from scipy.sparse import csr_matrix
        nx = Ahat.shape[0]
        rd = ((), (0,), (-1,), (0, -1))
        Lfdm = np.zeros((4, nx))
        Vfdm = np.zeros((4, nx, nx))
        for k in range(4):
            Abar = np.stack((Ahat, Bhat))
            if 0 in set(rd[k]):
                Abar[:, 0, 1:] = 0.0E0
                Abar[:, 1:, 0] = 0.0E0
            if -1 in set(rd[k]):
                Abar[:, -1, :-1] = 0.0E0
                Abar[:, :-1, -1] = 0.0E0

            Lfdm[k], Vfdm[k] = eigh(Abar[0], Abar[1])
            iord = np.argsort(abs(Vfdm[k][-1]) - abs(Vfdm[k][0]))
            Vfdm[k] = Vfdm[k][:, iord]
            Lfdm[k] = Lfdm[k][iord]

        def basis_bcs(V, Bhat, bc0, bc1):
            k0 = 1 if bc0 else 0
            k1 = -1 if bc1 else V.shape[1]
            rd = []
            if bc0 == 2:
                rd.append(0)
            if bc1 == 2:
                rd.append(V.shape[1]-1)
            if rd:
                Vbc = V.copy()
                Vbc[k0:k1, rd] = -V[k0:k1, k0:k1] @ ((V[k0:k1, k0:k1].T @ Bhat[k0:k1, rd]) @ V[np.ix_(rd, rd)])
                return Vbc
            else:
                return V

        def galerkin_bcs(Ahat, Vbc, bc0, bc1, ismass):
            A = Vbc.T @ Ahat @ Vbc
            k0 = 1 if bc0 == 2 else 0
            k1 = -1 if bc1 == 2 else A.shape[1]
            a = A.diagonal().copy()
            A[k0:k1, k0:k1] = 0.0E0
            np.fill_diagonal(A, a)
            if ismass:
                rd = []
                if bc0 == 2:
                    rd.append(0)
                if bc1 == 2:
                    rd.append(A.shape[1]-1)
                A[rd, k0:k1] = 0.0E0
                A[k0:k1, rd] = 0.0E0
            return csr_matrix(A)

        Vbc = []
        Afdm = []
        Abar = (Ahat, Bhat)
        for j in range(3):
            for i in range(3):
                k = (i > 0) + (j > 0) * 2
                Vbc.append(basis_bcs(Vfdm[k], Bhat, i, j))
                Afdm.append([galerkin_bcs(Ak, Vbc[-1], i, j, ismass) for ismass, Ak in enumerate(Abar)])

        return Afdm, Vbc

    @staticmethod
    def fdm_ipdg(Ahat, Bhat, N, eta=None):
        from scipy.linalg import eigh
        from scipy.sparse import csr_matrix
        from FIAT.reference_element import UFCInterval
        from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
        from FIAT.quadrature import GaussLegendreQuadratureLineRule

        if eta is None:
            eta = (N+1)*(N+2)

        cell = UFCInterval()
        elem = GaussLobattoLegendre(cell, N)

        # Interpolation onto GL nodes
        rule = GaussLegendreQuadratureLineRule(cell, N + 1)
        basis = elem.tabulate(0, rule.get_points())
        Jipdg = basis[(0,)]

        # Facet outward normal derivatives
        basis = elem.tabulate(1, cell.get_vertices())
        Dfacet = basis[(1,)]
        Dfacet[:, 0] = -Dfacet[:, 0]

        def fdm_weak(Ahat, Bhat, Dfacet, strong_bcs, weak_bcs, eta):
            rd = tuple()
            if strong_bcs[0]:
                rd += (0,)
            if strong_bcs[1]:
                rd += (Ahat.shape[0]-1,)
            k0 = strong_bcs[0]
            k1 = Ahat.shape[0] - strong_bcs[1]

            A = Ahat.copy()
            for j in (0, -1):
                if weak_bcs[j]:
                    mult = 1 + strong_bcs[j]
                    A[:, j] -= Dfacet[:, j] * (1.0E0/mult)
                    A[j, :] -= Dfacet[:, j] * (1.0E0/mult)
                    A[j, j] += eta * mult

            Vbc = np.eye(A.shape[0])
            _, Vbc[k0:k1, k0:k1] = eigh(A[k0:k1, k0:k1], Bhat[k0:k1, k0:k1])
            Vbc[k0:k1, rd] = -Vbc[k0:k1, k0:k1] @ ((Vbc[k0:k1, k0:k1].T @ Bhat[k0:k1, rd]) @ Vbc[np.ix_(rd, rd)])
            A = Vbc.T @ A @ Vbc
            B = Vbc.T @ Bhat @ Vbc
            a = A.diagonal().copy()
            A[k0:k1, k0:k1] = 0.0E0
            np.fill_diagonal(A, a)
            b = B.diagonal().copy()
            B[k0:k1, k0:k1] = 0.0E0
            np.fill_diagonal(B, b)
            B[rd, k0:k1] = 0.0E0
            B[k0:k1, rd] = 0.0E0
            return Vbc, csr_matrix(A), csr_matrix(B)

        Vbc = []
        Dbc = []
        Afdm = []
        for bc1 in range(3):
            for bc0 in range(3):
                weak_bcs = np.array([bc0 >= 1, bc1 >= 1])
                strong_bcs = np.array([bc0 == 2, bc1 == 2])
                Vk, Ak, Bk = fdm_weak(Ahat, Bhat, Dfacet, strong_bcs, weak_bcs, eta)
                Afdm.append([Ak, Bk])
                # Vbc rotates GL residuals into modes obtained from GLL
                Dbc.append(Vk.T @ Dfacet)
                Vbc.append(Jipdg.T @ Vk)

        return Afdm, Vbc, Dbc

    @staticmethod
    @lru_cache(maxsize=10)
    def assemble_matfree(ndim, nscal, N, Nq, needs_interior_facet, helm=False):
        # Assemble sparse 1D matrices and matrix-free kernels for basis transformation and stencil computation
        from firedrake.slate.slac.compiler import BLASLAPACK_LIB, BLASLAPACK_INCLUDE
        from pyop2 import op2

        nsym = (ndim * (ndim+1)) // 2

        Ahat, Bhat, Jhat, Dhat, _ = FDMPC.semhat(N, Nq)
        nx = Ahat.shape[0]
        ny = nx if ndim >= 2 else 1
        nz = nx if ndim >= 3 else 1
        nxyz = nx*ny*nz
        ntot = nscal*nxyz
        lwork = ntot
        nv = nx * nx

        nxq = Jhat.shape[1]
        nyq = nxq if ndim >= 2 else 1
        nzq = nxq if ndim >= 3 else 1
        nquad = nxq * nyq * nzq

        if needs_interior_facet:
            Afdm, Vbc, Dbc = FDMPC.fdm_ipdg(Ahat, Bhat, N)
        else:
            Afdm, Vbc = FDMPC.fdm_cg(Ahat, Bhat)
            Dbc = None

        Vsize = nv * len(Vbc)
        Vhex = ', '.join(map(float.hex, np.asarray(Vbc).flatten()))
        VX = f"V+((int)bcs[{ndim-1}])*{nv}"
        VY = f"V+((int)bcs[{ndim-2}])*{nv}" if ndim > 1 else "&one"
        VZ = f"V+((int)bcs[{ndim-3}])*{nv}" if ndim > 2 else "&one"

        kronmxv_code = """
        #include <petscsys.h>
        #include <petscblaslapack.h>

        static void kronmxv(PetscBLASInt tflag,
            PetscBLASInt mx, PetscBLASInt my, PetscBLASInt mz,
            PetscBLASInt nx, PetscBLASInt ny, PetscBLASInt nz, PetscBLASInt nel,
            PetscScalar  *A1, PetscScalar *A2, PetscScalar *A3,
            PetscScalar  *x , PetscScalar *y){

        PetscBLASInt m,n,k,s,p,lda;
        char TA1, TA2, TA3;
        char tran='T', notr='N';
        PetscScalar zero=0.0E0, one=1.0E0;

        if(tflag>0){
           TA1 = tran;
           TA2 = notr;
        }else{
           TA1 = notr;
           TA2 = tran;
        }
        TA3 = TA2;

        m = mx;  k = nx;  n = ny*nz*nel;
        lda = (tflag>0)? nx : mx;

        BLASgemm_(&TA1, &notr, &m,&n,&k, &one, A1,&lda, x,&k, &zero, y,&m);

        p = 0;  s = 0;
        m = mx;  k = ny;  n = my;
        lda = (tflag>0)? ny : my;
        for(PetscBLASInt i=0; i<nz*nel; i++){
           BLASgemm_(&notr, &TA2, &m,&n,&k, &one, y+p,&m, A2,&lda, &zero, x+s,&m);
           p += m*k;
           s += m*n;
        }

        p = 0;  s = 0;
        m = mx*my;  k = nz;  n = mz;
        lda = (tflag>0)? nz : mz;
        for(PetscBLASInt i=0; i<nel; i++){
           BLASgemm_(&notr, &TA3, &m,&n,&k, &one, x+p,&m, A3,&lda, &zero, y+s,&m);
           p += m*k;
           s += m*n;
        }
        return;
        }
        """

        transfer_code = f"""
        {kronmxv_code}

        void prolongation(PetscScalar *y,
                      PetscScalar *x,
                      PetscScalar *bcs){{
            PetscScalar V[{Vsize}] = {{ {Vhex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                    t0[j + {nxyz}*i] = x[i + {nscal}*j];

            kronmxv(1, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {VX},{VY},{VZ}, t0, t1);

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                   y[i + {nscal}*j] = t1[j + {nxyz}*i];
            return;
        }}

        void restriction(PetscScalar *y,
                      PetscScalar *x,
                      PetscScalar *bcs,
                      PetscScalar *w){{
            PetscScalar V[{Vsize}] = {{ {Vhex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                    t0[j + {nxyz}*i] = x[i + {nscal}*j] * w[i + {nscal}*j];

            kronmxv(0, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {VX},{VY},{VZ}, t0, t1);

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                   y[i + {nscal}*j] += t1[j + {nxyz}*i];
            return;
        }}
        """

        restrict_kernel = op2.Kernel(transfer_code, "restriction", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())
        prolong_kernel = op2.Kernel(transfer_code, "prolongation", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())

        nb = Jhat.size
        VJ = [Vk.T @ Jhat for Vk in Vbc]
        VD = [Vk.T @ Dhat for Vk in Vbc]
        Jhex = ', '.join(map(float.hex, np.asarray(VJ).flatten()))
        Dhex = ', '.join(map(float.hex, np.asarray(VD).flatten()))

        JX = f"J+(({IntType_c})bcs[{ndim-1}])*{nb}"
        JY = f"J+(({IntType_c})bcs[{ndim-2}])*{nb}" if ndim > 1 else "&one"
        JZ = f"J+(({IntType_c})bcs[{ndim-3}])*{nb}" if ndim > 2 else "&one"

        DX = f"D+(({IntType_c})bcs[{ndim-1}])*{nb}"
        DY = f"D+(({IntType_c})bcs[{ndim-2}])*{nb}" if ndim > 1 else "&one"
        DZ = f"D+(({IntType_c})bcs[{ndim-3}])*{nb}" if ndim > 2 else "&one"

        # FIXME I don't know how to use optional arguments here
        bcoef = "bcoef" if helm else "NULL"
        cargs = "PetscScalar *diag,"
        cargs += "PetscScalar *gcoef,"
        if helm:
            cargs += " PetscScalar *bcoef,"
        cargs += "PetscScalar *bcs"

        stencil_code = f"""
        {kronmxv_code}

        void mult3(PetscBLASInt n, PetscScalar *A, PetscScalar *B, PetscScalar *C){{
            for({IntType_c} i=0; i<n; i++)
                C[i] = A[i] * B[i];
            return;
        }}

        void mult_diag(PetscBLASInt m, PetscBLASInt n,
                       PetscScalar *A, PetscScalar *B, PetscScalar *C){{
            for({IntType_c} j=0; j<n; j++)
                for({IntType_c} i=0; i<m; i++)
                    C[i+m*j] = A[i] * B[i+m*j];
            return;
        }}

        void get_basis(PetscBLASInt dom, PetscScalar *J, PetscScalar *D, PetscScalar *B){{
            PetscScalar *basis[2] = {{J, D}};
            if(dom)
                for({IntType_c} j=0; j<2; j++)
                    for({IntType_c} i=0; i<2; i++)
                        mult_diag({nxq}, {nx}, basis[i]+{nxq*(nx-1)}*(dom-1), basis[j], B+(i+2*j)*{nb});
            else
                for({IntType_c} j=0; j<2; j++)
                    for({IntType_c} i=0; i<2; i++)
                        mult3({nb}, basis[i], basis[j], B+(i+2*j)*{nb});
            return;
        }}

        void get_band(PetscBLASInt dom1, PetscBLASInt dom2, PetscBLASInt dom3,
                      PetscScalar *JX, PetscScalar *DX,
                      PetscScalar *JY, PetscScalar *DY,
                      PetscScalar *JZ, PetscScalar *DZ,
                      PetscScalar *gcoef,
                      PetscScalar *bcoef,
                      PetscScalar *band){{

            PetscScalar BX[{4 * nb}];
            PetscScalar BY[{4 * nb}];
            PetscScalar BZ[{4 * nb}];
            PetscScalar t0[{nquad}], t1[{nquad}], t2[{nxyz}] = {{0.0E0}};
            PetscScalar scal;
            {IntType_c} k, ix, iy, iz;
            {IntType_c} ndiag = {nxyz}, nquad = {nquad}, nstencil = {2*ndim+1}, inc = 1;

            get_basis(dom1, JX, DX, BX);
            get_basis(dom2, JY, DY, BY);
            if({ndim}==3)
                get_basis(dom3, JZ, DZ, BZ);
            else
                BZ[0] = 1.0E0;

            if(bcoef){{
                BLAScopy_(&nquad, bcoef, &inc, t0, &inc);
                kronmxv(1, {nx}, {ny}, {nz}, {nxq}, {nyq}, {nzq}, 1, BX, BY, BZ, t0, t2);
            }}

            for({IntType_c} j=0; j<{ndim}; j++)
                for({IntType_c} i=0; i<{ndim}; i++){{
                    k = i + j + (i>0 && j>0 && {ndim}==3);
                    ix = (i == {ndim-1}) + 2 * (j == {ndim-1});
                    iy = (i == {ndim-2}) + 2 * (j == {ndim-2});
                    iz = (i == {ndim-3}) + 2 * (j == {ndim-3});
                    scal = (i == j) ? 1.0E0 : 0.5E0;
                    BLAScopy_(&nquad, gcoef+k*nquad, &inc, t0, &inc);
                    kronmxv(1, {nx}, {ny}, {nz}, {nxq}, {nyq}, {nzq}, 1, BX+ix*{nb}, BY+iy*{nb}, BZ+iz*{nb}, t0, t1);
                    BLASaxpy_(&ndiag, &scal, t1, &inc, t2, &inc);
                }}

            BLAScopy_(&ndiag, t2, &inc, band, &nstencil);
            return;
        }}

        void stencil({cargs}){{
            PetscScalar J[{9 * nb}] = {{ {Jhex} }};
            PetscScalar D[{9 * nb}] = {{ {Dhex} }};
            PetscScalar tcoef[{nquad * nsym}];
            PetscScalar one = 1.0E0;
            {IntType_c} i1, i2, i3, bcj;

            for({IntType_c} j=0; j<{nquad}; j++)
                for({IntType_c} i=0; i<{nsym}; i++)
                    tcoef[j + {nquad}*i] = gcoef[i + {nsym}*j];

            get_band(0, 0, 0, {JX}, {DX}, {JY}, {DY}, {JZ}, {DZ}, tcoef, {bcoef}, diag);

            for({IntType_c} j=0; j<{2*ndim}; j++){{
                i1 = (j/2 == {ndim-1}) * (1 + (j%2));
                i2 = (j/2 == {ndim-2}) * (1 + (j%2));
                i3 = (j/2 == {ndim-3}) * (1 + (j%2));
                bcj = ({IntType_c}) bcs[j/2];
                bcj = (j%2 == 0) ? bcj % 3 : bcj / 3;
                //if(bcj > 1)
                get_band(i1, i2, i3, {JX}, {DX}, {JY}, {DY}, {JZ}, {DZ}, tcoef, {bcoef}, diag + (j+1));
            }}
            return;
        }}
        """

        stencil_kernel = op2.Kernel(stencil_code, "stencil", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())
        return restrict_kernel, prolong_kernel, stencil_kernel, Afdm, Dbc

    @staticmethod
    def multiplicity(V):
        # Lawrence's magic code for calculating dof multiplicities
        shapes = (V.finat_element.space_dimension(),
                  np.prod(V.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        weight = firedrake.Function(V)
        firedrake.par_loop((domain, instructions), firedrake.dx,
                           {"w": (weight, op2.INC)}, is_loopy_kernel=True)
        return weight

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum_fun(V, interior_facet=False):
        node_map = V.interior_facet_node_map() if interior_facet else V.cell_node_map()
        nelh = node_map.values.shape[0]
        if node_map.offset is None:
            return lambda e: node_map.values[e], nelh
        else:
            layers = node_map.iterset.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1] - layers[0, 0] - 1
                nel = nelz * nelh
                return lambda e: node_map.values[e % nelh] + (e//nelh)*node_map.offset, nel
            else:
                k = 0
                nelz = layers[:nelh, 1] - layers[:nelh, 0] - 1
                nel = sum(nelz)
                layer_id = np.zeros((nel, 2))
                for e in range(0, nelh):
                    for l in range(0, nelz[e]):
                        layer_id[k, :] = [e, l]
                        k += 1
                return lambda e: node_map.values[layer_id[e, 0]] + layer_id[e, 1]*node_map.offset, nel

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum(V):
        node_map = V.cell_node_map()
        if node_map.offset is None:
            return node_map.values
        else:
            nelh = node_map.values.shape[0]
            layers = node_map.iterset.layers_array
            if(layers.shape[0] == 1):
                nelz = layers[0, 1]-layers[0, 0]-1
                nel = nelz * nelh
                gl = np.zeros((nelz,)+node_map.values.shape, dtype=PETSc.IntType)
                for k in range(0, nelz):
                    gl[k] = node_map.values + k*node_map.offset
                gl = np.reshape(gl, (nel, -1))
            else:
                k = 0
                nelz = layers[:nelh, 1]-layers[:nelh, 0]-1
                nel = sum(nelz)
                gl = np.zeros((nel, node_map.values.shape[1]), dtype=PETSc.IntType)
                for e in range(0, nelh):
                    for l in range(0, nelz[e]):
                        gl[k] = node_map.values[e] + l*node_map.offset
                        k += 1
            return gl

    @staticmethod
    @lru_cache(maxsize=10)
    def get_bc_flags(mesh, bcs, J):
        extruded = mesh.cell_set._extruded
        ndim = mesh.topological_dimension()
        nface = 2*ndim

        # Partition of unity at interior facets (fraction of volumes)
        DG0 = firedrake.FunctionSpace(mesh, 'DG', 0)
        if ndim == 1:
            DGT = firedrake.FunctionSpace(mesh, 'Lagrange', 1)
        else:
            DGT = firedrake.FunctionSpace(mesh, 'DGT', 0)
        cell2cell = FDMPC.glonum(DG0)
        face2cell = FDMPC.glonum(DGT)

        area = firedrake.FacetArea(mesh)
        vol = firedrake.CellVolume(mesh)
        tau = firedrake.interpolate(vol, DG0)
        v = firedrake.TestFunction(DGT)

        dFacet = firedrake.dS_h + firedrake.dS_v if extruded else firedrake.dS
        w = firedrake.assemble(((v('-') * tau('-') + v('+') * tau('+')) / area) * dFacet)

        rho = w.dat.data_ro_with_halos[face2cell] / tau.dat.data_ro[cell2cell]

        if extruded:
            ibot = 4
            itop = 5
            ivert = [0, 1, 2, 3]
            nelh = mesh.cell_set.sizes[1]
            layers = mesh.cell_set.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1] - layers[0, 0] - 1
                nel = nelh * nelz
                facetdata = np.zeros([nel, nface, 2], dtype=PETSc.IntType)
                facetdata[:, ivert, :] = np.tile(mesh.cell_to_facets.data, (nelz, 1, 1))
            else:
                nelz = layers[:nelh, 1] - layers[:nelh, 0] - 1
                nel = sum(nelz)
                facetdata = np.zeros([nel, nface, 2], dtype=PETSc.IntType)
                facetdata[:, ivert, :] = np.repeat(mesh.cell_to_facets.data, nelz, axis=0)
                for f in ivert:
                    bnd = np.isclose(rho[:, f], 0.0E0)
                    bnd &= (facetdata[:, f, 0] != 0)
                    facetdata[bnd, f, :] = [0, -8]

            bot = np.isclose(rho[:, ibot], 0.0E0)
            top = np.isclose(rho[:, itop], 0.0E0)
            facetdata[:, [ibot, itop], :] = -1
            facetdata[bot, ibot, :] = [0, -2]
            facetdata[top, itop, :] = [0, -4]
        else:
            facetdata = mesh.cell_to_facets.data

        flags = facetdata[:, :, 0]
        sub = facetdata[:, :, 1]

        # Boundary condition flags
        # 0 => Natural, do nothing
        # 1 => Strong Dirichlet
        # 2 => Interior facet
        labels = ()
        maskall = False
        for bc in bcs:
            if isinstance(bc, firedrake.DirichletBC):
                bs = bc.sub_domain
                if bs == "on_boundary":
                    maskall = True
                elif bs == "bottom":
                    labels += (-2,)
                elif bs == "top":
                    labels += (-4,)
                else:
                    labels += bs if type(bs) == tuple else (bs,)

        for it in J.integrals():
            itype = it.integral_type()
            if itype.startswith("exterior_facet"):
                bs = it.subdomain_id()
                if bs == "everywhere":
                    if itype == "exterior_facet_bottom":
                        labels += (-2,)
                    elif itype == "exterior_facet_top":
                        labels += (-4,)
                    else:
                        maskall = True
                else:
                    labels += bs if type(bs) == tuple else (bs,)

        labels = list(set(labels))

        fbc = np.isin(sub, labels).astype(PETSc.IntType)
        if maskall:
            fbc[sub >= -1] = 1

        fbc[flags != 0] = 2
        return fbc, cell2cell
