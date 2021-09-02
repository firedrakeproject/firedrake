from functools import lru_cache, partial

from pyop2 import op2
from pyop2.sparsity import get_preallocation

from ufl import FiniteElement, VectorElement, TensorElement
from ufl import FacetNormal, Jacobian, JacobianDeterminant, JacobianInverse
from ufl import as_tensor, diag_vector, dot, dx, indices, inner, inv
from ufl.algorithms.ad import expand_derivatives

from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.pmg import get_permuted_map, tensor_product_space_query
from firedrake.utils import IntType_c
from firedrake.dmhooks import get_function_space, get_appctx
import firedrake.dmhooks as dmhooks
import firedrake
import numpy

try:
    from scipy.linalg import eigh

    def sym_eig(A, B):
        return eigh(A, B)
except ImportError:
    import numpy.linalg as npla

    def sym_eig(A, B):
        L = npla.cholesky(B)
        Linv = npla.inv(L)
        C = numpy.dot(Linv, numpy.dot(A, Linv.T))
        Z, W = npla.eigh(C)
        V = numpy.dot(Linv.T, W)
        return Z, V


class FDMPC(PCBase):

    _prefix = "fdm_"

    def initialize(self, pc):
        A, P = pc.getOperators()

        # Read options
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        # opts = PETSc.Options(options_prefix)

        dm = pc.getDM()
        V = get_function_space(dm)
        use_tensorproduct, N, ndim, family, _ = tensor_product_space_query(V)

        if not use_tensorproduct or (family <= {"RTCE", "NCE"}):
            raise ValueError("The element %s is not supported by FDMPC." % (V.ufl_element(), ))

        needs_interior_facet = not (family <= {"Q", "Lagrange"})
        Nq = 2*N+1  # quadrature degree

        self.mesh = V.mesh()
        self.uf = firedrake.Function(V)
        self.uc = firedrake.Function(V)
        self.cell_node_map = get_permuted_map(V)

        # Get problem solution and bcs
        solverctx = get_appctx(dm)
        self.u = solverctx._problem.u
        self.bcs = solverctx.bcs_F

        if len(self.bcs) > 0:
            self.bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False)
                                                            for bc in self.bcs]))
        else:
            self.bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

        bcflags = self.get_bc_flags(V, self.mesh, self.bcs, solverctx._problem.J)

        self.weight = self.multiplicity(V)
        with self.weight.dat.vec as w:
            w.reciprocal()

        # Get form coefficients
        appctx = self.get_appctx(pc)
        eta = appctx.get("eta", (N+1)*(N+ndim))  # interior penalty parameter
        mu = appctx.get("viscosity", None)  # second order coefficient
        beta = appctx.get("reaction", None)  # zeroth order coefficient
        eta = float(eta)
        self.appctx = appctx

        # Obtain the FDM basis and transfer kernels (restriction and prolongation)
        # Afdm = 1D interval stiffness and mass matrices in the FDM basis for each direction and BC type
        # Dfdm = normal derivate matrices in the FDM basis
        Afdm, Dfdm, self.restrict_kernel, self.prolong_kernel = self.assemble_matfree(V, N, Nq, eta, needs_interior_facet)

        # Preallocate by calling the assembly routine on a PETSc Mat of type PREALLOCATOR
        prealloc = PETSc.Mat().create(comm=A.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(A.getSizes())
        prealloc.setUp()
        ndof = V.value_size * V.dof_dset.set.size

        # PDE coefficients interpolated on the quadrature nodes
        ele = V.ufl_element()
        ncomp = ele.value_size()
        bsize = V.value_size
        needs_hdiv = bsize != ncomp
        Gq, Bq, self._assemble_Gq, self._assemble_Bq = self.assemble_coef(mu, beta, Nq, diagonal=True, piola=needs_hdiv)

        # Assign arbitrary non-zero coefficients for preallocation
        Gq.dat.data[:] = 1.0E0
        if Bq is not None:
            Bq.dat.data[:] = 1.0E0

        self.assemble_kron(prealloc, V, Gq, Bq, Afdm, Dfdm, eta, bcflags, needs_interior_facet)
        nnz = get_preallocation(prealloc, ndof)
        self.Pmat = PETSc.Mat().createAIJ(A.getSizes(), nnz=nnz, comm=A.comm)
        self._assemble_Pmat = partial(self.assemble_kron, self.Pmat, V, Gq, Bq,
                                      Afdm, Dfdm, eta, bcflags, needs_interior_facet)

        prealloc.destroy()
        lgmap = V.dof_dset.lgmap
        self.Pmat.setBlockSize(V.value_size)
        self.Pmat.setLGMap(lgmap, lgmap)

        opc = pc
        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -fdm_pc_type ksp.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        # We set a DM on the constructed PC so one
        # can do patch solves with ASMPC.
        from firedrake.solving_utils import _SNESContext
        mat_type = "aij"
        dm = opc.getDM()
        octx = get_appctx(dm)
        oproblem = octx._problem
        self._ctx_ref = _SNESContext(oproblem, mat_type, mat_type, octx.appctx, options_prefix=options_prefix)

        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(self.Pmat, self.Pmat)
        self.pc = pc
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref, save=False):
            pc.setFromOptions()
        self.update(pc)

    def update(self, pc):
        self._assemble_Gq()
        self._assemble_Bq()
        self.Pmat.zeroEntries()
        self._assemble_Pmat()
        self.Pmat.zeroRowsColumnsLocal(self.bc_nodes)

    def applyTranspose(self, pc, x, y):
        # TODO trivial to implement reusing the code below
        pass

    def apply(self, pc, x, y):
        self.uc.assign(firedrake.zero())

        with self.uf.dat.vec_wo as xf:
            x.copy(xf)

        op2.par_loop(self.restrict_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.INC, self.cell_node_map),
                     self.uf.dat(op2.READ, self.cell_node_map),
                     self.weight.dat(op2.READ, self.cell_node_map))

        for bc in self.bcs:
            bc.zero(self.uc)

        dm = pc.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref), self.uc.dat.vec as x_, self.uf.dat.vec as y_:
            self.pc.apply(x_, y_)

        for bc in self.bcs:
            bc.zero(self.uf)

        op2.par_loop(self.prolong_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.WRITE, self.cell_node_map),
                     self.uf.dat(op2.READ, self.cell_node_map))

        with self.uc.dat.vec_ro as xc:
            xc.copy(y)

        y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]

    def view(self, pc, viewer=None):
        super(FDMPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    @staticmethod
    def pull_axis(x, pshape, idir):
        return numpy.reshape(numpy.moveaxis(numpy.reshape(x.copy(), pshape), idir, 0), x.shape)

    def assemble_kron(self, A, V, Gq, Bq, Afdm, Dfdm, eta, bcflags, needs_interior_facet):
        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map(self.bcs)

        ele = V.ufl_element()
        ncomp = ele.value_size()
        bsize = V.value_size
        ndim = V.mesh().topological_dimension()
        sdim = V.finat_element.space_dimension()

        needs_hdiv = bsize != ncomp
        if needs_hdiv:
            sdim = sdim // ncomp

        if needs_hdiv:
            # FIXME still need to pass mu
            mu = self.appctx.get("viscosity", None)
            Gfacet0, Gfacet1, Piola0, Piola1 = self.assemble_piola_facet(mu)
            jid, _, _, _ = self.get_facet_topology(Gfacet0.function_space())

        lexico_cell, nel = self.glonum_fun(V.cell_node_map())
        gid, _ = self.glonum_fun(Gq.cell_node_map())
        bid, _ = self.glonum_fun(Bq.cell_node_map()) if Bq is not None else (None, nel)

        # Build sparse cell matrices and assemble global matrix
        flag2id = numpy.kron(numpy.eye(ndim, ndim, dtype=PETSc.IntType), [[1], [2]])
        if needs_hdiv:
            pshape = [[Afdm[(k-i) % ncomp][0][0].size[0] for i in range(ndim)] for k in range(ncomp)]
        else:
            pshape = [Ak[0][0].size[0] for Ak in Afdm]

        for row in V.dof_dset.lgmap.indices:
            A.setValue(row, row, 0.0E0, imode)

        # We first deal with the reaction term if Bq is a tensor
        use_separate_reaction = False if Bq is None else (not needs_hdiv and len(Bq.ufl_shape) == 2)

        if use_separate_reaction:
            be = Afdm[0][0][1]
            for k in range(1, ndim):
                be = be.kron(Afdm[k][0][1])

            aptr = numpy.arange(0, (Bq.ufl_shape[0]+1)*Bq.ufl_shape[1], Bq.ufl_shape[1], dtype=PETSc.IntType)
            aidx = numpy.tile(numpy.arange(Bq.ufl_shape[1], dtype=PETSc.IntType), Bq.ufl_shape[0])
            for e in range(nel):
                adata = numpy.sum(Bq.dat.data_ro[bid(e)], axis=0)
                ae = PETSc.Mat().createAIJWithArrays(Bq.ufl_shape, (aptr, aidx, adata), comm=PETSc.COMM_SELF)
                ae = be.kron(ae)

                ie = lexico_cell(e)
                ie = numpy.repeat(ie*bsize, bsize) + numpy.tile(numpy.arange(bsize, dtype=PETSc.IntType), len(ie))
                indptr, indices, data = ae.getValuesCSR()
                rows = lgmap.apply(ie)
                cols = rows[indices]
                for i, row in enumerate(rows):
                    i0 = indptr[i]
                    i1 = indptr[i+1]
                    A.setValues(row, cols[i0:i1], data[i0:i1], imode)
                ae.destroy()
            Bq = None

        # Assemble the viscouos term and the reaction term if any
        for e in range(nel):
            ie = lexico_cell(e)
            if needs_hdiv:
                ie = numpy.reshape(ie, (ncomp, -1))

            mue = numpy.atleast_1d(numpy.sum(Gq.dat.data_ro[gid(e)], axis=0))
            bce = bcflags[e]

            if Bq is not None:
                bqe = numpy.atleast_1d(numpy.sum(Bq.dat.data_ro[bid(e)], axis=0))
                if len(bqe) == 1:
                    bqe = numpy.tile(bqe, ncomp)

            for k in range(ncomp):
                bcj = bce[k] if len(bce.shape) == 2 else bce
                muj = mue[k] if len(mue.shape) == 2 else mue
                fbc = bcj @ flag2id

                facet_perm = numpy.arange(ndim)
                if needs_hdiv:
                    facet_perm = (facet_perm-k) % ndim

                be = Afdm[facet_perm[0]][fbc[0]][1]
                ae = Afdm[facet_perm[0]][fbc[0]][0].copy()
                ae.scale(muj[0])
                if Bq is not None:
                    ae.axpy(bqe[k], be)

                if ndim > 1:
                    ae = ae.kron(Afdm[facet_perm[1]][fbc[1]][1])
                    ae.axpy(muj[1], be.kron(Afdm[facet_perm[1]][fbc[1]][0]))
                    if ndim > 2:
                        be = be.kron(Afdm[facet_perm[1]][fbc[1]][1])
                        ae = ae.kron(Afdm[facet_perm[2]][fbc[2]][1])
                        ae.axpy(muj[2], be.kron(Afdm[facet_perm[2]][fbc[2]][0]))

                indptr, indices, data = ae.getValuesCSR()
                rows = lgmap.apply(ie[k] if needs_hdiv else k+bsize*ie)
                cols = rows[indices]
                for i, row in enumerate(rows):
                    i0 = indptr[i]
                    i1 = indptr[i+1]
                    A.setValues(row, cols[i0:i1], data[i0:i1], imode)
                ae.destroy()

        istart = 1 if needs_hdiv else 0
        if needs_interior_facet:

            lexico_facet, nfacet, facet_cells, facet_data = self.get_facet_topology(V)
            rows = numpy.zeros((2*sdim,), dtype=PETSc.IntType)

            for f in range(nfacet):
                e0, e1 = facet_cells[f]
                idir = facet_data[f] // 2

                ie = lexico_facet(f)
                mu0 = numpy.atleast_1d(numpy.sum(Gq.dat.data_ro_with_halos[gid(e0)], axis=0))
                mu1 = numpy.atleast_1d(numpy.sum(Gq.dat.data_ro_with_halos[gid(e1)], axis=0))

                if needs_hdiv:
                    fid = numpy.reshape(jid(f), (2, -1))
                    fdof = fid[0][facet_data[f, 0]]
                    icell = numpy.reshape(lgmap.apply(ie), (2, ncomp, -1))
                    iord0 = numpy.insert(numpy.delete(numpy.arange(ndim), idir[0]), 0, idir[0])
                    iord1 = numpy.insert(numpy.delete(numpy.arange(ndim), idir[1]), 0, idir[1])

                for k in range(istart, ncomp):
                    if needs_hdiv:
                        k0 = iord0[k]
                        k1 = iord1[k]
                        facet_perm = numpy.insert(numpy.delete(numpy.arange(ndim), 0), k, 0)
                        mu = [Gfacet0.dat.data_ro_with_halos[fdof][idir[0]],
                              Gfacet1.dat.data_ro_with_halos[fdof][idir[1]]]
                        Piola = [Piola0.dat.data_ro_with_halos[fdof][k0],
                                 Piola1.dat.data_ro_with_halos[fdof][k1]]
                    else:
                        k0 = k
                        k1 = k
                        facet_perm = (idir[0]+numpy.arange(ndim)) % ndim
                        mu = [mu0[k0][idir[0]] if len(mu0.shape) > 1 else mu0[idir[0]],
                              mu1[k1][idir[1]] if len(mu1.shape) > 1 else mu1[idir[1]]]

                    Dfacet = Dfdm[facet_perm[0]]
                    offset = Dfacet.shape[0]
                    adense = numpy.zeros((2*offset, 2*offset), dtype=PETSc.RealType)
                    dense_indices = []
                    for j, jface in enumerate(facet_data[f]):
                        j0 = j * offset
                        j1 = j0 + offset
                        jj = j0 + (offset-1) * (jface % 2)
                        dense_indices.append(jj)
                        for i, iface in enumerate(facet_data[f]):
                            i0 = i * offset
                            i1 = i0 + offset
                            ii = i0 + (offset-1) * (iface % 2)

                            sij = 0.5E0 if (i == j) or (bool(k0) != bool(k1)) else -0.5E0
                            if needs_hdiv:
                                beta = [sij*numpy.dot(numpy.dot(mu[0], Piola[i]), Piola[j]),
                                        sij*numpy.dot(numpy.dot(mu[1], Piola[i]), Piola[j])]
                            else:
                                beta = [sij*mu[0], sij*mu[1]]

                            adense[ii, jj] += eta * sum(beta)
                            adense[i0:i1, jj] -= beta[i] * Dfacet[:, iface % 2]
                            adense[ii, j0:j1] -= beta[j] * Dfacet[:, jface % 2]

                    ae = FDMPC.fdm_numpy_to_petsc(adense, dense_indices, diag=False)
                    if ndim > 1:
                        # Here we are assuming that the mesh is oriented
                        ae = ae.kron(Afdm[facet_perm[1]][0][1])
                        if ndim > 2:
                            ae = ae.kron(Afdm[facet_perm[2]][0][1])

                    if needs_hdiv:
                        assert pshape[k0][idir[0]] == pshape[k1][idir[1]]
                        rows[:sdim] = self.pull_axis(icell[0][k0], pshape[k0], idir[0])
                        rows[sdim:] = self.pull_axis(icell[1][k1], pshape[k1], idir[1])
                    else:
                        icell = numpy.reshape(lgmap.apply(k+bsize*ie), (2, -1))
                        rows[:sdim] = self.pull_axis(icell[0], pshape, idir[0])
                        rows[sdim:] = self.pull_axis(icell[1], pshape, idir[1])

                    indptr, indices, data = ae.getValuesCSR()
                    cols = rows[indices]
                    for i, row in enumerate(rows):
                        i0 = indptr[i]
                        i1 = indptr[i+1]
                        A.setValues(row, cols[i0:i1], data[i0:i1], imode)
                    ae.destroy()

        A.assemble()

    def assemble_coef(self, mu, beta, Nq=0, diagonal=False, transpose=False, piola=False):
        ndim = self.mesh.topological_dimension()
        gdim = self.mesh.geometric_dimension()
        gshape = (ndim, ndim)

        if gdim == ndim:
            Finv = JacobianInverse(self.mesh)

            if piola:
                PF = (1/JacobianDeterminant(self.mesh)) * Jacobian(self.mesh)
                if mu is None:
                    i1, i2, i3, i4, j1, j2 = indices(6)
                    G = as_tensor(PF[j1, i1] * Finv[i2, j2] * PF[j1, i3] * Finv[i4, j2], (i1, i2, i3, i4))
                elif mu.ufl_shape == ():
                    i1, i2, i3, i4, j1, j2 = indices(6)
                    G = mu * as_tensor(PF[j1, i1] * Finv[i2, j2] * PF[j1, i3] * Finv[i4, j2], (i1, i2, i3, i4))
                elif len(mu.ufl_shape) == 4:
                    i1, i2, i3, i4, j1, j2, j3, j4 = indices(8)
                    G = as_tensor(PF[j1, i1] * Finv[i2, j2] * PF[j3, i3] * Finv[i4, j4] * mu[j1, j2, j3, j4], (i1, i2, i3, i4))
            else:
                if mu is None:
                    G = dot(Finv, Finv.T)
                elif mu.ufl_shape == ():
                    G = mu * dot(Finv, Finv.T)
                elif mu.ufl_shape == gshape:
                    G = dot(dot(Finv, mu), Finv.T)
                elif len(mu.ufl_shape) == 4:
                    i1, i2, i3, i4, j2, j4 = indices(6)
                    G = as_tensor(Finv[i2, j2] * Finv[i4, j4] * mu[i1, j2, i3, j4], (i1, i2, i3, i4))
                else:
                    raise ValueError("I don't know what to do with the homogeneity tensor")
        else:
            F = Jacobian(self.mesh)
            G = inv(dot(F.T, F))
            if mu:
                G = mu * G
            # I don't know how to use tensor viscosity on embedded manifolds

        if diagonal:
            if len(G.ufl_shape) == 2:
                G = diag_vector(G)
                Qe = VectorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default", dim=numpy.prod(G.ufl_shape))
            elif len(G.ufl_shape) == 4:
                if transpose:
                    G = as_tensor([[G[i, j, i, j] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[1])])
                else:
                    G = as_tensor([[G[i, j, i, j] for j in range(G.ufl_shape[1])] for i in range(G.ufl_shape[0])])
                Qe = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default", shape=G.ufl_shape)
            else:
                raise ValueError("I don't know how to get the diagonal of a tensor with shape ", G.ufl_shape)
        else:
            Qe = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                               quad_scheme="default", shape=G.ufl_shape, symmetry=True)

        Q = firedrake.FunctionSpace(self.mesh, Qe)
        q = firedrake.TestFunction(Q)
        Gq = firedrake.Function(Q)
        assemble_Gq = partial(firedrake.assemble, inner(G, q)*dx(degree=Nq), Gq)

        if beta is None:
            Bq = None
            assemble_Bq = lambda: None
        else:
            shape = beta.ufl_shape
            if len(shape) == 2:
                Qe = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default", shape=shape)
            elif len(shape) == 1:
                Qe = VectorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default", dim=shape[0])
            else:
                Qe = FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default")

            Q = firedrake.FunctionSpace(self.mesh, Qe)
            q = firedrake.TestFunction(Q)
            Bq = firedrake.Function(Q)
            assemble_Bq = partial(firedrake.assemble, inner(beta, q)*dx(degree=Nq), Bq)

        return Gq, Bq, assemble_Gq, assemble_Bq

    def assemble_piola_facet(self, mu):
        extruded = self.mesh.cell_set._extruded
        dS_int = firedrake.dS_h + firedrake.dS_v if extruded else firedrake.dS
        area = firedrake.FacetArea(self.mesh)

        Finv = JacobianInverse(self.mesh)
        vol = abs(JacobianDeterminant(self.mesh))
        i1, i2, i3, i4, j2, j4 = indices(6)
        if mu is None:
            I = firedrake.Identity(self.mesh.topological_dimension())
            G = vol * as_tensor(I[i1, i3] * Finv[i2, j2] * Finv[i4, j2], (i1, i2, i3, i4))
        else:
            G = vol * as_tensor(Finv[i2, j2] * Finv[i4, j4] * mu[i1, j2, i3, j4], (i1, i2, i3, i4))
        G = as_tensor([[[G[i, k, j, k] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[2])] for k in range(G.ufl_shape[3])])

        hinv = area / vol
        Finv = hinv * FacetNormal(self.mesh)
        # G = vol * as_tensor(Finv[j2] * Finv[j4] * mu[i1, j2, i3, j4], (i1, i3))
        DGT = firedrake.TensorFunctionSpace(self.mesh, "DGT", 0, shape=G.ufl_shape)
        test = firedrake.TestFunction(DGT)
        Gfacet0 = firedrake.assemble(inner(test('+'), G('+') / area) * dS_int)
        Gfacet1 = firedrake.assemble(inner(test('+'), G('-') / area) * dS_int)

        P = (1/JacobianDeterminant(self.mesh)) * Jacobian(self.mesh).T
        DGT = firedrake.TensorFunctionSpace(self.mesh, "DGT", 0, shape=P.ufl_shape)
        test = firedrake.TestFunction(DGT)
        Pfacet0 = firedrake.assemble(inner(test('+'), P('+') / area) * dS_int)
        Pfacet1 = firedrake.assemble(inner(test('+'), P('-') / area) * dS_int)
        return Gfacet0, Gfacet1, Pfacet0, Pfacet1

    @staticmethod
    @lru_cache(maxsize=10)
    def semhat(N, Nq):
        # Ahat = GLL stiffness matrix
        # Bhat = GLL mass matrix
        # Jhat = GLL(N) basis tabulated on the quadrature nodes
        # Dhat = first derivative of GLL(N) basis tabulated on the quadrature nodes
        # what = quadrature weights
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
        Ahat = Dhat @ numpy.diag(what) @ Dhat.T
        Bhat = Jhat @ numpy.diag(what) @ Jhat.T
        return Ahat, Bhat, Jhat, Dhat, what

    @staticmethod
    def fdm_numpy_to_petsc(A_numpy, dense_indices, diag=True):
        # Creates a SeqAIJ Mat from a dense matrix using the diagonal and a subset of rows and columns
        # If dense_indinces is empty, it includes the off-diagonal corners of the matrix
        n = A_numpy.shape[0]
        nbase = int(diag) + len(dense_indices)
        nnz = numpy.full((n,), nbase, dtype=PETSc.IntType)
        if dense_indices:
            nnz[dense_indices] = n
        else:
            nnz[[0, -1]] = 2

        imode = PETSc.InsertMode.INSERT
        A_petsc = PETSc.Mat().createAIJ(A_numpy.shape, nnz=nnz, comm=PETSc.COMM_SELF)
        if diag:
            for j, ajj in enumerate(A_numpy.diagonal()):
                A_petsc.setValue(j, j, ajj, imode)

        if dense_indices:
            idx = numpy.arange(n, dtype=PETSc.IntType)
            for j in dense_indices:
                A_petsc.setValues(j, idx, A_numpy[j], imode)
                A_petsc.setValues(idx, j, A_numpy[:][j], imode)
        else:
            A_petsc.setValue(0, n-1, A_numpy[0][-1], imode)
            A_petsc.setValue(n-1, 0, A_numpy[-1][0], imode)

        A_petsc.assemble()
        return A_petsc

    @staticmethod
    def fdm_cg(Ahat, Bhat):
        rd = (0, -1)
        kd = slice(1, -1)
        Vfdm = numpy.eye(Ahat.shape[0])
        if Vfdm.shape[0] > 2:
            _, Vfdm[kd, kd] = sym_eig(Ahat[kd, kd], Bhat[kd, kd])
            Vfdm[kd, rd] = -Vfdm[kd, kd] @ ((Vfdm[kd, kd].T @ Bhat[kd, rd]) @ Vfdm[numpy.ix_(rd, rd)])

        def apply_strong_bcs(Ahat, Bhat, bc0, bc1):
            k0 = 0 if bc0 == 1 else 1
            k1 = Ahat.shape[0] if bc1 == 1 else -1
            kk = slice(k0, k1)
            A = Ahat.copy()
            a = A.diagonal().copy()
            A[kk, kk] = 0.0E0
            numpy.fill_diagonal(A, a)

            B = Bhat.copy()
            b = B.diagonal().copy()
            B[kk, kk] = 0.0E0
            numpy.fill_diagonal(B, b)
            return [FDMPC.fdm_numpy_to_petsc(A, [0, A.shape[0]-1]),
                    FDMPC.fdm_numpy_to_petsc(B, [])]

        Afdm = []
        Ak = Vfdm.T @ Ahat @ Vfdm
        Bk = Vfdm.T @ Bhat @ Vfdm
        Bk[rd, kd] = 0.0E0
        Bk[kd, rd] = 0.0E0
        for bc1 in range(2):
            for bc0 in range(2):
                Afdm.append(apply_strong_bcs(Ak, Bk, bc0, bc1))

        return Afdm, Vfdm, None

    @staticmethod
    def fdm_ipdg(Ahat, Bhat, N, eta, gll=False):
        from FIAT.reference_element import UFCInterval
        from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
        from FIAT.quadrature import GaussLegendreQuadratureLineRule

        cell = UFCInterval()
        elem = GaussLobattoLegendre(cell, N)

        # Interpolation onto GL nodes
        rule = GaussLegendreQuadratureLineRule(cell, N + 1)
        basis = elem.tabulate(0, rule.get_points())
        Jipdg = basis[(0,)]

        # Facet normal derivatives
        basis = elem.tabulate(1, cell.get_vertices())
        Dfacet = basis[(1,)]
        Dfacet[:, 0] = -Dfacet[:, 0]

        rd = (0, -1)
        kd = slice(1, -1)
        Vfdm = numpy.eye(Ahat.shape[0])
        if Vfdm.shape[0] > 2:
            _, Vfdm[kd, kd] = sym_eig(Ahat[kd, kd], Bhat[kd, kd])
            Vfdm[kd, rd] = -Vfdm[kd, kd] @ ((Vfdm[kd, kd].T @ Bhat[kd, rd]) @ Vfdm[numpy.ix_(rd, rd)])

        def apply_weak_bcs(Ahat, Bhat, Dfacet, bcs, eta):
            Abc = Ahat.copy()
            for j in (0, -1):
                if bcs[j] == 1:
                    Abc[:, j] -= Dfacet[:, j]
                    Abc[j, :] -= Dfacet[:, j]
                    Abc[j, j] += eta

            return [FDMPC.fdm_numpy_to_petsc(Abc, [0, Abc.shape[0]-1]),
                    FDMPC.fdm_numpy_to_petsc(Bhat, [])]

        A = Vfdm.T @ Ahat @ Vfdm
        a = A.diagonal().copy()
        A[kd, kd] = 0.0E0
        numpy.fill_diagonal(A, a)

        B = Vfdm.T @ Bhat @ Vfdm
        b = B.diagonal().copy()
        B[kd, kd] = 0.0E0
        B[rd, kd] = 0.0E0
        B[kd, rd] = 0.0E0
        numpy.fill_diagonal(B, b)

        Dfdm = Vfdm.T @ Dfacet
        Afdm = []
        for bc1 in range(2):
            for bc0 in range(2):
                bcs = (bc0, bc1)
                Afdm.append(apply_weak_bcs(A, B, Dfdm, bcs, eta))

        if not gll:
            # Vfdm first rotates GL residuals into GLL space
            Vfdm = Jipdg.T @ Vfdm

        return Afdm, Vfdm, Dfdm

    @staticmethod
    @lru_cache(maxsize=10)
    def assemble_matfree(V, N, Nq, eta, needs_interior_facet):
        # Assemble sparse 1D matrices and matrix-free kernels for basis transformation
        from firedrake.slate.slac.compiler import BLASLAPACK_LIB, BLASLAPACK_INCLUDE

        bsize = V.value_size
        nscal = V.ufl_element().value_size()
        sdim = V.finat_element.space_dimension()
        ndim = V.ufl_domain().topological_dimension()
        lwork = bsize * sdim

        needs_hdiv = nscal != bsize

        if needs_hdiv:
            Ahat, Bhat, Jhat, Dhat, _ = FDMPC.semhat(N-1, Nq)
            Afdm, Vfdm, Dfdm = FDMPC.fdm_ipdg(Ahat, Bhat, N-1, eta)
            Afdm = [Afdm]*ndim
            Vfdm = [Vfdm]*ndim
            Dfdm = [Dfdm]*ndim
            Ahat, Bhat, Jhat, Dhat, _ = FDMPC.semhat(N, Nq)
            Afdm[0], Vfdm[0], Dfdm[0] = FDMPC.fdm_ipdg(Ahat, Bhat, N, eta, gll=True)
        else:
            Ahat, Bhat, Jhat, Dhat, _ = FDMPC.semhat(N, Nq)
            if needs_interior_facet:
                Afdm, Vfdm, Dfdm = FDMPC.fdm_ipdg(Ahat, Bhat, N, eta)
            else:
                Afdm, Vfdm, Dfdm = FDMPC.fdm_cg(Ahat, Bhat)
            Afdm = [Afdm]*ndim
            Dfdm = [Dfdm]*ndim
            Vfdm = [Vfdm]*ndim

        nx = Vfdm[0].shape[0]
        ny = Vfdm[1].shape[0] if ndim >= 2 else 1
        nz = Vfdm[2].shape[0] if ndim >= 3 else 1

        Vsize = sum([Vk.size for Vk in Vfdm])
        Vhex = ', '.join(map(float.hex, numpy.concatenate([numpy.asarray(Vk).flatten() for Vk in Vfdm])))
        VX = "V"
        VY = f"V+{nx*nx}" if ndim > 1 else "&one"
        VZ = f"V+{nx*nx+ny*ny}" if ndim > 2 else "&one"

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

        void prolongation(PetscScalar *restrict y, const PetscScalar *restrict x){{
            PetscScalar V[{Vsize}] = {{ {Vhex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                    t0[j + {sdim}*i] = x[i + {bsize}*j];

            kronmxv(1, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {VX},{VY},{VZ}, t0, t1);

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                   y[i + {bsize}*j] = t1[j + {sdim}*i];
            return;
        }}

        void restriction(PetscScalar *restrict y, const PetscScalar *restrict x,
                         const PetscScalar *restrict w){{
            PetscScalar V[{Vsize}] = {{ {Vhex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                    t0[j + {sdim}*i] = x[i + {bsize}*j] * w[i + {bsize}*j];

            kronmxv(0, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {VX},{VY},{VZ}, t0, t1);

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                    y[i + {bsize}*j] += t1[j + {sdim}*i];
            return;
        }}
        """

        restrict_kernel = op2.Kernel(transfer_code, "restriction", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())
        prolong_kernel = op2.Kernel(transfer_code, "prolongation", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())

        return Afdm, Dfdm, restrict_kernel, prolong_kernel

    @staticmethod
    def multiplicity(V):
        # Lawrence's magic code for calculating dof multiplicities
        shapes = (V.finat_element.space_dimension(),
                  numpy.prod(V.shape))
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
    def get_facet_topology(V):
        # Returns the 4-tuple of
        # lexico_facet: a function that maps an interior facet id with the nodes of the two cells sharing it
        # nfacets: the total number of interior facets owned by this process
        # facet_cells: the interior facet to cell map
        # facet_data: the local numbering of each interior facet with respect to the two cells sharing it
        mesh = V.mesh()
        intfacets = mesh.interior_facets
        facet_cells = intfacets.facet_cell_map.values
        facet_data = intfacets.local_facet_dat.data_ro

        facet_node_map = V.interior_facet_node_map()
        facet_values = facet_node_map.values
        nbase = facet_node_map.values.shape[0]

        if mesh.layers:
            layers = facet_node_map.iterset.layers_array
            if layers.shape[0] == 1:

                cell_node_map = V.cell_node_map()
                cell_values = cell_node_map.values
                cell_offset = cell_node_map.offset
                nelh = cell_values.shape[0]
                nelz = layers[0, 1] - layers[0, 0] - 1

                nh = nbase * nelz
                nv = nelh * (nelz - 1)
                nfacets = nh + nv
                facet_offset = facet_node_map.offset

                lexico_base = lambda e: facet_values[e % nbase] + (e//nbase)*facet_offset

                lexico_v = lambda e: numpy.append(cell_values[e % nelh] + (e//nelh)*cell_offset,
                                                  cell_values[e % nelh] + (e//nelh + 1)*cell_offset)

                lexico_facet = lambda e: lexico_base(e) if e < nh else lexico_v(e-nh)

                if nv:
                    facet_data = numpy.concatenate((numpy.tile(facet_data, (nelz, 1)),
                                                    numpy.tile(numpy.array([[5, 4]], facet_data.dtype), (nv, 1))), axis=0)

                    facet_cells_base = [facet_cells + nelh*k for k in range(nelz)]
                    facet_cells_base.append(numpy.array([[k, k+nelh] for k in range(nv)], facet_cells.dtype))
                    facet_cells = numpy.concatenate(facet_cells_base, axis=0)

            else:
                raise NotImplementedError("Not implemented for variable layers")
        else:
            lexico_facet = lambda e: facet_values[e]
            nfacets = nbase

        return lexico_facet, nfacets, facet_cells, facet_data

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum_fun(node_map):
        # Returns a function that maps the cell id to its global nodes
        nelh = node_map.values.shape[0]
        if node_map.offset is None:
            return lambda e: node_map.values_with_halo[e], nelh
        else:
            layers = node_map.iterset.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1] - layers[0, 0] - 1
                nel = nelz * nelh
                return lambda e: node_map.values_with_halo[e % nelh] + (e//nelh)*node_map.offset, nel
            else:
                k = 0
                nelz = layers[:, 1] - layers[:, 0] - 1
                nel = sum(nelz[:nelh])
                layer_id = numpy.zeros((sum(nelz), 2))
                for e in range(len(nelz)):
                    for l in range(nelz[e]):
                        layer_id[k, :] = [e, l]
                        k += 1
                return lambda e: node_map.values_with_halo[layer_id[e, 0]] + layer_id[e, 1]*node_map.offset, nel

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum(V):
        # Returns an array of global nodes for each cell id
        node_map = V.cell_node_map()
        if node_map.offset is None:
            return node_map.values
        else:
            nelh = node_map.values.shape[0]
            layers = node_map.iterset.layers_array
            if(layers.shape[0] == 1):
                nelz = layers[0, 1]-layers[0, 0]-1
                nel = nelz * nelh
                gl = numpy.zeros((nelz,)+node_map.values.shape, dtype=PETSc.IntType)
                for k in range(0, nelz):
                    gl[k] = node_map.values + k*node_map.offset
                gl = numpy.reshape(gl, (nel, -1))
            else:
                k = 0
                nelz = layers[:nelh, 1]-layers[:nelh, 0]-1
                nel = sum(nelz)
                gl = numpy.zeros((nel, node_map.values.shape[1]), dtype=PETSc.IntType)
                for e in range(0, nelh):
                    for l in range(0, nelz[e]):
                        gl[k] = node_map.values[e] + l*node_map.offset
                        k += 1
            return gl

    @staticmethod
    @lru_cache(maxsize=10)
    def get_bc_flags(V, mesh, bcs, J):
        # Returns boundary condition flags on each cell facet
        # 0 => Natural, do nothing
        # 1 => Strong / Weak Dirichlet
        # 2 => Interior facet

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
                facetdata = numpy.zeros([nel, nface, 2], dtype=PETSc.IntType)
                facetdata[:, ivert, :] = numpy.tile(mesh.cell_to_facets.data, (nelz, 1, 1))
            else:
                nelz = layers[:nelh, 1] - layers[:nelh, 0] - 1
                nel = sum(nelz)
                facetdata = numpy.zeros([nel, nface, 2], dtype=PETSc.IntType)
                facetdata[:, ivert, :] = numpy.repeat(mesh.cell_to_facets.data, nelz, axis=0)
                for f in ivert:
                    bnd = numpy.isclose(rho[:, f], 0.0E0)
                    bnd &= (facetdata[:, f, 0] != 0)
                    facetdata[bnd, f, :] = [0, -8]

            bot = numpy.isclose(rho[:, ibot], 0.0E0)
            top = numpy.isclose(rho[:, itop], 0.0E0)
            facetdata[:, [ibot, itop], :] = -1
            facetdata[bot, ibot, :] = [0, -2]
            facetdata[top, itop, :] = [0, -4]
        else:
            facetdata = mesh.cell_to_facets.data

        flags = facetdata[:, :, 0]
        sub = facetdata[:, :, 1]

        maskall = []
        comp = dict()
        for bc in bcs:
            if isinstance(bc, firedrake.DirichletBC):
                labels = comp.get(bc._indices, ())
                bs = bc.sub_domain
                if bs == "on_boundary":
                    maskall.append(bc._indices)
                elif bs == "bottom":
                    labels += (-2,)
                elif bs == "top":
                    labels += (-4,)
                else:
                    labels += bs if type(bs) == tuple else (bs,)
                comp[bc._indices] = labels

        # TODO add support for weak component BCs
        # The Neumann integral may still be present but it's zero
        J = expand_derivatives(J)
        for it in J.integrals():
            itype = it.integral_type()
            if itype.startswith("exterior_facet"):
                labels = comp.get((), ())
                bs = it.subdomain_id()
                if bs == "everywhere":
                    if itype == "exterior_facet_bottom":
                        labels += (-2,)
                    elif itype == "exterior_facet_top":
                        labels += (-4,)
                    else:
                        maskall.append(())
                else:
                    labels += bs if type(bs) == tuple else (bs,)
                comp[()] = labels

        labels = comp.get((), ())
        labels = list(set(labels))
        fbc = numpy.isin(sub, labels).astype(PETSc.IntType)

        if () in maskall:
            fbc[sub >= -1] = 1
        fbc[flags != 0] = 0

        others = set(comp.keys()) - {()}
        if others:
            # We have bcs on individual vector components
            fbc = numpy.tile(fbc, (V.value_size, 1, 1))
            for j in range(V.value_size):
                key = (j,)
                labels = comp.get(key, ())
                labels = list(set(labels))
                fbc[j] |= numpy.isin(sub, labels)
                if key in maskall:
                    fbc[j][sub >= -1] = 1

            fbc = numpy.transpose(fbc, (1, 0, 2))
        return fbc
