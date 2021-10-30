from functools import lru_cache, partial

from pyop2 import op2
from pyop2.sparsity import get_preallocation

from ufl import FiniteElement, VectorElement, TensorElement, BrokenElement
from ufl import Jacobian, JacobianDeterminant, JacobianInverse
from ufl import as_tensor, diag_vector, dot, dx, indices, inner
from ufl import grad, diff, replace, variable
from ufl.constantvalue import Zero
from ufl.algorithms.ad import expand_derivatives

from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.pmg import get_permuted_map, tensor_product_space_query
from firedrake.utils import IntType_c
from firedrake.dmhooks import get_function_space, get_appctx
import firedrake
import numpy
import numpy.linalg
from firedrake_citations import Citations

Citations().add("Brubeck2021", """
@misc{Brubeck2021,
  title={A scalable and robust vertex-star relaxation for high-order {FEM}},
  author={Brubeck, Pablo D. and Farrell, Patrick E.},
  archiveprefix = {arXiv},
  eprint = {2107.14758},
  primaryclass = {math.NA},
  year={2021}
}
""")

__all__ = ("FDMPC",)


def sym_eig(A, B):
    """
    numpy version of `scipy.linalg.eigh`
    """
    L = numpy.linalg.cholesky(B)
    Linv = numpy.linalg.inv(L)
    C = numpy.dot(Linv, numpy.dot(A, Linv.T))
    Z, W = numpy.linalg.eigh(C)
    V = numpy.dot(Linv.T, W)
    return Z, V


class FDMPC(PCBase):
    """
    A preconditioner for tensor-product elements that changes the shape
    functions so that the H^1 Riesz map is diagonalized in the interior of a
    Cartesian cell, and assembles a global sparse matrix on which other
    preconditioners, such as `ASMStarPC`, can be applied.

    Here we assume that the volume integrals in the Jacobian can be expressed as:

    inner(grad(v), alpha(grad(u)))*dx + inner(v, beta(u))*dx

    where alpha and beta are linear functions (tensor contractions).
    The sparse matrix is obtained by approximating alpha and beta by cell-wise
    constants and discarding the coefficients in alpha that couple together
    mixed derivatives and mixed components.

    For spaces that are not H^1-conforming, this preconditioner will use
    the symmetric interior-penalty DG method. The penalty coefficient can be
    provided in the application context, keyed on ``"eta"``.
    """

    _prefix = "fdm_"

    def initialize(self, pc):
        Citations().register("Brubeck2021")
        A, P = pc.getOperators()

        # Read options
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        # opts = PETSc.Options(options_prefix)

        dm = pc.getDM()
        V = get_function_space(dm)
        use_tensorproduct, N, ndim, sub_families, _ = tensor_product_space_query(V)

        needs_hdiv = sub_families < {"RTCF", "NCF"}
        needs_hcurl = sub_families < {"RTCE", "NCE"}
        if not use_tensorproduct or needs_hcurl:
            raise ValueError("FDMPC does not support the element %s" % V.ufl_element())

        needs_interior_facet = not (sub_families <= {"Q", "Lagrange"})
        Nq = 2*N+1  # quadrature degree, gives exact interval stiffness matrices for constant coefficients

        self.mesh = V.ufl_domain()
        self.uf = firedrake.Function(V)
        self.uc = firedrake.Function(V)
        self.cell_node_map = get_permuted_map(V)

        # Get Jacobian form and bcs
        solverctx = get_appctx(dm)
        self.J = solverctx.J
        self.bcs = solverctx.bcs_J

        if len(self.bcs) > 0:
            self.bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in self.bcs]))
        else:
            self.bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

        bcflags = self.get_bc_flags(self.bcs, self.J)

        self.weight = self.multiplicity(V)
        with self.weight.dat.vec as w:
            w.reciprocal()

        # Get the interior penalty parameter from the appctx
        appctx = self.get_appctx(pc)
        eta = float(appctx.get("eta", (N+1)*(N+ndim)))

        # Get the FDM transfer kernels (restriction and prolongation)
        # Afdm = sparse interval mass and stiffness matrices for each direction
        # Dfdm = tabulation of normal derivative of the FDM basis at the boundary for each direction
        Afdm, Dfdm, self.restrict_kernel, self.prolong_kernel = self.assemble_matfree(V, N, Nq, eta, needs_interior_facet, needs_hdiv)

        # Get coefficients w.r.t. the reference coordinates
        # we may use a lower quadrature degree, but using Nq is not so expensive
        coefficients, self.assembly_callables = self.assemble_coef(self.J, Nq, discard_mixed=True, cell_average=True, needs_hdiv=needs_hdiv)

        # Set arbitrary non-zero coefficients for preallocation
        for coef in coefficients.values():
            with coef.dat.vec as cvec:
                cvec.set(1.0E0)

        # Preallocate by calling the assembly routine on a PREALLOCATOR Mat
        prealloc = PETSc.Mat().create(comm=A.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(A.getSizes())
        prealloc.setUp()
        self.assemble_kron(prealloc, V, coefficients, Afdm, Dfdm, eta, bcflags, needs_hdiv)
        nnz = get_preallocation(prealloc, V.value_size * V.dof_dset.set.size)

        self.Pmat = PETSc.Mat().createAIJ(A.getSizes(), A.getBlockSize(), nnz=nnz, comm=A.comm)
        self.Pmat.setLGMap(V.dof_dset.lgmap)
        self._assemble_Pmat = partial(self.assemble_kron, self.Pmat, V,
                                      coefficients, Afdm, Dfdm, eta, bcflags, needs_hdiv)
        prealloc.destroy()

        opc = pc
        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -fdm_pc_type ksp.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        # We set a DM on the constructed PC so one
        # can do patch solves with ASMPC.
        dm = opc.getDM()
        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(self.Pmat, self.Pmat)
        self.pc = pc
        pc.setFromOptions()
        self.update(pc)

    def update(self, pc):
        for assemble_coef in self.assembly_callables:
            assemble_coef()
        self.Pmat.zeroEntries()
        self._assemble_Pmat()
        self.Pmat.zeroRowsColumnsLocal(self.bc_nodes)

    def apply(self, pc, x, y, transpose=False):
        with self.uc.dat.vec_wo as xc:
            xc.set(0.0E0)

        with self.uf.dat.vec_wo as xf:
            x.copy(xf)

        op2.par_loop(self.restrict_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.INC, self.cell_node_map),
                     self.uf.dat(op2.READ, self.cell_node_map),
                     self.weight.dat(op2.READ, self.cell_node_map))

        for bc in self.bcs:
            bc.zero(self.uc)

        with self.uc.dat.vec_ro as x_, self.uf.dat.vec_wo as y_:
            if transpose:
                self.pc.applyTranspose(x_, y_)
            else:
                self.pc.apply(x_, y_)

        for bc in self.bcs:
            bc.zero(self.uf)

        op2.par_loop(self.prolong_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.WRITE, self.cell_node_map),
                     self.uf.dat(op2.READ, self.cell_node_map))

        with self.uc.dat.vec_ro as xc:
            xc.copy(y)

        y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]

    def applyTranspose(self, pc, x, y):
        self.apply(pc, x, y, transpose=True)

    def view(self, pc, viewer=None):
        super(FDMPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    @staticmethod
    def pull_axis(x, pshape, idir):
        # permute x by reshaping into pshape and moving axis idir to the front
        return numpy.reshape(numpy.moveaxis(numpy.reshape(x.copy(), pshape), idir, 0), x.shape)

    def assemble_kron(self, A, V, coefficients, Afdm, Dfdm, eta, bcflags, needs_hdiv):
        """
        Assemble the stiffness matrix in the FDM basis using Kronecker products of interval matrices

        :arg A: the :class:`PETSc.Mat` to assemble
        :arg V: the :class:`firedrake.FunctionSpace` of the form arguments
        :arg coefficients: a ``dict`` mapping strings to :class:`firedrake.Functions` with the form coefficients
        :arg Bq: a :class:`firedrake.Function` with the zero-th order coefficients of the form
        :arg Afdm: the list with interval matrices returned by `FDMPC.assemble_matfree`
        :arg Dfdm: the list with normal derivatives matrices returned by `FDMPC.assemble_matfree`
        :arg eta: the SIPG penalty parameter as a ``float``
        :arg bcflags: the :class:`numpy.ndarray` with BC facet flags returned by `FDMPC.get_bc_flags`
        :arg needs_hdiv: a ``bool`` indicating whether the function space V is H(div)-conforming
        """
        Gq = coefficients.get("Gq", None)
        Bq = coefficients.get("Bq", None)

        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map(self.bcs)

        bsize = V.value_size
        ndim = V.ufl_domain().topological_dimension()
        ncomp = V.ufl_element().reference_value_size()
        sdim = (V.finat_element.space_dimension() * bsize) // ncomp  # dimension of a single component

        index_cell, nel = self.glonum_fun(V.cell_node_map())
        index_coef, _ = self.glonum_fun(Gq.cell_node_map())
        flag2id = numpy.kron(numpy.eye(ndim, ndim, dtype=PETSc.IntType), [[1], [2]])

        # pshape is the shape of the DOFs in the tensor product
        if needs_hdiv:
            pshape = [[Afdm[(k-i) % ncomp][0].size[0] for i in range(ndim)] for k in range(ncomp)]
        else:
            pshape = [Ak[0].size[0] for Ak in Afdm]

        # we need to preallocate the diagonal of Dirichlet nodes
        for row in V.dof_dset.lgmap.indices:
            A.setValue(row, row, 0.0E0, imode)

        # assemble zero-th order term separately, including off-diagonals (mixed components)
        # I cannot do this for hdiv elements as off-diagonals are not sparse, this is because
        # the FDM eigenbases for GLL(N) and GLL(N-1) are not orthogonal to each other
        use_separate_reaction = False if Bq is None else not needs_hdiv and Bq.ufl_shape
        if use_separate_reaction:
            bshape = Bq.ufl_shape
            assert (len(bshape) == 2) and (bshape[0] == bshape[1])
            # Be = Bhat kron ... kron Bhat
            Be = Afdm[0][0].copy()
            for k in range(1, ndim):
                Be = Be.kron(Afdm[k][0])

            aptr = numpy.arange(0, (bshape[0]+1)*bshape[1], bshape[1], dtype=PETSc.IntType)
            aidx = numpy.tile(numpy.arange(bshape[1], dtype=PETSc.IntType), bshape[0])
            for e in range(nel):
                # Ae = Be kron Bq[e]
                adata = numpy.sum(Bq.dat.data_ro[index_coef(e)], axis=0)
                Ae = PETSc.Mat().createAIJWithArrays(bshape, (aptr, aidx, adata), comm=PETSc.COMM_SELF)
                Ae = Be.kron(Ae)

                ie = index_cell(e)
                ie = numpy.repeat(ie*bsize, bsize) + numpy.tile(numpy.arange(bsize, dtype=ie.dtype), len(ie))
                indptr, indices, data = Ae.getValuesCSR()
                rows = lgmap.apply(ie)
                cols = rows[indices]
                for i, row in enumerate(rows):
                    i0 = indptr[i]
                    i1 = indptr[i+1]
                    A.setValues(row, cols[i0:i1], data[i0:i1], imode)
                Ae.destroy()
            Be.destroy()
            Bq = None

        # assemble the second order term and the zero-th order term if any,
        # discarding mixed derivatives and mixed components
        for e in range(nel):
            ie = numpy.reshape(index_cell(e), (ncomp//bsize, -1))

            bce = bcflags[e]
            # get second order coefficient on this cell
            je = index_coef(e)
            mue = numpy.atleast_1d(numpy.sum(Gq.dat.data_ro[je], axis=0))
            if Bq is not None:
                # get zero-th order coefficient on this cell
                bqe = numpy.atleast_1d(numpy.sum(Bq.dat.data_ro[je], axis=0))

            for k in range(ncomp):
                # for each component: compute the element stiffness matrix Ae
                muk = mue[k] if len(mue.shape) == 2 else mue
                bck = bce[k] if len(bce.shape) == 2 else bce
                fbc = numpy.dot(bck, flag2id)

                # permutation of dimensions with respect to the first vector component
                dim_perm = numpy.arange(ndim)
                if needs_hdiv:
                    dim_perm = (dim_perm-k) % ndim

                # Ae = mue[k][0] Ahat + bqe[k] Bhat
                Be = Afdm[dim_perm[0]][0].copy()
                Ae = Afdm[dim_perm[0]][1+fbc[0]].copy()
                Ae.scale(muk[0])
                if Bq is not None:
                    Ae.axpy(bqe[k], Be)

                if ndim > 1:
                    # Ae = Ae kron Bhat + mue[k][1] Bhat kron Ahat
                    Ae = Ae.kron(Afdm[dim_perm[1]][0])
                    Ae.axpy(muk[1], Be.kron(Afdm[dim_perm[1]][1+fbc[1]]))
                    if ndim > 2:
                        # Ae = Ae kron Bhat + mue[k][2] Bhat kron Bhat kron Ahat
                        Be = Be.kron(Afdm[dim_perm[1]][0])
                        Ae = Ae.kron(Afdm[dim_perm[2]][0])
                        Ae.axpy(muk[2], Be.kron(Afdm[dim_perm[2]][1+fbc[2]]))

                indptr, indices, data = Ae.getValuesCSR()
                rows = lgmap.apply(ie[0]*bsize+k if bsize == ncomp else ie[k])
                cols = rows[indices]
                for i, row in enumerate(rows):
                    i0 = indptr[i]
                    i1 = indptr[i+1]
                    A.setValues(row, cols[i0:i1], data[i0:i1], imode)
                Ae.destroy()
                Be.destroy()

        # assemble SIPG interior facet terms if the normal derivatives have been set up
        needs_interior_facet = any(Dk is not None for Dk in Dfdm)
        if needs_interior_facet:
            if ndim < V.ufl_domain().geometric_dimension():
                raise NotImplementedError("Interior facet integrals on immersed meshes are not implemented")
            index_facet, local_facet_data, nfacets = self.get_interior_facet_maps(V)
            rows = numpy.zeros((2*sdim,), dtype=PETSc.IntType)
            kstart = 1 if needs_hdiv else 0
            if needs_hdiv:
                Gq_facet = coefficients.get("Gq_facet", None)
                PT_facet = coefficients.get("PT_facet", None)
                index_coef, _, _ = self.get_interior_facet_maps(Gq_facet)
            else:
                index_coef, _, _ = self.get_interior_facet_maps(Gq)

            for e in range(nfacets):
                # for each interior facet: compute the SIPG stiffness matrix Ae
                ie = index_facet(e)
                je = numpy.reshape(index_coef(e), (2, -1))
                lfd = local_facet_data(e)
                idir = lfd // 2

                if needs_hdiv:
                    icell = numpy.reshape(lgmap.apply(ie), (2, ncomp, -1))
                    iord0 = numpy.insert(numpy.delete(numpy.arange(ndim), idir[0]), 0, idir[0])
                    iord1 = numpy.insert(numpy.delete(numpy.arange(ndim), idir[1]), 0, idir[1])
                    je = je[[0, 1], lfd]
                    Pfacet = PT_facet.dat.data_ro_with_halos[je]
                    Gfacet = Gq_facet.dat.data_ro_with_halos[je]
                else:
                    Gfacet = numpy.sum(Gq.dat.data_ro_with_halos[je], axis=1)

                for k in range(kstart, ncomp):
                    if needs_hdiv:
                        k0 = iord0[k]
                        k1 = iord1[k]
                        dim_perm = numpy.insert(numpy.delete(numpy.arange(ndim), 0), k, 0)
                        mu = Gfacet[[0, 1], idir]
                        Piola = Pfacet[[0, 1], [k0, k1]]
                    else:
                        k0 = k
                        k1 = k
                        dim_perm = (idir[0]+numpy.arange(ndim)) % ndim
                        mu = Gfacet[[0, 1], [k0, k1], idir] if len(Gfacet.shape) == 3 else Gfacet[[0, 1], idir]

                    Dfacet = Dfdm[dim_perm[0]]
                    offset = Dfacet.shape[0]
                    Adense = numpy.zeros((2*offset, 2*offset), dtype=PETSc.RealType)
                    dense_indices = []
                    for j, jface in enumerate(lfd):
                        j0 = j * offset
                        j1 = j0 + offset
                        jj = j0 + (offset-1) * (jface % 2)
                        dense_indices.append(jj)
                        for i, iface in enumerate(lfd):
                            i0 = i * offset
                            i1 = i0 + offset
                            ii = i0 + (offset-1) * (iface % 2)

                            sij = 0.5E0 if (i == j) or (bool(k0) != bool(k1)) else -0.5E0
                            if needs_hdiv:
                                smu = [sij*numpy.dot(numpy.dot(mu[0], Piola[i]), Piola[j]),
                                       sij*numpy.dot(numpy.dot(mu[1], Piola[i]), Piola[j])]
                            else:
                                smu = sij*mu

                            Adense[ii, jj] += eta * sum(smu)
                            Adense[i0:i1, jj] -= smu[i] * Dfacet[:, iface % 2]
                            Adense[ii, j0:j1] -= smu[j] * Dfacet[:, jface % 2]

                    Ae = FDMPC.numpy_to_petsc(Adense, dense_indices, diag=False)
                    if ndim > 1:
                        # assume that the mesh is oriented
                        Ae = Ae.kron(Afdm[dim_perm[1]][0])
                        if ndim > 2:
                            Ae = Ae.kron(Afdm[dim_perm[2]][0])

                    if needs_hdiv:
                        assert pshape[k0][idir[0]] == pshape[k1][idir[1]]
                        rows[:sdim] = self.pull_axis(icell[0][k0], pshape[k0], idir[0])
                        rows[sdim:] = self.pull_axis(icell[1][k1], pshape[k1], idir[1])
                    else:
                        icell = numpy.reshape(lgmap.apply(k+bsize*ie), (2, -1))
                        rows[:sdim] = self.pull_axis(icell[0], pshape, idir[0])
                        rows[sdim:] = self.pull_axis(icell[1], pshape, idir[1])

                    indptr, indices, data = Ae.getValuesCSR()
                    cols = rows[indices]
                    for i, row in enumerate(rows):
                        i0 = indptr[i]
                        i1 = indptr[i+1]
                        A.setValues(row, cols[i0:i1], data[i0:i1], imode)
                    Ae.destroy()
        A.assemble()

    def assemble_coef(self, J, quad_deg, discard_mixed=False, cell_average=False, needs_hdiv=False):
        """
        Return the coefficients of the Jacobian form arguments and their gradient with respect to the reference coordinates.

        :arg J: the Jacobian bilinear form
        :arg quad_deg: the quadrature degree used for the coefficients
        :arg discard_mixed: discard entries in second order coefficient with mixed derivatives and mixed components
        :arg cell_average: to return the coefficients as DG_0 Functions

        :returns: a 2-tuple of
            coefficients: a dictionary mapping strings to :class:`firedrake.Functions` with the coefficients of the form,
            assembly_callables: a list of assembly callables for each coefficient of the form
        """
        coefficients = {}
        assembly_callables = []

        mesh = J.ufl_domain()
        gdim = mesh.geometric_dimension()
        ndim = mesh.topological_dimension()
        Finv = JacobianInverse(mesh)

        if cell_average:
            family = "Discontinuous Lagrange" if ndim == 1 else "DQ"
            degree = 0
        else:
            family = "Quadrature"
            degree = quad_deg

        # extract coefficients directly from the bilinear form
        args_J = J.arguments()
        integrals_J = J.integrals_by_type("cell")
        mapping = args_J[0].ufl_element().mapping().lower()
        if mapping == 'identity':
            Piola = None
        elif mapping == 'covariant piola':
            Piola = Finv.T
        elif mapping == 'contravariant piola':
            Piola = Jacobian(mesh) / JacobianDeterminant(mesh)
            if ndim < gdim:
                Piola *= 1-2*mesh.cell_orientations()
        else:
            raise NotImplementedError("Unrecognized element mapping %s" % mapping)

        # get second order coefficient
        rep_dict = {grad(t): variable(grad(t)) for t in args_J}
        val = list(rep_dict.values())
        alpha = expand_derivatives(sum([diff(diff(replace(i.integrand(), rep_dict), val[0]), val[1]) for i in integrals_J]))

        # get zero-th order coefficent
        rep_dict = {t: variable(t) for t in args_J}
        val = list(rep_dict.values())
        beta = expand_derivatives(sum([diff(diff(replace(i.integrand(), rep_dict), val[0]), val[1]) for i in integrals_J]))

        # transform the second order coefficient alpha (physical gradient) to obtain G (reference gradient)
        if Piola is None:
            if len(alpha.ufl_shape) == 2:
                G = dot(dot(Finv, alpha), Finv.T)
            elif len(alpha.ufl_shape) == 4:
                i1, i2, i3, i4, j2, j4 = indices(6)
                G = as_tensor(Finv[i2, j2] * Finv[i4, j4] * alpha[i1, j2, i3, j4], (i1, i2, i3, i4))
            else:
                raise ValueError("TensorFunctionSpaces with more than 1 index are not supported.")
        else:
            if len(alpha.ufl_shape) == 4:
                i1, i2, i3, i4, j1, j2, j3, j4 = indices(8)
                G = as_tensor(Piola[j1, i1] * Finv[i2, j2] * Piola[j3, i3] * Finv[i4, j4] * alpha[j1, j2, j3, j4], (i1, i2, i3, i4))
            else:
                raise ValueError("TensorFunctionSpaces with more than 1 index are not supported.")

        if discard_mixed:
            # discard mixed derivatives and mixed components
            if len(G.ufl_shape) == 2:
                G = diag_vector(G)
                Qe = VectorElement(family, mesh.ufl_cell(), degree=degree,
                                   quad_scheme="default", dim=numpy.prod(G.ufl_shape))
            elif len(G.ufl_shape) == 4:
                G = as_tensor([[G[i, j, i, j] for j in range(G.ufl_shape[1])] for i in range(G.ufl_shape[0])])
                Qe = TensorElement(family, mesh.ufl_cell(), degree=degree,
                                   quad_scheme="default", shape=G.ufl_shape)
            else:
                raise ValueError("I don't know how to discard mixed entries of a tensor with shape ", G.ufl_shape)
        else:
            Qe = TensorElement(family, mesh.ufl_cell(), degree=degree,
                               quad_scheme="default", shape=G.ufl_shape, symmetry=True)

        # assemble second order coefficient
        Q = firedrake.FunctionSpace(mesh, Qe)
        q = firedrake.TestFunction(Q)
        Gq = firedrake.Function(Q)
        coefficients["Gq"] = Gq
        assembly_callables.append(partial(firedrake.assemble, inner(G, q)*dx(degree=quad_deg), Gq))

        # assemble zero-th order coefficient
        if isinstance(beta, Zero):
            Bq = None
        else:
            if Piola:
                # apply Piola transform and keep diagonal
                beta = diag_vector(dot(Piola.T, dot(beta, Piola)))
            shape = beta.ufl_shape
            if shape:
                Qe = TensorElement(family, mesh.ufl_cell(), degree=degree,
                                   quad_scheme="default", shape=shape)
            else:
                Qe = FiniteElement(family, mesh.ufl_cell(), degree=degree,
                                   quad_scheme="default")

            Q = firedrake.FunctionSpace(mesh, Qe)
            q = firedrake.TestFunction(Q)
            Bq = firedrake.Function(Q)
            coefficients["Bq"] = Bq
            assembly_callables.append(partial(firedrake.assemble, inner(beta, q)*dx(degree=quad_deg), Bq))

        if needs_hdiv:
            # make DGT functions with the second order coefficient
            # and the Piola transform matrix for each side of each facet
            extruded = mesh.cell_set._extruded
            dS_int = firedrake.dS_h(degree=quad_deg) + firedrake.dS_v(degree=quad_deg) if extruded else firedrake.dS(degree=quad_deg)
            ele = BrokenElement(FiniteElement("DGT", mesh.ufl_cell(), 0))
            area = firedrake.FacetArea(mesh)

            vol = abs(JacobianDeterminant(mesh))
            i1, i2, i3, i4, j2, j4 = indices(6)
            G = vol * as_tensor(Finv[i2, j2] * Finv[i4, j4] * alpha[i1, j2, i3, j4], (i1, i2, i3, i4))
            G = as_tensor([[[G[i, k, j, k] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[2])] for k in range(G.ufl_shape[3])])

            Q = firedrake.TensorFunctionSpace(mesh, ele, shape=G.ufl_shape)
            q = firedrake.TestFunction(Q)
            Gq_facet = firedrake.Function(Q)
            coefficients["Gq_facet"] = Gq_facet
            assembly_callables.append(partial(firedrake.assemble, ((inner(q('+'), G('+')) + inner(q('-'), G('-')))/area) * dS_int, Gq_facet))

            PT = Piola.T
            Q = firedrake.TensorFunctionSpace(mesh, ele, shape=PT.ufl_shape)
            q = firedrake.TestFunction(Q)
            PT_facet = firedrake.Function(Q)
            coefficients["PT_facet"] = PT_facet
            assembly_callables.append(partial(firedrake.assemble, ((inner(q('+'), PT('+')) + inner(q('-'), PT('-')))/area) * dS_int, PT_facet))
        return coefficients, assembly_callables

    @staticmethod
    def numpy_to_petsc(A_numpy, dense_indices, diag=True):
        # Create a SeqAIJ Mat from a dense matrix using the diagonal and a subset of rows and columns
        # If dense_indices is empty, then also include the off-diagonal corners of the matrix
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
    def semhat(N, Nq):
        """
        Setup for the GLL finite element method on the reference interval

        :arg N: polynomial degree of the GLL element
        :arg Nq: quadrature degree (Nq >= 2*N+1 gives exact integrals)

        :returns: 5-tuple of :class:`numpy.ndarray`
            Ahat: GLL(N) stiffness matrix
            Bhat: GLL(N) mass matrix
            Jhat: tabulation of the GLL(N) basis on the GL quadrature nodes
            Dhat: tabulation of the first derivative of the GLL(N) basis on the GL quadrature nodes
            what: GL quadrature weights
        """
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
        Ahat = numpy.dot(Dhat, numpy.dot(numpy.diag(what), Dhat.T))
        Bhat = numpy.dot(Jhat, numpy.dot(numpy.diag(what), Jhat.T))
        return Ahat, Bhat, Jhat, Dhat, what

    @staticmethod
    def fdm_setup_cg(Ahat, Bhat):
        """
        Setup for the fast diagonalization method for continuous Lagrange
        elements. Compute the FDM eigenvector basis and the sparsified interval
        stiffness and mass matrices.

        :arg Ahat: GLL stiffness matrix as a :class:`numpy.ndarray`
        :arg Bhat: GLL mass matrix as a :class:`numpy.ndarray`

        :returns: 3-tuple of:
            Afdm: a list of :class:`PETSc.Mats` with the sparse interval matrices
            Sfdm.T * Bhat * Sfdm, and bcs(Sfdm.T * Ahat * Sfdm) for every combination of either
            natural or strong Dirichlet BCs on each endpoint,
            Sfdm: the tabulation of Dirichlet eigenfunctions on the GLL nodes,
            Dfdm: None.
        """
        Sfdm = numpy.eye(Ahat.shape[0])
        if Sfdm.shape[0] > 2:
            rd = (0, -1)
            kd = slice(1, -1)
            _, Sfdm[kd, kd] = sym_eig(Ahat[kd, kd], Bhat[kd, kd])
            Sfdm[kd, rd] = -Sfdm[kd, kd] @ ((Sfdm[kd, kd].T @ Bhat[kd, rd]) @ Sfdm[numpy.ix_(rd, rd)])

        def apply_strong_bcs(Ahat, bc0, bc1):
            k0 = 0 if bc0 == 1 else 1
            k1 = Ahat.shape[0] if bc1 == 1 else -1
            kk = slice(k0, k1)
            A = Ahat.copy()
            a = A.diagonal().copy()
            A[kk, kk] = 0.0E0
            numpy.fill_diagonal(A, a)
            return FDMPC.numpy_to_petsc(A, [0, A.shape[0]-1])

        A = numpy.dot(Sfdm.T, numpy.dot(Ahat, Sfdm))
        B = numpy.dot(Sfdm.T, numpy.dot(Bhat, Sfdm))
        Afdm = [FDMPC.numpy_to_petsc(B, [])]
        for bc1 in range(2):
            for bc0 in range(2):
                Afdm.append(apply_strong_bcs(A, bc0, bc1))
        return Afdm, Sfdm, None

    @staticmethod
    def fdm_setup_ipdg(Ahat, Bhat, eta, gll=False):
        """
        Setup for the fast diagonalization method for the IP-DG formulation.
        Compute the FDM eigenvector basis, its normal derivative and the
        sparsified interval stiffness and mass matrices.

        :arg Ahat: GLL stiffness matrix as a :class:`numpy.ndarray`
        :arg Bhat: GLL mass matrix as a :class:`numpy.array`
        :arg eta: penalty coefficient as a `float`
        :arg gll: `bool` flag indicating whether to keep the eigenvectors in the GLL basis

        :returns: 3-tuple of:
            Afdm: a list of :class:`PETSc.Mats` with the sparse interval matrices
            Sfdm.T * Bhat * Sfdm, and bcs(Sfdm.T * Ahat * Sfdm) for every combination of either
            natural or weak Dirichlet BCs on each endpoint,
            Sfdm: the tabulation of Dirichlet eigenfunctions on the GL or GLL nodes,
            Dfdm: the tabulation of the normal derivatives of the Dirichlet eigenfunctions.
        """
        from FIAT.reference_element import UFCInterval
        from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
        from FIAT.quadrature import GaussLegendreQuadratureLineRule

        N = Ahat.shape[0]-1
        cell = UFCInterval()
        elem = GaussLobattoLegendre(cell, N)

        # Facet normal derivatives
        basis = elem.tabulate(1, cell.get_vertices())
        Dfacet = basis[(1,)]
        Dfacet[:, 0] = -Dfacet[:, 0]

        Sfdm = numpy.eye(Ahat.shape[0])
        if Sfdm.shape[0] > 2:
            rd = (0, -1)
            kd = slice(1, -1)
            _, Sfdm[kd, kd] = sym_eig(Ahat[kd, kd], Bhat[kd, kd])
            Sfdm[kd, rd] = -Sfdm[kd, kd] @ ((Sfdm[kd, kd].T @ Bhat[kd, rd]) @ Sfdm[numpy.ix_(rd, rd)])

        def apply_weak_bcs(Ahat, Dfacet, bcs, eta):
            Abc = Ahat.copy()
            for j in (0, -1):
                if bcs[j] == 1:
                    Abc[:, j] -= Dfacet[:, j]
                    Abc[j, :] -= Dfacet[:, j]
                    Abc[j, j] += eta
            return FDMPC.numpy_to_petsc(Abc, [0, Abc.shape[0]-1])

        A = numpy.dot(Sfdm.T, numpy.dot(Ahat, Sfdm))
        B = numpy.dot(Sfdm.T, numpy.dot(Bhat, Sfdm))
        Dfdm = numpy.dot(Sfdm.T, Dfacet)
        Afdm = [FDMPC.numpy_to_petsc(B, [])]
        for bc1 in range(2):
            for bc0 in range(2):
                Afdm.append(apply_weak_bcs(A, Dfdm, (bc0, bc1), eta))

        if not gll:
            # Interpolate eigenfunctions from GLL onto GL
            rule = GaussLegendreQuadratureLineRule(cell, N + 1)
            basis = elem.tabulate(0, rule.get_points())
            Jipdg = basis[(0,)]
            Sfdm = numpy.dot(Jipdg.T, Sfdm)
        return Afdm, Sfdm, Dfdm

    @staticmethod
    def assemble_matfree(V, N, Nq, eta, needs_interior_facet, needs_hdiv):
        """
        Setup of coefficient-independent quantities in the FDM

        Assemble the sparse interval stiffness matrices, tabulate normal derivatives,
        and get the transfer kernels between the space V and the FDM basis.

        :arg V: the :class:`FunctionSpace` of the problem
        :arg N: the degree of V
        :arg Nq: the quadrature degree for the assembly of interval matrices
        :arg eta: a `float` penalty parameter for the symmetric interior penalty method
        :arg needs_interior_facet: is the space V non-H^1-conforming?
        :arg needs_hdiv: is the space V H(div)-conforming?

        :returns: a 4-tuple with
            Afdm: a list of lists of interval matrices for each direction,
            Dfdm: a list with tabulations of the normal derivative for each direction
            restrict_kernel: a :class:`pyop2.Kernel` with the tensor product restriction from
            V onto the FDM space,
            prolong_kernel: a :class:`pyop2.Kernel` with the tensor product prolongation from
            the FDM space onto V
        """
        bsize = V.value_size
        nscal = V.ufl_element().value_size()
        sdim = V.finat_element.space_dimension()
        ndim = V.ufl_domain().topological_dimension()
        lwork = bsize * sdim

        if needs_hdiv:
            Ahat, Bhat, _, _, _ = FDMPC.semhat(N-1, Nq)
            Afdm, Sfdm, Dfdm = FDMPC.fdm_setup_ipdg(Ahat, Bhat, eta)
            Afdm = [Afdm]*ndim
            Sfdm = [Sfdm]*ndim
            Dfdm = [Dfdm]*ndim
            Ahat, Bhat, _, _, _ = FDMPC.semhat(N, Nq)
            Afdm[0], Sfdm[0], Dfdm[0] = FDMPC.fdm_setup_ipdg(Ahat, Bhat, eta, gll=True)
        else:
            Ahat, Bhat, _, _, _ = FDMPC.semhat(N, Nq)
            if needs_interior_facet:
                Afdm, Sfdm, Dfdm = FDMPC.fdm_setup_ipdg(Ahat, Bhat, eta)
            else:
                Afdm, Sfdm, Dfdm = FDMPC.fdm_setup_cg(Ahat, Bhat)
            Afdm = [Afdm]*ndim
            Sfdm = [Sfdm]*ndim
            Dfdm = [Dfdm]*ndim

        nx = Sfdm[0].shape[0]
        ny = Sfdm[1].shape[0] if ndim >= 2 else 1
        nz = Sfdm[2].shape[0] if ndim >= 3 else 1

        S_len = sum([Sk.size for Sk in Sfdm])
        S_hex = ', '.join(map(float.hex, numpy.concatenate([numpy.asarray(Sk).flatten() for Sk in Sfdm])))
        SX = "S"
        SY = f"S+{nx*nx}" if ndim > 1 else "&one"
        SZ = f"S+{nx*nx+ny*ny}" if ndim > 2 else "&one"

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

        BLASgemm_(&TA1, &notr, &m, &n, &k, &one, A1, &lda, x, &k, &zero, y, &m);

        p = 0;  s = 0;
        m = mx;  k = ny;  n = my;
        lda = (tflag>0)? ny : my;
        for(PetscBLASInt i=0; i<nz*nel; i++){
           BLASgemm_(&notr, &TA2, &m, &n, &k, &one, y+p, &m, A2, &lda, &zero, x+s, &m);
           p += m*k;
           s += m*n;
        }

        p = 0;  s = 0;
        m = mx*my;  k = nz;  n = mz;
        lda = (tflag>0)? nz : mz;
        for(PetscBLASInt i=0; i<nel; i++){
           BLASgemm_(&notr, &TA3, &m, &n, &k, &one, x+p, &m, A3, &lda, &zero, y+s, &m);
           p += m*k;
           s += m*n;
        }
        return;
        }
        """

        kernel_code = f"""
        {kronmxv_code}

        void prolongation(PetscScalar *restrict y, const PetscScalar *restrict x){{
            PetscScalar S[{S_len}] = {{ {S_hex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                    t0[j + {sdim}*i] = x[i + {bsize}*j];

            kronmxv(1, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {SX},{SY},{SZ}, t0, t1);

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                   y[i + {bsize}*j] = t1[j + {sdim}*i];
            return;
        }}

        void restriction(PetscScalar *restrict y, const PetscScalar *restrict x,
                         const PetscScalar *restrict w){{
            PetscScalar S[{S_len}] = {{ {S_hex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                    t0[j + {sdim}*i] = x[i + {bsize}*j] * w[i + {bsize}*j];

            kronmxv(0, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {SX},{SY},{SZ}, t0, t1);

            for({IntType_c} j=0; j<{sdim}; j++)
                for({IntType_c} i=0; i<{bsize}; i++)
                    y[i + {bsize}*j] += t1[j + {sdim}*i];
            return;
        }}
        """

        from firedrake.slate.slac.compiler import BLASLAPACK_LIB, BLASLAPACK_INCLUDE
        prolong_kernel = op2.Kernel(kernel_code, "prolongation", include_dirs=BLASLAPACK_INCLUDE.split(),
                                    ldargs=BLASLAPACK_LIB.split(), requires_zeroed_output_arguments=True)
        restrict_kernel = op2.Kernel(kernel_code, "restriction", include_dirs=BLASLAPACK_INCLUDE.split(),
                                     ldargs=BLASLAPACK_LIB.split(), requires_zeroed_output_arguments=True)
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
    def get_interior_facet_maps(V):
        """
        Extrude V.interior_facet_node_map and V.ufl_domain().interior_facets.local_facet_dat

        :arg V: a :class:`FunctionSpace`

        :returns: the 3-tuple of
            facet_to_nodes_fun: maps interior facets to the nodes of the two cells sharing it,
            local_facet_data_fun: maps interior facets to the local facet numbering in the two cells sharing it,
            nfacets: the total number of interior facets owned by this process
        """
        mesh = V.ufl_domain()
        intfacets = mesh.interior_facets
        facet_to_cells = intfacets.facet_cell_map.values
        local_facet_data = intfacets.local_facet_dat.data_ro

        facet_node_map = V.interior_facet_node_map()
        facet_to_nodes = facet_node_map.values
        nbase = facet_to_nodes.shape[0]

        if mesh.cell_set._extruded:
            facet_offset = facet_node_map.offset
            local_facet_data_h = numpy.array([5, 4], local_facet_data.dtype)

            cell_node_map = V.cell_node_map()
            cell_to_nodes = cell_node_map.values_with_halo
            cell_offset = cell_node_map.offset

            nelv = cell_node_map.values.shape[0]
            layers = facet_node_map.iterset.layers_array
            itype = cell_offset.dtype
            shift_h = numpy.array([[0], [1]], itype)

            if mesh.variable_layers:
                nv = 0
                to_base = []
                to_layer = []
                for f, cells in enumerate(facet_to_cells):
                    istart = max(layers[cells, 0])
                    iend = min(layers[cells, 1])
                    nz = iend-istart-1
                    nv += nz
                    to_base.append(numpy.full((nz,), f, itype))
                    to_layer.append(numpy.arange(nz, dtype=itype))

                nh = layers[:, 1]-layers[:, 0]-2
                to_base.append(numpy.repeat(numpy.arange(len(nh), dtype=itype), nh))
                to_layer += [numpy.arange(nf, dtype=itype) for nf in nh]

                to_base = numpy.concatenate(to_base)
                to_layer = numpy.concatenate(to_layer)
                nfacets = nv + sum(nh[:nelv])

                local_facet_data_fun = lambda e: local_facet_data[to_base[e]] if e < nv else local_facet_data_h
                facet_to_nodes_fun = lambda e: facet_to_nodes[to_base[e]] + to_layer[e]*facet_offset if e < nv else numpy.reshape(cell_to_nodes[to_base[e]] + numpy.kron(to_layer[e]+shift_h, cell_offset), (-1,))
            else:
                nelz = layers[0, 1]-layers[0, 0]-1
                nv = nbase * nelz
                nh = nelv * (nelz-1)
                nfacets = nv + nh

                local_facet_data_fun = lambda e: local_facet_data[e//nelz] if e < nv else local_facet_data_h
                facet_to_nodes_fun = lambda e: facet_to_nodes[e//nelz] + (e % nelz)*facet_offset if e < nv else numpy.reshape(cell_to_nodes[(e-nv)//(nelz-1)] + numpy.kron(((e-nv) % (nelz-1))+shift_h, cell_offset), (-1,))
        else:
            facet_to_nodes_fun = lambda e: facet_to_nodes[e]
            local_facet_data_fun = lambda e: local_facet_data[e]
            nfacets = nbase

        return facet_to_nodes_fun, local_facet_data_fun, nfacets

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum_fun(node_map):
        """
        Return a function that maps each topological entity to its nodes and the total number of entities.

        :arg node_map: a :class:`pyop2.Map` mapping entities to their nodes, including ghost entities.

        :returns: a 2-tuple with the map and the number of cells owned by this process
        """
        nelv = node_map.values.shape[0]
        if node_map.offset is None:
            return lambda e: node_map.values_with_halo[e], nelv
        else:
            layers = node_map.iterset.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1]-layers[0, 0]-1
                nel = nelz*nelv
                return lambda e: node_map.values_with_halo[e//nelz] + (e % nelz)*node_map.offset, nel
            else:
                nelz = layers[:, 1]-layers[:, 0]-1
                nel = sum(nelz[:nelv])
                to_base = numpy.repeat(numpy.arange(node_map.values_with_halo.shape[0], dtype=node_map.offset.dtype), nelz)
                to_layer = numpy.concatenate([numpy.arange(nz, dtype=node_map.offset.dtype) for nz in nelz])
                return lambda e: node_map.values_with_halo[to_base[e]] + to_layer[e]*node_map.offset, nel

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum(node_map):
        """
        Return an array with the nodes of each topological entity of a certain kind.

        :arg node_map: a :class:`pyop2.Map` mapping entities to their nodes, including ghost entities.

        :returns: a :class:`numpy.ndarray` whose rows are the nodes for each cell
        """
        if node_map.offset is None:
            return node_map.values_with_halo
        else:
            layers = node_map.iterset.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1]-layers[0, 0]-1
                to_layer = numpy.tile(numpy.arange(nelz, dtype=node_map.offset.dtype), len(node_map.values_with_halo))
            else:
                nelz = layers[:, 1]-layers[:, 0]-1
                to_layer = numpy.concatenate([numpy.arange(nz, dtype=node_map.offset.dtype) for nz in nelz])
            return numpy.repeat(node_map.values_with_halo, nelz, axis=0) + numpy.kron(to_layer.reshape((-1, 1)), node_map.offset)

    @staticmethod
    @lru_cache(maxsize=10)
    def get_bc_flags(bcs, J):
        # Return boundary condition flags on each cell facet
        # 0 => natural, do nothing
        # 1 => strong / weak Dirichlet
        V = J.arguments()[0].function_space()
        mesh = V.ufl_domain()

        if mesh.cell_set._extruded:
            layers = mesh.cell_set.layers_array
            nelv, nfacet, _ = mesh.cell_to_facets.data_with_halos.shape
            if layers.shape[0] == 1:
                nelz = layers[0, 1]-layers[0, 0]-1
                nel = nelv*nelz
            else:
                nelz = layers[:, 1]-layers[:, 0]-1
                nel = sum(nelz)
            # extrude cell_to_facets
            cell_to_facets = numpy.zeros((nel, nfacet+2, 2), dtype=mesh.cell_to_facets.data.dtype)
            cell_to_facets[:, :nfacet, :] = numpy.repeat(mesh.cell_to_facets.data_with_halos, nelz, axis=0)

            # get a function with a single node per facet
            # mark interior facets by assembling a surface integral
            dS_int = firedrake.dS_h(degree=0) + firedrake.dS_v(degree=0)
            DGT = firedrake.FunctionSpace(mesh, "DGT", 0)
            v = firedrake.TestFunction(DGT)
            w = firedrake.assemble((v('+')+v('-'))*dS_int)

            # mark the bottom and top boundaries with DirichletBCs
            markers = (-2, -4)
            subs = ("bottom", "top")
            bc_h = [firedrake.DirichletBC(DGT, marker, sub) for marker, sub in zip(markers, subs)]
            [bc.apply(w) for bc in bc_h]

            # index the function with the extruded cell_node_map
            marked_facets = w.dat.data_ro_with_halos[FDMPC.glonum(DGT.cell_node_map())]

            # complete the missing pieces of cell_to_facets
            interior = marked_facets > 0
            cell_to_facets[interior, :] = [1, -1]
            topbot = marked_facets < 0
            cell_to_facets[topbot, 0] = 0
            cell_to_facets[topbot, 1] = marked_facets[topbot].astype(cell_to_facets.dtype)
        else:
            cell_to_facets = mesh.cell_to_facets.data_with_halos

        flags = cell_to_facets[:, :, 0]
        sub = cell_to_facets[:, :, 1]

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

        # The Neumann integral may still be present but it's zero
        J = expand_derivatives(J)
        # Assume that every facet integral in the Jacobian imposes a
        # weak Dirichlet BC on all components
        # TODO add support for weak component BCs
        # FIXME for variable layers there is inconsistency between
        # ds_t/ds_b and DirichletBC(V, ubc, "top/bottom").
        # the labels here are the ones that DirichletBC would use
        for it in J.integrals():
            itype = it.integral_type()
            if itype.startswith("exterior_facet"):
                index = ()
                labels = comp.get(index, ())
                bs = it.subdomain_id()
                if bs == "everywhere":
                    if itype == "exterior_facet_bottom":
                        labels += (-2,)
                    elif itype == "exterior_facet_top":
                        labels += (-4,)
                    else:
                        maskall.append(index)
                else:
                    labels += bs if type(bs) == tuple else (bs,)
                comp[index] = labels

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
