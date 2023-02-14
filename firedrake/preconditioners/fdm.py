from functools import lru_cache, partial
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks
import firedrake
import numpy
import ufl
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

    @PETSc.Log.EventDecorator("FDMInit")
    def initialize(self, pc):
        from firedrake.assemble import allocate_matrix, assemble
        from firedrake.preconditioners.pmg import prolongation_matrix_matfree
        from firedrake.preconditioners.patch import bcdofs
        Citations().register("Brubeck2021")

        self.comm = pc.comm

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        # Get original Jacobian form and bcs
        octx = dmhooks.get_appctx(pc.getDM())
        mat_type = octx.mat_type
        oproblem = octx._problem
        J = oproblem.J
        bcs = tuple(oproblem.bcs)

        # Transform the problem into the space with FDM shape functions
        V = J.arguments()[0].function_space()
        element = V.ufl_element()
        e_fdm = element.reconstruct(variant="fdm")

        def interp_nullspace(I, nsp):
            if not nsp:
                return nsp
            vectors = []
            for x in nsp.getVecs():
                y = I.createVecLeft()
                I.mult(x, y)
                vectors.append(y)
            if nsp.hasConstant():
                y = I.createVecLeft()
                x = I.createVecRight()
                x.set(1.0E0)
                I.mult(x, y)
                vectors.append(y)
                x.destroy()
            return PETSc.NullSpace().create(constant=False, vectors=vectors, comm=nsp.getComm())

        # Matrix-free assembly of the transformed Jacobian
        if element == e_fdm:
            V_fdm, J_fdm, bcs_fdm = (V, J, bcs)
            Amat, _ = pc.getOperators()
            self._ctx_ref = octx
        else:
            V_fdm = firedrake.FunctionSpace(V.mesh(), e_fdm)
            J_fdm = ufl.replace(J, {t: t.reconstruct(function_space=V_fdm) for t in J.arguments()})
            bcs_fdm = tuple(bc.reconstruct(V=V_fdm) for bc in bcs)
            self.fdm_interp = prolongation_matrix_matfree(V, V_fdm, [], bcs_fdm)
            self.A = allocate_matrix(J_fdm, bcs=bcs_fdm, form_compiler_parameters=fcp, mat_type=mat_type,
                                     options_prefix=options_prefix)
            self._assemble_A = partial(assemble, J_fdm, tensor=self.A, bcs=bcs_fdm,
                                       form_compiler_parameters=fcp, mat_type=mat_type)
            self._assemble_A()
            Amat = self.A.petscmat

            omat, _ = pc.getOperators()
            inject = prolongation_matrix_matfree(V_fdm, V, [], [])
            Amat.setNullSpace(interp_nullspace(inject, omat.getNullSpace()))
            Amat.setTransposeNullSpace(interp_nullspace(inject, omat.getTransposeNullSpace()))
            Amat.setNearNullSpace(interp_nullspace(inject, omat.getNearNullSpace()))
            self.work_vec_x = Amat.createVecLeft()
            self.work_vec_y = Amat.createVecRight()

            self._ctx_ref = self.new_snes_ctx(pc, J_fdm, bcs_fdm, mat_type,
                                              fcp=fcp, options_prefix=options_prefix)

        if len(bcs) > 0:
            self.bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs]))
        else:
            self.bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

        # Assemble the FDM preconditioner with sparse local matrices
        Pmat, self._assemble_P = self.assemble_fdm_op(V_fdm, J_fdm, bcs_fdm, appctx)
        self._assemble_P()
        Pmat.setNullSpace(Amat.getNullSpace())
        Pmat.setTransposeNullSpace(Amat.getTransposeNullSpace())
        Pmat.setNearNullSpace(Amat.getNearNullSpace())

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -fdm_pc_type ksp.
        fdmpc = PETSc.PC().create(comm=pc.comm)
        fdmpc.incrementTabLevel(1, parent=pc)

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. multigrid or patch solves.
        fdm_dm = V_fdm.dm
        self._dm = fdm_dm

        fdmpc.setDM(fdm_dm)
        fdmpc.setOptionsPrefix(options_prefix)
        fdmpc.setOperators(A=Amat, P=Pmat)
        fdmpc.setUseAmat(True)
        self.pc = fdmpc

        with dmhooks.add_hooks(fdm_dm, self, appctx=self._ctx_ref, save=False):
            fdmpc.setFromOptions()

    @PETSc.Log.EventDecorator("FDMUpdate")
    def update(self, pc):
        if hasattr(self, "A"):
            self._assemble_A()
        self._assemble_P()

    def apply(self, pc, x, y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            if hasattr(self, "fdm_interp"):
                self.fdm_interp.multTranspose(x, self.work_vec_x)
                self.pc.apply(self.work_vec_x, self.work_vec_y)
                self.fdm_interp.mult(self.work_vec_y, y)
                y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]
            else:
                self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            if hasattr(self, "fdm_interp"):
                self.fdm_interp.multTranspose(x, self.work_vec_y)
                self.pc.applyTranspose(self.work_vec_y, self.work_vec_x)
                self.fdm_interp.mult(self.work_vec_x, y)
                y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]
            else:
                self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super(FDMPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    def assemble_fdm_op(self, V, J, bcs, appctx):
        """
        Assemble the sparse preconditioner with cell-wise constant coefficients.

        :arg V: the :class:`~.FunctionSpace` of the form arguments
        :arg J: the Jacobian bilinear form
        :arg bcs: an iterable of boundary conditions on V
        :arg appctx: the application context

        :returns: 2-tuple with the preconditioner :class:`PETSc.Mat` and its assembly callable
        """
        from pyop2.sparsity import get_preallocation
        from firedrake.preconditioners.pmg import get_line_elements
        try:
            line_elements = get_line_elements(V)
        except ValueError:
            raise ValueError("FDMPC does not support the element %s" % V.ufl_element())

        degree = max(e.degree() for e in line_elements)
        eta = float(appctx.get("eta", (degree+1)**2))
        quad_degree = 2*degree+1
        element = V.finat_element
        is_dg = element.entity_dofs() == element.entity_closure_dofs()

        Afdm = []  # sparse interval mass and stiffness matrices for each direction
        Dfdm = []  # tabulation of normal derivatives at the boundary for each direction
        bdof = []  # indices of point evaluation dofs for each direction
        for e in line_elements:
            Afdm[:0], Dfdm[:0], bdof[:0] = tuple(zip(fdm_setup_ipdg(e, eta)))
            if not (e.formdegree or is_dg):
                Dfdm[0] = None

        # coefficients w.r.t. the reference values
        coefficients, self.assembly_callables = self.assemble_coef(J, quad_degree)
        # set arbitrary non-zero coefficients for preallocation
        for coef in coefficients.values():
            with coef.dat.vec as cvec:
                cvec.set(1.0E0)

        bcflags = get_weak_bc_flags(J)

        # preallocate by calling the assembly routine on a PREALLOCATOR Mat
        sizes = (V.dof_dset.layout_vec.getSizes(),)*2
        block_size = V.dof_dset.layout_vec.getBlockSize()
        prealloc = PETSc.Mat().create(comm=self.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(sizes)
        prealloc.setUp()
        self.assemble_kron(prealloc, V, bcs, eta, coefficients, Afdm, Dfdm, bdof, bcflags)
        nnz = get_preallocation(prealloc, block_size * V.dof_dset.set.size)
        Pmat = PETSc.Mat().createAIJ(sizes, block_size, nnz=nnz, comm=self.comm)
        Pmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        assemble_P = partial(self.assemble_kron, Pmat, V, bcs, eta,
                             coefficients, Afdm, Dfdm, bdof, bcflags)
        prealloc.destroy()
        return Pmat, assemble_P

    def assemble_kron(self, A, V, bcs, eta, coefficients, Afdm, Dfdm, bdof, bcflags):
        """
        Assemble the stiffness matrix in the FDM basis using Kronecker products of interval matrices

        :arg A: the :class:`PETSc.Mat` to assemble
        :arg V: the :class:`~.FunctionSpace` of the form arguments
        :arg bcs: an iterable of :class:`~.DirichletBC` s
        :arg eta: a ``float`` penalty parameter for the symmetric interior penalty method
        :arg coefficients: a ``dict`` mapping strings to :class:`firedrake.function.Function` s with the form coefficients
        :arg Afdm: the list with sparse interval matrices
        :arg Dfdm: the list with normal derivatives matrices
        :arg bcflags: the :class:`numpy.ndarray` with BC facet flags returned by ``get_weak_bc_flags``
        """
        from firedrake.preconditioners.pmg import get_axes_shift
        Gq = coefficients.get("Gq")
        Bq = coefficients.get("Bq")
        Gq_facet = coefficients.get("Gq_facet")
        PT_facet = coefficients.get("PT_facet")

        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map(bcs)

        bsize = V.value_size
        ncomp = V.ufl_element().reference_value_size()
        sdim = (V.finat_element.space_dimension() * bsize) // ncomp  # dimension of a single component
        ndim = V.ufl_domain().topological_dimension()
        shift = get_axes_shift(V.finat_element) % ndim

        index_cell, nel = glonum_fun(V.cell_node_map())
        index_coef, _ = glonum_fun(Gq.cell_node_map())
        flag2id = numpy.kron(numpy.eye(ndim, ndim, dtype=PETSc.IntType), [[1], [2]])

        # pshape is the shape of the DOFs in the tensor product
        pshape = tuple(Ak[0].size[0] for Ak in Afdm)
        if shift:
            assert ncomp == ndim
            pshape = [tuple(numpy.roll(pshape, -shift*k)) for k in range(ncomp)]

        if A.getType() != PETSc.Mat.Type.PREALLOCATOR:
            A.zeroEntries()
            for assemble_coef in self.assembly_callables:
                assemble_coef()

        # insert the identity in the Dirichlet rows and columns
        for row in V.dof_dset.lgmap.indices[lgmap.indices < 0]:
            A.setValue(row, row, 1.0E0, imode)

        # assemble zero-th order term separately, including off-diagonals (mixed components)
        # I cannot do this for hdiv elements as off-diagonals are not sparse, this is because
        # the FDM eigenbases for GLL(N) and GLL(N-1) are not orthogonal to each other
        use_diag_Bq = Bq is None or len(Bq.ufl_shape) != 2
        if not use_diag_Bq:
            bshape = Bq.ufl_shape
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
                rows = lgmap.apply(ie)
                set_submat_csr(A, Ae, rows, imode)
                Ae.destroy()
            Be.destroy()
            Bq = None

        # assemble the second order term and the zero-th order term if any,
        # discarding mixed derivatives and mixed components
        for e in range(nel):
            ie = numpy.reshape(index_cell(e), (ncomp//bsize, -1))
            je = index_coef(e)
            bce = bcflags[e]

            # get second order coefficient on this cell
            mue = numpy.atleast_1d(numpy.sum(Gq.dat.data_ro[je], axis=0))
            if Bq is not None:
                # get zero-th order coefficient on this cell
                bqe = numpy.atleast_1d(numpy.sum(Bq.dat.data_ro[je], axis=0))

            for k in range(ncomp):
                # permutation of axes with respect to the first vector component
                axes = numpy.roll(numpy.arange(ndim), -shift*k)
                # for each component: compute the stiffness matrix Ae
                muk = mue[k] if len(mue.shape) == 2 else mue
                bck = bce[:, k] if len(bce.shape) == 2 else bce
                fbc = numpy.dot(bck, flag2id)

                # Ae = mue[k][0] Ahat + bqe[k] Bhat
                Be = Afdm[axes[0]][0].copy()
                Ae = Afdm[axes[0]][1+fbc[0]].copy()
                Ae.scale(muk[0])
                if Bq is not None:
                    Ae.axpy(bqe[k], Be)

                if ndim > 1:
                    # Ae = Ae kron Bhat + mue[k][1] Bhat kron Ahat
                    Ae = Ae.kron(Afdm[axes[1]][0])
                    Ae.axpy(muk[1], Be.kron(Afdm[axes[1]][1+fbc[1]]))
                    if ndim > 2:
                        # Ae = Ae kron Bhat + mue[k][2] Bhat kron Bhat kron Ahat
                        Be = Be.kron(Afdm[axes[1]][0])
                        Ae = Ae.kron(Afdm[axes[2]][0])
                        Ae.axpy(muk[2], Be.kron(Afdm[axes[2]][1+fbc[2]]))

                rows = lgmap.apply(ie[0]*bsize+k if bsize == ncomp else ie[k])
                set_submat_csr(A, Ae, rows, imode)
                Ae.destroy()
                Be.destroy()

        # assemble SIPG interior facet terms if the normal derivatives have been set up
        if any(Dk is not None for Dk in Dfdm):
            if ndim < V.ufl_domain().geometric_dimension():
                raise NotImplementedError("SIPG on immersed meshes is not implemented")
            index_facet, local_facet_data, nfacets = get_interior_facet_maps(V)
            index_coef, _, _ = get_interior_facet_maps(Gq_facet or Gq)
            rows = numpy.zeros((2, sdim), dtype=PETSc.IntType)

            for e in range(nfacets):
                # for each interior facet: compute the SIPG stiffness matrix Ae
                ie = index_facet(e)
                je = numpy.reshape(index_coef(e), (2, -1))
                lfd = local_facet_data(e)
                idir = lfd // 2

                if PT_facet:
                    icell = numpy.reshape(lgmap.apply(ie), (2, ncomp, -1))
                    iord0 = numpy.insert(numpy.delete(numpy.arange(ndim), idir[0]), 0, idir[0])
                    iord1 = numpy.insert(numpy.delete(numpy.arange(ndim), idir[1]), 0, idir[1])
                    je = je[[0, 1], lfd]
                    Pfacet = PT_facet.dat.data_ro_with_halos[je]
                    Gfacet = Gq_facet.dat.data_ro_with_halos[je]
                else:
                    Gfacet = numpy.sum(Gq.dat.data_ro_with_halos[je], axis=1)

                for k in range(ncomp):
                    axes = numpy.roll(numpy.arange(ndim), -shift*k)
                    Dfacet = Dfdm[axes[0]]
                    if Dfacet is None:
                        continue

                    if PT_facet:
                        k0 = iord0[k] if shift != 1 else ndim-1-iord0[-k-1]
                        k1 = iord1[k] if shift != 1 else ndim-1-iord1[-k-1]
                        Piola = Pfacet[[0, 1], [k0, k1]]
                        mu = Gfacet[[0, 1], idir]
                    else:
                        if len(Gfacet.shape) == 3:
                            mu = Gfacet[[0, 1], [k, k], idir]
                        elif len(Gfacet.shape) == 2:
                            mu = Gfacet[[0, 1], idir]
                        else:
                            mu = Gfacet

                    offset = Dfacet.shape[0]
                    Adense = numpy.zeros((2*offset, 2*offset), dtype=PETSc.RealType)
                    dense_indices = []
                    for j, jface in enumerate(lfd):
                        j0 = j * offset
                        j1 = j0 + offset
                        jj = j0 + bdof[axes[0]][jface % 2]
                        dense_indices.append(jj)
                        for i, iface in enumerate(lfd):
                            i0 = i * offset
                            i1 = i0 + offset
                            ii = i0 + bdof[axes[0]][iface % 2]
                            sij = 0.5E0 if i == j else -0.5E0
                            if PT_facet:
                                smu = [sij*numpy.dot(numpy.dot(mu[0], Piola[i]), Piola[j]),
                                       sij*numpy.dot(numpy.dot(mu[1], Piola[i]), Piola[j])]
                            else:
                                smu = sij*mu

                            Adense[ii, jj] += eta * sum(smu)
                            Adense[i0:i1, jj] -= smu[i] * Dfacet[:, iface % 2]
                            Adense[ii, j0:j1] -= smu[j] * Dfacet[:, jface % 2]

                    Ae = numpy_to_petsc(Adense, dense_indices, diag=False)
                    if ndim > 1:
                        # assume that the mesh is oriented
                        Ae = Ae.kron(Afdm[axes[1]][0])
                        if ndim > 2:
                            Ae = Ae.kron(Afdm[axes[2]][0])

                    if bsize == ncomp:
                        icell = numpy.reshape(lgmap.apply(k+bsize*ie), (2, -1))
                        rows[0] = pull_axis(icell[0], pshape, idir[0])
                        rows[1] = pull_axis(icell[1], pshape, idir[1])
                    else:
                        assert pshape[k0][idir[0]] == pshape[k1][idir[1]]
                        rows[0] = pull_axis(icell[0][k0], pshape[k0], idir[0])
                        rows[1] = pull_axis(icell[1][k1], pshape[k1], idir[1])

                    set_submat_csr(A, Ae, rows, imode)
                    Ae.destroy()
        A.assemble()

    def assemble_coef(self, J, quad_deg, discard_mixed=True, cell_average=True):
        """
        Return the coefficients of the Jacobian form arguments and their gradient with respect to the reference coordinates.

        :arg J: the Jacobian bilinear form
        :arg quad_deg: the quadrature degree used for the coefficients
        :arg discard_mixed: discard entries in second order coefficient with mixed derivatives and mixed components
        :arg cell_average: to return the coefficients as DG_0 Functions

        :returns: a 2-tuple of
            coefficients: a dictionary mapping strings to :class:`firedrake.function.Function` s with the coefficients of the form,
            assembly_callables: a list of assembly callables for each coefficient of the form
        """
        from ufl import inner, diff
        from ufl.algorithms.ad import expand_derivatives
        coefficients = {}
        assembly_callables = []

        mesh = J.ufl_domain()
        tdim = mesh.topological_dimension()
        Finv = ufl.JacobianInverse(mesh)
        dx = firedrake.dx(degree=quad_deg)

        if cell_average:
            family = "Discontinuous Lagrange" if tdim == 1 else "DQ"
            degree = 0
        else:
            family = "Quadrature"
            degree = quad_deg

        # extract coefficients directly from the bilinear form
        args_J = J.arguments()
        integrals_J = J.integrals_by_type("cell")
        mapping = args_J[0].ufl_element().mapping().lower()
        Piola = get_piola_tensor(mapping, mesh)

        # get second order coefficient
        ref_grad = [ufl.variable(ufl.grad(t)) for t in args_J]
        if Piola:
            replace_grad = {ufl.grad(t): ufl.dot(Piola, ufl.dot(dt, Finv)) for t, dt in zip(args_J, ref_grad)}
        else:
            replace_grad = {ufl.grad(t): ufl.dot(dt, Finv) for t, dt in zip(args_J, ref_grad)}

        alpha = expand_derivatives(sum([diff(diff(ufl.replace(i.integrand(), replace_grad),
                                             ref_grad[0]), ref_grad[1]) for i in integrals_J]))

        # get zero-th order coefficent
        ref_val = [ufl.variable(t) for t in args_J]
        if Piola:
            dummy_element = ufl.TensorElement("DQ", cell=mesh.ufl_cell(), degree=1, shape=Piola.ufl_shape)
            dummy_Piola = ufl.Coefficient(ufl.FunctionSpace(mesh, dummy_element))
            replace_val = {t: ufl.dot(dummy_Piola, s) for t, s in zip(args_J, ref_val)}
        else:
            replace_val = {t: s for t, s in zip(args_J, ref_val)}

        beta = expand_derivatives(sum([diff(diff(ufl.replace(i.integrand(), replace_val),
                                            ref_val[0]), ref_val[1]) for i in integrals_J]))
        if Piola:
            beta = ufl.replace(beta, {dummy_Piola: Piola})

        G = alpha
        if discard_mixed:
            # discard mixed derivatives and mixed components
            if len(G.ufl_shape) == 2:
                G = ufl.diag_vector(G)
            else:
                Gshape = G.ufl_shape
                Gshape = Gshape[:len(Gshape)//2]
                G = ufl.as_tensor(numpy.reshape([G[i+i] for i in numpy.ndindex(Gshape)], (Gshape[0], -1)))
            Qe = ufl.TensorElement(family, mesh.ufl_cell(), degree=degree, quad_scheme="default", shape=G.ufl_shape)
        else:
            Qe = ufl.TensorElement(family, mesh.ufl_cell(), degree=degree, quad_scheme="default", shape=G.ufl_shape, symmetry=True)

        # assemble second order coefficient
        Q = firedrake.FunctionSpace(mesh, Qe)
        q = firedrake.TestFunction(Q)
        Gq = firedrake.Function(Q)
        coefficients["Gq"] = Gq
        assembly_callables.append(partial(firedrake.assemble, inner(G, q)*dx, Gq))

        # assemble zero-th order coefficient
        if not isinstance(beta, ufl.constantvalue.Zero):
            if Piola:
                # keep diagonal
                beta = ufl.diag_vector(beta)
            shape = beta.ufl_shape
            Qe = ufl.FiniteElement(family, mesh.ufl_cell(), degree=degree, quad_scheme="default")
            if shape:
                Qe = ufl.TensorElement(Qe, shape=shape)
            Q = firedrake.FunctionSpace(mesh, Qe)
            q = firedrake.TestFunction(Q)
            Bq = firedrake.Function(Q)
            coefficients["Bq"] = Bq
            assembly_callables.append(partial(firedrake.assemble, inner(beta, q)*dx, Bq))

        if Piola:
            # make DGT functions with the second order coefficient
            # and the Piola tensor for each side of each facet
            extruded = mesh.cell_set._extruded
            dS_int = firedrake.dS_h(degree=quad_deg) + firedrake.dS_v(degree=quad_deg) if extruded else firedrake.dS(degree=quad_deg)
            ele = ufl.BrokenElement(ufl.FiniteElement("DGT", mesh.ufl_cell(), 0))
            area = ufl.FacetArea(mesh)

            replace_grad = {ufl.grad(t): ufl.dot(dt, Finv) for t, dt in zip(args_J, ref_grad)}
            alpha = expand_derivatives(sum([diff(diff(ufl.replace(i.integrand(), replace_grad),
                                                 ref_grad[0]), ref_grad[1]) for i in integrals_J]))
            vol = abs(ufl.JacobianDeterminant(mesh))
            G = vol * alpha
            G = ufl.as_tensor([[[G[i, k, j, k] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[2])] for k in range(G.ufl_shape[3])])

            Q = firedrake.TensorFunctionSpace(mesh, ele, shape=G.ufl_shape)
            q = firedrake.TestFunction(Q)
            Gq_facet = firedrake.Function(Q)
            coefficients["Gq_facet"] = Gq_facet
            assembly_callables.append(partial(firedrake.assemble, ((inner(q('+'), G('+')) + inner(q('-'), G('-')))/area)*dS_int, Gq_facet))

            PT = Piola.T
            Q = firedrake.TensorFunctionSpace(mesh, ele, shape=PT.ufl_shape)
            q = firedrake.TestFunction(Q)
            PT_facet = firedrake.Function(Q)
            coefficients["PT_facet"] = PT_facet
            assembly_callables.append(partial(firedrake.assemble, ((inner(q('+'), PT('+')) + inner(q('-'), PT('-')))/area)*dS_int, PT_facet))
        return coefficients, assembly_callables


def get_piola_tensor(mapping, domain):
    tdim = domain.topological_dimension()
    if mapping == 'identity':
        return None
    elif mapping == 'covariant piola':
        return ufl.JacobianInverse(domain).T * ufl.as_tensor(numpy.flipud(numpy.identity(tdim)))
    elif mapping == 'contravariant piola':
        sign = ufl.diag(ufl.as_tensor([-1]+[1]*(tdim-1)))
        return ufl.Jacobian(domain)*sign/ufl.JacobianDeterminant(domain)
    else:
        raise NotImplementedError("Unsupported element mapping %s" % mapping)


def pull_axis(x, pshape, idir):
    """permute x by reshaping into pshape and moving axis idir to the front"""
    return numpy.reshape(numpy.moveaxis(numpy.reshape(x.copy(), pshape), idir, 0), x.shape)


def set_submat_csr(A_global, A_local, global_indices, imode):
    """insert values from A_local to A_global on the diagonal block with indices global_indices"""
    indptr, indices, data = A_local.getValuesCSR()
    for i, row in enumerate(global_indices.flat):
        i0 = indptr[i]
        i1 = indptr[i+1]
        A_global.setValues(row, global_indices.flat[indices[i0:i1]], data[i0:i1], imode)


def numpy_to_petsc(A_numpy, dense_indices, diag=True, block=False):
    """
    Create a SeqAIJ Mat from a dense matrix using the diagonal and a subset of rows and columns.
    If dense_indices is empty, then also include the off-diagonal corners of the matrix.
    """
    n = A_numpy.shape[0]
    nbase = int(diag) if block else min(n, int(diag) + len(dense_indices))
    nnz = numpy.full((n,), nbase, dtype=PETSc.IntType)
    nnz[dense_indices] = len(dense_indices) if block else n

    imode = PETSc.InsertMode.INSERT
    A_petsc = PETSc.Mat().createAIJ(A_numpy.shape, nnz=(nnz, 0), comm=PETSc.COMM_SELF)

    idx = numpy.arange(n, dtype=PETSc.IntType)
    if block:
        values = A_numpy[dense_indices, :][:, dense_indices]
        A_petsc.setValues(dense_indices, dense_indices, values, imode)
    else:
        for j in dense_indices:
            A_petsc.setValues(j, idx, A_numpy[j, :], imode)
            A_petsc.setValues(idx, j, A_numpy[:, j], imode)

    if diag:
        idx = idx[:, None]
        values = A_numpy.diagonal()[:, None]
        A_petsc.setValuesRCV(idx, idx, values, imode)

    A_petsc.assemble()
    return A_petsc


@lru_cache(maxsize=10)
def fdm_setup_ipdg(fdm_element, eta):
    """
    Setup for the fast diagonalization method for the IP-DG formulation.
    Compute sparsified interval stiffness and mass matrices
    and tabulate the normal derivative of the shape functions.

    :arg fdm_element: a :class:`FIAT.FDMElement`
    :arg eta: penalty coefficient as a `float`

    :returns: 3-tuple of:
        Afdm: a list of :class:`PETSc.Mats` with the sparse interval matrices
        Bhat, and bcs(Ahat) for every combination of either natural or weak
        Dirichlet BCs on each endpoint.
        Dfdm: the tabulation of the normal derivatives of the Dirichlet eigenfunctions.
        bdof: the indices of PointEvaluation dofs.
    """
    from FIAT.quadrature import GaussLegendreQuadratureLineRule
    from FIAT.functional import PointEvaluation
    ref_el = fdm_element.get_reference_element()
    degree = fdm_element.degree()
    rule = GaussLegendreQuadratureLineRule(ref_el, degree+1)
    bdof = [k for k, f in enumerate(fdm_element.dual_basis()) if isinstance(f, PointEvaluation)]

    phi = fdm_element.tabulate(1, rule.get_points())
    Jhat = phi[(0, )]
    Dhat = phi[(1, )]
    Ahat = numpy.dot(numpy.multiply(Dhat, rule.get_weights()), Dhat.T)
    Bhat = numpy.dot(numpy.multiply(Jhat, rule.get_weights()), Jhat.T)

    # Facet normal derivatives
    basis = fdm_element.tabulate(1, ref_el.get_vertices())
    Dfacet = basis[(1,)]
    Dfacet[:, 0] = -Dfacet[:, 0]

    Afdm = [numpy_to_petsc(Bhat, bdof, block=True)]
    for bc in range(4):
        bcs = (bc % 2, bc//2)
        Abc = Ahat.copy()
        for k in range(2):
            if bcs[k] == 1:
                j = bdof[k]
                Abc[:, j] -= Dfacet[:, k]
                Abc[j, :] -= Dfacet[:, k]
                Abc[j, j] += eta
        Afdm.append(numpy_to_petsc(Abc, bdof))
    return Afdm, Dfacet, bdof


@lru_cache(maxsize=10)
def get_interior_facet_maps(V):
    """
    Extrude V.interior_facet_node_map and V.ufl_domain().interior_facets.local_facet_dat

    :arg V: a :class:`~.FunctionSpace`

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


def glonum(node_map):
    """
    Return an array with the nodes of each topological entity of a certain kind.

    :arg node_map: a :class:`pyop2.Map` mapping entities to their nodes, including ghost entities.

    :returns: a :class:`numpy.ndarray` whose rows are the nodes for each cell
    """
    if (node_map.offset is None) or (node_map.values_with_halo.size == 0):
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


def get_weak_bc_flags(J):
    """
    Return flags indicating whether the zero-th order coefficient on each facet of every cell is non-zero
    """
    from ufl.algorithms.ad import expand_derivatives
    mesh = J.ufl_domain()
    args_J = J.arguments()
    V = args_J[0].function_space()
    rvs = V.ufl_element().reference_value_shape()
    cell = mesh.ufl_cell()
    family = "CG" if cell.topological_dimension() == 1 else "DGT"
    degree = 1 if cell.topological_dimension() == 1 else 0
    Qe = ufl.FiniteElement(family, cell=cell, degree=degree)
    if rvs:
        Qe = ufl.TensorElement(Qe, shape=rvs)
    Q = firedrake.FunctionSpace(mesh, Qe)
    q = firedrake.TestFunction(Q)

    ref_args = [ufl.variable(t) for t in args_J]
    replace_args = {t: s for t, s in zip(args_J, ref_args)}

    forms = []
    md = {"quadrature_degree": 0}
    for it in J.integrals():
        itype = it.integral_type()
        if itype.startswith("exterior_facet"):
            beta = ufl.diff(ufl.diff(ufl.replace(it.integrand(), replace_args), ref_args[0]), ref_args[1])
            beta = expand_derivatives(beta)
            if rvs:
                beta = ufl.diag_vector(beta)
            ds_ext = ufl.Measure(itype, domain=mesh, subdomain_id=it.subdomain_id(), metadata=md)
            forms.append(ufl.inner(q, beta)*ds_ext)

    tol = 1E-8
    if len(forms):
        bq = firedrake.assemble(sum(forms))
        fbc = bq.dat.data_with_halos[glonum(Q.cell_node_map())]
        return (abs(fbc) > tol).astype(PETSc.IntType)
    else:
        return numpy.zeros(glonum(Q.cell_node_map()).shape, dtype=PETSc.IntType)
