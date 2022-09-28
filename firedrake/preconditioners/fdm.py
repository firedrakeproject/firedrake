from functools import partial, lru_cache
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks
import firedrake
import numpy
import ufl
import ctypes
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

__all__ = ("FDMPC", "PoissonFDMPC")


class FDMPC(PCBase):
    """
    A preconditioner for tensor-product elements that changes the shape
    functions so that the H(d) Riesz map is sparse in the interior of a
    Cartesian cell, and assembles a global sparse matrix on which other
    preconditioners, such as `ASMStarPC`, can be applied.

    Here we assume that the volume integrals in the Jacobian can be expressed as:

    inner(d(v), alpha(d(u)))*dx + inner(v, beta(u))*dx

    where alpha and beta are linear functions (tensor contractions).
    The sparse matrix is obtained by approximating (v, alpha u) and (v, beta u) as
    diagonal mass matrices
    """

    _prefix = "fdm_"

    _variant = "fdm_feec"

    @PETSc.Log.EventDecorator("FDMInit")
    def initialize(self, pc):
        from firedrake.assemble import allocate_matrix, assemble
        from firedrake.preconditioners.pmg import prolongation_matrix_matfree
        from firedrake.preconditioners.patch import bcdofs
        Citations().register("Brubeck2021")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        options = PETSc.Options(options_prefix)
        use_amat = options.getBool("pc_use_amat", True)
        use_ainv = options.getString("pc_type", "") == "mat"
        self.use_ainv = use_ainv

        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        # Get original Jacobian form and bcs
        octx = dmhooks.get_appctx(pc.getDM())
        mat_type = octx.mat_type
        oproblem = octx._problem

        J = oproblem.J
        if isinstance(J, firedrake.slate.Add):
            J = J.children[0].form
        assert type(J) == ufl.Form

        bcs = tuple(oproblem.bcs)

        # Transform the problem into the space with FDM shape functions
        V = J.arguments()[0].function_space()
        element = V.ufl_element()
        e_fdm = element.reconstruct(variant=self._variant)

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
        else:
            V_fdm = firedrake.FunctionSpace(V.mesh(), e_fdm)
            J_fdm = ufl.replace(J, {t: t.reconstruct(function_space=V_fdm) for t in J.arguments()})
            bcs_fdm = tuple(bc.reconstruct(V=V_fdm, g=0) for bc in bcs)
            self.fdm_interp = prolongation_matrix_matfree(V, V_fdm, [], bcs_fdm)
            Amat = None
            omat, _ = pc.getOperators()
            if use_amat:
                self.A = allocate_matrix(J_fdm, bcs=bcs_fdm, form_compiler_parameters=fcp, mat_type=mat_type,
                                         options_prefix=options_prefix)
                self._assemble_A = partial(assemble, J_fdm, tensor=self.A, bcs=bcs_fdm,
                                           form_compiler_parameters=fcp, mat_type=mat_type)
                self._assemble_A()
                Amat = self.A.petscmat
                inject = prolongation_matrix_matfree(V_fdm, V, [], [])
                Amat.setNullSpace(interp_nullspace(inject, omat.getNullSpace()))
                Amat.setTransposeNullSpace(interp_nullspace(inject, omat.getTransposeNullSpace()))
                Amat.setNearNullSpace(interp_nullspace(inject, omat.getNearNullSpace()))

            self.work_vec_x = omat.createVecLeft()
            self.work_vec_y = omat.createVecRight()

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
        fdmpc.setUseAmat(use_amat)
        self.pc = fdmpc

        with dmhooks.add_hooks(fdm_dm, self, appctx=self._ctx_ref, save=False):
            fdmpc.setFromOptions()

    def assemble_fdm_op(self, V, J, bcs, appctx):
        """
        Assemble the sparse preconditioner with cell-wise constant coefficients.

        :arg V: the :class:`firedrake.FunctionSpace` of the form arguments
        :arg J: the Jacobian bilinear form
        :arg bcs: an iterable of boundary conditions on V
        :arg appctx: the application context

        :returns: 2-tuple with the preconditioner :class:`PETSc.Mat` and its assembly callable
        """
        from pyop2.sparsity import get_preallocation

        self.is_interior_element = True
        self.is_facet_element = True
        entity_dofs = V.finat_element.entity_dofs()
        ndim = V.mesh().topological_dimension()
        for key in entity_dofs:
            v = sum(list(entity_dofs[key].values()), [])
            if len(v):
                edim = key
                try:
                    edim = sum(edim)
                except TypeError:
                    pass
                if edim == ndim:
                    self.is_facet_element = False
                else:
                    self.is_interior_element = False

        Vbig = V
        _, fdofs = split_dofs(V.finat_element)
        if self.is_facet_element:
            Vbig = firedrake.FunctionSpace(V.mesh(), unrestrict_element(V.ufl_element()))
            fdofs = restricted_dofs(V.finat_element, Vbig.finat_element)

        fdofs = numpy.add.outer(V.value_size*fdofs, numpy.arange(V.value_size, dtype=fdofs.dtype))
        dofs = numpy.arange(V.value_size*Vbig.finat_element.space_dimension(), dtype=fdofs.dtype)
        self.idofs = PETSc.IS().createGeneral(numpy.setdiff1d(dofs, fdofs, assume_unique=True), comm=PETSc.COMM_SELF)
        self.fdofs = PETSc.IS().createGeneral(fdofs, comm=PETSc.COMM_SELF)
        self.submats = [None for _ in range(7)]

        if self.is_interior_element:
            self.condense_element_mat = lambda Ae: Ae
        elif self.is_facet_element:
            self.condense_element_mat = lambda Ae: condense_element_mat(Ae, self.idofs, self.fdofs, self.submats)
        elif V.finat_element.formdegree == 0:
            i1 = PETSc.IS().createGeneral(dofs, comm=PETSc.COMM_SELF)
            self.condense_element_mat = lambda Ae: condense_element_pattern(Ae, self.idofs, i1, self.submats)
        else:
            self.condense_element_mat = lambda Ae: Ae

        addv = PETSc.InsertMode.ADD_VALUES
        _update_A = load_assemble_csr()
        _set_bc_values = load_set_bc_values()
        self.update_A = lambda A, B, rows: _update_A(A, B, rows, rows, addv)
        self.set_bc_values = lambda A, rows: _set_bc_values(A, rows.size, rows, addv)

        Afdm, Dfdm, quad_degree, eta = self.assemble_reference_tensors(Vbig, appctx)

        # coefficients w.r.t. the reference values
        coefficients, self.assembly_callables = self.assemble_coef(J, quad_degree)
        bcflags = None
        if eta:
            coefficients["eta"] = eta
            bcflags = get_weak_bc_flags(J)

        # preallocate by calling the assembly routine on a PREALLOCATOR Mat
        sizes = (V.dof_dset.layout_vec.getSizes(),)*2
        block_size = V.dof_dset.layout_vec.getBlockSize()
        prealloc = PETSc.Mat().create(comm=V.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(sizes)
        prealloc.setUp()
        prealloc.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
        self.assemble_kron(prealloc, V, bcs, coefficients, Afdm, Dfdm, bcflags)
        nnz = get_preallocation(prealloc, block_size * V.dof_dset.set.size)

        Pmat = PETSc.Mat().createAIJ(sizes, block_size, nnz=nnz, comm=V.comm)
        Pmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        assemble_P = partial(self.assemble_kron, Pmat, V, bcs,
                             coefficients, Afdm, Dfdm, bcflags)
        prealloc.destroy()
        return Pmat, assemble_P

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

    @PETSc.Log.EventDecorator("FDMRefTensor")
    def assemble_reference_tensors(self, V, appctx):
        import FIAT
        ndim = V.mesh().topological_dimension()
        formdegree = V.finat_element.formdegree
        degree = V.finat_element.degree
        try:
            degree = max(degree)
        except TypeError:
            pass
        if formdegree == ndim:
            degree = degree+1

        elements = sorted(get_base_elements(V.finat_element), key=lambda e: e.formdegree)
        cell = elements[0].get_reference_element()
        e0 = elements[0] if elements[0].formdegree == 0 else FIAT.FDMLagrange(cell, degree)
        e1 = elements[-1] if elements[-1].formdegree == 1 else FIAT.FDMDiscontinuousLagrange(cell, degree-1)

        if self.is_interior_element:
            e0 = FIAT.RestrictedElement(e0, restriction_domain="interior")
            eq = e0
        else:
            eq = FIAT.FDMQuadrature(cell, degree)

        quad_degree = 2*degree+1
        rule = FIAT.make_quadrature(cell, degree+1)
        pts = rule.get_points()
        wts = rule.get_weights()
        phi0 = e0.tabulate(1, pts)
        phi1 = e1.tabulate(0, pts)
        phiq = eq.tabulate(0, pts)
        moments = lambda v, u: numpy.dot(numpy.multiply(v, wts), u.T)
        A10 = moments(phi1[(0, )], phi0[(1, )])
        A11 = moments(phi1[(0, )], phi1[(0, )])
        A00 = moments(phiq[(0, )], phi0[(0, )])

        Qhat = mass_matrix(ndim, formdegree, A00, A11)
        Dhat = diff_matrix(ndim, formdegree, A00, A11, A10)
        Afdm = [block_mat([[Qhat], [Dhat]]).kron(petsc_sparse(numpy.eye(V.value_size)))]
        return Afdm, [], quad_degree, None

    def assemble_coef(self, J, quad_deg, discard_mixed=True, cell_average=True):
        """
        Obtain coefficients as the diagonal of a weighted mass matrix in V^k x V^{k+1}
        """
        from ufl.algorithms.ad import expand_derivatives

        mesh = J.ufl_domain()
        ndim = mesh.topological_dimension()
        args_J = J.arguments()
        e = args_J[0].ufl_element()
        if isinstance(e, (ufl.VectorElement, ufl.TensorElement)):
            e = e._sub_element
        e = unrestrict_element(e)
        sobolev = e.sobolev_space()

        V = args_J[0].function_space()
        degree = e.degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        qdeg = degree

        formdegree = V.finat_element.formdegree
        if formdegree == ndim:
            qfam = "DG" if ndim == 1 else "DQ"
            qdeg = 0
        elif formdegree == 0:
            qfam = "DG" if ndim == 1 else "RTCE" if ndim == 2 else "NCE"
        elif formdegree == 1 and ndim == 3:
            qfam = "NCF"
        else:
            qfam = "DQ L2"
            qdeg = degree-1

        interior_element = self.is_interior_element
        qvariant = "fdm_feec" if interior_element else "fdm_quadrature"
        elements = [e.reconstruct(variant=qvariant),
                    ufl.FiniteElement(qfam, cell=mesh.ufl_cell(), degree=qdeg, variant=qvariant)]

        elements = list(map(ufl.InteriorElement if interior_element else ufl.BrokenElement, elements))

        pbjacobi = True
        shape = V.shape
        if shape:
            if pbjacobi:
                shape = shape*2
            elements = [ufl.TensorElement(ele, shape=shape) for ele in elements]
        else:
            pbjacobi = False

        map_grad = None
        if sobolev == ufl.H1:
            map_grad = lambda p: p
        elif sobolev in [ufl.HCurl, ufl.HDiv]:
            u = ufl.Coefficient(ufl.FunctionSpace(mesh, e))
            du = ufl.variable(ufl.grad(u))
            dku = ufl.div(u) if sobolev == ufl.HDiv else ufl.curl(u)
            eps = expand_derivatives(ufl.diff(ufl.replace(expand_derivatives(dku), {ufl.grad(u): du}), du))
            if sobolev == ufl.HDiv:
                map_grad = lambda p: ufl.outer(p, eps/ndim)
            elif len(eps.ufl_shape) == 3:
                map_grad = lambda p: ufl.dot(p, eps/2)
            else:
                map_grad = lambda p: p*(eps/2)

        Jcell = expand_derivatives(ufl.Form(J.integrals_by_type("cell")))

        def make_args(W):
            v = firedrake.TestFunction(W)
            u = firedrake.TrialFunction(W)
            if pbjacobi:
                v = sum([v[:, j, ...] for j in range(v.ufl_shape[1])])
                u = sum([u[i, :, ...] for i in range(u.ufl_shape[0])])
            return v, u

        def bform(*args):
            gdim = Jcell.ufl_domain().geometric_dimension()
            replace_args = {t: v for t, v in zip(args_J, args)}
            replace_grad = {ufl.grad(t): ufl.zero(t.ufl_shape+(gdim,)) for t in args_J} if map_grad else dict()
            return ufl.replace(ufl.replace(Jcell, replace_grad), replace_args)

        def aform(*args):
            replace_args = {t: ufl.zero(t.ufl_shape) for t in args_J}
            replace_grad = {ufl.grad(t): map_grad(q) for t, q in zip(args_J, args)} if map_grad else dict()
            return ufl.replace(ufl.replace(Jcell, replace_grad), replace_args)

        def assembly_callable(form, tensor, diagonal=True):
            return partial(firedrake.assemble, form, tensor=tensor, diagonal=diagonal) if form.integrals() else tensor.dat.zero

        W = [firedrake.FunctionSpace(mesh, e) for e in elements]
        beta = firedrake.Function(W[0], name="beta")
        alpha = firedrake.Function(W[1], name="alpha")
        coefficients = {"beta": beta, "alpha": alpha}
        assembly_callables = [assembly_callable(bform(*make_args(W[0])), beta),
                              assembly_callable(aform(*make_args(W[1])), alpha)]
        return coefficients, assembly_callables

    @PETSc.Log.EventDecorator("FDMAssemble")
    def assemble_kron(self, A, V, bcs, coefficients, Afdm, Bfdm, bcflags):
        lgmap = V.local_to_global_map(bcs)

        if A.getType() != PETSc.Mat.Type.PREALLOCATOR:
            A.zeroEntries()

        self.set_bc_values(A, V.dof_dset.lgmap.indices[lgmap.indices < 0])

        bsize = V.value_size
        nrows = Afdm[0].getSize()[0]
        work_mat = PETSc.Mat().createAIJ((nrows, nrows), nnz=([bsize], [0]), comm=PETSc.COMM_SELF)

        index_cell, nel = glonum_fun(V.cell_node_map())
        if bsize > 1:
            _index_cell = index_cell
            ibase = numpy.arange(bsize, dtype=PETSc.IntType)
            index_cell = lambda e: numpy.add.outer(_index_cell(e)*bsize, ibase)

        Ae = None
        coefs_array = None
        coefs = [coefficients.get(k) for k in ("beta", "alpha")]
        dof_maps = [glonum_fun(ck.cell_node_map())[0] for ck in coefs]
        get_coefs = lambda e, out: numpy.concatenate([coef.dat.data_ro[dof_map(e)] for coef, dof_map in zip(coefs, dof_maps)], out=out)

        if A.getType() != PETSc.Mat.Type.PREALLOCATOR:
            for assemble_coef in self.assembly_callables:
                assemble_coef()
            for e in range(nel):
                coefs_array = get_coefs(e, coefs_array)
                Ae = self.element_mat(coefs_array, Afdm, work_mat, Ae=Ae)
                self.update_A(A, self.condense_element_mat(Ae), lgmap.apply(index_cell(e)))

        elif nel:
            coefs_array = get_coefs(0, coefs_array)
            shape = coefs_array.shape
            if len(shape) > 2:
                coefs_array = numpy.tile(numpy.eye(shape[1]), shape[:1] + (1,)*(len(shape)-1))
            else:
                coefs_array.fill(1.0E0)
            Ae = self.element_mat(coefs_array, Afdm, work_mat, Ae=Ae)
            if self.idofs:
                sort_interior_dofs(self.idofs, Ae)
            Ae = self.condense_element_mat(Ae)
            for e in range(nel):
                self.update_A(A, Ae, lgmap.apply(index_cell(e)))

        A.assemble()

    def element_mat(self, coefs_array, Afdm, work_mat, Ae=None):
        shape = coefs_array.shape
        shape += (1,)*(3-len(shape))
        indptr = numpy.arange(work_mat.getSize()[0]+1, dtype=PETSc.IntType)
        indices = numpy.tile(indptr[:-1].reshape((-1, shape[1])), (1, shape[2]))
        indptr *= shape[2]
        work_mat.zeroEntries()
        work_mat.setValuesCSR(indptr, indices, coefs_array)
        work_mat.assemble()
        Ae = work_mat.PtAP(Afdm[0], result=Ae)
        return Ae


@PETSc.Log.EventDecorator("FDMCondense")
def condense_element_mat(A, i0, i1, submats):
    isrows = [i0, i0, i1, i1]
    iscols = [i0, i1, i0, i1]
    submats[:4] = A.createSubMatrices(isrows, iscols=iscols, submats=submats[:4] if submats[0] else None)
    A00, A01, A10, A11 = submats[:4]

    # Assume that interior DOF list i0 is ordered such that A00 is block diagonal
    # with blocks of increasing dimension
    indptr, indices, data = A00.getValuesCSR()
    degree = numpy.diff(indptr)

    # TODO handle non-symmetric case with LU, requires scipy
    invchol = lambda X: numpy.linalg.inv(numpy.linalg.cholesky(X))
    nblocks = numpy.count_nonzero(degree == 1)
    zlice = slice(0, nblocks)
    numpy.sqrt(data[zlice], out=data[zlice])
    numpy.reciprocal(data[zlice], out=data[zlice])
    PETSc.Log.logFlops(2*nblocks)
    for k in range(2, degree[-1]+1):
        nblocks = numpy.count_nonzero(degree == k)
        zlice = slice(zlice.stop, zlice.stop + k*nblocks)
        data[zlice] = invchol(data[zlice].reshape((-1, k, k))).reshape((-1,))
        flops = ((k+1)**3 + 5*(k+1)-12)//3 + k**3
        PETSc.Log.logFlops(flops*nblocks)

    A00.setValuesCSR(indptr, indices, data)
    A00.assemble()
    submats[4] = A10.matTransposeMult(A00, result=submats[4])
    submats[5] = A00.matMult(A01, result=submats[5])
    submats[6] = submats[4].matMult(submats[5], result=submats[6])
    submats[6].aypx(-1, A11)
    return submats[6]


@PETSc.Log.EventDecorator("FDMCondense")
def condense_element_pattern(A, i0, i1, submats):
    isrows = [i0, i0, i1]
    iscols = [i0, i1, i0]
    submats[:3] = A.createSubMatrices(isrows, iscols=iscols, submats=submats[:3] if submats[0] else None)
    A00, A01, A10 = submats[:3]
    A00.zeroEntries()
    A00.assemble()
    submats[4] = A10.matTransposeMult(A00, result=submats[4])
    submats[5] = A00.matMult(A01, result=submats[5])
    submats[6] = submats[4].matMult(submats[5], result=submats[6])
    submats[6].aypx(-1, A)
    return submats[6]


@PETSc.Log.EventDecorator("LoadCode")
def load_c_code(code, name, **kwargs):
    from pyop2.compilation import load
    from pyop2.utils import get_petsc_dir
    cppargs = ["-I%s/include" % d for d in get_petsc_dir()]
    ldargs = (["-L%s/lib" % d for d in get_petsc_dir()]
              + ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()]
              + ["-lpetsc", "-lm"])
    funptr = load(code, "c", name,
                  cppargs=cppargs, ldargs=ldargs,
                  **kwargs)

    def get_pointer(obj):
        if isinstance(obj, (PETSc.Mat, PETSc.Vec)):
            return obj.handle
        elif isinstance(obj, numpy.ndarray):
            return obj.ctypes.data
        return obj

    @PETSc.Log.EventDecorator(name)
    def wrapper(*args):
        return funptr(*list(map(get_pointer, args)))
    return wrapper


@lru_cache(maxsize=1)
def load_assemble_csr():
    comm = PETSc.COMM_SELF
    code = """
#include <petsc.h>

PetscErrorCode setSubMatCSR(Mat A,
                            Mat B,
                            PetscInt *rindices,
                            PetscInt *cindices,
                            InsertMode addv)
{{
    PetscInt ncols;
    PetscInt *cols, *indices;
    PetscScalar *vals;

    PetscInt m, n;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    MatGetSize(B, &m, NULL);

    n = 0;
    for (PetscInt i = 0; i < m; i++) {{
        ierr = MatGetRow(B, i, &ncols, NULL, NULL);CHKERRQ(ierr);
        n = ncols > n ? ncols : n;
        ierr = MatRestoreRow(B, i, &ncols, NULL, NULL);CHKERRQ(ierr);
    }}
    PetscMalloc1(n, &indices);
    for (PetscInt i = 0; i < m; i++) {{
        ierr = MatGetRow(B, i, &ncols, &cols, &vals);CHKERRQ(ierr);
        for (PetscInt j = 0; j < ncols; j++) {{
            indices[j] = cindices[cols[j]];
        }}
        ierr = MatSetValues(A, 1, &rindices[i], ncols, indices, vals, addv);CHKERRQ(ierr);
        ierr = MatRestoreRow(B, i, &ncols, &cols, &vals);CHKERRQ(ierr);
    }}
    PetscFree(indices);
    PetscFunctionReturn(0);
}}
"""
    name = "setSubMatCSR"
    argtypes = [ctypes.c_voidp, ctypes.c_voidp,
                ctypes.c_voidp, ctypes.c_voidp, ctypes.c_int]
    return load_c_code(code, name, comm=comm, argtypes=argtypes,
                       restype=ctypes.c_int)


@lru_cache(maxsize=1)
def load_set_bc_values():
    comm = PETSc.COMM_SELF
    code = """
#include <petsc.h>

PetscErrorCode setSubDiagonal(Mat A,
                              PetscInt n,
                              PetscInt *indices,
                              InsertMode addv)
{{
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    for (PetscInt i = 0; i < n; i++) {{
        ierr = MatSetValue(A, indices[i], indices[i], 1.0E0, addv);CHKERRQ(ierr);
    }}
    PetscFunctionReturn(0);
}}
"""
    name = "setSubDiagonal"
    argtypes = [ctypes.c_voidp, ctypes.c_int,
                ctypes.c_voidp, ctypes.c_int]
    return load_c_code(code, name, comm=comm, argtypes=argtypes,
                       restype=ctypes.c_int)


def petsc_sparse(A_numpy, rtol=1E-10):
    Amax = max(A_numpy.min(), A_numpy.max(), key=abs)
    atol = rtol*Amax
    nnz = numpy.count_nonzero(abs(A_numpy) > atol, axis=1).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ(A_numpy.shape, nnz=(nnz, [0]), comm=PETSc.COMM_SELF)
    for row, Arow in enumerate(A_numpy):
        cols = numpy.argwhere(abs(Arow) > atol).astype(PETSc.IntType).flat
        A.setValues(row, cols, Arow[cols], PETSc.InsertMode.INSERT)
    A.assemble()
    return A


def block_mat(A_blocks):
    if len(A_blocks) == 1:
        if len(A_blocks[0]) == 1:
            return A_blocks[0][0]

    nrows = sum([Arow[0].size[0] for Arow in A_blocks])
    ncols = sum([Aij.size[1] for Aij in A_blocks[0]])
    nnz = numpy.concatenate([sum([numpy.diff(Aij.getValuesCSR()[0]) for Aij in Arow]) for Arow in A_blocks])
    A = PETSc.Mat().createAIJ((nrows, ncols), nnz=(nnz, [0]), comm=PETSc.COMM_SELF)
    imode = PETSc.InsertMode.INSERT
    insert_block = load_assemble_csr()
    iend = 0
    for Ai in A_blocks:
        istart = iend
        iend += Ai[0].size[0]
        rows = numpy.arange(istart, iend, dtype=PETSc.IntType)
        jend = 0
        for Aij in Ai:
            jstart = jend
            jend += Aij.size[1]
            cols = numpy.arange(jstart, jend, dtype=PETSc.IntType)
            insert_block(A, Aij, rows, cols, imode)

    A.assemble()
    return A


def unrestrict_element(ele):
    if isinstance(ele, ufl.VectorElement):
        return type(ele)(unrestrict_element(ele._sub_element), dim=ele.num_sub_elements())
    elif isinstance(ele, ufl.TensorElement):
        return type(ele)(unrestrict_element(ele._sub_element), shape=ele.value_shape(), symmetry=ele.symmetry())
    elif isinstance(ele, ufl.EnrichedElement):
        return type(ele)(*list(dict.fromkeys(unrestrict_element(e) for e in ele._elements)))
    elif isinstance(ele, ufl.TensorProductElement):
        return type(ele)(*(unrestrict_element(e) for e in ele.sub_elements()), cell=ele.cell())
    elif isinstance(ele, ufl.MixedElement):
        return type(ele)(*(unrestrict_element(e) for e in ele.sub_elements()))
    elif isinstance(ele, ufl.WithMapping):
        return type(ele)(unrestrict_element(ele.wrapee), ele.mapping())
    elif isinstance(ele, ufl.RestrictedElement):
        return unrestrict_element(ele._element)
    elif isinstance(ele, (ufl.HDivElement, ufl.HCurlElement, ufl.BrokenElement)):
        return type(ele)(unrestrict_element(ele._element))
    else:
        return ele


def restricted_dofs(celem, felem):
    """
    find which DOFs from felem are on celem

    :arg celem: the restricted :class:`finat.FiniteElement`
    :arg felem: the unrestricted :class:`finat.FiniteElement`

    :returns: :class:`numpy.array` with indices of felem that correspond to celem
    """
    csplit = split_dofs(celem)
    fsplit = split_dofs(felem)
    if len(csplit[0]) and len(csplit[1]):
        csplit = [numpy.concatenate(csplit)]
        fsplit = [numpy.concatenate(fsplit)]

    k = len(csplit[0]) == 0
    if len(csplit[k]) != len(fsplit[k]):
        raise ValueError("Finite elements have different DOFs")
    perm = numpy.empty_like(csplit[k])
    perm[csplit[k]] = numpy.arange(len(perm))
    return fsplit[k][perm]


def split_dofs(elem):
    entity_dofs = elem.entity_dofs()
    ndim = elem.cell.get_spatial_dimension()
    edofs = [[] for k in range(ndim+1)]
    for key in entity_dofs:
        vals = entity_dofs[key]
        edim = key
        try:
            edim = sum(edim)
        except TypeError:
            pass
        split = numpy.arange(0, len(vals)+1, 2**(ndim-edim))
        for r in range(len(split)-1):
            v = sum([vals[k] for k in range(*split[r:r+2])], [])
            edofs[edim].extend(sorted(v))

    return numpy.array(edofs[-1], dtype=PETSc.IntType), numpy.array(sum(reversed(edofs[:-1]), []), dtype=PETSc.IntType)


def sort_interior_dofs(idofs, A):
    Aii = A.createSubMatrix(idofs, idofs)
    indptr, indices, _ = Aii.getValuesCSR()
    n = idofs.getSize()
    visit = numpy.zeros((n, ), dtype=bool)
    perm = []
    degree = 0
    while not visit.all():
        degree += 1
        for i in range(n):
            if not visit[i]:
                neigh = indices[slice(*indptr[i:i+2])]
                if len(neigh) == degree:
                    visit[neigh] = True
                    perm.extend(neigh)

    idofs.setIndices(idofs.getIndices()[perm])


def mass_matrix(ndim, formdegree, B00, B11):
    B00 = petsc_sparse(B00)
    B11 = petsc_sparse(B11)
    if ndim == 1:
        return B11 if formdegree else B00
    elif ndim == 2:
        if formdegree == 0:
            return B00.kron(B00)
        elif formdegree == 1:
            B_blocks = [B00.kron(B11), B11.kron(B00)]
        else:
            return B11.kron(B11)
    elif ndim == 3:
        if formdegree == 0:
            return B00.kron(B00.kron(B00))
        elif formdegree == 1:
            B_blocks = [B00.kron(B00.kron(B11)), B00.kron(B11.kron(B00)), B11.kron(B00.kron(B00))]
        elif formdegree == 2:
            B_blocks = [B00.kron(B11.kron(B11)), B11.kron(B00.kron(B11)), B11.kron(B11.kron(B00))]
        else:
            return B11.kron(B11.kron(B11))

    nrows = sum(Bk.size[0] for Bk in B_blocks)
    ncols = sum(Bk.size[1] for Bk in B_blocks)
    csr_block = [Bk.getValuesCSR() for Bk in B_blocks]
    ishift = numpy.cumsum([0] + [csr[0][-1] for csr in csr_block])
    jshift = numpy.cumsum([0] + [Bk.size[1] for Bk in B_blocks])
    indptr = numpy.concatenate([csr[0][bool(shift):]+shift for csr, shift in zip(csr_block, ishift[:-1])])
    indices = numpy.concatenate([csr[1]+shift for csr, shift in zip(csr_block, jshift[:-1])])
    data = numpy.concatenate([csr[2] for csr in csr_block])
    return PETSc.Mat().createAIJ((nrows, ncols), csr=(indptr, indices, data), comm=PETSc.COMM_SELF)


def diff_matrix(ndim, formdegree, A00, A11, A10):
    A00 = petsc_sparse(A00)
    A11 = petsc_sparse(A11)
    A10 = petsc_sparse(A10)
    if formdegree == ndim:
        ncols = A10.size[0]**ndim
        A_zero = PETSc.Mat().createAIJ((1, ncols), nnz=([0],)*2, comm=PETSc.COMM_SELF)
        A_zero.assemble()
        return A_zero

    if ndim == 1:
        return A10
    elif ndim == 2:
        if formdegree == 0:
            A_blocks = [[A00.kron(A10)], [A10.kron(A00)]]
        elif formdegree == 1:
            A_blocks = [[A10.kron(A11), A11.kron(-A10)]]
    elif ndim == 3:
        if formdegree == 0:
            A_blocks = [[A00.kron(A00.kron(A10))], [A00.kron(A10.kron(A00))], [A10.kron(A00.kron(A00))]]
        elif formdegree == 1:
            nrows = (A10.getSize()[0])**2 * A10.getSize()[1]
            ncols = (A10.getSize()[1])**2 * A10.getSize()[0]
            A_zero = PETSc.Mat().createAIJ((nrows, ncols), nnz=([0],)*2, comm=PETSc.COMM_SELF)
            A_zero.assemble()
            A_blocks = [[A00.kron(A10.kron(-A11)), A00.kron(A11.kron(A10)), A_zero],
                        [A10.kron(A00.kron(-A11)), A_zero, A11.kron(A00.kron(A10))],
                        [A_zero, A10.kron(A11.kron(A00)), A11.kron(A10.kron(-A00))]]
        elif formdegree == 2:
            A_blocks = [[A10.kron(A11.kron(-A11)), A11.kron(A10.kron(A11)), A11.kron(A11.kron(A10))]]
    return block_mat(A_blocks)


def assemble_reference_tensor(A, Ahat, Vrows, Vcols, rmap, cmap, addv=None):
    if addv is None:
        addv = PETSc.InsertMode.INSERT_VALUES

    rindices, nel = glonum_fun(Vrows.cell_node_map())
    cindices, nel = glonum_fun(Vcols.cell_node_map())
    bsize = Vrows.value_size
    if bsize > 1:
        ibase = numpy.arange(bsize, dtype=PETSc.IntType)
        _rindices = rindices
        _cindices = cindices
        rindices = lambda e: numpy.add.outer(_rindices(e)*bsize, ibase)
        cindices = lambda e: numpy.add.outer(_cindices(e)*bsize, ibase)

    update_A = load_assemble_csr()
    for e in range(nel):
        update_A(A, Ahat, rmap.apply(rindices(e)), cmap.apply(cindices(e)), addv)
    A.assemble()


def get_base_elements(e):
    import finat
    import FIAT
    if isinstance(e, finat.EnrichedElement):
        return sum(list(map(get_base_elements, e.elements)), [])
    elif isinstance(e, finat.TensorProductElement):
        return sum(list(map(get_base_elements, e.factors)), [])
    elif isinstance(e, finat.cube.FlattenedDimensions):
        return get_base_elements(e.product)
    elif isinstance(e, (finat.HCurlElement, finat.HDivElement)):
        return get_base_elements(e.wrappee)
    elif isinstance(e, finat.finiteelementbase.FiniteElementBase):
        return get_base_elements(e.fiat_equivalent)
    elif isinstance(e, FIAT.RestrictedElement):
        return get_base_elements(e._element)
    return [e]


def diff_prolongator(Vf, Vc, fbcs=[], cbcs=[]):
    from tsfc.finatinterface import create_element
    from pyop2.sparsity import get_preallocation
    from firedrake.preconditioners.pmg import fiat_reference_prolongator

    ef = Vf.finat_element
    ec = Vc.finat_element
    if ef.formdegree - ec.formdegree != 1:
        raise ValueError("Expecting Vf = d(Vc)")

    elements = list(set(get_base_elements(ec) + get_base_elements(ef)))
    elements = sorted(elements, key=lambda e: e.formdegree)
    e0, e1 = elements[::len(elements)-1]

    degree = e0.degree()
    A11 = numpy.eye(degree)
    A00 = numpy.eye(degree+1)
    A10 = fiat_reference_prolongator(e1, e0, derivative=True)

    ndim = Vc.mesh().topological_dimension()
    Dhat = diff_matrix(ndim, ec.formdegree, A00, A11, A10)

    scalar_element = lambda e: e._sub_element if isinstance(e, (ufl.TensorElement, ufl.VectorElement)) else e
    fdofs = restricted_dofs(ef, create_element(unrestrict_element(scalar_element(Vf.ufl_element()))))
    cdofs = restricted_dofs(ec, create_element(unrestrict_element(scalar_element(Vc.ufl_element()))))
    fises = PETSc.IS().createGeneral(fdofs, comm=PETSc.COMM_SELF)
    cises = PETSc.IS().createGeneral(cdofs, comm=PETSc.COMM_SELF)
    Dhat = Dhat.createSubMatrix(fises, cises)
    if Vf.value_size > 1:
        Dhat = Dhat.kron(petsc_sparse(numpy.eye(Vf.value_size)))

    fmap = Vf.local_to_global_map(fbcs)
    cmap = Vc.local_to_global_map(cbcs)
    imode = PETSc.InsertMode.INSERT
    sizes = tuple(V.dof_dset.layout_vec.getSizes() for V in (Vf, Vc))
    block_size = Vf.dof_dset.layout_vec.getBlockSize()
    prealloc = PETSc.Mat().create(comm=Vf.comm)
    prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
    prealloc.setSizes(sizes)
    prealloc.setUp()
    assemble_reference_tensor(prealloc, Dhat, Vf, Vc, fmap, cmap, addv=imode)
    nnz = get_preallocation(prealloc, block_size * Vf.dof_dset.set.size)
    Dmat = PETSc.Mat().createAIJ(sizes, block_size, nnz=nnz, comm=Vf.comm)
    Dmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    assemble_reference_tensor(Dmat, Dhat, Vf, Vc, fmap, cmap, addv=imode)
    Dhat.destroy()
    prealloc.destroy()
    return Dmat


class PoissonFDMPC(FDMPC):
    """
    A preconditioner for tensor-product elements that changes the shape
    functions so that the H^1 Riesz map is sparse in the interior of a
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

    _variant = "fdm"

    def assemble_reference_tensors(self, V, appctx):
        from firedrake.preconditioners.pmg import get_line_elements
        try:
            line_elements, shifts = get_line_elements(V)
        except ValueError:
            raise ValueError("FDMPC does not support the element %s" % V.ufl_element())

        line_elements, = line_elements
        self.axes_shifts, = shifts

        degree = max(e.degree() for e in line_elements)
        quad_degree = 2*degree+1
        eta = float(appctx.get("eta", degree*(degree+1)))
        element = V.finat_element
        is_dg = element.entity_dofs() == element.entity_closure_dofs()

        Afdm = []  # sparse interval mass and stiffness matrices for each direction
        Dfdm = []  # tabulation of normal derivatives at the boundary for each direction
        for e in line_elements:
            Afdm[:0], Dfdm[:0] = tuple(zip(fdm_setup_ipdg(e, eta)))
            if not (e.formdegree or is_dg):
                Dfdm[0] = None

        return Afdm, Dfdm, quad_degree, eta

    def assemble_kron(self, A, V, bcs, coefficients, Afdm, Dfdm, bcflags):
        """
        Assemble the stiffness matrix in the FDM basis using Kronecker products of interval matrices

        :arg A: the :class:`PETSc.Mat` to assemble
        :arg V: the :class:`firedrake.FunctionSpace` of the form arguments
        :arg bcs: an iterable of :class:`firedrake.DirichletBCs`
        :arg coefficients: a ``dict`` mapping strings to :class:`firedrake.Functions` with the form coefficients
        :arg Afdm: the list with sparse interval matrices
        :arg Dfdm: the list with normal derivatives matrices
        :arg bcflags: the :class:`numpy.ndarray` with BC facet flags returned by `get_weak_bc_flags`
        """
        Gq = coefficients.get("Gq")
        Bq = coefficients.get("Bq")
        Gq_facet = coefficients.get("Gq_facet")
        PT_facet = coefficients.get("PT_facet")

        lgmap = V.local_to_global_map(bcs)
        bsize = V.value_size
        ncomp = V.ufl_element().reference_value_size()
        sdim = (V.finat_element.space_dimension() * bsize) // ncomp  # dimension of a single component
        ndim = V.ufl_domain().topological_dimension()
        shift = self.axes_shifts * bsize

        index_cell, nel = glonum_fun(V.cell_node_map())
        index_coef, _ = glonum_fun((Gq or Bq).cell_node_map())
        flag2id = numpy.kron(numpy.eye(ndim, ndim, dtype=PETSc.IntType), [[1], [2]])

        # pshape is the shape of the DOFs in the tensor product
        pshape = tuple(Ak[0].size[0] for Ak in Afdm)
        static_condensation = False
        if sdim != numpy.prod(pshape):
            static_condensation = True

        if set(shift) != {0}:
            assert ncomp == ndim
            pshape = [tuple(numpy.roll(pshape, -shift[k])) for k in range(ncomp)]

        if A.getType() != PETSc.Mat.Type.PREALLOCATOR:
            A.zeroEntries()
            for assemble_coef in self.assembly_callables:
                assemble_coef()

        self.set_bc_values(A, V.dof_dset.lgmap.indices[lgmap.indices < 0])

        # assemble zero-th order term separately, including off-diagonals (mixed components)
        # I cannot do this for hdiv elements as off-diagonals are not sparse, this is because
        # the FDM eigenbases for GLL(N) and GLL(N-1) are not orthogonal to each other
        use_diag_Bq = Bq is None or len(Bq.ufl_shape) != 2 or static_condensation
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
                self.update_A(A, Ae, rows)
                Ae.destroy()
            Be.destroy()
            Bq = None

        # assemble the second order term and the zero-th order term if any,
        # discarding mixed derivatives and mixed components
        mue = numpy.zeros((ncomp, ndim), dtype=PETSc.RealType)
        bqe = numpy.zeros((ncomp,), dtype=PETSc.RealType)
        for e in range(nel):
            ie = numpy.reshape(index_cell(e), (ncomp//bsize, -1))
            je = index_coef(e)
            bce = bcflags[e]

            # get second order coefficient on this cell
            if Gq is not None:
                mue.flat[:] = numpy.sum(Gq.dat.data_ro[je], axis=0)
            # get zero-th order coefficient on this cell
            if Bq is not None:
                bqe.flat[:] = numpy.sum(Bq.dat.data_ro[je], axis=0)

            for k in range(ncomp):
                # permutation of axes with respect to the first vector component
                axes = numpy.roll(numpy.arange(ndim), -shift[k])
                # for each component: compute the stiffness matrix Ae
                bck = bce[:, k] if len(bce.shape) == 2 else bce
                fbc = numpy.dot(bck, flag2id)

                if Gq is not None:
                    # Ae = mue[k][0] Ahat + bqe[k] Bhat
                    Be = Afdm[axes[0]][0].copy()
                    Ae = Afdm[axes[0]][1+fbc[0]].copy()
                    Ae.scale(mue[k][0])
                    if Bq is not None:
                        Ae.axpy(bqe[k], Be)

                    if ndim > 1:
                        # Ae = Ae kron Bhat + mue[k][1] Bhat kron Ahat
                        Ae = Ae.kron(Afdm[axes[1]][0])
                        if Gq is not None:
                            Ae.axpy(mue[k][1], Be.kron(Afdm[axes[1]][1+fbc[1]]))

                        if ndim > 2:
                            # Ae = Ae kron Bhat + mue[k][2] Bhat kron Bhat kron Ahat
                            Be = Be.kron(Afdm[axes[1]][0])
                            Ae = Ae.kron(Afdm[axes[2]][0])
                            if Gq is not None:
                                Ae.axpy(mue[k][2], Be.kron(Afdm[axes[2]][1+fbc[2]]))
                    Be.destroy()

                elif Bq is not None:
                    Ae = Afdm[axes[0]][0]
                    for m in range(1, ndim):
                        Ae = Ae.kron(Afdm[axes[m]][0])
                    Ae.scale(bqe[k])

                Ae = self.condense_element_mat(Ae)
                rows = lgmap.apply(ie[0]*bsize+k if bsize == ncomp else ie[k])
                self.update_A(A, Ae, rows)
                Ae.destroy()

        # assemble SIPG interior facet terms if the normal derivatives have been set up
        if any(Dk is not None for Dk in Dfdm):
            if static_condensation:
                raise NotImplementedError("Static condensation for SIPG not implemented")
            if ndim < V.ufl_domain().geometric_dimension():
                raise NotImplementedError("SIPG on immersed meshes is not implemented")
            eta = float(coefficients.get("eta"))
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
                    axes = numpy.roll(numpy.arange(ndim), -shift[k])
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
                        jj = j0 + (offset-1) * (jface % 2)
                        dense_indices.append(jj)
                        for i, iface in enumerate(lfd):
                            i0 = i * offset
                            i1 = i0 + offset
                            ii = i0 + (offset-1) * (iface % 2)

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

                    self.update_A(A, Ae, rows)
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
            coefficients: a dictionary mapping strings to :class:`firedrake.Functions` with the coefficients of the form,
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
        if not isinstance(alpha, ufl.constantvalue.Zero):
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

        # set arbitrary non-zero coefficients for preallocation
        for coef in coefficients.values():
            with coef.dat.vec as cvec:
                cvec.set(1.0E0)
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


def numpy_to_petsc(A_numpy, dense_indices, diag=True):
    """
    Create a SeqAIJ Mat from a dense matrix using the diagonal and a subset of rows and columns.
    If dense_indices is empty, then also include the off-diagonal corners of the matrix.
    """
    n = A_numpy.shape[0]
    nbase = int(diag) + len(dense_indices)
    nnz = numpy.full((n,), nbase, dtype=PETSc.IntType)
    if dense_indices:
        nnz[dense_indices] = n
    else:
        nnz[[0, -1]] = 2

    imode = PETSc.InsertMode.INSERT
    A_petsc = PETSc.Mat().createAIJ(A_numpy.shape, 1, nnz=(nnz, [0]), comm=PETSc.COMM_SELF)
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


@lru_cache(maxsize=10)
def fdm_setup_ipdg(fdm_element, eta):
    """
    Setup for the fast diagonalization method for the IP-DG formulation.
    Compute sparsified interval stiffness and mass matrices
    and tabulate the normal derivative of the shape functions.

    :arg fdm_element: a :class:`FIAT.FDMElement`
    :arg eta: penalty coefficient as a `float`

    :returns: 2-tuple of:
        Afdm: a list of :class:`PETSc.Mats` with the sparse interval matrices
        Bhat, and bcs(Ahat) for every combination of either natural or weak
        Dirichlet BCs on each endpoint.
        Dfdm: the tabulation of the normal derivatives of the Dirichlet eigenfunctions.
    """
    from FIAT.quadrature import GaussLegendreQuadratureLineRule
    ref_el = fdm_element.get_reference_element()
    degree = fdm_element.degree()
    rule = GaussLegendreQuadratureLineRule(ref_el, degree+1)

    phi = fdm_element.tabulate(1, rule.get_points())
    Jhat = phi[(0, )]
    Dhat = phi[(1, )]
    Ahat = numpy.dot(numpy.multiply(Dhat, rule.get_weights()), Dhat.T)
    Bhat = numpy.dot(numpy.multiply(Jhat, rule.get_weights()), Jhat.T)

    # Facet normal derivatives
    basis = fdm_element.tabulate(1, ref_el.get_vertices())
    Dfacet = basis[(1,)]
    Dfacet[:, 0] = -Dfacet[:, 0]

    Afdm = [numpy_to_petsc(Bhat, [])]
    for bc in range(4):
        bcs = (bc % 2, bc//2)
        Abc = Ahat.copy()
        for j in (0, -1):
            if bcs[j] == 1:
                Abc[:, j] -= Dfacet[:, j]
                Abc[j, :] -= Dfacet[:, j]
                Abc[j, j] += eta
        Afdm.append(numpy_to_petsc(Abc, [0, Abc.shape[0]-1]))
    return Afdm, Dfacet


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
        buffer = numpy.empty(node_map.values_with_halo.shape[1:], dtype=node_map.values_with_halo.dtype)
        layers = node_map.iterset.layers_array
        if layers.shape[0] == 1:
            nelz = layers[0, 1]-layers[0, 0]-1
            nel = nelz*nelv
            # return lambda e: node_map.values_with_halo[e//nelz] + (e % nelz)*node_map.offset

            def _glonum(buffer, node_map, nelz, e):
                numpy.copyto(buffer, node_map.values_with_halo[e//nelz])
                buffer += (e % nelz)*node_map.offset
                return buffer
            return lambda e: _glonum(buffer, node_map, nelz, e), nel
        else:
            nelz = layers[:, 1]-layers[:, 0]-1
            nel = sum(nelz[:nelv])
            to_base = numpy.repeat(numpy.arange(node_map.values_with_halo.shape[0], dtype=node_map.offset.dtype), nelz)
            to_layer = numpy.concatenate([numpy.arange(nz, dtype=node_map.offset.dtype) for nz in nelz])
            # return lambda e: node_map.values_with_halo[to_base[e]] + to_layer[e]*node_map.offset

            def _glonum(buffer, node_map, to_base, to_layer, e):
                numpy.copyto(buffer, node_map.values_with_halo[to_base[e]])
                buffer += to_layer[e]*node_map.offset
                return buffer
            return lambda e: _glonum(buffer, node_map, to_base, to_layer, e), nel


def glonum(node_map):
    """
    Return an array with the node map.

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
        form = sum(forms)
        if len(form.arguments()) == 1:
            bq = firedrake.assemble(form)
            fbc = bq.dat.data_with_halos[glonum(Q.cell_node_map())]
            return (abs(fbc) > tol).astype(PETSc.IntType)
    return numpy.zeros(glonum(Q.cell_node_map()).shape, dtype=PETSc.IntType)
