from functools import partial
from itertools import chain, product
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.pmg import (prolongation_matrix_matfree,
                                           evaluate_dual,
                                           get_permutation_to_line_elements)
from firedrake.preconditioners.facet_split import split_dofs, restricted_dofs
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.functionspace import FunctionSpace
from firedrake.ufl_expr import TestFunction, TestFunctions, TrialFunctions
from firedrake.utils import cached_property
from firedrake_citations import Citations
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.expand_indices import expand_indices
from tsfc.finatinterface import create_element
from pyop2.compilation import load
from pyop2.sparsity import get_preallocation
from pyop2.utils import get_petsc_dir

import firedrake.dmhooks as dmhooks
import ufl
import FIAT
import finat
import numpy
import ctypes
import operator

Citations().add("Brubeck2022a", """
@article{Brubeck2022a,
  title={A scalable and robust vertex-star relaxation for high-order {FEM}},
  author={Brubeck, Pablo D. and Farrell, Patrick E.},
  journal = {SIAM J. Sci. Comput.},
  volume = {44},
  number = {5},
  pages = {A2991-A3017},
  year = {2022},
  doi = {10.1137/21M1444187}
""")

Citations().add("Brubeck2022b", """
@misc{Brubeck2022b,
  title={{Multigrid solvers for the de Rham complex with optimal complexity in polynomial degree}},
  author={Brubeck, Pablo D. and Farrell, Patrick E.},
  archiveprefix = {arXiv},
  eprint = {2211.14284},
  primaryclass = {math.NA},
  year={2022}
""")


__all__ = ("FDMPC", "PoissonFDMPC")


class FDMPC(PCBase):
    """
    A preconditioner for tensor-product elements that changes the shape
    functions so that the H(d) (d in {grad, curl, div}) Riesz map is sparse on
    Cartesian cells, and assembles a global sparse matrix on which other
    preconditioners, such as `ASMStarPC`, can be applied.

    Here we assume that the volume integrals in the Jacobian can be expressed as:

    inner(d(v), alpha * d(u))*dx + inner(v, beta * u)*dx

    where alpha and beta are possibly tensor-valued.  The sparse matrix is
    obtained by approximating (v, alpha * u) and (v, beta * u) as diagonal mass
    matrices.

    The PETSc options inspected by this class are:
    - 'fdm_mat_type': can be either 'aij' or 'sbaij'
    - 'fdm_static_condensation': are we assembling the Schur complement on facets?
    """

    _prefix = "fdm_"
    _variant = "fdm"
    _citation = "Brubeck2022b"
    _cache = {}

    @PETSc.Log.EventDecorator("FDMInit")
    def initialize(self, pc):
        Citations().register(self._citation)
        self.comm = pc.comm
        Amat, Pmat = pc.getOperators()
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        options = PETSc.Options(options_prefix)

        use_amat = options.getBool("pc_use_amat", True)
        use_static_condensation = options.getBool("static_condensation", False)
        pmat_type = options.getString("mat_type", PETSc.Mat.Type.AIJ)

        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters") or {}
        self.appctx = appctx

        # Get original Jacobian form and bcs
        if Pmat.getType() == "python":
            ctx = Pmat.getPythonContext()
            J = ctx.a
            bcs = tuple(ctx.bcs)
            mat_type = "matfree"
        else:
            ctx = dmhooks.get_appctx(pc.getDM())
            J = ctx.Jp or ctx.J
            bcs = tuple(ctx._problem.bcs)
            mat_type = ctx.mat_type

        # TODO assemble Schur complements specified by a SLATE Tensor
        # we might extract the form on the interface-interface block like this:
        #
        # if isinstance(J, firedrake.slate.TensorBase) and use_static_condensation:
        #     J = J.children[0].form
        if not isinstance(J, ufl.Form):
            raise ValueError("Expecting a ufl.Form, not a %r" % type(J))

        # Transform the problem into the space with FDM shape functions
        V = J.arguments()[-1].function_space()
        element = V.ufl_element()
        e_fdm = element.reconstruct(variant=self._variant)

        if element == e_fdm:
            V_fdm, J_fdm, bcs_fdm = (V, J, bcs)
        else:
            # Reconstruct Jacobian and bcs with variant element
            V_fdm = FunctionSpace(V.mesh(), e_fdm)
            J_fdm = J(*(t.reconstruct(function_space=V_fdm) for t in J.arguments()), coefficients={})
            bcs_fdm = []
            for bc in bcs:
                W = V_fdm
                for index in bc._indices:
                    W = W.sub(index)
                bcs_fdm.append(bc.reconstruct(V=W, g=0))

            # Create a new _SNESContext in the variant space
            self._ctx_ref = self.new_snes_ctx(pc, J_fdm, bcs_fdm, mat_type,
                                              fcp=fcp, options_prefix=options_prefix)

            # Construct interpolation from variant to original spaces
            self.fdm_interp = prolongation_matrix_matfree(V_fdm, V, bcs_fdm, [])
            self.work_vec_x = Amat.createVecLeft()
            self.work_vec_y = Amat.createVecRight()
            if use_amat:
                from firedrake.assemble import allocate_matrix, TwoFormAssembler
                self.A = allocate_matrix(J_fdm, bcs=bcs_fdm, form_compiler_parameters=fcp,
                                         mat_type=mat_type, options_prefix=options_prefix)
                self._assemble_A = TwoFormAssembler(J_fdm, tensor=self.A, bcs=bcs_fdm,
                                                    form_compiler_parameters=fcp,
                                                    mat_type=mat_type).assemble
                self._assemble_A()
                Amat = self.A.petscmat

            if len(bcs) > 0:
                self.bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs]))
            else:
                self.bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

        # Assemble the FDM preconditioner with sparse local matrices
        Pmat, self.assembly_callables = self.allocate_matrix(V_fdm, J_fdm, bcs_fdm, fcp, pmat_type, use_static_condensation)
        Pmat.setNullSpace(Amat.getNullSpace())
        Pmat.setTransposeNullSpace(Amat.getTransposeNullSpace())
        Pmat.setNearNullSpace(Amat.getNearNullSpace())
        self._assemble_P()

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -fdm_pc_type ksp.
        fdmpc = PETSc.PC().create(comm=pc.comm)
        fdmpc.incrementTabLevel(1, parent=pc)

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. multigrid or patch solves.
        self._dm = V_fdm.dm
        fdmpc.setDM(self._dm)
        fdmpc.setOptionsPrefix(options_prefix)
        fdmpc.setOperators(A=Amat, P=Pmat)
        fdmpc.setUseAmat(use_amat)
        self.pc = fdmpc
        if hasattr(self, "_ctx_ref"):
            with dmhooks.add_hooks(self._dm, self, appctx=self._ctx_ref, save=False):
                fdmpc.setFromOptions()
        else:
            fdmpc.setFromOptions()

    @PETSc.Log.EventDecorator("FDMPrealloc")
    def allocate_matrix(self, V, J, bcs, fcp, pmat_type, use_static_condensation):
        """
        Allocate the FDM sparse preconditioner.

        :arg V: the :class:`.FunctionSpace` of the form arguments
        :arg J: the Jacobian bilinear form
        :arg bcs: an iterable of boundary conditions on V
        :arg fcp: form compiler parameters to assemble coefficients
        :arg pmat_type: the `PETSc.Mat.Type` for the blocks in the diagonal
        :arg use_static_condensation: are we assembling the statically-condensed Schur complement on facets?

        :returns: 2-tuple with the preconditioner :class:`PETSc.Mat` and a list of assembly callables
        """
        symmetric = pmat_type.endswith("sbaij")
        ifacet = [i for i, Vsub in enumerate(V) if is_restricted(Vsub.finat_element)[1]]
        if len(ifacet) == 0:
            Vfacet = None
            Vbig = V
            ebig = V.ufl_element()
            _, fdofs = split_dofs(V.finat_element)
        elif len(ifacet) == 1:
            Vfacet = V[ifacet[0]]
            ebig, = set(unrestrict_element(Vsub.ufl_element()) for Vsub in V)
            Vbig = FunctionSpace(V.mesh(), ebig)
            if len(V) > 1:
                dims = [Vsub.finat_element.space_dimension() for Vsub in V]
                assert sum(dims) == Vbig.finat_element.space_dimension()
            fdofs = restricted_dofs(Vfacet.finat_element, Vbig.finat_element)
        else:
            raise ValueError("Expecting at most one FunctionSpace restricted onto facets.")
        self.embedding_element = ebig

        if Vbig.value_size == 1:
            self.fises = PETSc.IS().createGeneral(fdofs, comm=PETSc.COMM_SELF)
        else:
            self.fises = PETSc.IS().createBlock(Vbig.value_size, fdofs, comm=PETSc.COMM_SELF)

        # Dictionary with kernel to compute the Schur complement
        self.schur_kernel = {}
        if V == Vbig and Vbig.finat_element.formdegree == 0:
            # If we are in H(grad), we just pad with zeros on the statically-condensed pattern
            self.schur_kernel[V] = SchurComplementPattern
        elif Vfacet and use_static_condensation:
            # If we are in a facet space, we build the Schur complement on its diagonal block
            if Vfacet.finat_element.formdegree == 0 and Vfacet.value_size == 1:
                self.schur_kernel[Vfacet] = SchurComplementDiagonal
            elif symmetric:
                self.schur_kernel[Vfacet] = SchurComplementBlockCholesky
            else:
                self.schur_kernel[Vfacet] = SchurComplementBlockQR

        # Create data structures needed for assembly
        self.lgmaps = {Vsub: Vsub.local_to_global_map([bc for bc in bcs if bc.function_space() == Vsub]) for Vsub in V}
        self.coefficients, assembly_callables = self.assemble_coefficients(J, fcp)
        self.assemblers = {}

        Pmats = {}
        addv = PETSc.InsertMode.ADD_VALUES
        # Store only off-diagonal blocks with more columns than rows to save memory
        Vsort = sorted(V, key=lambda Vsub: Vsub.dim())
        # Loop over all pairs of subspaces
        for Vrow, Vcol in product(Vsort, Vsort):
            if symmetric and (Vcol, Vrow) in Pmats:
                P = PETSc.Mat().createTranspose(Pmats[Vcol, Vrow])
            else:
                on_diag = Vrow == Vcol
                triu = on_diag and symmetric
                ptype = pmat_type if on_diag else PETSc.Mat.Type.AIJ
                sizes = tuple(Vsub.dof_dset.layout_vec.getSizes() for Vsub in (Vrow, Vcol))

                preallocator = PETSc.Mat().create(comm=self.comm)
                preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
                preallocator.setSizes(sizes)
                preallocator.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
                preallocator.setUp()
                self.set_values(preallocator, Vrow, Vcol, addv, triu=triu)

                preallocator.assemble()
                d_nnz, o_nnz = get_preallocation(preallocator, sizes[0][0])
                preallocator.destroy()
                if on_diag:
                    numpy.maximum(d_nnz, 1, out=d_nnz)

                P = PETSc.Mat().create(comm=self.comm)
                P.setType(ptype)
                P.setSizes(sizes)
                P.setPreallocationNNZ((d_nnz, o_nnz))
                P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
                if on_diag:
                    P.setOption(PETSc.Mat.Option.STRUCTURALLY_SYMMETRIC, True)
                if ptype.endswith("sbaij"):
                    P.setOption(PETSc.Mat.Option.IGNORE_LOWER_TRIANGULAR, True)
                P.setUp()
                # append callables to zero entries, insert element matrices, and apply BCs
                assembly_callables.append(P.zeroEntries)
                assembly_callables.append(partial(self.set_values, P, Vrow, Vcol, addv))
                if on_diag:
                    own = Vrow.dof_dset.layout_vec.getLocalSize()
                    bdofs = numpy.flatnonzero(self.lgmaps[Vrow].indices[:own] < 0).astype(PETSc.IntType)[:, None]
                    Vrow.dof_dset.lgmap.apply(bdofs, result=bdofs)
                    if len(bdofs) > 0:
                        vals = numpy.ones(bdofs.shape, dtype=PETSc.RealType)
                        assembly_callables.append(partial(P.setValuesRCV, bdofs, bdofs, vals, addv))
            Pmats[Vrow, Vcol] = P

        if len(V) == 1:
            Pmat = Pmats[V, V]
        else:
            Pmat = PETSc.Mat().createNest([[Pmats[Vrow, Vcol] for Vcol in V] for Vrow in V], comm=self.comm)
        assembly_callables.append(Pmat.assemble)
        return Pmat, assembly_callables

    @PETSc.Log.EventDecorator("FDMAssemble")
    def _assemble_P(self):
        for _assemble in self.assembly_callables:
            _assemble()

    @PETSc.Log.EventDecorator("FDMUpdate")
    def update(self, pc):
        if hasattr(self, "A"):
            self._assemble_A()
        self._assemble_P()

    def apply(self, pc, x, y):
        if hasattr(self, "fdm_interp"):
            self.fdm_interp.multTranspose(x, self.work_vec_x)
            with dmhooks.add_hooks(self._dm, self, appctx=self._ctx_ref):
                self.pc.apply(self.work_vec_x, self.work_vec_y)
            self.fdm_interp.mult(self.work_vec_y, y)
            y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]
        else:
            self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        if hasattr(self, "fdm_interp"):
            self.fdm_interp.multTranspose(x, self.work_vec_y)
            with dmhooks.add_hooks(self._dm, self, appctx=self._ctx_ref):
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

    def destroy(self, pc):
        if hasattr(self, "A"):
            self.A.petscmat.destroy()
        if hasattr(self, "pc"):
            self.pc.getOperators()[-1].destroy()
            self.pc.destroy()

    @PETSc.Log.EventDecorator("FDMCoefficients")
    def assemble_coefficients(self, J, fcp, block_diagonal=True):
        """
        Obtain coefficients for the auxiliary operator as the diagonal of a
        weighted mass matrix in broken(V^k) * broken(V^{k+1}).
        See Section 3.2 of Brubeck2022b.

        :arg J: the Jacobian bilinear :class:`ufl.Form`,
        :arg fcp: form compiler parameters to assemble the diagonal of the mass matrices.
        :arg block_diagonal: are we assembling the block diagonal of the mass matrices?

        :returns: a 2-tuple of a `dict` with the zero-th order and second
                  order coefficients keyed on ``"beta"`` and ``"alpha"``,
                  and a list of assembly callables.
        """
        coefficients = {}
        assembly_callables = []
        # Basic idea: take the original bilinear form and
        # replace the exterior derivatives with arguments in broken(V^{k+1}).
        # Then, replace the original arguments with arguments in broken(V^k).
        # Where the broken spaces have L2-orthogonal FDM basis functions.
        index = len(J.arguments()[-1].function_space())-1
        if index:
            splitter = ExtractSubBlock()
            J = splitter.split(J, argument_indices=(index, index))

        args_J = J.arguments()
        e = args_J[0].ufl_element()
        mesh = args_J[0].function_space().mesh()
        tdim = mesh.topological_dimension()
        if isinstance(e, (ufl.VectorElement, ufl.TensorElement)):
            e = e._sub_element
        e = unrestrict_element(e)
        sobolev = e.sobolev_space()

        # Replacement rule for the exterior derivative = grad(arg) * eps
        map_grad = None
        if sobolev == ufl.H1:
            map_grad = lambda p: p
        elif sobolev in [ufl.HCurl, ufl.HDiv]:
            u = ufl.Coefficient(ufl.FunctionSpace(mesh, e))
            du = ufl.variable(ufl.grad(u))
            dku = ufl.div(u) if sobolev == ufl.HDiv else ufl.curl(u)
            eps = expand_derivatives(ufl.diff(ufl.replace(expand_derivatives(dku), {ufl.grad(u): du}), du))
            if sobolev == ufl.HDiv:
                map_grad = lambda p: ufl.outer(p, eps/tdim)
            elif len(eps.ufl_shape) == 3:
                map_grad = lambda p: ufl.dot(p, eps/2)
            else:
                map_grad = lambda p: p*(eps/2)

        # Construct Z = broken(V^k) * broken(V^{k+1})
        V = args_J[0].function_space()
        fe = V.finat_element
        formdegree = fe.formdegree
        degree = fe.degree
        if type(degree) != int:
            degree, = set(degree)
        qdeg = degree
        if formdegree == tdim:
            qfam = "DG" if tdim == 1 else "DQ"
            qdeg = 0
        elif formdegree == 0:
            qfam = "DG" if tdim == 1 else "RTCE" if tdim == 2 else "NCE"
        elif formdegree == 1 and tdim == 3:
            qfam = "NCF"
        else:
            qfam = "DQ L2"
            qdeg = degree - 1

        qvariant = "fdm_quadrature"
        elements = [e.reconstruct(variant=qvariant),
                    ufl.FiniteElement(qfam, cell=mesh.ufl_cell(), degree=qdeg, variant=qvariant)]
        elements = list(map(ufl.BrokenElement, elements))
        if V.shape:
            elements = [ufl.TensorElement(ele, shape=V.shape) for ele in elements]
        Z = FunctionSpace(mesh, ufl.MixedElement(elements))

        # Transform the exterior derivative and the original arguments of J to arguments in Z
        args = (TestFunctions(Z), TrialFunctions(Z))
        repargs = {t: v[0] for t, v in zip(args_J, args)}
        repgrad = {ufl.grad(t): map_grad(v[1]) for t, v in zip(args_J, args)} if map_grad else {}
        Jcell = expand_indices(expand_derivatives(ufl.Form(J.integrals_by_type("cell"))))
        mixed_form = ufl.replace(ufl.replace(Jcell, repgrad), repargs)

        # Return coefficients and assembly callables
        if block_diagonal and V.shape:
            from firedrake.assemble import assemble
            M = assemble(mixed_form, mat_type="matfree", form_compiler_parameters=fcp)
            for iset, name in zip(Z.dof_dset.field_ises, ("beta", "alpha")):
                sub = M.petscmat.createSubMatrix(iset, iset)
                ctx = sub.getPythonContext()
                coefficients[name] = ctx._block_diagonal
                assembly_callables.append(ctx._assemble_block_diagonal)
        else:
            from firedrake.assemble import OneFormAssembler
            tensor = Function(Z.dual())
            coefficients["beta"] = tensor.subfunctions[0]
            coefficients["alpha"] = tensor.subfunctions[1]
            assembly_callables.append(OneFormAssembler(mixed_form, tensor=tensor, diagonal=True,
                                                       form_compiler_parameters=fcp).assemble)
        return coefficients, assembly_callables

    @PETSc.Log.EventDecorator("FDMRefTensor")
    def assemble_reference_tensor(self, V, transpose=False, sort_interior=False):
        """
        Return the reference tensor used in the diagonal factorisation of the
        sparse cell matrices.  See Section 3.2 of Brubeck2022b.

        :arg V: a :class:`.FunctionSpace`

        :returns: a :class:`PETSc.Mat` interpolating V^k * d(V^k) onto
                  broken(V^k) * broken(V^{k+1}) on the reference element.
        """
        value_size = V.value_size
        fe = V.finat_element
        tdim = fe.cell.get_spatial_dimension()
        formdegree = fe.formdegree
        degree = fe.degree
        if type(degree) != int:
            degree, = set(degree)
        if formdegree == tdim:
            degree = degree + 1
        is_interior, is_facet = is_restricted(fe)
        key = (value_size, tdim, degree, formdegree, is_interior, is_facet, transpose, sort_interior)
        cache = self._cache.setdefault("reference_tensor", {})
        try:
            return cache[key]
        except KeyError:
            pass

        if transpose:
            result = self.assemble_reference_tensor(V, transpose=False, sort_interior=sort_interior)
            result = PETSc.Mat().createTranspose(result).convert(result.getType())
            return cache.setdefault(key, result)

        if sort_interior:
            assert is_interior and not is_facet and not transpose
            # Sort DOFs to make A00 block diagonal with blocks of increasing dimension along the diagonal
            result = self.assemble_reference_tensor(V, transpose=transpose, sort_interior=False)
            if formdegree != 0:
                # Compute the stiffness matrix on the interior of a cell
                A00 = self._element_mass_matrix.PtAP(result)
                indptr, indices, _ = A00.getValuesCSR()
                degree = numpy.diff(indptr)
                # Sort by blocks
                uniq, u_index = numpy.unique(indices, return_index=True)
                perm = uniq[u_index.argsort(kind='stable')]
                # Sort by degree
                degree = degree[perm]
                perm = perm[degree.argsort(kind='stable')]
                A00.destroy()

                isperm = PETSc.IS().createGeneral(perm, comm=result.getComm())
                result = get_submat(result, iscol=isperm, permute=True)
                isperm.destroy()
            return cache.setdefault(key, result)

        short_key = key[:-3] + (False,) * 3
        try:
            result = cache[short_key]
        except KeyError:
            # Get CG(k) and DG(k-1) 1D elements from V
            elements = sorted(get_base_elements(fe), key=lambda e: e.formdegree)
            e0 = elements[0] if elements[0].formdegree == 0 else None
            e1 = elements[-1] if elements[-1].formdegree == 1 else None
            if e0 and is_interior:
                e0 = FIAT.RestrictedElement(e0, restriction_domain="interior")

            # Get broken(CG(k)) and DG(k-1) 1D elements from the coefficient spaces
            Q0 = self.coefficients["beta"].function_space().finat_element.element
            elements = sorted(get_base_elements(Q0), key=lambda e: e.formdegree)
            q0 = elements[0] if elements[0].formdegree == 0 else None
            q1 = elements[-1]
            if q1.formdegree != 1:
                Q1 = self.coefficients["alpha"].function_space().finat_element.element
                q1 = sorted(get_base_elements(Q1), key=lambda e: e.formdegree)[-1]

            # Interpolate V * d(V) -> space(beta) * space(alpha)
            comm = PETSc.COMM_SELF
            zero = PETSc.Mat()
            A00 = petsc_sparse(evaluate_dual(e0, q0), comm=comm) if e0 and q0 else zero
            A11 = petsc_sparse(evaluate_dual(e1, q1), comm=comm) if e1 else zero
            A10 = petsc_sparse(evaluate_dual(e0, q1, alpha=(1,)), comm=comm) if e0 else zero
            B_blocks = mass_blocks(tdim, formdegree, A00, A11)
            A_blocks = diff_blocks(tdim, formdegree, A00, A11, A10)
            result = block_mat(B_blocks + A_blocks, destroy_blocks=True)
            A00.destroy()
            A10.destroy()
            A11.destroy()
            if value_size != 1:
                eye = petsc_sparse(numpy.eye(value_size), comm=result.getComm())
                temp = result
                result = temp.kron(eye)
                temp.destroy()
                eye.destroy()

        if is_facet:
            cache[short_key] = result
            result = get_submat(result, iscol=self.fises)
        return cache.setdefault(key, result)

    @cached_property
    def _element_mass_matrix(self):
        Z = [self.coefficients[name].function_space() for name in ("beta", "alpha")]
        shape = (sum(V.finat_element.space_dimension() for V in Z),) + Z[0].shape
        data = numpy.ones(shape, dtype=PETSc.RealType)
        shape += (1,) * (3-len(shape))
        nrows = shape[0] * shape[1]
        ai = numpy.arange(nrows+1, dtype=PETSc.IntType)
        aj = numpy.tile(ai[:-1].reshape((-1, shape[1])), (1, shape[2]))
        if shape[2] > 1:
            ai *= shape[2]
            data = numpy.tile(numpy.eye(shape[2], dtype=data.dtype), shape[:1] + (1,)*(len(shape)-1))
        return PETSc.Mat().createAIJ((nrows, nrows), csr=(ai, aj, data), comm=PETSc.COMM_SELF)

    @PETSc.Log.EventDecorator("FDMSetValues")
    def set_values(self, A, Vrow, Vcol, addv, triu=False):
        """
        Assemble the stiffness matrix in the FDM basis using sparse reference
        tensors and diagonal mass matrices.

        :arg A: the :class:`PETSc.Mat` to assemble
        :arg Vrow: the :class:`.FunctionSpace` test space
        :arg Vcol: the :class:`.FunctionSpace` trial space
        :arg addv: a `PETSc.Mat.InsertMode`
        :arg triu: are we assembling only the upper triangular part?
        """
        key = (Vrow.ufl_element(), Vcol.ufl_element())
        try:
            assembler = self.assemblers[key]
        except KeyError:
            # Interpolation of basis and exterior derivative onto broken spaces
            C1 = self.assemble_reference_tensor(Vcol)
            R1 = self.assemble_reference_tensor(Vrow, transpose=True)
            M = self._element_mass_matrix
            # Element stiffness matrix = R1 * M * C1, see Equation (3.9) of Brubeck2022b
            element_kernel = TripleProductKernel(R1, M, C1, self.coefficients["beta"], self.coefficients["alpha"])

            schur_kernel = None
            if Vrow == Vcol:
                schur_kernel = self.schur_kernel.get(Vrow)
            if schur_kernel is not None:
                V0 = FunctionSpace(Vrow.mesh(), restrict_element(self.embedding_element, "interior"))
                C0 = self.assemble_reference_tensor(V0, sort_interior=True)
                R0 = self.assemble_reference_tensor(V0, sort_interior=True, transpose=True)
                # Only the facet block updates the coefficients in M
                element_kernel = schur_kernel(element_kernel,
                                              TripleProductKernel(R1, M, C0),
                                              TripleProductKernel(R0, M, C1),
                                              TripleProductKernel(R0, M, C0))

            assembler = SparseAssembler(element_kernel, Vrow, Vcol, self.lgmaps[Vrow], self.lgmaps[Vcol])
            self.assemblers.setdefault(key, assembler)
        assembler.assemble(A, addv=addv, triu=triu)


class SparseAssembler(object):

    _cache = {}

    @staticmethod
    def setSubMatCSR(comm, triu=False):
        """
        Compile C code to insert sparse submatrices and store in class cache

        :arg triu: are we inserting onto the upper triangular part of the matrix?

        :returns: a python wrapper for the matrix insertion function
        """
        cache = SparseAssembler._cache.setdefault("setSubMatCSR", {})
        key = triu
        try:
            return cache[key]
        except KeyError:
            return cache.setdefault(key, load_setSubMatCSR(comm, triu))

    def __init__(self, kernel, Vrow, Vcol, rmap, cmap):
        self.kernel = kernel
        m, n = kernel.result.getSize()

        spaces = [Vrow]
        row_shape = tuple() if Vrow.value_size == 1 else (Vrow.value_size,)
        map_rows = (self.map_block_indices, rmap) if row_shape else (rmap.apply,)
        rows = numpy.empty((m, ), dtype=PETSc.IntType).reshape((-1,) + row_shape)
        if Vcol == Vrow:
            cols = rows
            map_cols = (lambda *args, result=None: result, )
        else:
            spaces.append(Vcol)
            col_shape = tuple() if Vcol.value_size == 1 else (Vcol.value_size,)
            map_cols = (self.map_block_indices, cmap) if col_shape else (cmap.apply, )
            cols = numpy.empty((n, ), dtype=PETSc.IntType).reshape((-1,) + col_shape)
        spaces.extend(c.function_space() for c in kernel.coefficients)

        integral_type = kernel.integral_type
        if integral_type in ["cell", "interior_facet_horiz"]:
            get_map = operator.methodcaller("cell_node_map")
        elif integral_type in ["interior_facet", "interior_facet_vert"]:
            get_map = operator.methodcaller("interior_facet_node_map")
        else:
            raise NotImplementedError("Only for cell or interior facet integrals")
        self.node_maps = tuple(map(get_map, spaces))

        ncell = 2 if integral_type.startswith("interior_facet") else 1
        self.indices = tuple(numpy.empty((V.finat_element.space_dimension() * ncell,), dtype=PETSc.IntType) for V in spaces)
        self.map_rows = partial(*map_rows, self.indices[spaces.index(Vrow)], result=rows)
        self.map_cols = partial(*map_cols, self.indices[spaces.index(Vcol)], result=cols)
        self.kernel_args = self.indices[-len(kernel.coefficients):]
        self.set_indices = self.copy_indices

        node_map = self.node_maps[0]
        self.nel = node_map.values.shape[0]
        if node_map.offset is None:
            layers = None
        else:
            layers = node_map.iterset.layers_array
            layers = layers[:, 1]-layers[:, 0]-1
            if integral_type.endswith("horiz"):
                layers -= 1
                self.set_indices = self.copy_indices_horiz
            if layers.shape[0] != self.nel:
                layers = numpy.repeat(layers, self.nel)
        self.layers = layers

    def map_block_indices(self, lgmap, indices, result=None):
        bsize = result.shape[-1]
        numpy.copyto(result[:, 0], indices)
        result[:, 0] *= bsize
        numpy.add.outer(result[:, 0], numpy.arange(1, bsize, dtype=indices.dtype), out=result[:, 1:])
        return lgmap.apply(result, result=result)

    def copy_indices_horiz(self, e):
        for index, node_map in zip(self.indices, self.node_maps):
            index = index.reshape((2, -1))
            numpy.copyto(index, node_map.values_with_halo[e])
            index[1] += node_map.offset

    def copy_indices(self, e):
        for index, node_map in zip(self.indices, self.node_maps):
            numpy.copyto(index, node_map.values_with_halo[e])

    def add_offsets(self):
        for index, node_map in zip(self.indices, self.node_maps):
            index += node_map.offset

    def assemble(self, A, addv=None, triu=False):
        if A.getType() == PETSc.Mat.Type.PREALLOCATOR:
            kernel = lambda *args, result=None: result
        else:
            kernel = self.kernel
        result = self.kernel.result
        insert = self.setSubMatCSR(PETSc.COMM_SELF, triu=triu)

        # Core assembly loop
        if self.layers is None:
            for e in range(self.nel):
                self.set_indices(e)
                insert(A, kernel(*self.kernel_args, result=result),
                       self.map_rows(), self.map_cols(), addv)
        else:
            for e in range(self.nel):
                self.set_indices(e)
                for _ in range(self.layers[e]):
                    insert(A, kernel(*self.kernel_args, result=result),
                           self.map_rows(), self.map_cols(), addv)
                    self.add_offsets()


class ElementKernel(object):
    """
    A constant element kernel
    """
    def __init__(self, A, *coefficients):
        self.result = A
        self.coefficients = coefficients
        self.integral_type = "cell"

    def __call__(self, *args, result=None):
        return result or self.result

    def __del__(self):
        self.destroy()

    def destroy(self):
        pass


class TripleProductKernel(ElementKernel):
    """
    An element kernel to compute a triple matrix product A * B * C, where A and
    C are constant matrices and B is a block diagonal matrix with entries given
    by coefficients.
    """
    def __init__(self, A, B, C, *coefficients):
        self.work = None
        if len(coefficients) == 0:
            self.data = numpy.array([])
            self.update = lambda *args: args
        else:
            dshape = (-1, ) + coefficients[0].dat.data_ro.shape[1:]
            if numpy.prod(dshape[1:]) == 1:
                self.work = B.getDiagonal()
                self.data = self.work.array_w.reshape(dshape)
                self.update = partial(B.setDiagonal, self.work)
            else:
                indptr, indices, data = B.getValuesCSR()
                self.data = data.reshape(dshape)
                self.update = lambda *args: (B.setValuesCSR(indptr, indices, self.data), B.assemble())

        stops = numpy.zeros((len(coefficients) + 1,), dtype=PETSc.IntType)
        numpy.cumsum([c.function_space().finat_element.space_dimension() for c in coefficients], out=stops[1:])
        self.slices = [slice(*stops[k:k+2]) for k in range(len(coefficients))]

        self.product = partial(A.matMatMult, B, C)
        super().__init__(self.product(), *coefficients)

    def __call__(self, *indices, result=None):
        for c, i, z in zip(self.coefficients, indices, self.slices):
            numpy.take(c.dat.data_ro, i, axis=0, out=self.data[z])
        self.update()
        return self.product(result=result)

    def destroy(self):
        self.result.destroy()
        if isinstance(self.work, PETSc.Object):
            self.work.destroy()


class SchurComplementKernel(ElementKernel):
    """
    An element kernel to compute Schur complements that reuses work matrices and the
    symbolic factorization of the interior block.
    """
    def __init__(self, *kernels):
        self.children = kernels
        self.submats = [k.result for k in kernels]

        # Create dict of slices with the extents of the diagonal blocks
        A00 = self.submats[-1]
        degree = numpy.diff(A00.getValuesCSR()[0])
        istart = 0
        self.slices = {1: slice(0, 0)}
        unique_degree, counts = numpy.unique(degree, return_counts=True)
        for k, kdofs in sorted(zip(unique_degree, counts)):
            self.slices[k] = slice(istart, istart + k * kdofs)
            istart += k * kdofs

        self.blocks = sorted(degree for degree in self.slices if degree > 1)

        self.work = [None for _ in range(2)]
        coefficients = []
        for k in self.children:
            coefficients.extend(k.coefficients)
        coefficients = list(dict.fromkeys(coefficients))
        super().__init__(self.condense(), *coefficients)

    def __call__(self, *args, result=None):
        for k in self.children:
            k(*args, result=k.result)
        return self.condense(result=result)

    def destroy(self):
        for k in self.children:
            k.destroy()
        self.result.destroy()
        for obj in self.work:
            if isinstance(obj, PETSc.Object):
                obj.destroy()

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        return result


class SchurComplementPattern(SchurComplementKernel):

    def __call__(self, *args, result=None):
        k = self.children[0]
        k(*args, result=k.result)
        return self.condense(result=result)

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        """By default pad with zeros the statically condensed pattern"""
        structure = PETSc.Mat.Structure.SUBSET if result else None
        if result is None:
            _, A10, A01, A00 = self.submats
            result = A10.matMatMult(A00, A01, result=result)
        result.aypx(0.0, self.submats[0], structure=structure)
        return result


class SchurComplementDiagonal(SchurComplementKernel):

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        structure = PETSc.Mat.Structure.SUBSET if result else None
        A11, A10, A01, A00 = self.submats
        self.work[0] = A00.getDiagonal(result=self.work[0])
        self.work[0].reciprocal()
        self.work[0].scale(-1)
        A01.diagonalScale(L=self.work[0])
        result = A10.matMult(A01, result=result)
        result.axpy(1.0, A11, structure=structure)
        return result


class SchurComplementBlockCholesky(SchurComplementKernel):

    def __init__(self, K11, K10, K01, K00):
        # asssume that K10 = K01^T
        super().__init__(K11, K01, K00)

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        structure = PETSc.Mat.Structure.SUBSET if result else None
        A11, A01, A00 = self.submats
        indptr, indices, R = A00.getValuesCSR()

        zlice = self.slices[1]
        numpy.sqrt(R[zlice], out=R[zlice])
        numpy.reciprocal(R[zlice], out=R[zlice])
        flops = 2 * (zlice.stop - zlice.start)
        for k in self.blocks:
            Rk = R[self.slices[k]]
            A = Rk.reshape((-1, k, k))
            rinv = numpy.linalg.inv(numpy.linalg.cholesky(A))
            numpy.copyto(Rk, rinv.flat)
            flops += A.shape[0] * ((k**3)//3 + k**3)

        PETSc.Log.logFlops(flops)
        A00.setValuesCSR(indptr, indices, R)
        A00.assemble()
        self.work[0] = A00.matMult(A01, result=self.work[0])
        result = self.work[0].transposeMatMult(self.work[0], result=result)
        result.aypx(-1.0, A11, structure=structure)
        return result


class SchurComplementBlockQR(SchurComplementKernel):

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        structure = PETSc.Mat.Structure.SUBSET if result else None
        A11, A10, A01, A00 = self.submats
        indptr, indices, R = A00.getValuesCSR()
        Q = numpy.ones(R.shape, dtype=R.dtype)

        zlice = self.slices[1]
        numpy.reciprocal(R[zlice], out=R[zlice])
        flops = zlice.stop - zlice.start
        for k in self.blocks:
            zlice = self.slices[k]
            A = R[zlice].reshape((-1, k, k))
            q, r = numpy.linalg.qr(A, mode="complete")
            numpy.copyto(Q[zlice], q.flat)
            rinv = numpy.linalg.inv(r)
            numpy.copyto(R[zlice], rinv.flat)
            flops += A.shape[0] * ((4*k**3)//3 + k**3)

        PETSc.Log.logFlops(flops)
        A00.setValuesCSR(indptr, indices, Q)
        A00.assemble()
        self.work[0] = A00.transposeMatMult(A01, result=self.work[0])
        A00.setValuesCSR(indptr, indices, R)
        A00.assemble()
        A00.scale(-1.0)
        result = A10.matMatMult(A00, self.work[0], result=result)
        result.axpy(1.0, A11, structure=structure)
        return result


class SchurComplementBlockSVD(SchurComplementKernel):

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        structure = PETSc.Mat.Structure.SUBSET if result else None
        A11, A10, A01, A00 = self.submats
        indptr, indices, U = A00.getValuesCSR()
        V = numpy.ones(U.shape, dtype=U.dtype)
        self.work[0] = A00.getDiagonal(result=self.work[0])
        D = self.work[0]
        dslice = self.slices[1]
        numpy.sign(D.array_r[dslice], out=U[dslice])
        flops = dslice.stop - dslice.start
        for k in self.blocks:
            bslice = self.slices[k]
            A = U[bslice].reshape((-1, k, k))
            u, s, v = numpy.linalg.svd(A, full_matrices=False)
            dslice = slice(dslice.stop, dslice.stop + k * A.shape[0])
            numpy.copyto(D.array_w[dslice], s.flat)
            numpy.copyto(U[bslice], numpy.transpose(u, axes=(0, 2, 1)).flat)
            numpy.copyto(V[bslice], numpy.transpose(v, axes=(0, 2, 1)).flat)
            flops += A.shape[0] * ((4*k**3)//3 + 4*k**3)

        PETSc.Log.logFlops(flops)
        D.sqrtabs()
        D.reciprocal()
        A00.setValuesCSR(indptr, indices, V)
        A00.assemble()
        A00.diagonalScale(R=D)
        self.work[1] = A10.matMult(A00, result=self.work[1])
        D.scale(-1.0)
        A00.setValuesCSR(indptr, indices, U)
        A00.assemble()
        A00.diagonalScale(L=D)
        result = self.work[1].matMatMult(A00, A01, result=result)
        result.axpy(1.0, A11, structure=structure)
        return result


class SchurComplementBlockInverse(SchurComplementKernel):

    @PETSc.Log.EventDecorator("FDMCondense")
    def condense(self, result=None):
        structure = PETSc.Mat.Structure.SUBSET if result else None
        A11, A10, A01, A00 = self.submats
        indptr, indices, R = A00.getValuesCSR()

        zlice = self.slices[1]
        numpy.reciprocal(R[zlice], out=R[zlice])
        flops = zlice.stop - zlice.start
        for k in self.blocks:
            Rk = R[self.slices[k]]
            A = Rk.reshape((-1, k, k))
            rinv = numpy.linalg.inv(A)
            numpy.copyto(Rk, rinv.flat)
            flops += A.shape[0] * (k**3)

        PETSc.Log.logFlops(flops)
        A00.setValuesCSR(indptr, indices, R)
        A00.assemble()
        A00.scale(-1.0)
        result = A10.matMatMult(A00, A01, result=result)
        result.axpy(1.0, A11, structure=structure)
        return result


@PETSc.Log.EventDecorator("LoadCode")
def load_c_code(code, name, **kwargs):
    petsc_dir = get_petsc_dir()
    cppargs = ["-I%s/include" % d for d in petsc_dir]
    ldargs = (["-L%s/lib" % d for d in petsc_dir]
              + ["-Wl,-rpath,%s/lib" % d for d in petsc_dir]
              + ["-lpetsc", "-lm"])
    return load(code, "c", name, cppargs=cppargs, ldargs=ldargs, **kwargs)


def load_setSubMatCSR(comm, triu=False):
    """Insert one sparse matrix into another sparse matrix.
       Done in C for efficiency, since it loops over rows."""
    if triu:
        name = "setSubMatCSR_SBAIJ"
        select_cols = "icol -= (icol < irow) * (1 + icol);"
    else:
        name = "setSubMatCSR_AIJ"
        select_cols = ""
    code = f"""
#include <petsc.h>

PetscErrorCode {name}(Mat A,
                      Mat B,
                      PetscInt *rindices,
                      PetscInt *cindices,
                      InsertMode addv)
{{
    PetscInt ncols, irow, icol;
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
        irow = rindices[i];
        for (PetscInt j = 0; j < ncols; j++) {{
            icol = cindices[cols[j]];
            {select_cols}
            indices[j] = icol;
        }}
        ierr = MatSetValues(A, 1, &irow, ncols, indices, vals, addv);CHKERRQ(ierr);
        ierr = MatRestoreRow(B, i, &ncols, &cols, &vals);CHKERRQ(ierr);
    }}
    PetscFree(indices);
    PetscFunctionReturn(0);
}}
"""
    argtypes = [ctypes.c_voidp, ctypes.c_voidp,
                ctypes.c_voidp, ctypes.c_voidp, ctypes.c_int]
    funptr = load_c_code(code, name, comm=comm, argtypes=argtypes,
                         restype=ctypes.c_int)

    @PETSc.Log.EventDecorator(name)
    def wrapper(A, B, rows, cols, addv):
        return funptr(A.handle, B.handle, rows.ctypes.data, cols.ctypes.data, addv)

    return wrapper


def is_restricted(finat_element):
    """Determine if an element is a restriction onto interior or facets"""
    tdim = finat_element.cell.get_dimension()
    idofs = len(finat_element.entity_dofs()[tdim][0])
    is_interior = idofs == finat_element.space_dimension()
    is_facet = idofs == 0
    return is_interior, is_facet


def petsc_sparse(A_numpy, rtol=1E-10, comm=None):
    """Convert dense numpy matrix into a sparse PETSc matrix"""
    atol = rtol * abs(max(A_numpy.min(), A_numpy.max(), key=abs))
    sparsity = abs(A_numpy) > atol
    nnz = numpy.count_nonzero(sparsity, axis=1).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ(A_numpy.shape, nnz=(nnz, 0), comm=comm)
    rows, cols = numpy.nonzero(sparsity)
    rows = rows.astype(PETSc.IntType)
    cols = cols.astype(PETSc.IntType)
    vals = A_numpy[sparsity]
    A.setValuesRCV(rows[:, None], cols[:, None], vals[:, None], PETSc.InsertMode.INSERT)
    A.assemble()
    return A


def kron3(A, B, C, scale=None):
    """Returns scale * kron(A, kron(B, C))"""
    temp = B.kron(C)
    if scale is not None:
        temp.scale(scale)
    result = A.kron(temp)
    temp.destroy()
    return result


def get_submat(A, isrow=None, iscol=None, permute=False):
    """Return the sub matrix A[isrow, iscol]"""
    needs_rows = isrow is None
    needs_cols = iscol is None
    if needs_rows and needs_cols:
        return A
    size = A.getSize()
    if needs_rows:
        isrow = PETSc.IS().createStride(size[0], step=1, comm=A.getComm())
    if needs_cols:
        iscol = PETSc.IS().createStride(size[1], step=1, comm=A.getComm())
    if permute:
        submat = A.permute(isrow, iscol)
    else:
        submat = A.createSubMatrix(isrow, iscol)
    if needs_rows:
        isrow.destroy()
    if needs_cols:
        iscol.destroy()
    return submat


def block_mat(A_blocks, destroy_blocks=False):
    """Return a concrete Mat corresponding to a block matrix given as a list of lists.
       Optionally, destroys the input Mats if a new Mat is created."""
    if len(A_blocks) == 1:
        if len(A_blocks[0]) == 1:
            return A_blocks[0][0]

    result = PETSc.Mat().createNest(A_blocks, comm=A_blocks[0][0].getComm())
    # A nest Mat would not allow us to take matrix-matrix products
    result = result.convert(mat_type=A_blocks[0][0].getType())
    if destroy_blocks:
        for row in A_blocks:
            for mat in row:
                mat.destroy()
    return result


def mass_blocks(tdim, formdegree, B00, B11):
    """Construct mass block matrix on reference cell from 1D mass matrices B00 and B11.
       The 1D matrices may come with different test and trial spaces."""
    if tdim == 1:
        B_diag = [B11 if formdegree else B00]
    elif tdim == 2:
        if formdegree == 0:
            B_diag = [B00.kron(B00)]
        elif formdegree == 1:
            B_diag = [B00.kron(B11), B11.kron(B00)]
        else:
            B_diag = [B11.kron(B11)]
    elif tdim == 3:
        if formdegree == 0:
            B_diag = [kron3(B00, B00, B00)]
        elif formdegree == 1:
            B_diag = [kron3(B00, B00, B11), kron3(B00, B11, B00), kron3(B11, B00, B00)]
        elif formdegree == 2:
            B_diag = [kron3(B00, B11, B11), kron3(B11, B00, B11), kron3(B11, B11, B00)]
        else:
            B_diag = [kron3(B11, B11, B11)]

    n = len(B_diag)
    if n == 1:
        return [B_diag]
    else:
        zero = PETSc.Mat()
        return [[B_diag[i] if i == j else zero for j in range(n)] for i in range(n)]


def diff_blocks(tdim, formdegree, A00, A11, A10):
    """Construct exterior derivative block matrix on reference cell from 1D
       mass matrices A00 and A11, and exterior derivative moments A10.
       The 1D matrices may come with different test and trial spaces."""
    if formdegree == tdim:
        ncols = A10.shape[0]**tdim
        zero = PETSc.Mat().createAIJ((1, ncols), nnz=(0, 0), comm=A10.getComm())
        zero.assemble()
        A_blocks = [[zero]]
    elif tdim == 1:
        A_blocks = [[A10]]
    elif tdim == 2:
        if formdegree == 0:
            A_blocks = [[A00.kron(A10)], [A10.kron(A00)]]
        elif formdegree == 1:
            A_blocks = [[A10.kron(A11), A11.kron(A10)]]
            A_blocks[-1][-1].scale(-1)
    elif tdim == 3:
        if formdegree == 0:
            A_blocks = [[kron3(A00, A00, A10)], [kron3(A00, A10, A00)], [kron3(A10, A00, A00)]]
        elif formdegree == 1:
            zero = PETSc.Mat()
            A_blocks = [[kron3(A00, A10, A11, scale=-1), kron3(A00, A11, A10), zero],
                        [kron3(A10, A00, A11, scale=-1), zero, kron3(A11, A00, A10)],
                        [zero, kron3(A10, A11, A00), kron3(A11, A10, A00, scale=-1)]]
        elif formdegree == 2:
            A_blocks = [[kron3(A10, A11, A11, scale=-1), kron3(A11, A10, A11), kron3(A11, A11, A10)]]
    return A_blocks


def tabulate_exterior_derivative(Vc, Vf, cbcs=[], fbcs=[], comm=None):
    """
    Tabulate exterior derivative: Vc -> Vf as an explicit sparse matrix.
    Works for any tensor-product basis. These are the same matrices one needs for HypreAMS and friends.
    """
    if comm is None:
        comm = Vf.comm
    ec = Vc.finat_element
    ef = Vf.finat_element
    if ef.formdegree - ec.formdegree != 1:
        raise ValueError("Expecting Vf = d(Vc)")

    elements = sorted(get_base_elements(ec), key=lambda e: e.formdegree)
    c0, c1 = elements[::len(elements)-1]
    elements = sorted(get_base_elements(ef), key=lambda e: e.formdegree)
    f0, f1 = elements[::len(elements)-1]
    if f0.formdegree != 0:
        f0 = None
    if c1.formdegree != 1:
        c1 = None

    tdim = Vc.mesh().topological_dimension()
    zero = PETSc.Mat()
    A00 = petsc_sparse(evaluate_dual(c0, f0), comm=PETSc.COMM_SELF) if f0 else zero
    A11 = petsc_sparse(evaluate_dual(c1, f1), comm=PETSc.COMM_SELF) if c1 else zero
    A10 = petsc_sparse(evaluate_dual(c0, f1, alpha=(1,)), comm=PETSc.COMM_SELF)
    Dhat = block_mat(diff_blocks(tdim, ec.formdegree, A00, A11, A10), destroy_blocks=True)
    A00.destroy()
    A10.destroy()
    A11.destroy()

    if any(is_restricted(ec)) or any(is_restricted(ef)):
        scalar_element = lambda e: e._sub_element if isinstance(e, (ufl.TensorElement, ufl.VectorElement)) else e
        fdofs = restricted_dofs(ef, create_element(unrestrict_element(scalar_element(Vf.ufl_element()))))
        cdofs = restricted_dofs(ec, create_element(unrestrict_element(scalar_element(Vc.ufl_element()))))
        temp = Dhat
        fises = PETSc.IS().createGeneral(fdofs, comm=temp.getComm())
        cises = PETSc.IS().createGeneral(cdofs, comm=temp.getComm())
        Dhat = temp.createSubMatrix(fises, cises)
        temp.destroy()
        fises.destroy()
        cises.destroy()

    if Vf.value_size > 1:
        temp = Dhat
        eye = petsc_sparse(numpy.eye(Vf.value_size, dtype=PETSc.RealType), comm=temp.getComm())
        Dhat = temp.kron(eye)
        temp.destroy()
        eye.destroy()

    sizes = tuple(V.dof_dset.layout_vec.getSizes() for V in (Vf, Vc))
    block_size = Vf.dof_dset.layout_vec.getBlockSize()
    preallocator = PETSc.Mat().create(comm=comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
    preallocator.setSizes(sizes)
    preallocator.setUp()

    insert = PETSc.InsertMode.INSERT
    rmap = Vf.local_to_global_map(fbcs)
    cmap = Vc.local_to_global_map(cbcs)
    assembler = SparseAssembler(ElementKernel(Dhat), Vf, Vc, rmap, cmap)
    assembler.assemble(preallocator, addv=insert)
    preallocator.assemble()

    nnz = get_preallocation(preallocator, sizes[0][0])
    preallocator.destroy()
    Dmat = PETSc.Mat().createAIJ(sizes, block_size, nnz=nnz, comm=comm)
    Dmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

    assembler.assemble(Dmat, addv=insert)
    Dmat.assemble()
    Dhat.destroy()
    return Dmat


def restrict_element(ele, restriction_domain):
    """Get an element that is not restricted and return the restricted element."""
    if isinstance(ele, ufl.VectorElement):
        return type(ele)(restrict_element(ele._sub_element, restriction_domain), dim=ele.num_sub_elements())
    elif isinstance(ele, ufl.TensorElement):
        return type(ele)(restrict_element(ele._sub_element, restriction_domain), shape=ele._shape, symmetry=ele.symmetry())
    elif isinstance(ele, ufl.MixedElement):
        return type(ele)(*(restrict_element(e, restriction_domain) for e in ele.sub_elements()))
    else:
        return ele[restriction_domain]


def unrestrict_element(ele):
    """Get an element that might or might not be restricted and
       return the parent unrestricted element."""
    if isinstance(ele, ufl.VectorElement):
        return type(ele)(unrestrict_element(ele._sub_element), dim=ele.num_sub_elements())
    elif isinstance(ele, ufl.TensorElement):
        return type(ele)(unrestrict_element(ele._sub_element), shape=ele._shape, symmetry=ele.symmetry())
    elif isinstance(ele, ufl.MixedElement):
        return type(ele)(*(unrestrict_element(e) for e in ele.sub_elements()))
    elif isinstance(ele, ufl.RestrictedElement):
        return unrestrict_element(ele._element)
    else:
        return ele


def get_base_elements(e):
    if isinstance(e, finat.EnrichedElement):
        return list(chain.from_iterable(map(get_base_elements, e.elements)))
    elif isinstance(e, finat.TensorProductElement):
        return list(chain.from_iterable(map(get_base_elements, e.factors)))
    elif isinstance(e, finat.FlattenedDimensions):
        return get_base_elements(e.product)
    elif isinstance(e, (finat.HCurlElement, finat.HDivElement)):
        return get_base_elements(e.wrappee)
    elif isinstance(e, finat.finiteelementbase.FiniteElementBase):
        return get_base_elements(e.fiat_equivalent)
    elif isinstance(e, FIAT.RestrictedElement):
        return get_base_elements(e._element)
    return [e]


class PoissonFDMPC(FDMPC):
    """
    A preconditioner for tensor-product elements that changes the shape
    functions so that the H^1 Riesz map is sparse in the interior of a
    Cartesian cell, and assembles a global sparse matrix on which other
    preconditioners, such as `ASMStarPC`, can be applied.

    Here we assume that the volume integrals in the Jacobian can be expressed as:

    inner(grad(v), alpha(grad(u)))*dx + inner(v, beta(u))*dx

    where alpha and beta are possibly tensor-valued.
    The sparse matrix is obtained by approximating alpha and beta by cell-wise
    constants and discarding the coefficients in alpha that couple together
    mixed derivatives and mixed components.

    For spaces that are not H^1-conforming, this preconditioner will use
    the symmetric interior-penalty DG method. The penalty coefficient can be
    provided in the application context, keyed on ``"eta"``.
    """

    _variant = "fdm_ipdg"
    _citation = "Brubeck2022a"

    def assemble_reference_tensor(self, V):
        try:
            _, line_elements, shifts = get_permutation_to_line_elements(V)
        except ValueError:
            raise ValueError("FDMPC does not support the element %s" % V.ufl_element())

        line_elements, = line_elements
        axes_shifts, = shifts

        degree = max(e.degree() for e in line_elements)
        eta = float(self.appctx.get("eta", degree*(degree+1)))
        element = V.finat_element
        is_dg = element.entity_dofs() == element.entity_closure_dofs()

        Afdm = []  # sparse interval mass and stiffness matrices for each direction
        Dfdm = []  # tabulation of normal derivatives at the boundary for each direction
        bdof = []  # indices of point evaluation dofs for each direction
        cache = self._cache.setdefault("ipdg_reference_tensor", {})
        for e in line_elements:
            key = (e.degree(), eta)
            try:
                rtensor = cache[key]
            except KeyError:
                rtensor = cache.setdefault(key, fdm_setup_ipdg(e, eta, comm=PETSc.COMM_SELF))
            Afdm[:0], Dfdm[:0], bdof[:0] = tuple(zip(rtensor))
            if not is_dg and e.degree() == degree:
                # do not apply SIPG along continuous directions
                Dfdm[0] = None
        return Afdm, Dfdm, bdof, axes_shifts

    @PETSc.Log.EventDecorator("FDMSetValues")
    def set_values(self, A, Vrow, Vcol, addv, triu=False):
        """
        Assemble the stiffness matrix in the FDM basis using Kronecker products of interval matrices

        :arg A: the :class:`PETSc.Mat` to assemble
        :arg Vrow: the :class:`.FunctionSpace` test space
        :arg Vcol: the :class:`.FunctionSpace` trial space
        :arg addv: a `PETSc.Mat.InsertMode`
        :arg triu: are we assembling only the upper triangular part?
        """
        set_submat = SparseAssembler.setSubMatCSR(PETSc.COMM_SELF, triu=triu)
        update_A = lambda A, Ae, rindices: set_submat(A, Ae, rindices, rindices, addv)
        condense_element_mat = lambda x: x

        def cell_to_global(lgmap, cell_to_local, cell_index, result=None):
            # Be careful not to create new arrays
            result = cell_to_local(cell_index, result=result)
            return lgmap.apply(result, result=result)

        bsize = Vrow.dof_dset.layout_vec.getBlockSize()
        cell_to_local, nel = extrude_node_map(Vrow.cell_node_map(), bsize=bsize)
        get_rindices = partial(cell_to_global, self.lgmaps[Vrow], cell_to_local)
        Afdm, Dfdm, bdof, axes_shifts = self.assemble_reference_tensor(Vrow)

        Gq = self.coefficients.get("alpha")
        Bq = self.coefficients.get("beta")
        bcflags = self.coefficients.get("bcflags")
        Gq_facet = self.coefficients.get("Gq_facet")
        PT_facet = self.coefficients.get("PT_facet")

        V = Vrow
        bsize = V.value_size
        ncomp = V.ufl_element().reference_value_size()
        sdim = (V.finat_element.space_dimension() * bsize) // ncomp  # dimension of a single component
        tdim = V.mesh().topological_dimension()
        shift = axes_shifts * bsize

        index_coef, _ = extrude_node_map((Gq or Bq).cell_node_map())
        index_bc, _ = extrude_node_map(bcflags.cell_node_map())
        flag2id = numpy.kron(numpy.eye(tdim, tdim, dtype=PETSc.IntType), [[1], [2]])

        # pshape is the shape of the DOFs in the tensor product
        pshape = tuple(Ak[0].size[0] for Ak in Afdm)
        static_condensation = False
        if sdim != numpy.prod(pshape):
            static_condensation = True

        if set(shift) != {0}:
            assert ncomp == tdim
            pshape = [tuple(numpy.roll(pshape, -shift[k])) for k in range(ncomp)]

        # assemble zero-th order term separately, including off-diagonals (mixed components)
        # I cannot do this for hdiv elements as off-diagonals are not sparse, this is because
        # the FDM eigenbases for CG(k) and CG(k-1) are not orthogonal to each other
        use_diag_Bq = Bq is None or len(Bq.ufl_shape) != 2 or static_condensation
        rindices = None
        if not use_diag_Bq:
            bshape = Bq.ufl_shape
            # Be = Bhat kron ... kron Bhat
            Be = Afdm[0][0].copy()
            for k in range(1, tdim):
                Be = Be.kron(Afdm[k][0])

            aptr = numpy.arange(0, (bshape[0]+1)*bshape[1], bshape[1], dtype=PETSc.IntType)
            aidx = numpy.tile(numpy.arange(bshape[1], dtype=PETSc.IntType), bshape[0])
            for e in range(nel):
                # Ae = Be kron Bq[e]
                adata = numpy.sum(Bq.dat.data_ro[index_coef(e)], axis=0)
                Ae = PETSc.Mat().createAIJWithArrays(bshape, (aptr, aidx, adata), comm=PETSc.COMM_SELF)
                Ae = Be.kron(Ae)
                rindices = get_rindices(e, result=rindices)
                update_A(A, Ae, rindices)
                Ae.destroy()
            Be.destroy()
            Bq = None

        # assemble the second order term and the zero-th order term if any,
        # discarding mixed derivatives and mixed components
        ae = numpy.zeros((ncomp, tdim), dtype=PETSc.RealType)
        be = numpy.zeros((ncomp,), dtype=PETSc.RealType)
        je = None
        for e in range(nel):
            je = index_coef(e, result=je)
            bce = bcflags.dat.data_ro_with_halos[index_bc(e)] > 1E-8
            # get coefficients on this cell
            if Gq is not None:
                ae[:] = numpy.sum(Gq.dat.data_ro[je], axis=0)
            if Bq is not None:
                be[:] = numpy.sum(Bq.dat.data_ro[je], axis=0)

            rindices = get_rindices(e, result=rindices)
            rows = numpy.reshape(rindices, (-1, bsize))
            rows = numpy.transpose(rows)
            rows = numpy.reshape(rows, (ncomp, -1))
            # for each component: compute the stiffness matrix Ae
            for k in range(ncomp):
                # permutation of axes with respect to the first vector component
                axes = numpy.roll(numpy.arange(tdim), -shift[k])
                bck = bce[:, k] if len(bce.shape) == 2 else bce
                fbc = numpy.dot(bck, flag2id)

                if Gq is not None:
                    # Ae = ae[k][0] Ahat + be[k] Bhat
                    Be = Afdm[axes[0]][0].copy()
                    Ae = Afdm[axes[0]][1+fbc[0]].copy()
                    Ae.scale(ae[k][0])
                    if Bq is not None:
                        Ae.axpy(be[k], Be)

                    if tdim > 1:
                        # Ae = Ae kron Bhat + ae[k][1] Bhat kron Ahat
                        Ae = Ae.kron(Afdm[axes[1]][0])
                        if Gq is not None:
                            Ae.axpy(ae[k][1], Be.kron(Afdm[axes[1]][1+fbc[1]]))

                        if tdim > 2:
                            # Ae = Ae kron Bhat + ae[k][2] Bhat kron Bhat kron Ahat
                            Be = Be.kron(Afdm[axes[1]][0])
                            Ae = Ae.kron(Afdm[axes[2]][0])
                            if Gq is not None:
                                Ae.axpy(ae[k][2], Be.kron(Afdm[axes[2]][1+fbc[2]]))
                    Be.destroy()

                elif Bq is not None:
                    Ae = Afdm[axes[0]][0]
                    for m in range(1, tdim):
                        Ae = Ae.kron(Afdm[axes[m]][0])
                    Ae.scale(be[k])

                Ae = condense_element_mat(Ae)
                update_A(A, Ae, rows[k].astype(PETSc.IntType))
                Ae.destroy()

        # assemble SIPG interior facet terms if the normal derivatives have been set up
        if any(Dk is not None for Dk in Dfdm):
            if static_condensation:
                raise NotImplementedError("Static condensation for SIPG not implemented")
            if tdim < V.mesh().geometric_dimension():
                raise NotImplementedError("SIPG on immersed meshes is not implemented")
            eta = float(self.appctx.get("eta"))

            lgmap = self.lgmaps[V]
            index_facet, local_facet_data, nfacets = extrude_interior_facet_maps(V)
            index_coef, _, _ = extrude_interior_facet_maps(Gq_facet or Gq)
            rows = numpy.zeros((2, sdim), dtype=PETSc.IntType)

            for e in range(nfacets):
                # for each interior facet: compute the SIPG stiffness matrix Ae
                ie = index_facet(e)
                je = numpy.reshape(index_coef(e), (2, -1))
                lfd = local_facet_data(e)
                idir = lfd // 2

                if PT_facet:
                    icell = numpy.reshape(lgmap.apply(ie), (2, ncomp, -1))
                    iord0 = numpy.insert(numpy.delete(numpy.arange(tdim), idir[0]), 0, idir[0])
                    iord1 = numpy.insert(numpy.delete(numpy.arange(tdim), idir[1]), 0, idir[1])
                    je = je[[0, 1], lfd]
                    Pfacet = PT_facet.dat.data_ro_with_halos[je]
                    Gfacet = Gq_facet.dat.data_ro_with_halos[je]
                else:
                    Gfacet = numpy.sum(Gq.dat.data_ro_with_halos[je], axis=1)

                for k in range(ncomp):
                    axes = numpy.roll(numpy.arange(tdim), -shift[k])
                    Dfacet = Dfdm[axes[0]]
                    if Dfacet is None:
                        continue

                    if PT_facet:
                        k0 = iord0[k] if shift[1] != 1 else tdim-1-iord0[-k-1]
                        k1 = iord1[k] if shift[1] != 1 else tdim-1-iord1[-k-1]
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
                    if tdim > 1:
                        # assume that the mesh is oriented
                        Ae = Ae.kron(Afdm[axes[1]][0])
                        if tdim > 2:
                            Ae = Ae.kron(Afdm[axes[2]][0])

                    if bsize == ncomp:
                        icell = numpy.reshape(lgmap.apply(k+bsize*ie), (2, -1))
                        rows[0] = pull_axis(icell[0], pshape, idir[0])
                        rows[1] = pull_axis(icell[1], pshape, idir[1])
                    else:
                        assert pshape[k0][idir[0]] == pshape[k1][idir[1]]
                        rows[0] = pull_axis(icell[0][k0], pshape[k0], idir[0])
                        rows[1] = pull_axis(icell[1][k1], pshape[k1], idir[1])

                    update_A(A, Ae, rows)
                    Ae.destroy()

    @PETSc.Log.EventDecorator("FDMCoefficients")
    def assemble_coefficients(self, J, fcp):
        from firedrake.assemble import OneFormAssembler
        coefficients = {}
        assembly_callables = []

        args_J = J.arguments()
        V = args_J[-1].function_space()
        mesh = V.mesh()
        tdim = mesh.topological_dimension()
        Finv = ufl.JacobianInverse(mesh)

        degree = V.ufl_element().degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        quad_deg = fcp.get("degree", 2*degree+1)
        dx = ufl.dx(degree=quad_deg, domain=mesh)
        family = "Discontinuous Lagrange" if tdim == 1 else "DQ"
        DG = ufl.FiniteElement(family, mesh.ufl_cell(), degree=0)

        # extract coefficients directly from the bilinear form
        integrals_J = J.integrals_by_type("cell")
        mapping = args_J[0].ufl_element().mapping().lower()
        Piola = get_piola_tensor(mapping, mesh)

        # get second order coefficient
        ref_grad = [ufl.variable(ufl.grad(t)) for t in args_J]
        if Piola:
            replace_grad = {ufl.grad(t): ufl.dot(Piola, ufl.dot(dt, Finv)) for t, dt in zip(args_J, ref_grad)}
        else:
            replace_grad = {ufl.grad(t): ufl.dot(dt, Finv) for t, dt in zip(args_J, ref_grad)}
        alpha = expand_derivatives(sum([ufl.diff(ufl.diff(ufl.replace(i.integrand(), replace_grad),
                                                 ref_grad[0]), ref_grad[1]) for i in integrals_J]))
        # discard mixed derivatives and mixed components
        if len(alpha.ufl_shape) == 2:
            alpha = ufl.diag_vector(alpha)
        else:
            ashape = alpha.ufl_shape
            ashape = ashape[:len(ashape)//2]
            alpha = ufl.as_tensor(numpy.reshape([alpha[i+i] for i in numpy.ndindex(ashape)], (ashape[0], -1)))

        # assemble second order coefficient
        if not isinstance(alpha, ufl.constantvalue.Zero):
            Q = FunctionSpace(mesh, ufl.TensorElement(DG, shape=alpha.ufl_shape))
            tensor = coefficients.setdefault("alpha", Function(Q.dual()))
            assembly_callables.append(OneFormAssembler(ufl.inner(TestFunction(Q), alpha)*dx, tensor=tensor,
                                                       form_compiler_parameters=fcp).assemble)

        # get zero-th order coefficent
        ref_val = [ufl.variable(t) for t in args_J]
        if Piola:
            dummy_element = ufl.TensorElement(family, cell=mesh.ufl_cell(), degree=1, shape=Piola.ufl_shape)
            dummy_Piola = ufl.Coefficient(ufl.FunctionSpace(mesh, dummy_element))
            replace_val = {t: ufl.dot(dummy_Piola, s) for t, s in zip(args_J, ref_val)}
        else:
            replace_val = {t: s for t, s in zip(args_J, ref_val)}
        beta = expand_derivatives(sum(ufl.diff(ufl.diff(ufl.replace(i.integrand(), replace_val),
                                               ref_val[0]), ref_val[1]) for i in integrals_J))
        if Piola:
            beta = ufl.replace(beta, {dummy_Piola: Piola})
        # assemble zero-th order coefficient
        if not isinstance(beta, ufl.constantvalue.Zero):
            if Piola:
                # keep diagonal
                beta = ufl.diag_vector(beta)
            Q = FunctionSpace(mesh, ufl.TensorElement(DG, shape=beta.ufl_shape) if beta.ufl_shape else DG)
            tensor = coefficients.setdefault("beta", Function(Q.dual()))
            assembly_callables.append(OneFormAssembler(ufl.inner(TestFunction(Q), beta)*dx, tensor=tensor,
                                                       form_compiler_parameters=fcp).assemble)

        family = "CG" if tdim == 1 else "DGT"
        degree = 1 if tdim == 1 else 0
        DGT = ufl.BrokenElement(ufl.FiniteElement(family, cell=mesh.ufl_cell(), degree=degree))
        if Piola:
            # make DGT functions with the second order coefficient
            # and the Piola tensor for each side of each facet
            extruded = mesh.cell_set._extruded
            dS_int = ufl.dS_h(degree=quad_deg) + ufl.dS_v(degree=quad_deg) if extruded else ufl.dS(degree=quad_deg)
            area = ufl.FacetArea(mesh)
            ifacet_inner = lambda v, u: ((ufl.inner(v('+'), u('+')) + ufl.inner(v('-'), u('-')))/area)*dS_int

            replace_grad = {ufl.grad(t): ufl.dot(dt, Finv) for t, dt in zip(args_J, ref_grad)}
            alpha = expand_derivatives(sum(ufl.diff(ufl.diff(ufl.replace(i.integrand(), replace_grad),
                                                    ref_grad[0]), ref_grad[1]) for i in integrals_J))
            G = alpha
            G = ufl.as_tensor([[[G[i, k, j, k] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[2])] for k in range(G.ufl_shape[3])])
            G = G * abs(ufl.JacobianDeterminant(mesh))

            Q = FunctionSpace(mesh, ufl.TensorElement(DGT, shape=G.ufl_shape))
            tensor = coefficients.setdefault("Gq_facet", Function(Q.dual()))
            assembly_callables.append(OneFormAssembler(ifacet_inner(TestFunction(Q), G), tensor=tensor,
                                                       form_compiler_parameters=fcp).assemble)
            PT = Piola.T
            Q = FunctionSpace(mesh, ufl.TensorElement(DGT, shape=PT.ufl_shape))
            tensor = coefficients.setdefault("PT_facet", Function(Q.dual()))
            assembly_callables.append(OneFormAssembler(ifacet_inner(TestFunction(Q), PT), tensor=tensor,
                                                       form_compiler_parameters=fcp).assemble)

        # make DGT functions with BC flags
        shape = V.ufl_element().reference_value_shape()
        Q = FunctionSpace(mesh, ufl.TensorElement(DGT, shape=shape) if shape else DGT)
        test = TestFunction(Q)

        ref_args = [ufl.variable(t) for t in args_J]
        replace_args = {t: s for t, s in zip(args_J, ref_args)}

        forms = []
        md = {"quadrature_degree": 0}
        for it in J.integrals():
            itype = it.integral_type()
            if itype.startswith("exterior_facet"):
                beta = ufl.diff(ufl.diff(ufl.replace(it.integrand(), replace_args), ref_args[0]), ref_args[1])
                beta = expand_derivatives(beta)
                if beta.ufl_shape:
                    beta = ufl.diag_vector(beta)
                ds_ext = ufl.Measure(itype, domain=mesh, subdomain_id=it.subdomain_id(), metadata=md)
                forms.append(ufl.inner(test, beta)*ds_ext)

        tensor = coefficients.setdefault("bcflags", Function(Q.dual()))
        if len(forms):
            form = sum(forms)
            if len(form.arguments()) == 1:
                assembly_callables.append(OneFormAssembler(form, tensor=tensor,
                                                           form_compiler_parameters=fcp).assemble)
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


def numpy_to_petsc(A_numpy, dense_indices, diag=True, block=False, comm=None):
    """
    Create a SeqAIJ Mat from a dense matrix using the diagonal and a subset of rows and columns.
    If dense_indices is empty, then also include the off-diagonal corners of the matrix.
    """
    n = A_numpy.shape[0]
    nbase = int(diag) if block else min(n, int(diag) + len(dense_indices))
    nnz = numpy.full((n,), nbase, dtype=PETSc.IntType)
    nnz[dense_indices] = len(dense_indices) if block else n

    imode = PETSc.InsertMode.INSERT
    A_petsc = PETSc.Mat().createAIJ(A_numpy.shape, nnz=(nnz, 0), comm=comm)
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


def fdm_setup_ipdg(fdm_element, eta, comm=None):
    """
    Setup for the fast diagonalisation method for the IP-DG formulation.
    Compute sparsified interval stiffness and mass matrices
    and tabulate the normal derivative of the shape functions.

    :arg fdm_element: a :class:`FIAT.FDMElement`
    :arg eta: penalty coefficient as a `float`
    :arg comm: a :class:`PETSc.Comm`

    :returns: 3-tuple of:
        Afdm: a list of :class:`PETSc.Mats` with the sparse interval matrices
        Bhat, and bcs(Ahat) for every combination of either natural or weak
        Dirichlet BCs on each endpoint.
        Dfdm: the tabulation of the normal derivatives of the Dirichlet eigenfunctions.
        bdof: the indices of the vertex degrees of freedom.
    """
    ref_el = fdm_element.get_reference_element()
    degree = fdm_element.degree()
    rule = FIAT.quadrature.make_quadrature(ref_el, degree+1)
    edof = fdm_element.entity_dofs()
    bdof = edof[0][0] + edof[0][1]

    phi = fdm_element.tabulate(1, rule.get_points())
    Jhat = phi[(0, )]
    Dhat = phi[(1, )]
    Ahat = numpy.dot(numpy.multiply(Dhat, rule.get_weights()), Dhat.T)
    Bhat = numpy.dot(numpy.multiply(Jhat, rule.get_weights()), Jhat.T)

    # Facet normal derivatives
    basis = fdm_element.tabulate(1, ref_el.get_vertices())
    Dfacet = basis[(1,)]
    Dfacet[:, 0] = -Dfacet[:, 0]

    Afdm = [numpy_to_petsc(Bhat, bdof, block=True, comm=comm)]
    for bc in range(4):
        bcs = (bc % 2, bc//2)
        Abc = Ahat.copy()
        for k in range(2):
            if bcs[k] == 1:
                j = bdof[k]
                Abc[:, j] -= Dfacet[:, k]
                Abc[j, :] -= Dfacet[:, k]
                Abc[j, j] += eta
        Afdm.append(numpy_to_petsc(Abc, bdof, comm=comm))
    return Afdm, Dfacet, bdof


def extrude_interior_facet_maps(V):
    """
    Extrude V.interior_facet_node_map and V.mesh().interior_facets.local_facet_dat

    :arg V: a :class:`.FunctionSpace`

    :returns: the 3-tuple of
        facet_to_nodes_fun: maps interior facets to the nodes of the two cells sharing it,
        local_facet_data_fun: maps interior facets to the local facet numbering in the two cells sharing it,
        nfacets: the total number of interior facets owned by this process
    """
    if isinstance(V, (Function, Cofunction)):
        V = V.function_space()
    mesh = V.mesh()
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


def extrude_node_map(node_map, bsize=1):
    """
    Construct a (possibly vector-valued) cell to node map from an un-extruded scalar map.

    :arg node_map: a :class:`pyop2.Map` mapping entities to their local dofs, including ghost entities.
    :arg bsize: the block size

    :returns: a 2-tuple with the cell to node map and the number of cells owned by this process
    """
    nel = node_map.values.shape[0]
    if node_map.offset is None:
        def _scalar_map(map_values, e, result=None):
            if result is None:
                result = numpy.empty_like(map_values[e])
            numpy.copyto(result, map_values[e])
            return result

        scalar_map = partial(_scalar_map, node_map.values_with_halo)
    else:
        layers = node_map.iterset.layers_array
        if layers.shape[0] == 1:
            def _scalar_map(map_values, offset, nelz, e, result=None):
                if result is None:
                    result = numpy.empty_like(offset)
                numpy.copyto(result, offset)
                result *= (e % nelz)
                result += map_values[e // nelz]
                return result

            nelz = layers[0, 1]-layers[0, 0]-1
            nel *= nelz
            scalar_map = partial(_scalar_map, node_map.values_with_halo, node_map.offset, nelz)
        else:
            def _scalar_map(map_values, offset, to_base, to_layer, e, result=None):
                if result is None:
                    result = numpy.empty_like(offset)
                numpy.copyto(result, offset)
                result *= to_layer[e]
                result += map_values[to_base[e]]
                return result

            nelz = layers[:, 1]-layers[:, 0]-1
            nel = sum(nelz[:nel])
            to_base = numpy.repeat(numpy.arange(node_map.values_with_halo.shape[0], dtype=node_map.offset.dtype), nelz)
            to_layer = numpy.concatenate([numpy.arange(nz, dtype=node_map.offset.dtype) for nz in nelz])
            scalar_map = partial(_scalar_map, node_map.values_with_halo, node_map.offset, to_base, to_layer)

    if bsize == 1:
        return scalar_map, nel

    def vector_map(bsize, ibase, e, result=None):
        index = None
        if result is not None:
            index = result[:, 0]
        index = scalar_map(e, result=index)
        index *= bsize
        return numpy.add.outer(index, ibase, out=result)

    ibase = numpy.arange(bsize, dtype=node_map.values.dtype)
    return partial(vector_map, bsize, ibase), nel
