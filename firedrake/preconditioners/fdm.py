from functools import partial
from itertools import product
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.pmg import (prolongation_matrix_matfree,
                                           fiat_reference_prolongator,
                                           get_permutation_to_line_elements)
from firedrake.preconditioners.facet_split import split_dofs, restricted_dofs
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.function import Function
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
import ctypes
import numpy
import ufl
import FIAT
import finat

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
    functions so that the H(d) Riesz map is sparse on Cartesian cells,
    and assembles a global sparse matrix on which other preconditioners,
    such as `ASMStarPC`, can be applied.

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

    @staticmethod
    def setSubMatCSR(comm, triu=False):
        """
        Compile C code to insert sparse submatrices and store in class cache

        :arg triu: are we inserting onto the upper triangular part of the matrix?

        :returns: a python wrapper for the matrix insertion function
        """
        cache = FDMPC._cache.setdefault("setSubMatCSR", {})
        key = triu
        try:
            return cache[key]
        except KeyError:
            return cache.setdefault(key, load_setSubMatCSR(comm, triu))

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
            J_fdm = J(*[t.reconstruct(function_space=V_fdm) for t in J.arguments()], coefficients={})
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
        Pmat, self._assemble_P = self.allocate_matrix(V_fdm, J_fdm, bcs_fdm, fcp, pmat_type, use_static_condensation)
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
        :arg pmat_type: the preconditioner `PETSc.Mat.Type`
        :arg use_static_condensation: are we assembling the statically-condensed Schur complement on facets?

        :returns: 2-tuple with the preconditioner :class:`PETSc.Mat` and its assembly callable
        """
        ifacet = [i for i, Vsub in enumerate(V) if is_restricted(Vsub.finat_element)[1]]
        if len(ifacet) == 0:
            Vfacet = None
            Vbig = V
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

        value_size = Vbig.value_size
        if value_size != 1:
            fdofs = numpy.add.outer(value_size * fdofs, numpy.arange(value_size, dtype=fdofs.dtype))
        dofs = numpy.arange(value_size * Vbig.finat_element.space_dimension(), dtype=fdofs.dtype)
        idofs = numpy.setdiff1d(dofs, fdofs, assume_unique=True)
        self.ises = tuple(PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF) for indices in (idofs, fdofs))
        self.submats = [None for _ in range(7)]

        # Dictionary with the parent space and a method to form the Schur complement
        self.get_static_condensation = {}
        if Vfacet and use_static_condensation:
            # If we are in a facet space, we build the Schur complement on its diagonal block
            diagonal_interior = Vfacet.finat_element.formdegree == 0 and value_size == 1
            get_schur = schur_complement_diagonal if diagonal_interior else schur_complement_block_qr
            self.get_static_condensation[Vfacet] = Vbig, lambda A: condense_element_mat(A, self.ises[0], self.ises[1],
                                                                                        self.submats, get_schur)

        elif len(fdofs) and V.finat_element.formdegree == 0:
            # If we are in H(grad), we just pad with zeros on the statically-condensed pattern
            i1 = PETSc.IS().createGeneral(dofs, comm=PETSc.COMM_SELF)
            self.get_static_condensation[V] = Vbig, lambda Ae: condense_element_pattern(Ae, self.ises[0], i1, self.submats)

        @PETSc.Log.EventDecorator("FDMGetIndices")
        def cell_to_global(lgmap, cell_to_local, cell_index, result=None):
            # Be careful not to create new arrays
            result = cell_to_local(cell_index, result=result)
            return lgmap.apply(result, result=result)

        # Create data structures needed for assembly
        self.cell_to_global = {}
        self.lgmaps = {}
        bc_rows = {}
        for Vsub in V:
            lgmap = Vsub.local_to_global_map([bc for bc in bcs if bc.function_space() == Vsub])
            bsize = Vsub.dof_dset.layout_vec.getBlockSize()
            cell_to_local, nel = extrude_node_map(Vsub.cell_node_map(), bsize=bsize)
            self.cell_to_global[Vsub] = partial(cell_to_global, lgmap, cell_to_local)
            self.lgmaps[Vsub] = lgmap

            own = Vsub.dof_dset.layout_vec.getLocalSize()
            bdofs = numpy.nonzero(lgmap.indices[:own] < 0)[0].astype(PETSc.IntType)
            bc_rows[Vsub] = Vsub.dof_dset.lgmap.apply(bdofs, result=bdofs)
        self.nel = nel

        coefficients, assembly_callables = self.assemble_coefficients(J, fcp)
        coeffs = [coefficients.get(name) for name in ("beta", "alpha")]
        cdata = [c.dat.data_ro for c in coeffs]
        cmaps = [extrude_node_map(c.cell_node_map())[0] for c in coeffs]
        cindices = [cmap(0) if self.nel else None for cmap in cmaps]

        @PETSc.Log.EventDecorator("FDMGetCoeffs")
        def get_coeffs(e, result=None):
            # Get vector for betas and alphas on a cell
            if result is None:
                return numpy.concatenate([c[cmap(e, result=idx)] for c, cmap, idx in zip(cdata, cmaps, cindices)], out=result)
            numpy.take(cdata[0], cmaps[0](e, result=cindices[0]), axis=0, out=result[:cindices[0].size])
            numpy.take(cdata[1], cmaps[1](e, result=cindices[1]), axis=0, out=result[cindices[0].size:])
            return result

        self.get_coeffs = get_coeffs
        self.work_mats = {}

        Pmats = {}
        addv = PETSc.InsertMode.ADD_VALUES
        symmetric = pmat_type.endswith("sbaij")

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
                if ptype.endswith("sbaij"):
                    P.setOption(PETSc.Mat.Option.IGNORE_LOWER_TRIANGULAR, True)
                P.setUp()
            Pmats[Vrow, Vcol] = P

        if len(V) == 1:
            Pmat = Pmats[V, V]
        else:
            Pmat = PETSc.Mat().createNest([[Pmats[Vrow, Vcol] for Vcol in V] for Vrow in V], comm=self.comm)

        @PETSc.Log.EventDecorator("FDMAssemble")
        def assemble_P():
            for _assemble in assembly_callables:
                _assemble()
            for Vrow, Vcol in product(Vsort, Vsort):
                P = Pmats[Vrow, Vcol]
                if P.getType().endswith("aij"):
                    P.zeroEntries()
                    if Vrow == Vcol and len(bc_rows[Vrow]) > 0:
                        rows = bc_rows[Vrow][:, None]
                        vals = numpy.ones(rows.shape, dtype=PETSc.RealType)
                        P.setValuesRCV(rows, rows, vals, addv)
                    self.set_values(P, Vrow, Vcol, addv)
            Pmat.assemble()

        return Pmat, assemble_P

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
        objs = []
        if hasattr(self, "A"):
            objs.append(self.A)
        if hasattr(self, "pc"):
            objs.append(self.pc)
            objs.append(self.pc.getOperators()[-1])
        if hasattr(self, "submats"):
            objs.extend(self.submats)
        if hasattr(self, "work_mats"):
            objs.extend(list(self.work_mats.values()))
        if hasattr(self, "ises"):
            objs.extend(self.ises)
        for obj in objs:
            if hasattr(obj, "destroy"):
                obj.destroy()

    @cached_property
    def _element_mass_matrix(self):
        data = self.get_coeffs(0)
        data.fill(1.0E0)
        shape = data.shape + (1,)*(3-len(data.shape))
        nrows = shape[0] * shape[1]
        ai = numpy.arange(nrows+1, dtype=PETSc.IntType)
        aj = numpy.tile(ai[:-1].reshape((-1, shape[1])), (1, shape[2]))
        if shape[2] > 1:
            ai *= shape[2]
            data = numpy.tile(numpy.eye(shape[2], dtype=data.dtype), shape[:1] + (1,)*(len(shape)-1))
        Me = PETSc.Mat().createAIJ((nrows, nrows), bsize=shape[2], csr=(ai, aj, data), comm=PETSc.COMM_SELF)
        return self.work_mats.setdefault("mass_matrix", Me)

    @cached_property
    def _element_mass_diagonal(self):
        return self.work_mats.setdefault("mass_diagonal", self._element_mass_matrix.getDiagonal())

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
        if self.nel == 0:
            # This MPI rank does not own any elements, nothing to be done
            return

        Vbig = None
        condense_element_mat = lambda x: x
        set_submat = self.setSubMatCSR(PETSc.COMM_SELF, triu=triu)
        get_rindices = self.cell_to_global[Vrow]
        if Vrow == Vcol:
            get_cindices = lambda e, result=None: result
            update_A = lambda Ae, rindices, cindices: set_submat(A, Ae, rindices, rindices, addv)
            Vbig, condense_element_mat = self.get_static_condensation.get(Vrow, (Vbig, condense_element_mat))
        else:
            get_cindices = self.cell_to_global[Vcol]
            update_A = lambda Ae, rindices, cindices: set_submat(A, Ae, rindices, cindices, addv)

        Me = self._element_mass_matrix
        # interpolation of basis and exterior derivative onto broken spaces
        ctensor = self.assemble_reference_tensor(Vbig or Vcol)
        rtensor = self.assemble_reference_tensor(Vbig or Vrow, transpose=True)
        # element matrix obtained via Equation (3.9) of Brubeck2022b
        assemble_element_mat = partial(rtensor.matMatMult, Me, ctensor)
        try:
            Ae = self.work_mats[Vrow, Vcol]
        except KeyError:
            Ae = self.work_mats.setdefault((Vrow, Vcol), assemble_element_mat())

        insert = PETSc.InsertMode.INSERT
        if A.getType() == PETSc.Mat.Type.PREALLOCATOR:
            # Empty kernel for preallocation
            if Vbig is not None:
                sort_interior_dofs(self.ises[0], Ae)
            Se = condense_element_mat(Ae)
            element_kernel = lambda e, result=None: result
            condense_element_mat = lambda Ae: Se
        elif Me.getBlockSize() == 1:
            # Kernel with diagonal mass matrix
            diagonal = self._element_mass_diagonal
            data = diagonal.array_w.reshape((-1,) + Vrow.shape)

            def element_kernel(e, result=None):
                self.get_coeffs(e, result=data)
                Me.setDiagonal(diagonal, addv=insert)
                return assemble_element_mat(result=result)
        else:
            # Kernel with block diagonal mass matrix
            ai, aj, data = Me.getValuesCSR()
            data = data.reshape((-1,) + Vrow.shape * 2)

            def element_kernel(e, result=None):
                self.get_coeffs(e, result=data)
                Me.setValuesCSR(ai, aj, data, addv=insert)
                Me.assemble()
                return assemble_element_mat(result=result)

        cindices = None
        rindices = None
        # Core assembly loop
        for e in range(self.nel):
            cindices = get_cindices(e, result=cindices)
            rindices = get_rindices(e, result=rindices)
            Ae = element_kernel(e, result=Ae)
            update_A(condense_element_mat(Ae), rindices, cindices)

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
        formdegree = V.finat_element.formdegree
        degree = e.degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
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

        # Return coefficients and assembly callables, and cache them class
        key = (mixed_form.signature(), mesh)
        cache = self._cache.setdefault("coefficients", {})
        try:
            return cache[key]
        except KeyError:
            if block_diagonal and V.shape:
                from firedrake.assemble import assemble
                M = assemble(mixed_form, mat_type="matfree",
                             form_compiler_parameters=fcp)
                coefficients = {}
                assembly_callables = []
                for iset, name in zip(Z.dof_dset.field_ises, ("beta", "alpha")):
                    sub = M.petscmat.createSubMatrix(iset, iset)
                    ctx = sub.getPythonContext()
                    coefficients[name] = ctx._block_diagonal
                    assembly_callables.append(ctx._assemble_block_diagonal)
            else:
                from firedrake.assemble import OneFormAssembler
                tensor = Function(Z)
                coefficients = {"beta": tensor.sub(0), "alpha": tensor.sub(1)}
                assembly_callables = [OneFormAssembler(mixed_form, tensor=tensor, diagonal=True,
                                                       form_compiler_parameters=fcp).assemble]
            return cache.setdefault(key, (coefficients, assembly_callables))

    @PETSc.Log.EventDecorator("FDMRefTensor")
    def assemble_reference_tensor(self, V, transpose=False):
        """
        Return the reference tensor used in the diagonal factorisation of the
        sparse cell matrices.  See Section 3.2 of Brubeck2022b.

        :arg V: a :class:`.FunctionSpace`

        :returns: a :class:`PETSc.Mat` interpolating V^k * d(V^k) onto
                  broken(V^k) * broken(V^{k+1}) on the reference element.
        """
        tdim = V.mesh().topological_dimension()
        value_size = V.value_size
        formdegree = V.finat_element.formdegree
        degree = V.finat_element.degree
        try:
            degree = max(degree)
        except TypeError:
            pass
        if formdegree == tdim:
            degree = degree + 1
        is_interior, is_facet = is_restricted(V.finat_element)
        key = (degree, tdim, formdegree, value_size, is_interior, is_facet, transpose)
        cache = self._cache.setdefault("reference_tensor", {})
        try:
            return cache[key]
        except KeyError:
            if transpose:
                result = self.assemble_reference_tensor(V, transpose=False)
                result = PETSc.Mat().createTranspose(result).convert(result.getType())
                return cache.setdefault(key, result)

            full_key = (degree, tdim, formdegree, value_size, False, False, False)
            if is_facet and full_key in cache:
                result = cache[full_key]
                noperm = PETSc.IS().createGeneral(numpy.arange(result.getSize()[0], dtype=PETSc.IntType), comm=result.getComm())
                result = result.createSubMatrix(noperm, self.ises[1])
                noperm.destroy()
                return cache.setdefault(key, result)

            elements = sorted(get_base_elements(V.finat_element), key=lambda e: e.formdegree)
            ref_el = elements[0].get_reference_element()
            eq = FIAT.FDMQuadrature(ref_el, degree)
            e0 = elements[0] if elements[0].formdegree == 0 else FIAT.FDMLagrange(ref_el, degree)
            e1 = elements[-1] if elements[-1].formdegree == 1 else FIAT.FDMDiscontinuousLagrange(ref_el, degree-1)
            if is_interior:
                e0 = FIAT.RestrictedElement(e0, restriction_domain="interior")

            comm = PETSc.COMM_SELF
            A00 = petsc_sparse(fiat_reference_prolongator(e0, eq), comm=comm)
            A10 = petsc_sparse(fiat_reference_prolongator(e0, e1, derivative=True), comm=comm)
            A11 = petsc_sparse(numpy.eye(e1.space_dimension(), dtype=PETSc.RealType), comm=comm)
            B_blocks = mass_blocks(tdim, formdegree, A00, A11)
            A_blocks = diff_blocks(tdim, formdegree, A00, A11, A10)
            result = block_mat(B_blocks + A_blocks, destroy_blocks=True)
            A00.destroy()
            A10.destroy()
            A11.destroy()

            if value_size != 1:
                eye = petsc_sparse(numpy.eye(value_size), comm=comm)
                temp = result
                result = temp.kron(eye)
                temp.destroy()
                eye.destroy()

            if is_facet:
                cache[full_key] = result
                noperm = PETSc.IS().createGeneral(numpy.arange(result.getSize()[0], dtype=PETSc.IntType), comm=result.getComm())
                result = result.createSubMatrix(noperm, self.ises[1])
                noperm.destroy()

            return cache.setdefault(key, result)


@PETSc.Log.EventDecorator("FDMGetSchur")
def schur_complement_diagonal(submats):
    """
    Used in static condensation. Take in blocks A00, A01, A10, A11,
    return the Schur complement A11 - A10 * inv(A00) * A01.

    Assumes A00 is diagonal.
    """
    structure = PETSc.Mat.Structure.SUBSET if submats[-1] else None
    A00, A01, A10, A11 = submats[:4]
    submats[4] = A00.getDiagonal(result=submats[4])
    submats[4].reciprocal()
    submats[4].scale(-1)
    A01.diagonalScale(L=submats[4])
    submats[-1] = A10.matMult(A01, result=submats[-1])
    submats[-1].axpy(1.0, A11, structure=structure)
    return submats[-1]


@PETSc.Log.EventDecorator("FDMGetSchur")
def schur_complement_block_inv(submats):
    """
    Used in static condensation. Take in blocks A00, A01, A10, A11,
    return A11 - A10 * inv(A00) * A01.

    Assumes that interior DOFs have been reordered to make A00
    block diagonal with blocks of increasing dimension.
    """
    structure = PETSc.Mat.Structure.SUBSET if submats[-1] else None
    A00, A01, A10, A11 = submats[:4]
    indptr, indices, R = A00.getValuesCSR()
    degree = numpy.diff(indptr)

    nblocks = numpy.count_nonzero(degree == 1)
    zlice = slice(0, nblocks)
    numpy.reciprocal(R[zlice], out=R[zlice])
    flops = nblocks
    for k in range(2, degree[-1]+1):
        nblocks = numpy.count_nonzero(degree == k)
        zlice = slice(zlice.stop, zlice.stop + k*nblocks)
        A = R[zlice].reshape((-1, k, k))
        R[zlice] = numpy.linalg.inv(A).reshape((-1,))
        flops += nblocks * (k**3)

    PETSc.Log.logFlops(flops)
    A00.setValuesCSR(indptr, indices, R)
    A00.assemble()
    A00.scale(-1.0)
    submats[-1] = A10.matMatMult(A00, A01, result=submats[-1])
    submats[-1].axpy(1.0, A11, structure=structure)
    return submats[-1]


@PETSc.Log.EventDecorator("FDMGetSchur")
def schur_complement_block_cholesky(submats):
    """
    Used in static condensation. Take in blocks A00, A01, A10, A11,
    return A11 - A10 * inv(A00) * A01.

    Assumes that interior DOFs have been reordered to make A00
    block diagonal with blocks of increasing dimension.
    """
    structure = PETSc.Mat.Structure.SUBSET if submats[-1] else None
    A00, A01, A10, A11 = submats[:4]
    indptr, indices, R = A00.getValuesCSR()
    degree = numpy.diff(indptr)

    nblocks = numpy.count_nonzero(degree == 1)
    zlice = slice(0, nblocks)
    numpy.sqrt(R[zlice], out=R[zlice])
    numpy.reciprocal(R[zlice], out=R[zlice])
    flops = 2*nblocks
    for k in range(2, degree[-1]+1):
        nblocks = numpy.count_nonzero(degree == k)
        zlice = slice(zlice.stop, zlice.stop + k*nblocks)
        A = R[zlice].reshape((-1, k, k))
        R[zlice] = numpy.linalg.inv(numpy.linalg.cholesky(A)).reshape((-1))
        flops += nblocks * ((k**3)//3 + k**3)

    PETSc.Log.logFlops(flops)
    A00.setValuesCSR(indptr, indices, R)
    A00.assemble()
    submats[4] = A10.matTransposeMult(A00, result=submats[4])
    A00.scale(-1.0)
    submats[-1] = submats[4].matMatMult(A00, A01, result=submats[-1])
    submats[-1].axpy(1.0, A11, structure=structure)
    return submats[-1]


@PETSc.Log.EventDecorator("FDMGetSchur")
def schur_complement_block_qr(submats):
    """
    Used in static condensation. Take in blocks A00, A01, A10, A11,
    return A11 - A10 * inv(A00) * A01.

    Assumes that interior DOFs have been reordered to make A00
    block diagonal with blocks of increasing dimension.
    """
    structure = PETSc.Mat.Structure.SUBSET if submats[-1] else None
    A00, A01, A10, A11 = submats[:4]
    indptr, indices, R = A00.getValuesCSR()
    degree = numpy.diff(indptr)
    Q = numpy.ones(R.shape, dtype=R.dtype)

    nblocks = numpy.count_nonzero(degree == 1)
    zlice = slice(0, nblocks)
    numpy.reciprocal(R[zlice], out=R[zlice])
    flops = nblocks
    for k in range(2, degree[-1]+1):
        nblocks = numpy.count_nonzero(degree == k)
        zlice = slice(zlice.stop, zlice.stop + k*nblocks)
        A = R[zlice].reshape((-1, k, k))
        q, r = numpy.linalg.qr(A, mode="complete")
        Q[zlice] = q.reshape((-1,))
        R[zlice] = numpy.linalg.inv(r).reshape((-1,))
        flops += nblocks * ((4*k**3)//3 + k**3)

    PETSc.Log.logFlops(flops)
    A00.setValuesCSR(indptr, indices, Q)
    A00.assemble()
    submats[4] = A00.transposeMatMult(A01, result=submats[4])
    A00.setValuesCSR(indptr, indices, R)
    A00.assemble()
    A00.scale(-1.0)
    submats[-1] = A10.matMatMult(A00, submats[4], result=submats[-1])
    submats[-1].axpy(1.0, A11, structure=structure)
    return submats[-1]


@PETSc.Log.EventDecorator("FDMGetSchur")
def schur_complement_block_svd(submats):
    """
    Used in static condensation. Take in blocks A00, A01, A10, A11,
    return A11 - A10 * inv(A00) * A01.

    Assumes that interior DOFs have been reordered to make A00
    block diagonal with blocks of increasing dimension.
    """
    structure = PETSc.Mat.Structure.SUBSET if submats[-1] else None
    A00, A01, A10, A11 = submats[:4]
    indptr, indices, U = A00.getValuesCSR()
    degree = numpy.diff(indptr)
    V = numpy.ones(U.shape, dtype=U.dtype)
    submats[4] = A00.getDiagonal(result=submats[4])
    D = submats[4]

    nblocks = numpy.count_nonzero(degree == 1)
    bslice = slice(0, nblocks)
    dslice = slice(0, nblocks)
    numpy.sign(D.array_r[dslice], out=U[bslice])

    flops = nblocks
    for k in range(2, degree[-1]+1):
        nblocks = numpy.count_nonzero(degree == k)
        bslice = slice(bslice.stop, bslice.stop + k*nblocks)
        dslice = slice(dslice.stop, dslice.stop + nblocks)
        A = U[bslice].reshape((-1, k, k))

        u, s, v = numpy.linalg.svd(A, full_matrices=False)
        D.array_w[dslice] = s.reshape((-1,))
        U[bslice] = numpy.transpose(u, axes=(0, 2, 1)).reshape((-1,))
        V[bslice] = numpy.transpose(v, axes=(0, 2, 1)).reshape((-1,))
        flops += nblocks * ((4*k**3)//3 + 4*k**3)

    PETSc.Log.logFlops(flops)
    D.sqrtabs()
    D.reciprocal()
    A00.setValuesCSR(indptr, indices, V)
    A00.assemble()
    A00.diagonalScale(R=D)
    submats[5] = A10.matMult(A00, result=submats[5])
    D.scale(-1.0)
    A00.setValuesCSR(indptr, indices, U)
    A00.assemble()
    A00.diagonalScale(L=D)
    submats[-1] = submats[5].matMatMult(A00, A01, result=submats[-1])
    submats[-1].axpy(1.0, A11, structure=structure)
    return submats[-1]


@PETSc.Log.EventDecorator("FDMCondense")
def condense_element_mat(A, i0, i1, submats, get_schur_complement):
    """Return the Schur complement associated to indices in i1, condensing i0 out"""
    isrows = [i0, i0, i1, i1]
    iscols = [i0, i1, i0, i1]
    submats[:4] = A.createSubMatrices(isrows, iscols=iscols, submats=submats[:4] if submats[0] else None)
    return get_schur_complement(submats)


@PETSc.Log.EventDecorator("FDMCondense")
def condense_element_pattern(A, i0, i1, submats):
    """Add zeroes on the statically condensed pattern so that you can run ICC(0)"""
    isrows = [i0, i0, i1]
    iscols = [i0, i1, i0]
    structure = PETSc.Mat.Structure.SUBSET if submats[3] else None
    submats[:3] = A.createSubMatrices(isrows, iscols=iscols, submats=submats[:3] if submats[0] else None)
    A00, A01, A10 = submats[:3]
    A00.scale(0.0)
    submats[3] = A10.matMatMult(A00, A01, result=submats[3])
    submats[3].axpy(1.0, A, structure=structure)
    return submats[3]


@PETSc.Log.EventDecorator("LoadCode")
def load_c_code(code, name, **kwargs):
    cppargs = ["-I%s/include" % d for d in get_petsc_dir()]
    ldargs = (["-L%s/lib" % d for d in get_petsc_dir()]
              + ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()]
              + ["-lpetsc", "-lm"])
    funptr = load(code, "c", name,
                  cppargs=cppargs, ldargs=ldargs,
                  **kwargs)

    def get_pointer(obj):
        if isinstance(obj, PETSc.Object):
            return obj.handle
        elif isinstance(obj, numpy.ndarray):
            return obj.ctypes.data
        return obj

    @PETSc.Log.EventDecorator(name)
    def wrapper(*args):
        return funptr(*list(map(get_pointer, args)))
    return wrapper


def load_setSubMatCSR(comm, triu=False):
    """Insert one sparse matrix into another sparse matrix.
       Done in C for efficiency, since it loops over rows."""
    if triu:
        name = "setSubMatCSR_SBAIJ"
        select_cols = "icol < irow ? -1: icol"
    else:
        name = "setSubMatCSR_AIJ"
        select_cols = "icol"
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
            indices[j] = {select_cols};
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
    return load_c_code(code, name, comm=comm, argtypes=argtypes,
                       restype=ctypes.c_int)


def is_restricted(finat_element):
    """Determine if an element is a restriction onto interior or facets"""
    is_interior = True
    is_facet = True
    cell_dim = finat_element.cell.get_dimension()
    entity_dofs = finat_element.entity_dofs()
    for dim in sorted(entity_dofs):
        if any(len(entity_dofs[dim][entity]) > 0 for entity in entity_dofs[dim]):
            if dim == cell_dim:
                is_facet = False
            else:
                is_interior = False
    return is_interior, is_facet


def sort_interior_dofs(idofs, A):
    """Permute `idofs` to have A[idofs, idofs] with square blocks of
       increasing dimension along its diagonal."""
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
    Aii.destroy()


def petsc_sparse(A_numpy, rtol=1E-10, comm=None):
    """Convert dense numpy matrix into a sparse PETSc matrix"""
    atol = rtol * max(A_numpy.min(), A_numpy.max(), key=abs)
    sparsity = abs(A_numpy) > atol
    nnz = numpy.count_nonzero(sparsity, axis=1).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ(A_numpy.shape, nnz=(nnz, 0), comm=comm)
    for row, (Arow, Srow) in enumerate(zip(A_numpy, sparsity)):
        cols = numpy.argwhere(Srow).astype(PETSc.IntType).flat
        A.setValues(row, cols, Arow[cols], PETSc.InsertMode.INSERT)
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
        zero = PETSc.Mat().createAIJ(B_diag[0].getSize(), nnz=(0, 0), comm=B_diag[0].getComm())
        zero.assemble()
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
            size = tuple(A11.getSize()[k] * A10.getSize()[k] * A00.getSize()[k] for k in range(2))
            zero = PETSc.Mat().createAIJ(size, nnz=(0, 0), comm=A10.getComm())
            zero.assemble()
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

    elements = list(set(get_base_elements(ec) + get_base_elements(ef)))
    elements = sorted(elements, key=lambda e: e.formdegree)
    e0, e1 = elements[::len(elements)-1]

    degree = e0.degree()
    tdim = Vc.mesh().topological_dimension()
    A00 = petsc_sparse(numpy.eye(degree+1, dtype=PETSc.RealType), comm=PETSc.COMM_SELF)
    A10 = petsc_sparse(fiat_reference_prolongator(e0, e1, derivative=True), comm=PETSc.COMM_SELF)
    A11 = petsc_sparse(numpy.eye(degree, dtype=PETSc.RealType), comm=PETSc.COMM_SELF)
    Dhat = block_mat(diff_blocks(tdim, ec.formdegree, A00, A11, A10), destroy_blocks=True)
    A00.destroy()
    A10.destroy()
    A11.destroy()

    if any(is_restricted(ec)) or any(is_restricted(ef)):
        scalar_element = lambda e: e._sub_element if isinstance(e, (ufl.TensorElement, ufl.VectorElement)) else e
        fdofs = restricted_dofs(ef, create_element(unrestrict_element(scalar_element(Vf.ufl_element()))))
        cdofs = restricted_dofs(ec, create_element(unrestrict_element(scalar_element(Vc.ufl_element()))))
        fises = PETSc.IS().createGeneral(fdofs, comm=PETSc.COMM_SELF)
        cises = PETSc.IS().createGeneral(cdofs, comm=PETSc.COMM_SELF)
        temp = Dhat
        Dhat = temp.createSubMatrix(fises, cises)
        temp.destroy()
        fises.destroy()
        cises.destroy()

    if Vf.value_size > 1:
        temp = Dhat
        eye = petsc_sparse(numpy.eye(Vf.value_size, dtype=PETSc.RealType), comm=PETSc.COMM_SELF)
        Dhat = temp.kron(eye)
        temp.destroy()
        eye.destroy()

    rmap = Vf.local_to_global_map(fbcs)
    cmap = Vc.local_to_global_map(cbcs)
    rlocal, nel = extrude_node_map(Vf.cell_node_map(), bsize=Vf.value_size)
    clocal, nel = extrude_node_map(Vc.cell_node_map(), bsize=Vc.value_size)

    def cell_to_global(lgmap, cell_to_local, e, result=None):
        result = cell_to_local(e, result=result)
        return lgmap.apply(result, result=result)

    imode = PETSc.InsertMode.INSERT
    update_Dmat = FDMPC.setSubMatCSR(PETSc.COMM_SELF, triu=False)

    sizes = tuple(V.dof_dset.layout_vec.getSizes() for V in (Vf, Vc))
    block_size = Vf.dof_dset.layout_vec.getBlockSize()
    preallocator = PETSc.Mat().create(comm=comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
    preallocator.setSizes(sizes)
    preallocator.setUp()

    rindices = None
    cindices = None
    for e in range(nel):
        rindices = cell_to_global(rmap, rlocal, e, result=rindices)
        cindices = cell_to_global(cmap, clocal, e, result=cindices)
        update_Dmat(preallocator, Dhat, rindices, cindices, imode)

    preallocator.assemble()
    nnz = get_preallocation(preallocator, sizes[0][0])
    preallocator.destroy()
    Dmat = PETSc.Mat().createAIJ(sizes, block_size, nnz=nnz, comm=comm)
    Dmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

    for e in range(nel):
        rindices = cell_to_global(rmap, rlocal, e, result=rindices)
        cindices = cell_to_global(cmap, clocal, e, result=cindices)
        update_Dmat(Dmat, Dhat, rindices, cindices, imode)

    Dmat.assemble()
    Dhat.destroy()
    return Dmat


def unrestrict_element(ele):
    """Get an element that might or might not be restricted and
       return the parent unrestricted element."""
    if isinstance(ele, ufl.VectorElement):
        return type(ele)(unrestrict_element(ele._sub_element), dim=ele.num_sub_elements())
    elif isinstance(ele, ufl.TensorElement):
        return type(ele)(unrestrict_element(ele._sub_element), shape=ele._shape, symmetry=ele.symmetry())
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


def get_base_elements(e):
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
            _, line_elements, shifts = get_permutation_to_line_elements(V.finat_element)
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
        set_submat = self.setSubMatCSR(PETSc.COMM_SELF, triu=triu)
        update_A = lambda A, Ae, rindices: set_submat(A, Ae, rindices, rindices, addv)
        condense_element_mat = lambda x: x

        get_rindices = self.cell_to_global[Vrow]
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
        rindices = None
        use_diag_Bq = Bq is None or len(Bq.ufl_shape) != 2 or static_condensation
        if not use_diag_Bq:
            bshape = Bq.ufl_shape
            # Be = Bhat kron ... kron Bhat
            Be = Afdm[0][0].copy()
            for k in range(1, tdim):
                Be = Be.kron(Afdm[k][0])

            aptr = numpy.arange(0, (bshape[0]+1)*bshape[1], bshape[1], dtype=PETSc.IntType)
            aidx = numpy.tile(numpy.arange(bshape[1], dtype=PETSc.IntType), bshape[0])
            for e in range(self.nel):
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
        # discarding mixed derivatives and mixed componentsget_weak_bc_flags(J)
        mue = numpy.zeros((ncomp, tdim), dtype=PETSc.RealType)
        bqe = numpy.zeros((ncomp,), dtype=PETSc.RealType)

        for e in range(self.nel):
            je = index_coef(e)
            bce = bcflags.dat.data_ro_with_halos[index_bc(e)] > 1E-8

            rindices = get_rindices(e, result=rindices)
            rows = numpy.reshape(rindices, (-1, bsize))
            rows = numpy.transpose(rows)
            rows = numpy.reshape(rows, (ncomp, -1))

            # get second order coefficient on this cell
            if Gq is not None:
                numpy.sum(Gq.dat.data_ro[je], axis=0, out=mue)
            # get zero-th order coefficient on this cell
            if Bq is not None:
                numpy.sum(Bq.dat.data_ro[je], axis=0, out=bqe)

            for k in range(ncomp):
                # permutation of axes with respect to the first vector component
                axes = numpy.roll(numpy.arange(tdim), -shift[k])
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

                    if tdim > 1:
                        # Ae = Ae kron Bhat + mue[k][1] Bhat kron Ahat
                        Ae = Ae.kron(Afdm[axes[1]][0])
                        if Gq is not None:
                            Ae.axpy(mue[k][1], Be.kron(Afdm[axes[1]][1+fbc[1]]))

                        if tdim > 2:
                            # Ae = Ae kron Bhat + mue[k][2] Bhat kron Bhat kron Ahat
                            Be = Be.kron(Afdm[axes[1]][0])
                            Ae = Ae.kron(Afdm[axes[2]][0])
                            if Gq is not None:
                                Ae.axpy(mue[k][2], Be.kron(Afdm[axes[2]][1+fbc[2]]))
                    Be.destroy()

                elif Bq is not None:
                    Ae = Afdm[axes[0]][0]
                    for m in range(1, tdim):
                        Ae = Ae.kron(Afdm[axes[m]][0])
                    Ae.scale(bqe[k])

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
            tensor = coefficients.setdefault("alpha", Function(Q))
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
        beta = expand_derivatives(sum([ufl.diff(ufl.diff(ufl.replace(i.integrand(), replace_val),
                                                ref_val[0]), ref_val[1]) for i in integrals_J]))
        if Piola:
            beta = ufl.replace(beta, {dummy_Piola: Piola})
        # assemble zero-th order coefficient
        if not isinstance(beta, ufl.constantvalue.Zero):
            if Piola:
                # keep diagonal
                beta = ufl.diag_vector(beta)
            Q = FunctionSpace(mesh, ufl.TensorElement(DG, shape=beta.ufl_shape) if beta.ufl_shape else DG)
            tensor = coefficients.setdefault("beta", Function(Q))
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
            alpha = expand_derivatives(sum([ufl.diff(ufl.diff(ufl.replace(i.integrand(), replace_grad),
                                                     ref_grad[0]), ref_grad[1]) for i in integrals_J]))
            G = alpha
            G = ufl.as_tensor([[[G[i, k, j, k] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[2])] for k in range(G.ufl_shape[3])])
            G = G * abs(ufl.JacobianDeterminant(mesh))

            Q = FunctionSpace(mesh, ufl.TensorElement(DGT, shape=G.ufl_shape))
            tensor = coefficients.setdefault("Gq_facet", Function(Q))
            assembly_callables.append(OneFormAssembler(ifacet_inner(TestFunction(Q), G), tensor=tensor,
                                                       form_compiler_parameters=fcp).assemble)
            PT = Piola.T
            Q = FunctionSpace(mesh, ufl.TensorElement(DGT, shape=PT.ufl_shape))
            tensor = coefficients.setdefault("PT_facet", Function(Q))
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

        if len(forms):
            form = sum(forms)
            if len(form.arguments()) == 1:
                tensor = coefficients.setdefault("bcflags", Function(Q))
                assembly_callables.append(OneFormAssembler(form, tensor=tensor,
                                                           form_compiler_parameters=fcp).assemble)
        # set arbitrary non-zero coefficients for preallocation
        for coef in coefficients.values():
            with coef.dat.vec as cvec:
                cvec.set(1.0E0)
        self.coefficients = coefficients
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
    if isinstance(V, Function):
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
