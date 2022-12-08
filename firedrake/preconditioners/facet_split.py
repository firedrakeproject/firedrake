from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks
import numpy


__all__ = ['FacetSplitPC']


def get_permutation_map(V, W):
    from firedrake.preconditioners.fdm import glonum_fun, restricted_dofs
    bsize = V.value_size
    V_indices = None
    V_map, nel = glonum_fun(V.cell_node_map(), bsize=bsize)
    perm = numpy.empty((V.dof_count, ), dtype=PETSc.IntType)
    perm.fill(-1)

    offset = 0
    for Wsub in W:
        W_indices = None
        W_map, nel = glonum_fun(Wsub.cell_node_map(), bsize=bsize)
        rdofs = restricted_dofs(Wsub.finat_element, V.finat_element)

        for e in range(nel):
            V_indices = V_map(e, result=V_indices)
            W_indices = W_map(e, result=W_indices)
            perm[V_indices[rdofs]] = W_indices + offset

        offset += Wsub.dof_dset.set.size * bsize
        del rdofs
        del W_indices
    del V_indices
    perm = V.dof_dset.lgmap.apply(perm, result=perm)
    own = V.dof_dset.set.size * bsize
    return perm[:own]


def get_permutation_project(V, W):
    from firedrake import Function, split
    ownership_ranges = V.dof_dset.layout_vec.getOwnershipRanges()
    start, end = ownership_ranges[V.comm.rank:V.comm.rank+2]
    v = Function(V)
    w = Function(W)
    with w.dat.vec_wo as wvec:
        wvec.setArray(numpy.linspace(0.0E0, 1.0E0, end-start, dtype=PETSc.RealType))
    w_expr = sum(split(w))
    try:
        v.interpolate(w_expr)
    except NotImplementedError:
        rtol = 1.0 / max(numpy.diff(ownership_ranges))**2
        v.project(w_expr, solver_parameters={
                  "mat_type": "matfree",
                  "ksp_type": "cg",
                  "ksp_atol": 0,
                  "ksp_rtol": rtol,
                  "pc_type": "jacobi", })
    return numpy.rint((end-start-1)*v.dat.data_ro.reshape((-1,))+start).astype(PETSc.IntType)


class FacetSplitPC(PCBase):

    needs_python_pmat = False
    _prefix = "facet_"

    _permutation_cache = {}

    def get_permutation(self, V, W):
        from mpi4py import MPI
        key = (V, W)
        if key not in self._permutation_cache:
            indices = get_permutation_map(V, W)
            if V.comm.allreduce(numpy.all(indices[:-1] <= indices[1:]), MPI.PROD):
                self._permutation_cache[key] = None
            else:
                self._permutation_cache[key] = indices
        return self._permutation_cache[key]

    def initialize(self, pc):

        from ufl import (InteriorElement, FacetElement, MixedElement, TensorElement, VectorElement,
                         FiniteElement, EnrichedElement, TensorProductElement, HCurl)
        from firedrake import FunctionSpace, TestFunctions, TrialFunctions
        from firedrake.assemble import allocate_matrix, TwoFormAssembler
        from firedrake.solving_utils import _SNESContext
        from functools import partial

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")

        ctx = dmhooks.get_appctx(pc.getDM())
        if ctx is None:
            raise ValueError("No context found.")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r", ctx)

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        options = PETSc.Options(options_prefix)
        mat_type = options.getString("mat_type", "submatrix")

        problem = ctx._problem
        a = problem.Jp or problem.J
        V = a.arguments()[-1].function_space()
        assert len(V) == 1

        # W = V_interior * V_facet
        scalar_element = V.ufl_element()
        if isinstance(scalar_element, (TensorElement, VectorElement)):
            scalar_element = scalar_element._sub_element
        tensorize = lambda e, shape: TensorElement(e, shape=shape) if shape else e

        def get_facet_element(e):
            cell = e.cell()
            if e.sobolev_space() == HCurl and isinstance(cell, ufl.TensorProductCell):
                sub_cells = cell.sub_cells()
                degree = max(e.degree())
                variant = e.variant()
                Qc_elt = FiniteElement("Q", sub_cells[0], degree, variant=variant)
                Qd_elt = FiniteElement("RTCE", sub_cells[0], degree, variant=variant)
                Id_elt = FiniteElement("DG", sub_cells[1], degree - 1, variant=variant)
                Ic_elt = FiniteElement("CG", sub_cells[1], degree, variant=variant)
                return EnrichedElement(HCurl(TensorProductElement(FacetElement(Qc_elt), Id_elt, cell=cell)),
                                       HCurl(TensorProductElement(Qd_elt, FacetElement(Ic_elt), cell=cell)),
                                       HCurl(TensorProductElement(FacetElement(Qd_elt), InteriorElement(Ic_elt), cell=cell)))
            return FacetElement(e)

        elements = [tensorize(restriction(scalar_element), V.shape) for restriction in (InteriorElement, get_facet_element)]

        W = FunctionSpace(V.mesh(), MixedElement(elements))
        if W.dim() != V.dim():
            raise ValueError("Dimensions of the original and decomposed spaces do not match")

        self.perm = None
        self.iperm = None
        indices = self.get_permutation(V, W)
        if indices is not None:
            self.perm = PETSc.IS().createGeneral(indices, comm=V.comm)
            self.iperm = self.perm.invertPermutation()

        mixed_operator = a(sum(TestFunctions(W)), sum(TrialFunctions(W)), coefficients={})
        mixed_bcs = tuple(bc.reconstruct(V=W[-1], g=0) for bc in problem.bcs)

        def _permute_nullspace(nsp):
            if nsp is None or self.iperm is None:
                return nsp
            vecs = [vec.duplicate() for vec in nsp.getVecs()]
            for vec in vecs:
                vec.permute(self.iperm)
            return PETSc.NullSpace().create(constant=nsp.constant, vectors=vecs, comm=nsp.comm)

        if mat_type != "submatrix":
            self.mixed_op = allocate_matrix(mixed_operator,
                                            bcs=mixed_bcs,
                                            form_compiler_parameters=fcp,
                                            mat_type=mat_type,
                                            options_prefix=options_prefix)
            self._assemble_mixed_op = TwoFormAssembler(mixed_operator, tensor=self.mixed_op,
                                                       form_compiler_parameters=fcp,
                                                       bcs=mixed_bcs).assemble
            self._assemble_mixed_op()
            mixed_opmat = self.mixed_op.petscmat

            if False:
                # FIXME
                mixed_opmat.setNullSpace(_permute_nullspace(P.getNullSpace()))
                mixed_opmat.setNearNullSpace(_permute_nullspace(P.getNearNullSpace()))
                mixed_opmat.setTransposeNullSpace(_permute_nullspace(P.getTransposeNullSpace()))

        elif self.perm:
            self._permute_op = partial(PETSc.Mat().createSubMatrixVirtual, P, self.iperm, self.iperm)
            mixed_opmat = self._permute_op()
        else:
            mixed_opmat = P

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -facet_pc_type ksp.
        scpc = PETSc.PC().create(comm=pc.comm)
        scpc.incrementTabLevel(1, parent=pc)

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. fieldsplit.
        mixed_dm = W.dm
        self._dm = mixed_dm

        # Create new appctx
        self._ctx_ref = self.new_snes_ctx(pc,
                                          mixed_operator,
                                          mixed_bcs,
                                          mat_type,
                                          fcp,
                                          options_prefix=options_prefix)

        scpc.setDM(mixed_dm)
        scpc.setOptionsPrefix(options_prefix)
        scpc.setOperators(A=mixed_opmat, P=mixed_opmat)
        with dmhooks.add_hooks(mixed_dm, self, appctx=self._ctx_ref, save=False):
            scpc.setFromOptions()
        self.pc = scpc

    def update(self, pc):
        if hasattr(self, "mixed_op"):
            self._assemble_mixed_op()
        elif hasattr(self, "_permute_op"):
            for mat in self.pc.getOperators():
                mat.destroy()
            P = self._permute_op()
            self.pc.setOperators(A=P, P=P)
        self.pc.setUp()

    def apply(self, pc, x, y):
        if self.perm:
            x.permute(self.iperm)
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(x, y)
        if self.perm:
            x.permute(self.perm)
            y.permute(self.perm)

    def applyTranspose(self, pc, x, y):
        if self.perm:
            x.permute(self.iperm)
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(x, y)
        if self.perm:
            x.permute(self.perm)
            y.permute(self.perm)

    def view(self, pc, viewer=None):
        super(FacetSplitPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC using interior-facet decomposition\n")
            self.pc.view(viewer)

    def destroy(self, pc):
        if hasattr(self, "pc"):
            if hasattr(self, "_permute_op"):
                for mat in self.pc.getOperators():
                    mat.destroy()
            self.pc.destroy()
        if hasattr(self, "iperm"):
            if self.iperm:
                self.iperm.destroy()
        if hasattr(self, "perm"):
            if self.perm:
                self.perm.destroy()
