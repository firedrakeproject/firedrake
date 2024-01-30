from functools import partial
from mpi4py import MPI
from pyop2 import op2, PermutedMap
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import firedrake.dmhooks as dmhooks
import numpy

__all__ = ['FacetSplitPC']


class FacetSplitPC(PCBase):
    """ A preconditioner that splits a function into interior and facet DOFs.

    Internally this creates a PETSc PC object that can be controlled
    by options using the extra options prefix ``facet_``.

    This allows for statically-condensed preconditioners to be applied to
    linear systems involving the matrix applied to the full set of DOFs. Code
    generated for the matrix-free operator evaluation in the space with full
    DOFs will run faster than the one with interior-facet decoposition, since
    the full element has a simpler structure.
    """

    needs_python_pmat = False
    _prefix = "facet_"

    _permutation_cache = {}

    def get_permutation(self, V, W):
        key = (V, W)
        if key not in self._permutation_cache:
            indices = get_permutation_map(V, W)
            if V._comm.allreduce(numpy.all(indices[:-1] <= indices[1:]), MPI.PROD):
                self._permutation_cache[key] = None
            else:
                self._permutation_cache[key] = indices
        return self._permutation_cache[key]

    def initialize(self, pc):
        from finat.ufl import RestrictedElement, MixedElement, TensorElement, VectorElement
        from firedrake import FunctionSpace, TestFunctions, TrialFunctions
        from firedrake.assemble import allocate_matrix, TwoFormAssembler

        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")
        self.work_vec_x = P.createVecLeft()
        self.work_vec_y = P.createVecRight()

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        options = PETSc.Options(options_prefix)
        mat_type = options.getString("mat_type", "submatrix")

        if P.getType() == "python":
            ctx = P.getPythonContext()
            a = ctx.a
            bcs = tuple(ctx.row_bcs)
        else:
            ctx = dmhooks.get_appctx(pc.getDM())
            a = ctx.Jp or ctx.J
            bcs = tuple(ctx._problem.bcs)

        V = a.arguments()[-1].function_space()
        assert len(V) == 1, "Interior-facet decomposition of mixed elements is not supported"

        def restrict(ele, restriction_domain):
            if isinstance(ele, VectorElement):
                return type(ele)(restrict(ele._sub_element, restriction_domain), dim=ele.num_sub_elements)
            elif isinstance(ele, TensorElement):
                return type(ele)(restrict(ele._sub_element, restriction_domain), shape=ele._shape, symmetry=ele._symmety)
            else:
                return RestrictedElement(ele, restriction_domain)

        # W = V[interior] * V[facet]
        W = FunctionSpace(V.mesh(), MixedElement([restrict(V.ufl_element(), d) for d in ("interior", "facet")]))
        assert W.dim() == V.dim(), "Dimensions of the original and decomposed spaces do not match"

        mixed_operator = a(sum(TestFunctions(W)), sum(TrialFunctions(W)), coefficients={})
        mixed_bcs = tuple(bc.reconstruct(V=W[-1], g=0) for bc in bcs)

        self.perm = None
        self.iperm = None
        indices = self.get_permutation(V, W)
        if indices is not None:
            self.perm = PETSc.IS().createGeneral(indices, comm=V._comm)
            self.iperm = self.perm.invertPermutation()

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

            def _permute_nullspace(nsp):
                if not (nsp.handle and self.iperm):
                    return nsp
                vecs = [vec.duplicate() for vec in nsp.getVecs()]
                for vec in vecs:
                    self.work_vec_x.getArray(readonly=False)[:] = vec.getArray(readonly=True)[:]
                    self._vec_permute(self.work_vec_x, vec, self.iperm)
                return PETSc.NullSpace().create(constant=nsp.hasConstant(), vectors=vecs, comm=nsp.getComm())

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
            self._vec_permute(x, self.work_vec_x, self.iperm)
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(self.work_vec_x, self.work_vec_y)
        if self.perm:
            self._vec_permute(self.work_vec_y, y, self.perm)

    def applyTranspose(self, pc, x, y):
        if self.perm:
            self._vec_permute(x, self.work_vec_y, self.iperm)
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(self.work_vec_y, self.work_vec_x)
        if self.perm:
            self._vec_permute(self.work_vec_x, y, self.perm)

    def _vec_permute(self, vec_in, vec_out, perm):
        array_in = vec_in.getArray(readonly=True)
        array_out = vec_out.getArray(readonly=False)
        with perm as perm_indices:
            array_out[:] = array_in[perm_indices]

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


def split_dofs(elem):
    """ Split DOFs into interior and facet DOF, where facets are sorted by entity.
    """
    entity_dofs = elem.entity_dofs()
    ndim = elem.cell.get_spatial_dimension()
    edofs = [[], []]
    for key in sorted(entity_dofs.keys()):
        vals = entity_dofs[key]
        edim = key
        try:
            edim = sum(edim)
        except TypeError:
            pass
        for k in sorted(vals.keys()):
            edofs[edim < ndim].extend(sorted(vals[k]))
    return tuple(numpy.array(e, dtype=PETSc.IntType) for e in edofs)


def restricted_dofs(celem, felem):
    """ Find which DOFs from felem are on celem
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
    perm[csplit[k]] = numpy.arange(len(perm), dtype=perm.dtype)
    return fsplit[k][perm]


def get_permutation_map(V, W):
    perm = numpy.empty((V.dof_count, ), dtype=PETSc.IntType)
    perm.fill(-1)
    vdat = V.make_dat(val=perm)

    offset = 0
    wdats = []
    for Wsub in W:
        val = numpy.arange(offset, offset + Wsub.dof_count, dtype=PETSc.IntType)
        wdats.append(Wsub.make_dat(val=val))
        offset += Wsub.dof_dset.layout_vec.sizes[0]

    sizes = [Wsub.finat_element.space_dimension() * Wsub.value_size for Wsub in W]
    eperm = numpy.concatenate([restricted_dofs(Wsub.finat_element, V.finat_element) for Wsub in W])
    pmap = PermutedMap(V.cell_node_map(), eperm)

    kernel_code = f"""
    void permutation(PetscInt *restrict x,
                     const PetscInt *restrict xi,
                     const PetscInt *restrict xf){{
        for(PetscInt i=0; i<{sizes[0]}; i++) x[i] = xi[i];
        for(PetscInt i=0; i<{sizes[1]}; i++) x[i+{sizes[0]}] = xf[i];
        return;
    }}
    """
    kernel = op2.Kernel(kernel_code, "permutation", requires_zeroed_output_arguments=False)
    op2.par_loop(kernel, V.mesh().cell_set,
                 vdat(op2.WRITE, pmap),
                 wdats[0](op2.READ, W[0].cell_node_map()),
                 wdats[1](op2.READ, W[1].cell_node_map()))

    own = V.dof_dset.layout_vec.sizes[0]
    perm = perm.reshape((-1, ))
    perm = V.dof_dset.lgmap.apply(perm, result=perm)
    return perm[:own]
