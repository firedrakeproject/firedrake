from functools import partial
import pyop3 as op3
from pyop3.mpi import MPI, temp_internal_comm
from finat.ufl import MixedElement
from firedrake.function import Function
from firedrake.pack import pack
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.bcs import restricted_function_space
import firedrake.dmhooks as dmhooks
import numpy


__all__ = ['FacetSplitPC']


class FacetSplitPC(PCBase):
    """A preconditioner that splits a function into interior and facet DOFs.

    Internally this creates a PETSc PC object that can be controlled
    by options using the extra options prefix ``facet_``.

    This allows for statically-condensed preconditioners to be applied to
    linear systems involving the matrix applied to the full set of DOFs. Code
    generated for the matrix-free operator evaluation in the space with full
    DOFs will run faster than the one with interior-facet decomposition, since
    the full element has a simpler structure.
    """

    needs_python_pmat = False
    _prefix = "facet_"

    _index_cache = {}

    def get_indices(self, V, W):
        key = (V, W)
        if key not in self._index_cache:
            indices = get_restriction_indices(V, W)
            with temp_internal_comm(V.comm) as icomm:
                if icomm.allreduce(len(indices) == V.dof_count and numpy.all(indices[:-1] <= indices[1:]), MPI.PROD):
                    self._index_cache[key] = None
                else:
                    self._index_cache[key] = indices
        return self._index_cache[key]

    def initialize(self, pc):
        from firedrake import FunctionSpace, TestFunction, TrialFunction, split

        prefix = pc.getOptionsPrefix() or ""
        options_prefix = prefix + self._prefix
        options = PETSc.Options(options_prefix)
        mat_type = options.getString("mat_type", "submatrix")
        domains = options.getString("restriction_domain", "interior,facet")
        domains = domains.split(",")

        a, bcs = self.form(pc)
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")
        V = a.arguments()[-1].function_space()
        if len(V) != 1:
            raise ValueError("Decomposition of mixed elements is not supported")

        element = V.ufl_element()
        elements = [element[domain] for domain in domains]
        W = FunctionSpace(V.mesh(), elements[0] if len(elements) == 1 else MixedElement(*elements))
        if V.boundary_set:
            W = restricted_function_space(W, [V.boundary_set]*len(W))

        args = (TestFunction(W), TrialFunction(W))
        if len(W) > 1:
            args = tuple(sum(split(arg)) for arg in args)
        mixed_operator = a(*args)
        mixed_bcs = tuple(bc.reconstruct(V=W[-1], g=0) for bc in bcs)

        _, P = pc.getOperators()

        self.work_vecs = None
        indices = self.get_indices(V, W)
        self.subset = None
        self.needs_zeroing = False
        if indices is not None:
            self.needs_zeroing = len(indices) < V.dof_count
            self.subset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)

        if mat_type != "submatrix":
            from firedrake.assemble import get_assembler
            form_assembler = get_assembler(mixed_operator, bcs=mixed_bcs,
                                           form_compiler_parameters=fcp,
                                           mat_type=mat_type,
                                           options_prefix=options_prefix)
            self.P = form_assembler.allocate()
            self._assemble_mixed_op = form_assembler.assemble
            self._assemble_mixed_op(tensor=self.P)
            self.mixed_opmat = self.P.petscmat
            self.set_nullspaces(pc)
            self.work_vecs = self.mixed_opmat.createVecs()
        elif self.subset:
            global_indices = V.axes.lgmap(V.block_shape).apply(self.subset.indices)
            self._global_iperm = PETSc.IS().createGeneral(global_indices, comm=pc.comm)
            self._permute_op = partial(PETSc.Mat().createSubMatrixVirtual, P, self._global_iperm, self._global_iperm)
            self.mixed_opmat = self._permute_op()
            self.set_nullspaces(pc)
            self.work_vecs = self.mixed_opmat.createVecs()
        else:
            self.mixed_opmat = P

        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -facet_pc_type ksp.
        scpc = PETSc.PC().create(comm=pc.comm)
        scpc.incrementTabLevel(1, parent=pc)

        # We set a DM and an appropriate SNESContext on the constructed PC so one
        # can do e.g. fieldsplit.
        mixed_dm = W.dm

        # Create new appctx
        self._ctx_ref = self.new_snes_ctx(pc,
                                          mixed_operator,
                                          mixed_bcs,
                                          mat_type,
                                          fcp,
                                          options_prefix=options_prefix)

        scpc.setDM(mixed_dm)
        scpc.setOptionsPrefix(options_prefix)
        scpc.setOperators(A=self.mixed_opmat, P=self.mixed_opmat)
        self.pc = scpc
        with dmhooks.add_hooks(mixed_dm, self, appctx=self._ctx_ref, save=False):
            scpc.setFromOptions()

    def set_nullspaces(self, pc):
        _, P = pc.getOperators()
        Pmat = self.mixed_opmat

        def _restrict_nullspace(nsp):
            if not (nsp.handle and self.subset):
                return nsp
            vecs = []
            for x in nsp.getVecs():
                y = Pmat.createVecRight()
                self.restrict(x, y)
                y.normalize()
                vecs.append(y)
            return PETSc.NullSpace().create(constant=nsp.hasConstant(), vectors=vecs, comm=nsp.getComm())

        Pmat.setNullSpace(_restrict_nullspace(P.getNullSpace()))
        Pmat.setNearNullSpace(_restrict_nullspace(P.getNearNullSpace()))
        Pmat.setTransposeNullSpace(_restrict_nullspace(P.getTransposeNullSpace()))

    def update(self, pc):
        if hasattr(self, "_permute_op"):
            for mat in self.pc.getOperators():
                mat.destroy()
            P = self._permute_op()
            self.pc.setOperators(A=P, P=P)
            self.mixed_opmat = P
        elif hasattr(self, "P"):
            self._assemble_mixed_op(tensor=self.P)

    def prolong(self, x, y):
        if x is not y:
            if self.needs_zeroing:
                y.set(0.0)
            array_x = x.getArray(readonly=True)
            array_y = y.getArray(readonly=False)
            with self.subset as subset_indices:
                array_y[subset_indices] = array_x[:]

    def restrict(self, x, y):
        if x is not y:
            array_x = x.getArray(readonly=True)
            array_y = y.getArray(readonly=False)
            with self.subset as subset_indices:
                array_y[:] = array_x[subset_indices]

    def apply(self, pc, x, y):
        dm = self.pc.getDM()
        xwork, ywork = self.work_vecs or (x, y)
        self.restrict(x, xwork)
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(xwork, ywork)
        self.prolong(ywork, y)

    def applyTranspose(self, pc, x, y):
        dm = self.pc.getDM()
        xwork, ywork = self.work_vecs or (x, y)
        self.restrict(x, xwork)
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(xwork, ywork)
        self.prolong(ywork, y)

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
        if hasattr(self, "subset"):
            if self.subset:
                self.subset.destroy()


def split_dofs(elem):
    """ Split DOFs into interior and facet DOF, where facets are sorted by entity.
    """
    dim = elem.cell.get_spatial_dimension()
    entity_dofs = elem.entity_dofs()
    edofs = [[], []]
    for key in sorted(entity_dofs.keys()):
        vals = entity_dofs[key]
        edim = key
        try:
            edim = sum(edim)
        except TypeError:
            pass
        for k in sorted(vals.keys()):
            edofs[edim < dim].extend(sorted(vals[k]))
    return tuple(numpy.array(e, dtype=PETSc.IntType) for e in edofs)


def restricted_dofs(celem, felem):
    """ Find which DOFs from felem are on celem
    :arg celem: the restricted :class:`finat.FiniteElement`
    :arg felem: the unrestricted :class:`finat.FiniteElement`
    :returns: :class:`numpy.array` with indices of felem that correspond to celem
    """
    indices = numpy.full((celem.space_dimension(),), -1, dtype=PETSc.IntType)
    cdofs = celem.entity_dofs()
    fdofs = felem.entity_dofs()
    for dim in sorted(cdofs):
        for entity in sorted(cdofs[dim]):
            ndofs = len(cdofs[dim][entity])
            indices[cdofs[dim][entity]] = fdofs[dim][entity][:ndofs]
    return indices


def get_restriction_indices(V, W):
    """Return the list of dofs in the space V such that W = V[indices].
    """
    v_func = Function(V, val=numpy.arange(V.axes.size, dtype=PETSc.IntType), dtype=PETSc.IntType)
    w_func = Function(W, val=numpy.full(W.axes.size, -1, dtype=PETSc.IntType), dtype=PETSc.IntType)

    vsize = sum(Vsub.finat_element.space_dimension() for Vsub in V)
    eperm = numpy.concatenate([restricted_dofs(Wsub.finat_element, V.finat_element) for Wsub in W])
    if len(eperm) < vsize:
        eperm = numpy.concatenate((eperm, numpy.setdiff1d(numpy.arange(vsize, dtype=PETSc.IntType), eperm)))
    eperm_dat = op3.Dat.from_array(eperm, prefix="perm", buffer_kwargs={"constant": True})

    # TODO: now we can pass permutation to pack do we need this?

    # debug
    c = V.mesh().cells.owned.iter()
    b = pack(v_func, c, "cell")[eperm_dat]
    a = pack(w_func, c, "cell").reshape(b.axes.materialize())

    # import pyop3.extras.debug
    # pyop3.extras.debug.enable_conditional_breakpoints()
    op3.loop(
        c,
        a.assign(b),
        eager=True,
    )
    return w_func.dat.data_ro
