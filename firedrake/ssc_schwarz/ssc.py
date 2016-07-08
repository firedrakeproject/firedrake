from __future__ import absolute_import
from firedrake import DirichletBC, \
    FunctionSpace, VectorFunctionSpace, Function, assemble, InitializedPC

from firedrake.utils import cached_property
from firedrake.petsc import PETSc
from pyop2 import op2
from mpi4py import MPI
from . import sscutils
import numpy

import ufl
from ufl.algorithms import map_integrands, MultiFunction
from .patches import get_cell_facet_patches, get_dof_patches, \
    g2l_begin, g2l_end, l2g_begin, l2g_end, apply_patch


__all__ = ["PatchPC", "P1PC"]


class ArgumentReplacer(MultiFunction):
    def __init__(self, test, trial):
        self.args = {0: test, 1: trial}
        super(ArgumentReplacer, self).__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        return self.args[o.number()]


def mpi_type(dtype, dim):
    try:
        tdict = MPI.__TypeDict__
    except AttributeError:
        tdict = MPI._typedict

    btype = tdict[dtype.char]
    if dim == 1:
        return btype
    typ = btype.Create_contiguous(dim)
    typ.Commit()
    return typ


class PatchPC(InitializedPC):

    def initialSetUp(self, pc):
        A, P = pc.getOperators()
        ctx = P.getPythonContext()

        self.ctx = ctx
        a = ctx.a
        self.a = a
        bcs = ctx.row_bcs
        test, trial = a.arguments()

        V = test.function_space()
        self.V = V

        mesh = a.ufl_domain()
        self.mesh = mesh

        self.ksps = []

        self._mpi_type = mpi_type(numpy.dtype(PETSc.ScalarType), V.dim)
        dm = V._dm
        self._sf = dm.getDefaultSF()

        local = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        size = V.dof_dset.total_size * V.dim
        local.setSizes((size, size), bsize=V.dim)
        local.setUp()
        self._local = local

        if bcs is None:
            self.bcs = ()
            bcs = numpy.zeros(0, dtype=numpy.int32)
        else:
            try:
                bcs = tuple(bcs)
            except TypeError:
                bcs = (bcs, )
            self.bcs = bcs
            bcs = numpy.unique(numpy.concatenate([bc.nodes for bc in bcs]))
            bcs = bcs[bcs < V.dof_dset.size]

        dof_section = V._dm.getDefaultSection()
        dm = mesh._plex
        cells, facets = get_cell_facet_patches(dm, mesh._cell_numbering)
        d, g, b = get_dof_patches(dm, dof_section,
                                  V.cell_node_map().values_with_halo,
                                  bcs, cells, facets)
        self.bc_nodes = PETSc.IS().createBlock(V.dim, bcs, comm=PETSc.COMM_SELF)
        self.cells = []
        for i in range(len(cells)):
            self.cells.append(PETSc.IS().createGeneral(cells[i], comm=PETSc.COMM_SELF))
        self.facets = facets
        tmp = []
        for i in range(len(d)):
            tmp.append(PETSc.IS().createBlock(V.dim, d[i], comm=PETSc.COMM_SELF))
        self.dof_patches = tmp

        tmp = []
        for i in range(len(g)):
            tmp.append(PETSc.IS().createBlock(V.dim, g[i], comm=PETSc.COMM_SELF))
        self.glob_patches = tmp

        tmp = []
        for i in range(len(b)):
            tmp.append(PETSc.IS().createBlock(V.dim, b[i], comm=PETSc.COMM_SELF))
        self.bc_patches = tmp

        # Now the patch vectors:
        self._bs = []
        self._ys = []
        for i, m in enumerate(self.matrices):
            ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            pfx = pc.getOptionsPrefix()
            ksp.setOptionsPrefix(pfx + "sub_")
            ksp.setType(ksp.Type.PREONLY)
            ksp.setOperators(m, m)
            ksp.setFromOptions()
            self.ksps.append(ksp)
            size = self.glob_patches[i].getSize()
            bs = self.glob_patches[i].getBlockSize()
            b = PETSc.Vec().create(comm=PETSc.COMM_SELF)
            b.setSizes((size, size), bsize=bs)
            b.setUp()
            self._bs.append(b)
            self._ys.append(b.duplicate())

    @cached_property
    def kernels(self):
        from firedrake.tsfc_interface import compile_form
        kernels = compile_form(self.a, "subspace_form")
        compiled_kernels = []
        for k in kernels:
            # Don't want to think about mixed yet
            assert k.indices == (0, 0)
            kinfo = k.kinfo
            assert kinfo.integral_type == "cell"
            assert not kinfo.oriented
            compiled_kernels.append(kinfo)
        assert len(compiled_kernels) == 1
        return tuple(compiled_kernels)

    @cached_property
    def matrix_callable(self):
        return sscutils.matrix_callable(self.kernels, self.V, self.mesh.coordinates,
                                        *self.a.coefficients())

    @cached_property
    def matrices(self):
        mats = []
        coords = self.mesh.coordinates
        carg = coords.dat._data.ctypes.data
        cmap = coords.cell_node_map()._values.ctypes.data
        coeffs = self.a.coefficients()
        args = []
        for n in self.kernels[0].coefficient_map:
            c = coeffs[n]
            args.append(c.dat._data.ctypes.data)
            args.append(c.cell_node_map()._values.ctypes.data)
        for i in range(len(self.dof_patches)):
            mat = PETSc.Mat().create(comm=PETSc.COMM_SELF)
            size = self.glob_patches[i].getSize()
            bs = self.glob_patches[i].getBlockSize()
            mat.setSizes(((size, size), (size, size)),
                         bsize=bs)
            mat.setType(mat.Type.DENSE)
            mat.setOptionsPrefix("scp_")
            mat.setFromOptions()
            mat.setUp()
            marg = mat.handle
            mmap = self.dof_patches[i].getBlockIndices().ctypes.data
            cells = self.cells[i].getIndices().ctypes.data
            end = self.cells[i].getSize()
            with PETSc.Log.Event("Fill mat"):
                self.matrix_callable(0, end, cells, marg, mmap, mmap, carg, cmap, *args)
                mat.assemble()
                mat.zeroRowsColumns(self.bc_patches[i])
            mats.append(mat)
        return tuple(mats)

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Vertex-patch preconditioner, all subsolves identical", comm=comm)
        self.ksps[0].view(viewer)

    def apply(self, pc, x, y):
        apply_patch(self, x, y)


class P1PC(InitializedPC):

    def initialSetUp(self, pc):
        self.pc = PETSc.PC().create()
        self.pc.setOptionsPrefix(pc.getOptionsPrefix() + "lo_")
        A, P = pc.getOperators()
        ctx = P.getPythonContext()

        a = ctx.a
        self.a = a
        bcs = ctx.row_bcs
        test, trial = a.arguments()
        mesh = a.ufl_domain()
        self.mesh = mesh

        V = test.function_space()

        assert V == trial.function_space()
        self.V = V
        if V.rank == 0:
            self.P1 = FunctionSpace(mesh, "CG", 1)
        elif V.rank == 1:
            assert len(V.shape) == 1
            self.P1 = VectorFunctionSpace(mesh, "CG", 1, dim=V.shape[0])
        else:
            raise NotImplementedError

        if bcs is None:
            self.bcs = ()
            bcs = numpy.zeros(0, dtype=numpy.int32)
        else:
            try:
                bcs = tuple(bcs)
            except TypeError:
                bcs = (bcs, )
            self.bcs = bcs
            bcs = numpy.unique(numpy.concatenate([bc.nodes for bc in bcs]))
            bcs = bcs[bcs < V.dof_dset.size]

        self.A_p1 = assemble(self.P1_form, bcs=self.P1_bcs)
        self.A_p1.force_evaluation()
        op = self.A_p1.PETScMatHandle

        self.pc.setOperators(op, op)
        self.pc.setUp()
        self.pc.setFromOptions()

        self.transfer = self.transfer_op
        self.work1 = self.transfer.createVecLeft()
        self.work2 = self.transfer.createVecLeft()

        # dof_section = V._dm.getDefaultSection()
        # dm = mesh._plex
        # cells, facets = get_cell_facet_patches(dm, mesh._cell_numbering)
        # d, g, b = get_dof_patches(dm, dof_section,
        #                           V.cell_node_map().values_with_halo,
        #                           bcs, cells, facets)
        self.bc_nodes = PETSc.IS().createBlock(V.dim, bcs, comm=PETSc.COMM_SELF)
        # self.cells = []
        # for i in range(len(cells)):
        #     self.cells.append(PETSc.IS().createGeneral(cells[i], comm=PETSc.COMM_SELF))
        # self.facets = facets
        # tmp = []
        # for i in range(len(d)):
        #     tmp.append(PETSc.IS().createBlock(V.dim, d[i], comm=PETSc.COMM_SELF))
        # self.dof_patches = tmp

    def subsequentSetUp(self, pc):
        assemble(self.P1_form, self.P1_bcs, tensor=self.A_p1)

    @cached_property
    def P1_form(self):
        from firedrake.ufl_expr import TestFunction, TrialFunction
        tst = TestFunction(self.P1)
        trl = TrialFunction(self.P1)
        mapper = ArgumentReplacer(tst, trl)
        return map_integrands.map_integrand_dags(mapper, self.a)

    @cached_property
    def P1_bcs(self):
        bcs = []
        for bc in self.bcs:
            val = Function(self.P1)
            val.interpolate(ufl.as_ufl(bc.function_arg))
            bcs.append(DirichletBC(self.P1, val, bc.sub_domain, method=bc.method))
        return tuple(bcs)

    def transfer_kernel(self, restriction=True):
        """Compile a kernel that will map between Pk and P1.

        :kwarg restriction: If True compute a restriction operator, if
             False, a prolongation operator.
        :returns: a PyOP2 kernel.

        The prolongation maps a solution in P1 into Pk using the natural
        embedding.  The restriction maps a residual in the dual of Pk into
        the dual of P1 (it is the dual of the prolongation), computed
        using linearity of the test function.
        """
        # Mapping of a residual in Pk into a residual in P1
        from coffee import base as coffee
        from tsfc.coffee import generate as generate_coffee, SCALAR_TYPE
        from tsfc.kernel_interface import prepare_coefficient, prepare_arguments
        from gem import gem, impero_utils as imp
        import ufl
        import numpy

        Pk = self.V
        P1 = self.P1
        # Pk should be at least the same size as P1
        assert Pk.fiat_element.space_dimension() >= P1.fiat_element.space_dimension()
        # In the general case we should compute this by doing:
        # numpy.linalg.solve(Pkmass, PkP1mass)
        matrix = numpy.dot(Pk.fiat_element.dual.to_riesz(P1.fiat_element.get_nodal_basis()),
                           P1.fiat_element.get_coeffs().T).T

        if restriction:
            Vout, Vin = P1, Pk
            weights = gem.Literal(matrix)
            name = "Pk_P1_mapper"
        else:
            # Prolongation
            Vout, Vin = Pk, P1
            weights = gem.Literal(matrix.T)
            name = "P1_Pk_mapper"

        funargs = []
        Pke = Vin.fiat_element
        P1e = Vout.fiat_element

        assert Vin.shape == Vout.shape

        shape = (P1e.space_dimension(), ) + Vout.shape + (Pke.space_dimension(), ) + Vin.shape

        outarg = coffee.Decl(SCALAR_TYPE, coffee.Symbol("A", rank=shape))
        i = gem.Index()
        j = gem.Index()
        pre = [i]
        post = [j]
        extra = []
        for _ in Vin.shape:
            extra.append(gem.Index())
        indices = pre + extra + post + extra

        indices = tuple(indices)
        outgem = [gem.Indexed(gem.Variable("A", shape), indices)]

        funargs.append(outarg)

        exprs = [gem.Indexed(weights, (i, j))]

        ir = imp.compile_gem(outgem, exprs, indices)

        body = generate_coffee(ir, {})
        function = coffee.FunDecl("void", name, funargs, body,
                                  pred=["static", "inline"])

        return op2.Kernel(function, name=function.name)

    @cached_property
    def transfer_op(self):
        sp = op2.Sparsity((self.P1.dof_dset,
                           self.V.dof_dset),
                          (self.P1.cell_node_map(),
                           self.V.cell_node_map()),
                          "P1_Pk_mapper")
        mat = op2.Mat(sp, PETSc.ScalarType)
        matarg = mat(op2.WRITE, (self.P1.cell_node_map(self.P1_bcs)[op2.i[0]],
                                 self.V.cell_node_map(self.bcs)[op2.i[1]]))
        # HACK HACK HACK, this seems like it might be a pyop2 bug
        sh = matarg._block_shape
        assert len(sh) == 1 and len(sh[0]) == 1 and len(sh[0][0]) == 2
        a, b = sh[0][0]
        nsh = (((a*self.P1.dof_dset.cdim, b*self.V.dof_dset.cdim), ), )
        matarg._block_shape = nsh
        op2.par_loop(self.transfer_kernel(), self.mesh.cell_set,
                     matarg)
        mat.assemble()
        mat._force_evaluation()
        return mat.handle

    def view(self, pc, viewer=None):
        if viewer is not None:
            comm = viewer.comm
        else:
            comm = pc.comm

        PETSc.Sys.Print("Low-order P1, inner pc follows", comm=comm)
        self.pc.view(viewer)

    def apply(self, pc, x, y):
        with PETSc.Log.Stage("P1PC apply"):
            y.set(0)
            self.work1.set(0)
            self.work2.set(0)
            self.transfer.mult(x, self.work1)
            self.pc.apply(self.work1, self.work2)
            self.transfer.multTranspose(self.work2, y)
            indices = self.bc_nodes.getIndices()
            y.array[indices] = x.array_r[indices]
