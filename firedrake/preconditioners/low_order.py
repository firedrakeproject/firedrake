from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
import numpy
from itertools import chain

from ufl.algorithms import MultiFunction, map_integrands

import firedrake
from pyop2 import op2


__all__ = ("P1PC", )


class ArgumentReplacer(MultiFunction):
    def __init__(self, arg_map):
        self.arg_map = arg_map
        super(ArgumentReplacer, self).__init__()

    expr = MultiFunction.reuse_if_untouched

    def argument(self, o):
        return self.arg_map[o]


def transfer_kernel(Pk, P1):
    """Compile a kernel that will map between Pk and P1.
    :returns: a PyOP2 kernel.

    The prolongation maps a solution in P1 into Pk using the natural
    embedding.  The restriction maps a residual in the dual of Pk into
    the dual of P1 (it is the dual of the prolongation), computed
    using linearity of the test function.
    """
    # Mapping of a residual in Pk into a residual in P1
    from coffee import base as coffee
    from tsfc.coffee import generate as generate_coffee
    from tsfc.parameters import default_parameters
    from gem import gem, impero_utils as imp
    from firedrake.utils import ScalarType_c

    # Pk should be at least the same size as P1
    assert Pk.finat_element.space_dimension() >= P1.finat_element.space_dimension()
    # In the general case we should compute this by doing:
    # numpy.linalg.solve(Pkmass, PkP1mass)
    Pke = Pk.finat_element._element
    P1e = P1.finat_element._element
    # TODO, rework to use finat.
    matrix = numpy.dot(Pke.dual.to_riesz(P1e.get_nodal_basis()),
                       P1e.get_coeffs().T).T

    Vout, Vin = P1, Pk
    weights = gem.Literal(matrix)
    name = "Pk_P1_mapper"

    funargs = []

    assert Vin.shape == Vout.shape

    shape = (P1e.space_dimension() * Vout.value_size,
             Pke.space_dimension() * Vin.value_size)
    outarg = coffee.Decl(ScalarType_c, coffee.Symbol("A", rank=shape))
    i = gem.Index()
    j = gem.Index()
    k = gem.Index()
    indices = i, j, k
    A = gem.Variable("A", shape)

    outgem = [gem.Indexed(gem.reshape(A,
                                      (P1e.space_dimension(), Vout.value_size),
                                      (Pke.space_dimension(), Vin.value_size)),
                          (i, k, j, k))]

    funargs.append(outarg)

    expr = gem.Indexed(weights, (i, j))

    outgem, = imp.preprocess_gem(outgem)
    ir = imp.compile_gem([(outgem, expr)], indices)

    index_names = [(i, "i"), (j, "j"), (k, "k")]
    precision = default_parameters()["precision"]
    body = generate_coffee(ir, index_names, precision, ScalarType_c)
    function = coffee.FunDecl("void", name, funargs, body,
                              pred=["static", "inline"])

    return op2.Kernel(function, name=function.name)


def restriction_matrix(Pk, P1, Pk_bcs, P1_bcs):
    sp = op2.Sparsity((P1.dof_dset,
                       Pk.dof_dset),
                      (P1.cell_node_map(),
                       Pk.cell_node_map()))
    mat = op2.Mat(sp, PETSc.ScalarType)

    rlgmap, clgmap = mat.local_to_global_maps
    rlgmap = P1.local_to_global_map(P1_bcs, lgmap=rlgmap)
    clgmap = Pk.local_to_global_map(Pk_bcs, lgmap=clgmap)
    unroll = any(bc.function_space().component is not None
                 for bc in chain(P1_bcs, Pk_bcs) if bc is not None)
    matarg = mat(op2.WRITE, (P1.cell_node_map(), Pk.cell_node_map()),
                 lgmaps=(rlgmap, clgmap), unroll_map=unroll)
    mesh = Pk.ufl_domain()
    op2.par_loop(transfer_kernel(Pk, P1), mesh.cell_set,
                 matarg)
    mat.assemble()
    return mat.handle


class P1PC(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        _, P = pc.getOperators()
        assert P.type == "python"
        context = P.getPythonContext()
        (self.J, self.bcs) = (context.a, context.row_bcs)

        test, trial = self.J.arguments()
        if test.function_space() != trial.function_space():
            raise NotImplementedError("test and trial spaces must be the same")

        Pk = test.function_space()
        element = Pk.ufl_element()
        shape = element.value_shape()
        mesh = Pk.ufl_domain()
        if len(shape) == 0:
            P1 = firedrake.FunctionSpace(mesh, "CG", 1)
        elif len(shape) == 1:
            P1 = firedrake.VectorFunctionSpace(mesh, "CG", 1, dim=shape[0])
        else:
            P1 = firedrake.TensorFunctionSpace(mesh, "CG", 1, shape=shape,
                                               symmetry=element.symmetry())

        # TODO: A smarter low-order operator would also interpolate
        # any coefficients to the coarse space.
        mapper = ArgumentReplacer({test: firedrake.TestFunction(P1),
                                   trial: firedrake.TrialFunction(P1)})
        self.lo_J = map_integrands.map_integrand_dags(mapper, self.J)

        lo_bcs = []
        for bc in self.bcs:
            # Don't actually need the value, since it's only used for
            # killing parts of the restriction matrix.
            lo_bcs.append(firedrake.DirichletBC(P1, firedrake.zero(P1.shape),
                                                bc.sub_domain,
                                                method=bc.method))

        self.lo_bcs = tuple(lo_bcs)

        mat_type = PETSc.Options().getString(pc.getOptionsPrefix() + "lo_mat_type",
                                             firedrake.parameters["default_matrix_type"])
        self.lo_op = firedrake.assemble(self.lo_J, bcs=self.lo_bcs,
                                        mat_type=mat_type)
        A, P = pc.getOperators()
        nearnullsp = P.getNearNullSpace()
        if nearnullsp.handle != 0:
            # Actually have a near nullspace
            tmp = firedrake.Function(Pk)
            low = firedrake.Function(P1)
            vecs = []
            for vec in nearnullsp.getVecs():
                with tmp.dat.vec as v:
                    vec.copy(v)
                low.interpolate(tmp)
                with low.dat.vec_ro as v:
                    vecs.append(v.copy())
            nullsp = PETSc.NullSpace().create(vectors=vecs, comm=pc.comm)
            self.lo_op.petscmat.setNearNullSpace(nullsp)
        lo = PETSc.PC().create(comm=pc.comm)
        lo.incrementTabLevel(1, parent=pc)
        lo.setOperators(self.lo_op.petscmat, self.lo_op.petscmat)
        lo.setOptionsPrefix(pc.getOptionsPrefix() + "lo_")
        lo.setFromOptions()
        self.lo = lo
        self.restriction = restriction_matrix(Pk, P1, self.bcs, self.lo_bcs)

        self.work = self.lo_op.petscmat.createVecs()
        if len(self.bcs) > 0:
            bc_nodes = numpy.unique(numpy.concatenate([bc.nodes for bc in self.bcs]))
            bc_nodes = bc_nodes[bc_nodes < Pk.dof_dset.size]
            bc_iset = PETSc.IS().createBlock(numpy.prod(shape), bc_nodes,
                                             comm=PETSc.COMM_SELF)
            self.bc_indices = bc_iset.getIndices()
            bc_iset.destroy()
        else:
            self.bc_indices = numpy.empty(0, dtype=numpy.int32)

    def update(self, pc):
        firedrake.assemble(self.lo_J, bcs=self.lo_bcs, tensor=self.lo_op)

    def apply(self, pc, x, y):
        work1, work2 = self.work
        # MatMult zeros output vector first.
        self.restriction.mult(x, work1)
        # PC application may not zero output vector first.
        work2.set(0)
        self.lo.apply(work1, work2)
        # MatMultTranspose zeros output vector first.
        self.restriction.multTranspose(work2, y)
        # That was all done orthogonal to the BC subspace, so now
        # carry the boundary values over:
        y.array[self.bc_indices] = x.array_r[self.bc_indices]

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Haven't coded lo-applyTranspose")
        # TODO is this right?
        work1, work2 = self.work
        self.restriction.mult(x, work1)
        work2.set(0)
        self.lo.applyTranspose(work1, work2)
        tmp = y.duplicate()
        self.restriction.multTranspose(work2, tmp)
        # That was all done orthogonal to the BC subspace, so now
        # carry the boundary values over:
        y.array[self.bc_indices] = x.array_r[self.bc_indices]

    def view(self, pc, viewer=None):
        if viewer is None:
            viewer = PETSc.Viewer.STDOUT
        viewer.printfASCII("Low-order PC\n")
        self.lo.view(viewer)
