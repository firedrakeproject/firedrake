from pyop2.datatypes import IntType
import numpy
from pyop2.mpi import COMM_SELF
from mpi4py import MPI
import firedrake
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.utils import cached_property
from firedrake.preconditioners import geneoimpl


__all__ = ("GenEOPC", )


def mpi_type(dtype, n=1):
    """Return an MPI Datatype for a given numpy type.

    :arg dtype: The base numpy type.
    :arg n: Length (if > 1, creates a contiguous MPI type of length n)
    :returns: An MPI Datatype, which should be freed after use."""
    try:
        tdict = MPI.__TypeDict__
    except AttributeError:
        tdict = MPI._typedict
    base = tdict[dtype.char]
    if n == 1:
        mtype = base.Dup()
    else:
        mtype = base.Create_contiguous(n)
        mtype.Commit()
    return mtype


def dof_multiplicity(V):
    """For each dof in V, return its multiplicity.

    :arg V: the function space
    :returns: An IS counting the number of processes each dof appears on.
    """
    sf = V.dm.getDefaultSF()
    degrees = sf.computeDegree()
    _, leaves, _ = sf.getGraph()
    leafdata = numpy.full(leaves.shape, -1, dtype=degrees.dtype)

    datatype = mpi_type(degrees.dtype)
    sf.bcastBegin(datatype, degrees, leafdata)
    sf.bcastEnd(datatype, degrees, leafdata)
    datatype.Free()

    assert all(leafdata >= 1)
    leafdata = numpy.repeat(leafdata, V.value_size)
    return PETSc.IS().createGeneral(leafdata, comm=COMM_SELF)


def domain_intersections(V):
    """Return a list of ISes, one for each rank, containing those
    nodes on my rank which intersect with that rank.

    :arg V: The function space to compute intersections of.
    :returns: A list of ISes (of length comm.size)
    """
    sf = V.dm.getDefaultSF()

    comm = V.comm
    degrees = sf.computeDegree()
    maxdegree = numpy.asarray([degrees.max()])
    comm.Allreduce(MPI.IN_PLACE, maxdegree, op=MPI.MAX)
    maxdegree, = maxdegree

    leafdata = numpy.full(V.node_set.total_size, comm.rank, dtype=IntType)
    rootdata = numpy.full(sum(degrees), -1, dtype=IntType)

    datatype = mpi_type(leafdata.dtype)

    sf.gatherBegin(datatype, leafdata, rootdata)
    sf.gatherEnd(datatype, leafdata, rootdata)

    datatype.Free()
    assert all(rootdata >= 0)

    _, leaves, _ = sf.getGraph()
    nleaves, = leaves.shape
    ghosted = numpy.full((nleaves, maxdegree), -2, dtype=IntType)

    unrolled = numpy.full((len(degrees), maxdegree), -1, dtype=IntType)
    offset = 0
    for i, degree in enumerate(degrees):
        unrolled[i, :degree] = rootdata[offset:offset+degree]
        offset += degree

    datatype = mpi_type(unrolled.dtype, maxdegree)

    sf.bcastBegin(datatype, unrolled, ghosted)
    sf.bcastEnd(datatype, unrolled, ghosted)

    datatype.Free()

    # FIXME: this should be sparse!
    intersections = [list() for _ in range(comm.size)]

    for node, ranks in enumerate(ghosted):
        for rank in ranks:
            if rank < 0 or rank == comm.rank:
                continue
            intersections[rank].append(node)

    lgmap = V.dof_dset.lgmap
    return tuple(PETSc.IS().createGeneral(lgmap.applyBlock(nodes), comm=COMM_SELF)
                 for nodes in intersections)


class GenEOPC(PCBase):

    @cached_property
    def multiplicities(self):
        return dof_multiplicity(self.V)

    @cached_property
    def intersections(self):
        return domain_intersections(self.V)

    def initialize(self, pc):
        A, P = pc.getOperators()
        if P.type != PETSc.Mat.Type.PYTHON:
            raise NotImplementedError("Only for python matrices")
        ctx = P.getPythonContext()

        test, trial = ctx.a.arguments()

        if test.function_space() != trial.function_space():
            raise ValueError("Only for SPD systems")
        mesh = test.ufl_domain()
        overlap, depth = mesh._distribution_parameters.get("overlap_type", (None, None))
        if overlap is not firedrake.DistributedMeshOverlapType.NONE:
            raise NotImplementedError("Only for meshes with no overlap right now, sorry!")

        V = test.function_space()
        if V.value_size is not 1:
            raise NotImplementedError("Only for scalar problems right now, sorry!")
        self.V = V
        P = firedrake.assemble(ctx.a, bcs=ctx.row_bcs, mat_type="is").M.handle

        A = firedrake.assemble(ctx.a, bcs=ctx.row_bcs, mat_type="aij").M.handle

        iset = (PETSc.IS().createGeneral(V.dof_dset.lgmap.indices, comm=COMM_SELF), )
        localDirichlet, = A.createSubMatrices(iset, iset)

        bcs = ctx.row_bcs
        bcnodes = numpy.unique(numpy.concatenate([bc.nodes for bc in bcs]))
        # Disgusting hack. We put 1 on the diagonal with
        # INSERT_VALUES, but when doing MatConvert, the MatIS has
        # forgotten this, so Dirichlet nodes shared across N processes
        # get "N" on the diagonal, rather than 1. By putting 1/N on
        # the diagonal on each process, the global matrix is "right".
        multiplicities = self.multiplicities
        intersections = self.intersections
        if len(bcnodes) > 0:
            P.setValuesLocalRCV(bcnodes.reshape(-1, 1),
                                bcnodes.reshape(-1, 1),
                                (1/multiplicities.indices[bcnodes]).reshape(-1, 1),
                                addv=PETSc.InsertMode.INSERT_VALUES)
        P.assemble()

        # TODO: geneo4petsc makes some objects on COMM_WORLD
        geneo = PETSc.PC().create(comm=pc.comm)

        prefix = pc.getOptionsPrefix()
        geneo.setOptionsPrefix(prefix + "geneo_")

        geneo.setOperators(P, P)
        geneo.setType("geneo")

        geneoimpl.setup(geneo, localDirichlet,
                        multiplicities, intersections)

        geneo.setFromOptions()
        geneo.setUp()
        geneo.incrementTabLevel(1, parent=pc)
        self.geneo = geneo

    def update(self, pc):
        # FIXME: This rebuilds more than it needs to. It should just
        # re-solve the eigenproblems.
        # FIXME: This doesn't update with the new linearisation.
        self.geneo.update()

    def apply(self, pc, x, y):
        self.geneo.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.geneo.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super().view(viewer)
        viewer.printfASCII("GenEO preconditioner, using geneo4PETSc\n")
        self.geneo.view()
