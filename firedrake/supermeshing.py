# Code for projections and other fun stuff involving supermeshes.
from firedrake.mg.utils import get_level
from firedrake.petsc import PETSc
import numpy
from pyop2.datatypes import IntType, ScalarType
from pyop2.sparsity import get_preallocation

__all__ = ["assemble_mixed_mass_matrix"]

def assemble_mixed_mass_matrix(V_A, V_B):
    """
    Construct the mixed mass matrix of two function spaces,
    using the TrialFunction from V_A and the TestFunction 
    from V_B.
    """

    if len(V_A) > 1 or len(V_B) > 1:
        raise NotImplementedError("Sorry, only implemented for non-mixed spaces")
    mesh_A = V_A.mesh()
    mesh_B = V_B.mesh()

    (mh_A, level_A) = get_level(mesh_A)
    (mh_B, level_B) = get_level(mesh_B)

    if mesh_A is not mesh_B:
        if (mh_A is None or mh_B is None) or (mh_A is not mh_B):
            msg = """
Sorry, only implemented for non-nested hierarchies for now. You need to
call libsupermesh's intersection finder here to compute the likely cell
coverings that we fetch from the hierarchy.
"""

            raise NotImplementedError(msg)

    if abs(level_A - level_B) > 1:
        raise NotImplementedError("Only works for transferring between adjacent levels for now.")

    # What are the cells of B that (probably) intersect with a given cell in A?
    if level_A > level_B:
        cell_map = mh_A.fine_to_coarse_cells[level_A]
    else:
        cell_map = mh_A.coarse_to_fine_cells[level_A]

    def likely(cell_A):
        return cell_map[cell_A]

    # for cell_A in range(mesh_A.num_cells()):
    #     print("likely(%s) = %s" % (cell_A, likely(cell_A)))


    # Preallocate sparsity pattern for mixed mass matrix from likely() function:
    # For each cell_A, find dofs_A.
    #   For each cell_B in likely(cell_B), 
    #     Find dofs_B.
    #     For dof_B in dofs_B:
    #         nnz[dof_B] += len(dofs_A)
    preallocator = PETSc.Mat().create(comm=mesh_A.comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)

    rset = V_B.dof_dset
    cset = V_A.dof_dset

    nrows = rset.layout_vec.getSizes()
    ncols = cset.layout_vec.getSizes()

    preallocator.setLGMap(rmap=rset.scalar_lgmap, cmap=cset.scalar_lgmap)
    preallocator.setSizes(size=(nrows, ncols), bsize=1)
    preallocator.setUp()

    zeros = numpy.zeros((V_B.cell_node_map().arity, V_A.cell_node_map().arity), dtype=ScalarType)
    for cell_A, dofs_A in enumerate(V_A.cell_node_map().values):
        for cell_B in likely(cell_A):
            if cell_B >= mesh_B.cell_set.size:
                # In halo region
                continue
            dofs_B = V_B.cell_node_map().values[cell_B, :]
            preallocator.setValuesLocal(dofs_B, dofs_A, zeros)
    preallocator.assemble()

    dnnz, onnz = get_preallocation(preallocator, nrows[0])

    # Unroll from block to AIJ
    dnnz = dnnz * cset.cdim
    dnnz = numpy.repeat(dnnz, rset.cdim)
    onnz = onnz * cset.cdim
    onnz = numpy.repeat(onnz, cset.cdim)
    preallocator.destroy()

    assert V_A.value_size == V_B.value_size
    rdim = V_B.dof_dset.cdim
    cdim = V_A.dof_dset.cdim

    #
    # Preallocate M_AB.
    #
    mat = PETSc.Mat().create(comm=mesh_A.comm)
    mat.setType(PETSc.Mat.Type.AIJ)
    rsizes = tuple(n * rdim for n in nrows)
    csizes = tuple(c * cdim for c in ncols)
    mat.setSizes(size=(rsizes, csizes),
                 bsize=(rdim, cdim))
    mat.setPreallocationNNZ((dnnz, onnz))
    mat.setLGMap(rmap=rset.lgmap, cmap=cset.lgmap)
    # TODO: Boundary conditions not handled.
    mat.setOption(mat.Option.IGNORE_OFF_PROC_ENTRIES, False)
    mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    mat.setOption(mat.Option.KEEP_NONZERO_PATTERN, True)
    mat.setOption(mat.Option.UNUSED_NONZERO_LOCATION_ERR, False)
    mat.setOption(mat.Option.IGNORE_ZERO_ENTRIES, True)
    mat.setUp()

    vertices_A = mesh_A.coordinates.dat._data
    vertex_map_A = mesh_A.coordinates.cell_node_map().values
    vertices_B = mesh_B.coordinates.dat._data
    vertex_map_B = mesh_B.coordinates.cell_node_map().values
    # Magic number! 22 in 2D, 81 in 3D
    # TODO: need to be careful in "complex" mode, libsupermesh needs real coordinates.
    tris_C = numpy.empty((22, 3, 2), dtype=numpy.float64)
    for cell_A in range(len(V_A.cell_node_map().values)):
        for cell_B in likely(cell_A):
            if cell_B >= mesh_B.cell_set.size:
                # In halo region
                continue
            tri_A = vertices_A[vertex_map_A[cell_A, :], :].flatten()
            tri_B = vertices_B[vertex_map_B[cell_B, :], :].flatten()
            """
            libsupermesh_intersect_tris_real(&tri_A[0], &tri_B[0], &tris_C[0], &ntris);
            if (ntris == 0)
                continue;
            double MAB[NB][NA];
            for (int c = 0; c < ntris; c++) {
                cell_S = tris_C + c*6;
                evaluate V_A at dofs(A) in cell_S;
                assemble mass A-B on cell_S;
                evaluate V_B at dofs(B) in cell_S;
                for ( int i = 0; i < NB; i++ ) {
                    for (int j = 0; j < NA; j++) {
                        MAB[i][j] = 0;
                        for ( int k = 0; k < NB; k++) {
                            for ( int l = 0; l < NA; l++) {
                                MAB[i][j] += R_BS[k][i] * M_SS[k][l] * R_AS[l][j];
                            }
                        }
                    }
                }
            }
            """
    # Compute M_AB:
    # For cell_A in mesh_A:
    #     For cell_B in likely(cell_A):
    #         mesh_S = supermesh(cell_A, cell_B)
    #         if mesh_S is empty: continue
    #         For cell_S in mesh_S:
    #             evaluate basis functions of cell_A at dofs(A) of cell_S -> R_AS matrix
    #             scale precomputed mass matrix to get M_SS
    #                   (or mixed mass matrix if V_A, V_B have different finite elements)
    #             evaluate basis functions of cell_B at dofs(B) of cell_S -> R_BS matrix
    #             compute out = R_BS^T @ M_SS @ R_AS with dense matrix triple product
    #             stuff out into relevant part of M_AB (given by outer(dofs_B, dofs_A))

    return mat
