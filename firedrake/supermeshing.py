# Code for projections and other fun stuff involving supermeshes.
from firedrake.mg.utils import get_level

__all__ = ["assemble_mixed_mass_matrix"]

def assemble_mixed_mass_matrix(V_A, V_B):
    """
    Construct the mixed mass matrix of two function spaces,
    using the TrialFunction from V_A and the TestFunction 
    from V_B.
    """

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

    for cell_A in range(mesh_A.num_cells()):
        print("likely(%s) = %s" % (cell_A, likely(cell_A)))

    # Preallocate sparsity pattern for mixed mass matrix from likely() function:
    # For each cell_A, find dofs_A.
    # For each cell_B in likely(cell_B), 
    #     Find dofs_B.
    #     For dof_B in dofs_B:
    #         nnz[dof_B] += len(dofs_A)
    #
    # Preallocate M_AB.
    #
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
