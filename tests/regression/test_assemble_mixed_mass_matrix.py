from firedrake import *
from firedrake.supermeshing import *

def test_assemble_mixed_mass_matrix():
    base = UnitTriangleMesh()
    mh = MeshHierarchy(base, 2)
    mesh_A = mh[-2]
    mesh_B = mh[-1]

    ele = FiniteElement("CG", base.ufl_cell(), 1)
    V_A = FunctionSpace(mesh_A, ele)
    V_B = FunctionSpace(mesh_B, ele)

    assemble_mixed_mass_matrix(V_A, V_B)

if __name__ == "__main__":
    test_assemble_mixed_mass_matrix()
