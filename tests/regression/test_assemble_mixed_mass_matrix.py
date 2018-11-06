from firedrake import *
from firedrake.supermeshing import *
import numpy as np

def test_assemble_mixed_mass_matrix():
    base = UnitTriangleMesh()
    mh = MeshHierarchy(base, 1)
    mesh_A = mh[-2]
    mesh_B = mh[-2]

    ele_A = FiniteElement("CG", base.ufl_cell(), 3)
    V_A = FunctionSpace(mesh_A, ele_A)
    ele_B = FiniteElement("CG", base.ufl_cell(), 2)
    V_B = FunctionSpace(mesh_B, ele_B)

    M = assemble_mixed_mass_matrix(V_A, V_B)
    M = M[:,:]

    M_ex = assemble(inner(TrialFunction(V_A), TestFunction(V_B)) * dx)
    M_ex.force_evaluation()
    M_ex = M_ex.M.handle[:,:]
    print("M_ex\n", M_ex)
    print("M\n", M)
    assert np.allclose(M_ex, M)


if __name__ == "__main__":
    test_assemble_mixed_mass_matrix()
