"""This tests the assembly of RHSs containing facet integrals"""
import pytest
from firedrake import *


def test_rhs():
    m = UnitSquareMesh(2, 2)
    mesh = ExtrudedMesh(m, layers=2, layer_height=0.5)
    V = FunctionSpace(mesh, "CG", 2)

    v = TestFunction(V)

    assemble(v*ds_b).dat._force_evaluation()
    assemble(v*ds_t).dat._force_evaluation()
    assemble(v*ds_tb).dat._force_evaluation()
    assemble(v*ds_v).dat._force_evaluation()
    assemble(v('+')*dS_h).dat._force_evaluation()
    assemble(v('-')*dS_v).dat._force_evaluation()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
