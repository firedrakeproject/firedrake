"""This tests the assembly of LHSs containing facet integrals"""
import pytest
from firedrake import *


def test_lhs():
    m = UnitSquareMesh(2, 2)
    mesh = ExtrudedMesh(m, layers=2, layer_height=0.5)
    V = FunctionSpace(mesh, "CG", 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    assemble(u*v*ds_b).M._force_evaluation()
    assemble(u*v*ds_t).M._force_evaluation()
    assemble(u*v*ds_tb).M._force_evaluation()
    assemble(u*v*ds_v).M._force_evaluation()
    assemble(u('-')*v('+')*dS_h).M._force_evaluation()
    assemble(u('+')*v('-')*dS_v).M._force_evaluation()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
