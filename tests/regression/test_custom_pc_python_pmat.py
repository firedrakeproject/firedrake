from firedrake import *
from firedrake.petsc import PETSc
import pytest
import numpy


class MyPC(PCBase):

    needs_python_pmat = True

    def initialize(self, pc):
        pass

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        x.copy(y)

    def applyTranspose(self, pc, x, y):
        x.copy(y)


class AnyPC(MyPC):

    needs_python_pmat = False


def test_python_pc_valueerror():
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)

    A = assemble(inner(u, v) * dx).M.handle
    pc = PETSc.PC().create(comm=mesh.comm)
    pc.setType(pc.Type.PYTHON)
    pc.setOperators(A, A)
    obj = MyPC()
    with pytest.raises(ValueError):
        obj.setUp(pc)


def test_any_pc_fine():
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    uh = Function(V)

    solve(inner(u, v) * dx == conj(v) * dx, uh,
          solver_parameters={"ksp_type": "cg",
                             "pc_type": "python",
                             "pc_python_type": __name__ + "." + "AnyPC"})
    assert numpy.allclose(uh.dat.data_ro, 1)
