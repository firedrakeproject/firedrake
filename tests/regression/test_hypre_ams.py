import pytest
import numpy
from firedrake import *

def test_homogeneous_field_linear():
    mesh = UnitCubeMesh(5,5,5)
    V = FunctionSpace(mesh, "N1curl" , 1)
    V0 = VectorFunctionSpace(mesh, "DG" , 0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u),curl(v))*dx
    L = inner(Constant((0.,0.,0.)),v)*dx

    class ConstantField(Expression):
      def __init__(self, B0):
        self._B0 = B0
        super().__init__()
      def eval(self, value, X):
        value[0] = -0.5*self._B0*(X[1]-0.5)
        value[1] =  0.5*self._B0*(X[0]-0.5)
        value[2] = 0.0
      def value_shape(self):
        return (3,)

    bc = DirichletBC(V, ConstantField(1.), (1,2,3,4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_itmax': '30',
              'ksp_rtol': '1e-15',
              'ksp_atol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'HypreAMS_preconditioner.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True,
             }

    A = Function(V)
    solve(a == L, A, bc, solver_parameters=params)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0.,0.,1.)), atol=1e-6)

def test_homogeneous_field_matfree():
    mesh = UnitCubeMesh(5,5,5)
    V = FunctionSpace(mesh, "N1curl" , 1)
    V0 = VectorFunctionSpace(mesh, "DG" , 0)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u),curl(v))*dx
    L = inner(Constant((0.,0.,0.)),v)*dx

    class ConstantField(Expression):
      def __init__(self, B0):
        self._B0 = B0
        super().__init__()
      def eval(self, value, X):
        value[0] = -0.5*self._B0*(X[1]-0.5)
        value[1] =  0.5*self._B0*(X[0]-0.5)
        value[2] = 0.0
      def value_shape(self):
        return (3,)

    bc = DirichletBC(V, ConstantField(1.), (1,2,3,4))

    params = {'snes_type': 'ksponly',
              'mat_type': 'matfree',
              'ksp_type': 'cg',
              'ksp_itmax': '30',
              'ksp_rtol': '1e-15',
              'ksp_atol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.AssembledPC',
              'assembled_pc_type': 'python',
              'assembled_pc_python_type': 'HypreAMS_preconditioner.HypreAMS',
              'assembled_pc_hypre_ams_zero_beta_poisson': True,
             }

    A = Function(V)
    solve(a == L, A, bc, solver_parameters=params)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0.,0.,1.)), atol=1e-6)


def test_homogeneous_field_nonlinear():
    mesh = UnitCubeMesh(5,5,5)
    V = FunctionSpace(mesh, "N1curl" , 1)
    V0 = VectorFunctionSpace(mesh, "DG" , 0)

    u = Function(V)
    v = TestFunction(V)

    a = inner(curl(u),curl(v))*dx
    L = inner(Constant((0.,0.,0.)),v)*dx

    class ConstantField(Expression):
      def __init__(self, B0):
        self._B0 = B0
        super().__init__()
      def eval(self, value, X):
        value[0] = -0.5*self._B0*(X[1]-0.5)
        value[1] =  0.5*self._B0*(X[0]-0.5)
        value[2] = 0.0
      def value_shape(self):
        return (3,)

    bc = DirichletBC(V, ConstantField(1.), (1,2,3,4))

    params = {'snes_type': 'ksponly',
              'ksp_type': 'cg',
              'ksp_itmax': '30',
              'ksp_rtol': '1e-15',
              'ksp_atol': '1e-15',
              'pc_type': 'python',
              'pc_python_type': 'HypreAMS_preconditioner.HypreAMS',
              'pc_hypre_ams_zero_beta_poisson': True,
             }

    A = Function(V)
    solve(a - L == 0, A, bc, solver_parameters=params)
    B = project(curl(A), V0)
    assert numpy.allclose(B.dat.data_ro, numpy.array((0.,0.,1.)), atol=1e-6)
