import os
import pytest
import numpy as np
from firedrake import *
from petsc4py import PETSc


cwd = os.path.abspath(os.path.dirname(__file__))


def _test_submesh_mixed_cell_base():
    dim = 2
    mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"))
    mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
    x_t, y_t = SpatialCoordinate(mesh_t)
    n_t = FacetNormal(mesh_t)
    mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
    x_q, y_q = SpatialCoordinate(mesh_q)
    n_q = FacetNormal(mesh_q)
    # pgfplot(f, "mesh_tri.dat", degree=2)
    dx_t = Measure("dx", mesh_t)
    dx_q = Measure("dx", mesh_q)
    ds_t = Measure("ds", mesh_t)
    ds_q = Measure("ds", mesh_q)
    A_t = assemble(Constant(1) * dx_t)
    A_q = assemble(Constant(1) * dx_q)
    assert abs(A_t + A_q -1.0) < 1.e-13
    HDiv_t = FunctionSpace(mesh_t, "BDM", 3)
    HDiv_q = FunctionSpace(mesh_q, "RTCF", 3)
    hdiv_t = Function(HDiv_t).interpolate(as_vector([x_t**2, y_t**2]))
    hdiv_q = Function(HDiv_q).project(as_vector([x_q**2, y_q**2]), solver_parameters={"ksp_rtol": 1.e-13})
    v_t = assemble(dot(hdiv_q, as_vector([x_q, y_q])) * ds_t(0))
    v_q = assemble(dot(hdiv_t, as_vector([x_t, y_t])) * ds_q(0))
    assert abs(v_q - v_t) < 1.e-13
    v_t = assemble(dot(hdiv_q, as_vector([x_t, y_t])) * ds_t(0))
    v_q = assemble(dot(hdiv_t, as_vector([x_q, y_q])) * ds_q(0))
    assert abs(v_q - v_t) < 1.e-13
    v_t = assemble(dot(hdiv_q, as_vector([x_q, y_t])) * ds_t(0))
    v_q = assemble(dot(hdiv_t, as_vector([x_t, y_q])) * ds_q(0))
    assert abs(v_q - v_t) < 1.e-13
    v = assemble(inner(n_t, as_vector([888., 999.])) * ds_t(0))
    assert abs(v) < 1.e-13
    v = assemble(inner(n_q, as_vector([888., 999.])) * ds_q(0))
    assert abs(v) < 1.e-13
    v = assemble(inner(n_q, as_vector([888., 999.])) * ds_t(0))
    assert abs(v) < 1.e-13
    v = assemble(inner(n_t, as_vector([888., 999.])) * ds_q(0))
    assert abs(v) < 1.e-13
    v = assemble(dot(n_q + n_t, n_q + n_t) * ds_t(0))
    assert abs(v) < 1.e-30
    v = assemble(dot(n_q + n_t, n_q + n_t) * ds_q(0))
    assert abs(v) < 1.e-30


def test_submesh_mixed_cell_base_one_process():
    _test_submesh_mixed_cell_base()


@pytest.mark.parallel(nprocs=2)
def test_submesh_mixed_cell_base_two_processes():
    _test_submesh_mixed_cell_base()


@pytest.mark.parallel(nprocs=3)
def test_submesh_mixed_cell_base_three_processes():
    _test_submesh_mixed_cell_base()


def test_submesh_mixed_cell_assemble():
    dim = 2
    label_interf = 0
    mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"))
    mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
    x_t, y_t = SpatialCoordinate(mesh_t)
    n_t = FacetNormal(mesh_t)
    mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
    x_q, y_q = SpatialCoordinate(mesh_q)
    n_q = FacetNormal(mesh_q)
    V_t = FunctionSpace(mesh_t, "P", 4)
    V_q = FunctionSpace(mesh_q, "Q", 3)
    V = V_t * V_q
    u = TrialFunction(V)
    v = TestFunction(V)
    u_t, u_q = split(u)
    v_t, v_q = split(v)
    dx_t = Measure("dx", mesh_t)
    dx_q = Measure("dx", mesh_q)
    ds_t = Measure("ds", mesh_t)
    ds_q = Measure("ds", mesh_q)
    # Test against the base cases.
    c = x_t**2 * y_t**2
    a = c * inner(u_t, v_q) * ds_t(label_interf)
    A = assemble(a)
    c_ref = x_q**2 * y_q**2
    a_ref = c_ref * inner(TrialFunction(V_t), TestFunction(V_q)) * ds_t(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(A.M[1][0].values, A_ref.M.values)
    c = x_t**2 * y_q**2
    a = c * inner(u_q, v_t) * ds_t(label_interf)
    A = assemble(a)
    c_ref = x_q**2 * y_t**2
    a_ref = c_ref * inner(TrialFunction(V_q), TestFunction(V_t)) * ds_t(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(A.M[0][1].values, A_ref.M.values)
    c = dot(n_t, n_t)
    a = c * inner(u_t, v_q) * ds_q(label_interf)
    A = assemble(a)
    c_ref = dot(n_q, n_q)
    a_ref = c_ref * inner(TrialFunction(V_t), TestFunction(V_q)) * ds_q(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(A.M[1][0].values, A_ref.M.values)
    c = dot(n_t, n_q)
    a = c * inner(u_q, v_t) * ds_q(label_interf)
    A = assemble(a)
    c_ref = dot(n_q, n_t)
    a_ref = c_ref * inner(TrialFunction(V_q), TestFunction(V_t)) * ds_q(label_interf)
    A_ref = assemble(a_ref)
    assert np.allclose(A.M[0][1].values, A_ref.M.values)
