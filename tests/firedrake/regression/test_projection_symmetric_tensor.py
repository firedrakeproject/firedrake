import pytest
import numpy as np
from firedrake import *


@pytest.fixture(params=["triangles", "quadrilaterals"], scope="module")
def mesh(request):
    base = UnitSquareMesh(1, 1, quadrilateral=(request.param == "quadrilaterals"))
    mh = MeshHierarchy(base, 1)
    mesh = mh[-1]
    return mesh


@pytest.mark.parametrize(('degree', 'family', 'tdim'), [
    (1, 'Lagrange', 3),
    (2, 'Lagrange', 3),
    (3, 'Lagrange', 3),
    (0, 'Quadrature', 2),
    (1, 'Quadrature', 2),
    (2, 'Quadrature', 2)])
def test_projection_symmetric_tensor(mesh, degree, family, tdim):
    shape = (tdim, tdim)
    remove = 2 if tdim == 2 else [3, 6, 7]
    if family == 'Quadrature':
        Nq = 2*degree+1
        Q = FunctionSpace(mesh, TensorElement(family, mesh.ufl_cell(), degree=Nq,
                                              quad_scheme="default", shape=shape,
                                              symmetry=None))
        Qs = FunctionSpace(mesh, TensorElement(family, mesh.ufl_cell(), degree=Nq,
                                               quad_scheme="default", shape=shape,
                                               symmetry=True))
        sp = {"mat_type": "matfree",
              "ksp_type": "preonly",
              "pc_type": "jacobi"}
        fcp = {"quadrature_degree": Nq}
    else:
        Q = TensorFunctionSpace(mesh, family, degree=degree, shape=shape, symmetry=None)
        Qs = TensorFunctionSpace(mesh, family, degree=degree, shape=shape, symmetry=True)
        sp = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}
        fcp = {}

    x, y = SpatialCoordinate(mesh)
    bcomp = [x, y, x+y]
    b = as_vector(bcomp[:tdim])
    G = (x+y)*Identity(tdim) + outer(b, b)
    P = project(G, Q, solver_parameters=sp, form_compiler_parameters=fcp, use_slate_for_inverse=False)
    Ps = project(G, Qs, solver_parameters=sp, form_compiler_parameters=fcp, use_slate_for_inverse=False)
    X = np.delete(np.reshape(P.dat.data_ro, (-1, Q.value_size)), remove, 1)
    assert np.isclose(Ps.dat.data_ro, X).all()
