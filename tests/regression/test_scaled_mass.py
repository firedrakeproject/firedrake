import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.mark.parametrize('expr',
                         ['f',
                          '2*f',
                          'tanh(f)',
                          '2 * tanh(f)',
                          'f + tanh(f)',
                          'cos(f) + sin(f)',
                          'cos(f)*cos(f) + sin(f)*sin(f)',
                          'tanh(f) + cos(f) + sin(f)',
                          '1.0/tanh(f) + 1.0/f',
                          'sqrt(f*f)',
                          '1.0/tanh(sqrt(f*f)) + 1.0/f + sqrt(f*f)'],
                         ids=lambda x: 'expr=(%s)' % x)
@pytest.mark.parametrize('value',
                         [1, 10, 20, -1, -10, -20],
                         ids=lambda x: 'f=(%d)' % x)
@pytest.mark.parametrize('typ',
                         ['Function', 'Constant'])
@pytest.mark.parametrize('fs_type',
                         ['scalar', 'vector', 'tensor'])
def test_math_functions(mesh, expr, value, typ, fs_type):
    if typ == 'Function':
        if fs_type == 'vector':
            V = VectorFunctionSpace(mesh, 'CG', 1)
        elif fs_type == 'tensor':
            V = TensorFunctionSpace(mesh, 'CG', 1)
        else:
            V = FunctionSpace(mesh, 'CG', 1)
        f = Function(V)
        f.assign(value)
        if fs_type == 'vector':
            f = dot(f, f)
        elif fs_type == 'tensor':
            f = inner(f, f)
    elif typ == 'Constant':
        if fs_type == 'vector':
            f = Constant([value, value])
            f = dot(f, f)
        elif fs_type == 'tensor':
            f = Constant([[value, value], [value, value]])
            f = inner(f, f)
        else:
            f = Constant(value)

    H = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(H)
    v = TestFunction(H)

    C = eval(expr)

    a = (C)*inner(u, v) * dx
    L = (C)*conj(v) * dx
    actual = Function(H)
    solve(a == L, actual)

    assert norm(assemble(actual - 1)) < 1e-6


@pytest.fixture(scope="module",
                params=["triangle", "tet"])
def m(request):
    if request.param == "triangle":
        return UnitTriangleMesh()
    elif request.param == "tet":
        return UnitTetrahedronMesh()


@pytest.mark.parametrize("value",
                         [-1, 1, 2],
                         ids=lambda x: "Scaling[%d]" % x)
@pytest.mark.parametrize("typ",
                         ["number", "Constant", "Function"],
                         ids=lambda x: "Type=%s" % x)
@pytest.mark.parametrize("degree",
                         [0, 1, 2],
                         ids=lambda x: "DG(%d)" % x)
def test_scalar_scaled_mass(m, value, typ, degree):
    if typ == "number":
        c = value
    elif typ == "Constant":
        c = Constant(value)
    elif typ == "Function":
        V = FunctionSpace(m, "DG", 0)
        c = Function(V)
        c.assign(value)

    V = FunctionSpace(m, "DG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)

    scaled = assemble(c*inner(u, v) * dx)

    assert np.allclose(mass.M.values * value, scaled.M.values)

    scaled_sum = assemble(c*inner(u, v) * dx + inner(u, v) * dx)

    assert np.allclose(mass.M.values * (value + 1), scaled_sum.M.values)


@pytest.mark.parametrize("value",
                         [-1, 1, 2],
                         ids=lambda x: "Scaling[%d]" % x)
@pytest.mark.parametrize("typ",
                         ["number", "Constant", "Function"],
                         ids=lambda x: "Type=%s" % x)
@pytest.mark.parametrize("degree",
                         [1, 2],
                         ids=lambda x: "(%d)" % x)
@pytest.mark.parametrize("space",
                         ["DG", "RT", "BDM", "N1curl", "N2curl"])
def test_vector_scaled_mass(m, value, typ, degree, space):
    if typ == "number":
        c = value
    elif typ == "Constant":
        c = Constant(value)
    elif typ == "Function":
        V = FunctionSpace(m, "DG", 0)
        c = Function(V)
        c.assign(value)

    if space == "DG":
        V = VectorFunctionSpace(m, space, degree)
    else:
        V = FunctionSpace(m, space, degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)

    scaled = assemble(c*inner(u, v) * dx)

    assert np.allclose(mass.M.values * value, scaled.M.values)

    scaled_sum = assemble(c * inner(u, v) * dx + inner(u, v) * dx)

    assert np.allclose(mass.M.values * (value + 1), scaled_sum.M.values)


@pytest.mark.parametrize("value",
                         [-1, 1, 2],
                         ids=lambda x: "Scaling[%d]" % x)
@pytest.mark.parametrize("typ",
                         ["number", "Constant", "Function"],
                         ids=lambda x: "Type=%s" % x)
@pytest.mark.parametrize("degree",
                         [0, 1, 2],
                         ids=lambda x: "(%d)" % x)
def test_tensor_scaled_mass(m, value, typ, degree):
    if typ == "number":
        c = value
    elif typ == "Constant":
        c = Constant(value)
    elif typ == "Function":
        V = FunctionSpace(m, "DG", 0)
        c = Function(V)
        c.assign(value)

    V = TensorFunctionSpace(m, "DG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    mass = assemble(inner(u, v) * dx)
    scaled = assemble(c * inner(u, v) * dx)

    assert np.allclose(mass.M.values * value, scaled.M.values)

    scaled_sum = assemble(c * inner(u, v) * dx + inner(u, v) * dx)

    assert np.allclose(mass.M.values * (value + 1), scaled_sum.M.values)
