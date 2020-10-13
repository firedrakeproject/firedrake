from firedrake import *

def test_extruded_extend():
    Nx, Nz = 10, 4
    base_mesh = UnitIntervalMesh(Nx)
    mesh = ExtrudedMesh(base_mesh, layers=Nz, layer_height=1.0/Nx)
    Q1D = FunctionSpace(base_mesh, 'CG', 1)
    x = SpatialCoordinate(base_mesh)
    f1D = Function(Q1D).interpolate(cos(2.0 * pi* x[0]))
    fextended = ExtrudedExtendFunction(mesh, f1D)
    x,y = SpatialCoordinate(mesh)
    Q2D = FunctionSpace(mesh, 'CG', 1)
    f = Function(Q2D).interpolate(cos(2.0 * pi* x))
    assert errornorm(f, fextended) < 1.0e-10

def test_extruded_extend_topbottom():
    Nx, Nz = 10, 4
    base_mesh = UnitIntervalMesh(Nx)
    mesh = ExtrudedMesh(base_mesh, layers=Nz, layer_height=1.0/Nx)
    x,y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V).interpolate(cos(2.0 * pi* x) * y)
    fbottom = Function(V).interpolate(0.0 * x)
    ftop = Function(V).interpolate(cos(2.0 * pi* x) * 0.4)
    fbottomextend = ExtrudedExtendFunction(mesh, f, extend_type='bottom')
    ftopextend = ExtrudedExtendFunction(mesh, f, extend_type='top')
    assert errornorm(fbottom, fbottomextend) < 1.0e-10
    assert errornorm(ftop, ftopextend) < 1.0e-10

