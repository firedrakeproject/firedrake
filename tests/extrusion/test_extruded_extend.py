from firedrake import *

def standard2d():
    Nx, Nz = 4, 3
    base_mesh = UnitIntervalMesh(Nx)
    mesh = ExtrudedMesh(base_mesh, layers=Nz, layer_height=1.0/Nz)
    return mesh

def standard3d():
    Nx, Ny, Nz = 4, 3, 3
    base_mesh = UnitSquareMesh(Nx, Ny, quadrilateral=True)
    mesh = ExtrudedMesh(base_mesh, layers=Nz, layer_height=1.0/Nz)
    return mesh

def test_extruded_extend_scalar():
    mesh = standard2d()
    Vbase = FunctionSpace(mesh._base_mesh,'P',1)
    x = SpatialCoordinate(mesh._base_mesh)
    fbase = Function(Vbase).interpolate(cos(2.0 * pi* x[0]))
    fextended = ExtrudedExtendFunction(mesh, fbase)
    x,y = SpatialCoordinate(mesh)
    Vextruded = FunctionSpace(mesh,'Q',1)
    f = Function(Vextruded).interpolate(cos(2.0 * pi* x))
    assert errornorm(f, fextended) < 1.0e-10
    fextendedagain = ExtrudedExtendFunction(mesh, fbase, target=fextended)
    assert errornorm(f, fextendedagain) < 1.0e-10

def test_extruded_extend_topbottom():
    mesh = standard2d()
    x,y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh,'Q',1)
    f = Function(V).interpolate(cos(2.0 * pi* x) * y)
    fbottom = Function(V).interpolate(0.0 * x)
    ftop = Function(V).interpolate(cos(2.0 * pi* x))
    fbottomextend = ExtrudedExtendFunction(mesh, f, extend='bottom')
    ftopextend = ExtrudedExtendFunction(mesh, f, extend='top')
    assert errornorm(fbottom, fbottomextend) < 1.0e-10
    assert errornorm(ftop, ftopextend) < 1.0e-10

