from firedrake import *

def standardextruded2d():
    Nx, Nz = 4, 3
    base_mesh = UnitIntervalMesh(Nx)
    mesh = ExtrudedMesh(base_mesh, layers=Nz, layer_height=1.0/Nz)
    return mesh

def standardextruded3d():
    Nx, Ny, Nz = 4, 3, 3
    base_mesh = UnitSquareMesh(Nx, Ny, quadrilateral=True)
    mesh = ExtrudedMesh(base_mesh, layers=Nz, layer_height=1.0/Nz)
    return mesh

def test_extruded_extend_scalar():
    mesh = standardextruded2d()
    xbase = SpatialCoordinate(mesh._base_mesh)
    Vbase = FunctionSpace(mesh._base_mesh,'P',1)
    fbase = Function(Vbase).interpolate(cos(2.0 * pi* xbase[0]))
    fextruded = extrude_function(mesh, fbase)
    x,y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh,'Q',1)
    f = Function(V).interpolate(cos(2.0 * pi* x))
    assert errornorm(f, fextruded) < 1.0e-10
    fextrudedagain = extrude_function(mesh, fbase, target=fextruded)
    assert errornorm(f, fextrudedagain) < 1.0e-10

def test_extruded_extend_vector_tensor():
    mesh = standardextruded3d()
    xbase,ybase = SpatialCoordinate(mesh._base_mesh)
    Vbase = VectorFunctionSpace(mesh._base_mesh,FiniteElement('Q',quadrilateral,1))
    fbase = Function(Vbase).interpolate(as_vector([cos(2.0 * pi * xbase),cos(2.0 * pi * ybase)]))
    fextruded = extrude_function(mesh, fbase)
    x,y,z = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh,'Q',1, dim=2)  # note dimension matches base_mesh
    f = Function(V).interpolate(as_vector([cos(2.0 * pi * x),cos(2.0 * pi * y)]))
    assert errornorm(f, fextruded) < 1.0e-10
    Tbase = TensorFunctionSpace(mesh._base_mesh,FiniteElement('Q',quadrilateral,1))
    fbase = Function(Tbase).interpolate(as_tensor([[cos(2.0 * pi * xbase),cos(2.0 * pi * ybase)],
                                                   [xbase,ybase]]))
    fextruded = extrude_function(mesh, fbase)
    x,y,z = SpatialCoordinate(mesh)
    T = TensorFunctionSpace(mesh,'Q',1, shape=(2,2))  # note dimension matches base_mesh
    f = Function(T).interpolate(as_tensor([[cos(2.0 * pi * x),cos(2.0 * pi * y)],
                                           [x,y]]))
    assert errornorm(f, fextruded) < 1.0e-10

def test_extruded_extend_topbottom():
    mesh = standardextruded2d()
    x,y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh,'Q',1)
    f = Function(V).interpolate(cos(2.0 * pi* x) * y)
    fbottom = Function(V).interpolate(0.0 * x)
    ftop = Function(V).interpolate(cos(2.0 * pi* x))
    fbottomextruded = extrude_function(mesh, f, extend='bottom')
    ftopextruded = extrude_function(mesh, f, extend='top')
    assert errornorm(fbottom, fbottomextruded) < 1.0e-10
    assert errornorm(ftop, ftopextruded) < 1.0e-10

def test_extruded_extend_topbottom_vector():
    mesh = standardextruded2d()
    x,y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh,'Q',1)
    f = Function(V).interpolate(as_vector([cos(2.0 * pi* x) * y,x]))
    fbottom = Function(V).interpolate(as_vector([0.0 * x,x]))
    ftop = Function(V).interpolate(as_vector([cos(2.0 * pi* x),x]))
    fbottomextruded = extrude_function(mesh, f, extend='bottom')
    ftopextruded = extrude_function(mesh, f, extend='top')
    assert errornorm(fbottom, fbottomextruded) < 1.0e-10
    assert errornorm(ftop, ftopextruded) < 1.0e-10

