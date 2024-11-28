from firedrake import *


def test_with_mapping():
    """
    This test creates a deformed domain and checks that the function
    values are entirely horizontal if identity mapping is used.
    """

    n = 10
    m = 5
    mesh = UnitIntervalMesh(n)
    mesh = ExtrudedMesh(mesh, m)

    x, z = SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([x, z*(1-0.5*x)]))
    mesh.coordinates.assign(f)

    horiz_h = FiniteElement("CG", interval, 1)
    horiz_v = FiniteElement("DG", interval, 0)
    horiz = TensorProductElement(horiz_h, horiz_v)
    horiz_hdiv = HDivElement(horiz)
    Vorig = FunctionSpace(mesh, horiz_hdiv)
    remapped = WithMapping(horiz_hdiv, "identity")
    V = FunctionSpace(mesh, remapped)

    x, z = SpatialCoordinate(mesh)
    uexp = as_vector([cos(x+z), sin(x-z)])

    uorig = Function(Vorig).project(uexp)
    # Piola map means that functions in horizontal space
    # have some vertical component
    vval = assemble(uorig[1]*uorig[1]*dx)
    assert vval > 1.0e-3

    # No piola map means that functions in horizontal space
    # have no vertical component
    uremapped = Function(V).project(uexp)
    revval = assemble(uremapped[1]*uremapped[1]*dx)
    assert revval < 1.0e-6
