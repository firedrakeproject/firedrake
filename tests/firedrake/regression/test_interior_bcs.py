from os.path import abspath, dirname, join

from firedrake import *

cwd = abspath(dirname(__file__))


def test_interior_bc():
    mesh = Mesh(join(cwd, "..", "meshes", "square_with_embedded_line.msh"))
    V = FunctionSpace(mesh, "P", 1)

    v = TestFunction(V)
    uh = Function(V)
    F = inner(grad(uh), grad(v))*dx

    # 0 on left boundary, 10 on right boundary
    bcs = [DirichletBC(V, 0, 1),
           DirichletBC(V, 10, 2)]

    solve(F == 0, uh, bcs=bcs)

    x, y = SpatialCoordinate(mesh)
    expect = 10*x
    assert errornorm(expect, uh) < 1e-10

    # Now put a no-penetration boundary in the interior

    bcs = [DirichletBC(V, 10, 1),
           DirichletBC(V, 10, 2),
           DirichletBC(V, 0, 5)]

    expect = conditional(x < 0.5, -20*x + 10, 20*x - 10)

    solve(F == 0, uh, bcs=bcs)
    assert errornorm(expect, uh) < 1e-10
