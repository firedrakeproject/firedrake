from firedrake import *
import pytest

def test_domains():

    mesh = UnitSquareMesh(2, 2)

    mesh2 = Mesh(Function(mesh.coordinates))

    V = FunctionSpace(mesh, "BDM", 1)
    u = Function(V)
    v = TestFunction(V)

    V2 = FunctionSpace(mesh2, "BDM", 1)

    u2 = Function(V2)
    v2 = TestFunction(V2)
    form = inner(u, v)*dx(domain=mesh) + inner(u2, v2)*dx(domain=mesh2)

    print(form)
    print(form.ufl_domains())
    print(assemble(form))
