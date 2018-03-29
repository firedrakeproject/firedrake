import pytest
import numpy as np
from firedrake import *


def test_first_shape_derivative():
    mesh = UnitSquareMesh(6, 6)
    X = SpatialCoordinate(mesh)
    x, y = X
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    dX = TestFunction(mesh.coordinates.function_space())

    J = u * u * dx
    computed = assemble(derivative(J, X)).dat.data
    actual = assemble(u * u * div(dX) * dx).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)    

    J = inner(grad(u), grad(u)) * dx
    computed = assemble(derivative(J, X)).dat.data
    dJdX = -2*inner(dot(grad(dX), grad(u)), grad(u)) * dx + inner(grad(u), grad(u)) * div(dX) * dx
    actual = assemble(dJdX).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)    


    f = x * y * sin(x) * cos(y)
    J = f * dx
    computed = assemble(derivative(J, X)).dat.data
    dJdX = div(f*dX) * dx
    actual = assemble(dJdX).dat.data
    assert np.allclose(computed, actual, rtol=1e-14)    

def test_mixed_derivatives():
    mesh = UnitSquareMesh(6, 6)
    X = SpatialCoordinate(mesh)
    x, y = X
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    v = TrialFunction(V)
    dX = TestFunction(mesh.coordinates.function_space())

    J = u * u * dx
    computed1 = assemble(derivative(derivative(J, X), u)).M.values
    computed2 = assemble(derivative(derivative(J, u), X)).M.values
    actual = assemble(2 * u * v * div(dX) * dx).M.values
    assert np.allclose(computed1, actual, rtol=1e-14)    
    assert np.allclose(computed2.T, actual, rtol=1e-14)    

    J = inner(grad(u), grad(u)) * dx
    computed1 = assemble(derivative(derivative(J, X), u)).M.values
    computed2 = assemble(derivative(derivative(J, u), X)).M.values
    actual = assemble(2*inner(grad(u), grad(v)) * div(dX) * dx
                      - 2*inner(dot(grad(dX), grad(u)), grad(v)) * dx 
                      - 2*inner(grad(u), dot(grad(dX), grad(v))) * dx).M.values
    assert np.allclose(computed1, actual, rtol=1e-14)    
    assert np.allclose(computed2.T, actual, rtol=1e-14)    

    
def test_second_shape_derivative():
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    X = SpatialCoordinate(mesh)
    x, y = X
    u.interpolate(x*x+y*x+y*y+sin(x)+cos(x))
    dX1 = TestFunction(mesh.coordinates.function_space())
    dX2 = TrialFunction(mesh.coordinates.function_space())

    J = u * u * dx
    computed = assemble(derivative(derivative(J, X, dX1), X, dX2)).M.values
    actual = assemble(u * u * div(dX1) * div(dX2) * dx - u * u * tr(grad(dX1)*grad(dX2)) * dx).M.values
    assert np.allclose(computed, actual, rtol=1e-14)    
    

if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
