import numpy as np
from firedrake import *
import pyop3 as op3 

GEM = {
    "pyop3_compiler_parameters": {"backend": "gem"},
    "form_compiler_parameters": {"backend": "gem"},
}
mesh = UnitSquareMesh(3, 3)

def demo1():
    V = VectorFunctionSpace(mesh, "P", 1)

    f = Function(V)
    g = Function(V)

    f.dat.assign(2*g.dat+3, eager=True, eager_strategy="compile", compiler_parameters={"backend": "gem"})

def demo2():
    V = FunctionSpace(mesh, "P", 1)

    c = Constant(10)
    v = TestFunction(V)

    # do a regular one first to make sure that it still runs
    # assemble(c*v*dx)

    gem_a = assemble(
        c*v*dx, 
        **GEM,
    )

    print(gem_a.dat.data_ro)

def demo3():
    mesh = UnitSquareMesh(3, 3, quadrilateral = True)
    V = FunctionSpace(mesh, "P", 2)

    x, y = SpatialCoordinate(mesh)
    v = TestFunction(V)

    gem_b = assemble(
        x*y*v*dx,
        **GEM,
    )

    # The P2 basis functions sum to 1, so the entries should sum to the integral of x*y
    # over the unit square - 1/4.
    total = gem_b.dat.data_ro.sum()
    print(f"sum of entries = {total!r}, expected 0.25")
    assert np.isclose(total, 0.25)

def demo4():
    V = FunctionSpace(mesh, "P", 2)

    c = Constant(10)
    v = TestFunction(V)

    gpu = op3.CUDAGPU()

    with op3.offloading(gpu):
        gem_a = assemble(
            c*v*dx, 
            **GEM,
        )

def demo5():
    mesh = UnitSquareMesh(3, 3, quadrilateral = True)
    V = FunctionSpace(mesh, "P", 2)

    x, y = SpatialCoordinate(mesh)
    v = TestFunction(V)

    gpu = op3.CUDAGPU()

    with op3.offloading(gpu):
        gem_b = assemble(
            x*y*v*dx,
            **GEM,
        )

        # The P2 basis functions sum to 1, so the entries should sum to the integral of x*y
        # over the unit square - 1/4.
        total = gem_b.dat.data_ro.sum()

    print(f"sum of entries = {total!r}, expected 0.25")
    assert np.isclose(total, 0.25)

def demo6():
    # Mass operator: the action of the mass matrix on a coefficient.
    V = FunctionSpace(mesh, "P", 3)

    u = Function(V).assign(10)
    v = TestFunction(V)

    gpu = op3.CUDAGPU()

    with op3.offloading(gpu):
        gem_b = assemble(
            u*v*dx,
            **GEM,
        )

    # The P2 basis functions sum to 1, so the entries should sum to the integral of u = 10
    # over the unit square - 10.
    total = gem_b.dat.data_ro.sum()
    print(f"sum of entries = {total!r}, expected 10")
    assert np.isclose(total, 10)

def demo7():
    # Poisson operator: the action of the stiffness matrix on a coefficient.
    V = FunctionSpace(mesh, "P", 3)

    u = Function(V).assign(10)
    v = TestFunction(V)

    gpu = op3.CUDAGPU()

    with op3.offloading(gpu):
        gem_b = assemble(
            inner(grad(u), grad(v))*dx,
            **GEM,
        )

    # The gradient of a constant function is zero, so every entry should be zero.
    largest = abs(gem_b.dat.data_ro).max()
    print(f"largest entry = {largest!r}, expected 0")
    assert np.isclose(largest, 0)

def demo8():
    # Poisson operator on a function of the coordinates. No coefficient is needed.
    V = FunctionSpace(mesh, "P", 2)

    x, y = SpatialCoordinate(mesh)
    v = TestFunction(V)

    gpu = op3.CUDAGPU()

    with op3.offloading(gpu):
        gem_b = assemble(
            inner(grad(x*y), grad(v))*dx,
            **GEM,
        )

    total = gem_b.dat.data_ro.sum()
    norm = np.linalg.norm(gem_b.dat.data_ro)
    print(f"sum of entries = {total!r}, expected 0")
    print(f"norm of entries = {norm!r}, expected 0.48784962490072176")
    assert np.isclose(total, 0)
    assert np.isclose(norm, 0.48784962490072176)

def demo9():
    # MFE for gem.Identity
    V = FunctionSpace(mesh, "P", 1)

    x, y = SpatialCoordinate(mesh)
    v = TestFunction(V)

    gem_b = assemble(
        inner(grad(x*y), grad(v))*dx,
        **GEM,
    )

    total = gem_b.dat.data_ro.sum()
    norm = np.linalg.norm(gem_b.dat.data_ro)
    print(f"sum of entries = {total!r}, expected 0")
    print(f"norm of entries = {norm!r}, expected 0.608016857754889")
    assert np.isclose(total, 0)
    assert np.isclose(norm, 0.608016857754889)

def demo10():
    # Hyperelasticity
    V = VectorFunctionSpace(mesh, "P", 2)

    x, y = SpatialCoordinate(mesh)
    u = Function(V).interpolate(as_vector([0.1*x*y, 0.05*x*x]))
    v = TestFunction(V)

    mu, lmbda = Constant(1.0), Constant(1.25)
    F = Identity(2) + grad(u)
    J = det(F)
    psi = mu/2*(tr(F.T*F) - 2) - mu*ln(J) + lmbda/2*ln(J)**2

    gpu = op3.CUDAGPU()

    with op3.offloading(gpu):
        gem_b = assemble(
            derivative(psi*dx, u, v),
            **GEM,
        )
        data = gem_b.dat.data_ro.copy()

    total = data.sum(axis=0)
    norm = np.linalg.norm(data)
    print(f"sum of entries = {total!r}, expected 0")
    print(f"norm of entries = {norm!r}, expected 0.15636381794828516")
    assert np.allclose(total, 0)
    assert np.isclose(norm, 0.15636381794828516)

# demo1()
# demo2()
# demo3()
# demo4()
demo5()
# demo6()
# demo7()
# demo8()
# demo9()
# demo10()
