from firedrake import *
import pytest
import numpy as np


def test_interpolate_operator():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    expression = x * y
    # [test_interpolate_operator 1]
    # create a symbolic expression for the interpolation operation.
    f_i = interpolate(expression, V)

    # assemble the interpolation to get the result
    f = assemble(f_i)
    # [test_interpolate_operator 2]
    assert isinstance(f, Function)

    # [test_interpolate_operator 3]
    f = Function(V)
    f.interpolate(expression)
    # [test_interpolate_operator 4]
    f2 = f
    # [test_interpolate_operator 5]
    f = Function(V)
    f.assign(assemble(interpolate(expression, V)))
    # [test_interpolate_operator 6]
    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)

    W = FunctionSpace(mesh, "RT", 1)
    g = project(as_vector((sin(x), cos(y))), W)
    # [test_interpolate_operator 7]
    # g is a vector-valued Function, e.g. on an H(div) function space
    f = assemble(interpolate(sqrt(3.2 * div(g)), V))
    # [test_interpolate_operator 8]

    # [test_interpolate_operator 9]
    trace = FunctionSpace(mesh, "HDiv Trace", 0)
    f = assemble(interpolate(expression, trace))
    # [test_interpolate_operator 10]

    U = FunctionSpace(mesh, "CG", 3)
    g = Function(U).interpolate(expression)
    # [test_interpolate_operator 11]
    A = assemble(interpolate(TrialFunction(U), V))
    # [test_interpolate_operator 12]
    h = assemble(A @ g)
    # [test_interpolate_operator 13]
    assert np.allclose(h.dat.data_ro, f2.dat.data_ro)

    # [test_interpolate_operator 14]
    Istar1 = interpolate(TestFunction(U), TrialFunction(V.dual()))
    # [test_interpolate_operator 15]
    Istar2 = adjoint(interpolate(TrialFunction(U), V))
    # [test_interpolate_operator 16]
    cofunc = assemble(inner(1, TestFunction(V)) * dx)  # a cofunction in V*
    res1 = assemble(interpolate(TestFunction(U), cofunc))  # a cofunction in U*
    # [test_interpolate_operator 17]
    res2 = assemble(action(Istar1, cofunc))  # same as res1
    # [test_interpolate_operator 18]
    u = Function(U)
    # [test_interpolate_operator 19]
    interpolate(u, cofunc)
    # [test_interpolate_operator 20]

    res3 = assemble(action(Istar2, cofunc))  # same as res1
    assert isinstance(res1, Cofunction)
    assert np.allclose(res1.dat.data_ro, res2.dat.data_ro)
    assert np.allclose(res1.dat.data_ro, res3.dat.data_ro)


def test_interpolate_external():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    expression = x * y

    def mydata(points):
        return [x*y for x, y in points]

    # [test_interpolate_external 1]
    # First, grab the mesh.
    m = V.mesh()

    # Now make the VectorFunctionSpace corresponding to V.
    W = VectorFunctionSpace(m, V.ufl_element())

    # Next, interpolate the coordinates onto the nodes of W.
    X = assemble(interpolate(m.coordinates, W))

    # Make an output function.
    f = Function(V)

    # Use the external data function to interpolate the values of f.
    f.dat.data[:] = mydata(X.dat.data_ro)
    # [test_interpolate_external 2]
    f2 = assemble(interpolate(expression, V))
    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)


def test_line_integral():
    # [test_line_integral 1]
    # Start with a simple field exactly represented in the function space over
    # the unit square domain.
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    f = Function(V).interpolate(x * y)

    # We create a 1D mesh immersed 2D from (0, 0) to (1, 1) which we call "line".
    # Note that it only has 1 cell
    cells = np.asarray([[0, 1]])
    vertex_coords = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    plex = mesh.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
    line = mesh.Mesh(plex, dim=2)
    # [test_line_integral 2]
    x, y = SpatialCoordinate(line)
    V_line = FunctionSpace(line, "CG", 2)
    f_line = Function(V_line).interpolate(x * y)
    assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)  # for sanity
    f_line.zero()
    assert np.isclose(assemble(f_line * dx), 0)  # sanity again

    # [test_line_integral 3]
    # We want to calculate the line integral of f along it. To do this we
    # create a function space on the line mesh...
    V_line = FunctionSpace(line, "CG", 2)

    # ... and interpolate our function f onto it.
    f_line = assemble(interpolate(f, V_line))

    # The integral of f along the line is then a simple form expression which
    # we assemble:
    assemble(f_line * dx)  # this outputs sqrt(2) / 3
    # [test_line_integral 4]
    assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)


def test_cross_mesh():
    def correct_indent():
        # [test_cross_mesh 1]
        # These meshes only share some of their domain
        src_mesh = UnitSquareMesh(2, 2)
        dest_mesh = UnitSquareMesh(3, 3, quadrilateral=True)
        dest_mesh.coordinates.dat.data_wo[:] *= 2

        # We consider a simple function on our source mesh...
        x_src, y_src = SpatialCoordinate(src_mesh)
        V_src = FunctionSpace(src_mesh, "CG", 2)
        f_src = Function(V_src).interpolate(x_src**2 + y_src**2)

        # ... and want to interpolate into a function space on our target mesh ...
        V_dest = FunctionSpace(dest_mesh, "Q", 2)
        # [test_cross_mesh 2]
        return src_mesh, dest_mesh, f_src, V_dest

    src_mesh, dest_mesh, f_src, V_dest = correct_indent()

    with pytest.raises(DofNotDefinedError):
        # [test_cross_mesh 3]
        # ... but get a DofNotDefinedError if we try
        f_dest = assemble(interpolate(f_src, V_dest))  # raises DofNotDefinedError
        # [test_cross_mesh 4]

    with pytest.raises(DofNotDefinedError):
        # as will the interpolate method of a Function
        f_dest = Function(V_dest).interpolate(f_src)

    # [test_cross_mesh 5]
    # Setting the allow_missing_dofs keyword allows the interpolation to proceed.
    f_dest = assemble(interpolate(f_src, V_dest, allow_missing_dofs=True))
    # [test_cross_mesh 6]

    dest_eval05 = PointEvaluator(dest_mesh, [[0.5, 0.5]])
    assert np.isclose(dest_eval05.evaluate(f_dest), 0.5)

    # or
    # [test_cross_mesh 7]
    f_dest = Function(V_dest).interpolate(f_src, allow_missing_dofs=True)
    # [test_cross_mesh 8]

    # We get values at the points in the destination mesh as we would expect
    dest_eval05.evaluate(f_dest)  # returns 0.5**2 + 0.5**2 = 0.5

    assert np.isclose(dest_eval05.evaluate(f_dest), 0.5)

    # By default the missing points are set to 0.0
    # [test_cross_mesh 9]
    dest_eval15 = PointEvaluator(dest_mesh, [[1.5, 1.5]])
    dest_eval15.evaluate(f_dest)  # returns 0.0
    # [test_cross_mesh 10]

    assert np.isclose(dest_eval15.evaluate(f_dest), 0.0)

    # We can alternatively specify a value to use for missing points:
    # [test_cross_mesh 11]
    f_dest = assemble(interpolate(
        f_src, V_dest, allow_missing_dofs=True, default_missing_val=np.nan
    ))
    dest_eval15.evaluate(f_dest)  # returns np.nan
    # [test_cross_mesh 12]

    assert np.isclose(dest_eval05.evaluate(f_dest), 0.5)
    assert np.isnan(dest_eval15.evaluate(f_dest))

    # If we supply an output function and don't set default_missing_val
    # then any points outside the domain are left as they were.
    # [test_cross_mesh 13]
    x_dest, y_dest = SpatialCoordinate(dest_mesh)
    f_dest = Function(V_dest).interpolate(x_dest + y_dest)
    dest_eval05.evaluate(f_dest)  # returns x_dest + y_dest = 1.0
    # [test_cross_mesh 14]

    assert np.isclose(dest_eval05.evaluate(f_dest), 1.0)

    # [test_cross_mesh 15]
    assemble(interpolate(f_src, V_dest, allow_missing_dofs=True), tensor=f_dest)
    dest_eval05.evaluate(f_dest)  # now returns x_src^2 + y_src^2 = 0.5
    # [test_cross_mesh 16]

    assert np.isclose(dest_eval05.evaluate(f_dest), 0.5)

    dest_eval15.evaluate(f_dest)  # still returns x_dest + y_dest = 3.0

    f_dest.zero()
    f_dest.interpolate(x_dest + y_dest)
    assert np.isclose(dest_eval05.evaluate(f_dest), 1.0)  # x_dest + y_dest = 1.0

    # Similarly, using the interpolate method on a function will not overwrite
    # the pre-existing values if default_missing_val is not set
    # [test_cross_mesh 17]
    f_dest.interpolate(f_src, allow_missing_dofs=True)
    # [test_cross_mesh 18]

    assert np.isclose(dest_eval05.evaluate(f_dest), 0.5)  # x_src^2 + y_src^2 = 0.5
    assert np.isclose(dest_eval15.evaluate(f_dest), 3.0)  # x_dest + y_dest = 3.0
