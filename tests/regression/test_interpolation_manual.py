from firedrake import *
from firedrake.__future__ import *
import pytest
import numpy as np


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

    assert np.isclose(f_dest.at(0.5, 0.5), 0.5)

    # or
    # [test_cross_mesh 7]
    f_dest = Function(V_dest).interpolate(f_src, allow_missing_dofs=True)
    # [test_cross_mesh 8]

    # We get values at the points in the destination mesh as we would expect
    f_dest.at(0.5, 0.5)  # returns 0.5**2 + 0.5**2 = 0.5

    assert np.isclose(f_dest.at(0.5, 0.5), 0.5)

    # By default the missing points are set to 0.0
    # [test_cross_mesh 9]
    f_dest.at(1.5, 1.5)  # returns 0.0
    # [test_cross_mesh 10]

    assert np.isclose(f_dest.at(1.5, 1.5), 0.0)

    # We can alternatively specify a value to use for missing points:
    # [test_cross_mesh 11]
    f_dest = assemble(interpolate(
        f_src, V_dest, allow_missing_dofs=True, default_missing_val=np.nan
    ))
    f_dest.at(1.5, 1.5)  # returns np.nan
    # [test_cross_mesh 12]

    assert np.isclose(f_dest.at(0.5, 0.5), 0.5)
    assert np.isnan(f_dest.at(1.5, 1.5))

    # When creating interpolators, the allow_missing_dofs keyword argument is
    # set when creating the interpolator, rather than when calling interpoalate.
    # [test_cross_mesh 13]
    interpolator = Interpolator(f_src, V_dest, allow_missing_dofs=True)
    # [test_cross_mesh 14]

    # A default missing value can be specified when calling interpolate.
    # [test_cross_mesh 15]
    f_dest = assemble(interpolator.interpolate(default_missing_val=np.nan))
    # [test_cross_mesh 16]

    # If we supply an output function and don't set default_missing_val
    # then any points outside the domain are left as they were.
    # [test_cross_mesh 17]
    x_dest, y_dest = SpatialCoordinate(dest_mesh)
    f_dest = Function(V_dest).interpolate(x_dest + y_dest)
    f_dest.at(0.5, 0.5)  # returns x_dest + y_dest = 1.0
    # [test_cross_mesh 18]

    assert np.isclose(f_dest.at(0.5, 0.5), 1.0)

    # [test_cross_mesh 19]
    assemble(interpolator.interpolate(), tensor=f_dest)
    f_dest.at(0.5, 0.5)  # now returns x_src^2 + y_src^2 = 0.5
    # [test_cross_mesh 20]

    assert np.isclose(f_dest.at(0.5, 0.5), 0.5)

    f_dest.at(1.5, 1.5)  # still returns x_dest + y_dest = 3.0

    f_dest.zero()
    f_dest.interpolate(x_dest + y_dest)
    assert np.isclose(f_dest.at(0.5, 0.5), 1.0)  # x_dest + y_dest = 1.0

    # [test_cross_mesh 21]
    # Similarly, using the interpolate method on a function will not overwrite
    # the pre-existing values if default_missing_val is not set
    f_dest.interpolate(f_src, allow_missing_dofs=True)
    # [test_cross_mesh 22]

    assert np.isclose(f_dest.at(0.5, 0.5), 0.5)  # x_src^2 + y_src^2 = 0.5
    assert np.isclose(f_dest.at(1.5, 1.5), 3.0)  # x_dest + y_dest = 3.0
