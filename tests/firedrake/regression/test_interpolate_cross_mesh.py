from firedrake import *
from firedrake.__future__ import *
from firedrake.petsc import DEFAULT_PARTITIONER
from firedrake.ufl_expr import extract_unique_domain
import numpy as np
import pytest
from functools import reduce
from operator import mul, add
import subprocess


def allgather(comm, coords):
    """Gather all coordinates from all ranks."""
    coords = coords.copy()
    coords = comm.allgather(coords)
    coords = np.concatenate(coords)
    return coords


def unitsquaresetup():
    m_src = UnitSquareMesh(2, 3)
    m_dest = UnitSquareMesh(3, 5, quadrilateral=True)
    coords = np.array(
        [[0.56, 0.6], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.726, 0.6584]]
    )  # fairly arbitrary
    # add the coordinates of the mesh vertices to test boundaries
    vertices_src = allgather(m_src.comm, m_src.coordinates.dat.data_ro)
    coords = np.concatenate((coords, vertices_src))
    vertices_dest = allgather(m_dest.comm, m_dest.coordinates.dat.data_ro)
    coords = np.concatenate((coords, vertices_dest))
    return m_src, m_dest, coords


def make_high_order(m_low_order, degree):
    """Make a higher order mesh from a lower order mesh."""
    f = Function(VectorFunctionSpace(m_low_order, "CG", degree))
    f.interpolate(SpatialCoordinate(m_low_order))
    return Mesh(f)


@pytest.fixture(
    params=[
        "unitsquare",
        "circlemanifold",
        "circlemanifold_to_high_order",
        pytest.param(
            "unitsquare_from_high_order",
            marks=pytest.mark.xfail(
                # CalledProcessError is so the parallel tests correctly xfail
                raises=(subprocess.CalledProcessError, NotImplementedError),
                reason="Cannot yet interpolate from high order meshes to other meshes.",
            ),
        ),
        "unitsquare_to_high_order",
        "extrudedcube",
        "unitsquare_vfs",
        "unitsquare_tfs",
        "unitsquare_N1curl_source",
        pytest.param(
            "unitsquare_SminusDiv_destination",
            marks=pytest.mark.xfail(
                # CalledProcessError is so the parallel tests correctly xfail
                raises=(subprocess.CalledProcessError, NotImplementedError),
                reason="Can only interpolate into spaces with point evaluation nodes",
            ),
        ),
        "unitsquare_Regge_source",
        # This test fails in complex mode
        pytest.param("spheresphere", marks=pytest.mark.skipcomplex),
        "sphereextrudedsphereextruded",
    ]
)
def parameters(request):
    if request.param == "unitsquare":
        m_src, m_dest, coords = unitsquaresetup()
        expr_src = reduce(mul, SpatialCoordinate(m_src))
        expr_dest = reduce(mul, SpatialCoordinate(m_dest))
        expected = reduce(mul, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "circlemanifold":
        m_src = UnitSquareMesh(2, 3)
        # shift to cover the whole circle
        m_src.coordinates.dat.data[:] *= 2
        m_src.coordinates.dat.data[:] -= 1
        m_dest = CircleManifoldMesh(1000)  # want close to actual unit circle
        coords = np.array(
            [
                [0, 1],
                [1, 0],
                [-1, 0],
                [0, -1],
                [sqrt(2) / 2, sqrt(2) / 2],
                [-sqrt(3) / 2, 1 / 2],
            ]
        )  # points that ought to be on the unit circle
        # only add target mesh vertices since they are common to both meshes
        vertices_dest = allgather(m_dest.comm, m_dest.coordinates.dat.data_ro)
        coords = np.concatenate((coords, vertices_dest))
        expr_src = reduce(mul, SpatialCoordinate(m_src))
        expr_dest = reduce(mul, SpatialCoordinate(m_dest))
        expected = reduce(mul, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "circlemanifold_to_high_order":
        if COMM_WORLD.size > 1 and DEFAULT_PARTITIONER == "simple":
            # TODO: This failure should be investigated
            pytest.skip(reason="This test fails in parallel when using the simple partitioner")
        m_src = UnitSquareMesh(2, 3)
        # shift to cover the whole circle
        m_src.coordinates.dat.data[:] *= 2
        m_src.coordinates.dat.data[:] -= 1
        m_dest = CircleManifoldMesh(1000, degree=2)  # note degree!
        # Function.at often gets conflicting answers across boundaries for this
        # mesh, so we lower the tolerance a bit for this test
        m_dest.tolerance = 0.1
        coords = np.array(
            [
                [0, 1],
                [1, 0],
                [-1, 0],
                [0, -1],
                [sqrt(2) / 2, sqrt(2) / 2],
                [-sqrt(3) / 2, 1 / 2],
            ]
        )  # points that ought to be on the unit circle
        # only add target mesh vertices since they are common to both meshes
        vertices_dest = allgather(m_dest.comm, m_dest.coordinates.dat.data_ro)
        coords = np.concatenate((coords, vertices_dest))
        expr_src = reduce(mul, SpatialCoordinate(m_src))
        expr_dest = reduce(mul, SpatialCoordinate(m_dest))
        expected = reduce(mul, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "unitsquare_from_high_order":
        m_low_order, m_dest, coords = unitsquaresetup()
        m_src = make_high_order(m_low_order, 2)
        expr_src = reduce(mul, SpatialCoordinate(m_src))
        expr_dest = reduce(mul, SpatialCoordinate(m_dest))
        expected = reduce(mul, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "unitsquare_to_high_order":
        m_src, m_low_order, coords = unitsquaresetup()
        m_dest = make_high_order(m_low_order, 2)
        expr_src = reduce(mul, SpatialCoordinate(m_src))
        expr_dest = reduce(mul, SpatialCoordinate(m_dest))
        expected = reduce(mul, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "extrudedcube":
        m_src = ExtrudedMesh(UnitSquareMesh(2, 3), 2)
        m_dest = UnitCubeMesh(3, 5, 7)
        coords = np.array(
            [
                [0.56, 0.06, 0.6],
                [0.726, 0.6584, 0.951],
            ]
        )  # fairly arbitrary
        vertices_src = allgather(m_src.comm, m_src.coordinates.dat.data_ro)
        coords = np.concatenate((coords, vertices_src))
        vertices_dest = allgather(m_dest.comm, m_dest.coordinates.dat.data_ro)
        coords = np.concatenate((coords, vertices_dest))
        # we use add to avoid TSFC complaints about too many indices for sum
        # factorisation when interpolating expressions of SpatialCoordinates(m_src)
        # into V_dest
        expr_src = reduce(add, SpatialCoordinate(m_src))
        expr_dest = reduce(add, SpatialCoordinate(m_dest))
        expected = reduce(add, coords.T)
        V_src = FunctionSpace(m_src, "CG", 2)
        V_dest = FunctionSpace(m_dest, "CG", 3)
        V_dest_2 = FunctionSpace(m_dest, "CG", 1)
    elif request.param == "unitsquare_vfs":
        m_src, m_dest, coords = unitsquaresetup()
        expr_src = 2 * SpatialCoordinate(m_src)
        expr_dest = 2 * SpatialCoordinate(m_dest)
        expected = 2 * coords
        V_src = VectorFunctionSpace(m_src, "CG", 3)
        V_dest = VectorFunctionSpace(m_dest, "CG", 4)
        V_dest_2 = VectorFunctionSpace(m_dest, "DQ", 2)
    elif request.param == "unitsquare_tfs":
        m_src, m_dest, coords = unitsquaresetup()
        expr_src = outer(SpatialCoordinate(m_src), SpatialCoordinate(m_src))
        expr_dest = outer(SpatialCoordinate(m_dest), SpatialCoordinate(m_dest))
        expected = np.asarray(
            [np.outer(coords[i], coords[i]) for i in range(len(coords))]
        )
        V_src = TensorFunctionSpace(m_src, "CG", 3)
        V_dest = TensorFunctionSpace(m_dest, "CG", 4)
        V_dest_2 = TensorFunctionSpace(m_dest, "DQ", 2)
    elif request.param == "unitsquare_N1curl_source":
        m_src, m_dest, coords = unitsquaresetup()
        expr_src = 2 * SpatialCoordinate(m_src)
        expr_dest = 2 * SpatialCoordinate(m_dest)
        expected = 2 * coords
        V_src = FunctionSpace(m_src, "N1curl", 2)  # Not point evaluation nodes
        V_dest = VectorFunctionSpace(m_dest, "CG", 4)
        V_dest_2 = VectorFunctionSpace(m_dest, "DQ", 2)
    elif request.param == "unitsquare_SminusDiv_destination":
        m_src, m_dest, coords = unitsquaresetup()
        expr_src = 2 * SpatialCoordinate(m_src)
        expr_dest = 2 * SpatialCoordinate(m_dest)
        expected = 2 * coords
        V_src = VectorFunctionSpace(m_src, "CG", 2)
        V_dest = FunctionSpace(m_dest, "SminusDiv", 2)  # Not point evaluation nodes
        V_dest_2 = VectorFunctionSpace(m_dest, "DQ", 2)
    elif request.param == "unitsquare_Regge_source":
        m_src, m_dest, coords = unitsquaresetup()
        expr_src = outer(SpatialCoordinate(m_src), SpatialCoordinate(m_src))
        expr_dest = outer(SpatialCoordinate(m_dest), SpatialCoordinate(m_dest))
        expected = np.asarray(
            [np.outer(coords[i], coords[i]) for i in range(len(coords))]
        )
        V_src = FunctionSpace(m_src, "Regge", 2, variant="point")  # Not point evaluation nodes
        V_dest = TensorFunctionSpace(m_dest, "CG", 4)
        V_dest_2 = TensorFunctionSpace(m_dest, "DQ", 2)
    elif request.param == "spheresphere":
        m_src = UnitCubedSphereMesh(5, name="src_sphere")
        m_dest = UnitIcosahedralSphereMesh(5, name="dest_sphere")
        coords = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, -1, 0],
                [sqrt(2) / 2, sqrt(2) / 2, 0],
                [-sqrt(3) / 2, 1 / 2, 0],
            ]
        )  # points that ought to be on the unit circle, at z=0
        # We don't add source and target mesh vertices since no amount of mesh
        # tolerance loading allows .at to avoid getting different results on different
        # processes for this mesh pair.
        # Function.at often gets conflicting answers across boundaries for this
        # mesh, so we lower the tolerance a bit for this test
        m_dest.tolerance = 0.1
        # We use add to avoid TSFC complaints about too many indices for sum
        # factorisation when interpolating expressions of SpatialCoordinates(m_src)
        # into V_dest
        expr_src = reduce(add, SpatialCoordinate(m_src))
        expr_dest = reduce(add, SpatialCoordinate(m_dest))
        expected = reduce(add, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "sphereextrudedsphereextruded":
        if COMM_WORLD.size > 1 and DEFAULT_PARTITIONER == "simple":
            # TODO: This failure should be investigated
            pytest.skip(reason="This test hangs in parallel when using the simple partitioner")
        m_src = ExtrudedMesh(UnitIcosahedralSphereMesh(1), 2, extrusion_type="radial")
        # Note we need to use the same base sphere otherwise it's hard to check
        # anything really
        m_dest = ExtrudedMesh(UnitIcosahedralSphereMesh(1), 3, extrusion_type="radial")
        coords = np.array(
            [
                [0, 1.5, 0],
                [1.5, 0, 0],
                [-1.5, 0, 0],
                [0, -1.5, 0],
                [sqrt(2) / 2 + sqrt(2) / 4, sqrt(2) / 2 + sqrt(2) / 4, 0],
                [-sqrt(3) / 2 - sqrt(3) / 4, 1.0, 0],
            ]
        )  # points that ought to be on in the meshes, at z=0
        # We don't add source and target mesh vertices since no amount of mesh
        # tolerance loading allows .at to avoid getting different results on different
        # processes for this mesh pair.
        # Function.at often gets conflicting answers across boundaries for this
        # mesh, so we lower the tolerance a bit for this test
        m_dest.tolerance = 0.1
        # We use add to avoid TSFC complaints about too many indices for sum
        # factorisation when interpolating expressions of SpatialCoordinates(m_src)
        # into V_dest
        expr_src = reduce(add, SpatialCoordinate(m_src))
        expr_dest = reduce(add, SpatialCoordinate(m_dest))
        expected = reduce(add, coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    else:
        raise ValueError("Unknown mesh pair")
    return m_src, m_dest, coords, expr_src, expr_dest, expected, V_src, V_dest, V_dest_2


def test_interpolate_cross_mesh(parameters):
    (
        m_src,
        m_dest,
        coords,
        expr_src,
        expr_dest,
        expected,
        V_src,
        V_dest,
        V_dest_2,
    ) = parameters
    get_expected_values(
        m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest
    )
    get_expected_values(
        m_src, m_dest, V_src, V_dest_2, coords, expected, expr_src, expr_dest
    )


def test_interpolate_unitsquare_mixed():
    # this has to be in its own test because UFL expressions on mixed function
    # spaces are not supported.

    m_src = UnitSquareMesh(2, 3)
    m_dest = UnitSquareMesh(3, 5, quadrilateral=True)

    coords = np.array(
        [[0.56, 0.6], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.726, 0.6584]]
    )  # fairly arbitrary
    # add the coordinates of the mesh vertices to test boundaries
    vertices_src = allgather(m_src.comm, m_src.coordinates.dat.data_ro)
    coords = np.concatenate((coords, vertices_src))
    vertices_dest = allgather(m_dest.comm, m_dest.coordinates.dat.data_ro)
    coords = np.concatenate((coords, vertices_dest))

    expr_1 = reduce(add, SpatialCoordinate(m_src))
    expected_1 = reduce(add, coords.T)
    expr_2 = reduce(mul, SpatialCoordinate(m_src))
    expected_2 = reduce(mul, coords.T)

    V_1 = FunctionSpace(m_src, "CG", 1)
    V_2 = FunctionSpace(m_src, "CG", 2)
    V_src = V_1 * V_2
    V_3 = FunctionSpace(m_dest, "CG", 3)
    V_4 = FunctionSpace(m_dest, "CG", 4)
    V_dest = V_3 * V_4
    f_src = Function(V_src)
    f_src.subfunctions[0].interpolate(expr_1)
    f_src.subfunctions[1].interpolate(expr_2)

    f_dest = assemble(interpolate(f_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = np.asarray(f_dest.at(coords))
    assert np.allclose(got[:, 0], expected_1)
    assert np.allclose(got[:, 1], expected_2)

    # can create interpolator and use it
    interpolator = Interpolator(TestFunction(V_src), V_dest)
    f_dest = assemble(interpolator.interpolate(f_src))
    assert extract_unique_domain(f_dest) is m_dest
    got = np.asarray(f_dest.at(coords))
    assert np.allclose(got[:, 0], expected_1)
    assert np.allclose(got[:, 1], expected_2)
    f_dest = Function(V_dest)
    assemble(interpolator.interpolate(f_src), tensor=f_dest)
    assert extract_unique_domain(f_dest) is m_dest
    got = np.asarray(f_dest.at(coords))
    assert np.allclose(got[:, 0], expected_1)
    assert np.allclose(got[:, 1], expected_2)
    cofunc_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
    cofunc_src = assemble(interpolator.interpolate(cofunc_dest, adjoint=True))
    assert not np.allclose(f_src.dat.data_ro[0], cofunc_src.dat.data_ro[0])
    assert not np.allclose(f_src.dat.data_ro[1], cofunc_src.dat.data_ro[1])

    # Can't go from non-mixed to mixed
    V_src_2 = VectorFunctionSpace(m_src, "CG", 1)
    assert V_src_2.value_shape == V_src.value_shape
    f_src_2 = Function(V_src_2)
    with pytest.raises(NotImplementedError):
        assemble(interpolate(f_src_2, V_dest))


def test_exact_refinement():
    # With an exact mesh refinement, we can do error checks to see if our
    # forward and adjoint interpolations are exact where we expect them to
    # be.
    m_coarse = UnitSquareMesh(2, 2)
    m_fine = UnitSquareMesh(8, 8)
    V_coarse = FunctionSpace(m_coarse, "CG", 2)
    V_fine = FunctionSpace(m_fine, "CG", 2)
    # V_fine entirely spans V_coarse, i.e. V_scr is a subset of V_fine so we
    # expect no interpolation error.

    # expr_in_V_coarse this is exactly representable in V_coarse
    x, y = SpatialCoordinate(m_coarse)
    expr_in_V_coarse = x**2 + y**2 + 1
    f_coarse = Function(V_coarse).interpolate(expr_in_V_coarse)

    # expr_in_V_fine is also exactly representable in V_fine and is
    # symbolically the same as expr_in_V_coarse
    x, y = SpatialCoordinate(m_fine)
    expr_in_V_fine = x**2 + y**2 + 1
    f_fine = Function(V_fine).interpolate(expr_in_V_fine)

    # If we now interpolate f_coarse into V_fine we should get a function
    # which has no interpolation error versus f_fine because we were able to
    # exactly represent expr_in_V_coarse in V_coarse and V_coarse is a subset
    # of V_fine
    interpolator_coarse_to_fine = Interpolator(TestFunction(V_coarse), V_fine)
    f_coarse_on_fine = assemble(interpolator_coarse_to_fine.interpolate(f_coarse))
    assert np.allclose(f_coarse_on_fine.dat.data_ro, f_fine.dat.data_ro)

    # Adjoint interpolation takes us from V_fine^* to V_coarse^* so we should
    # also get an exact result here.
    cofunction_coarse = assemble(inner(f_coarse, TestFunction(V_coarse)) * dx)
    cofunction_fine = assemble(inner(f_fine, TestFunction(V_fine)) * dx)
    cofunction_fine_on_coarse = assemble(interpolator_coarse_to_fine.interpolate(
        cofunction_fine, adjoint=True
    ))
    assert np.allclose(
        cofunction_fine_on_coarse.dat.data_ro, cofunction_coarse.dat.data_ro
    )

    # Now we test with expressions which are NOT exactly representable in the
    # function spaces by introducing a cube term. This can't be represented
    # with a 2nd degree polynomial basis
    x, y = SpatialCoordinate(m_fine)
    expr_fine = x**3 + x**2 + y**2 + 1
    f_fine = Function(V_fine).interpolate(expr_fine)
    x, y = SpatialCoordinate(m_coarse)
    expr_coarse = x**3 + x**2 + y**2 + 1
    f_coarse = Function(V_coarse).interpolate(expr_coarse)

    # We still expect interpolation from V_coarse to V_fine to produce a result
    # which is consistent with building a function in V_coarse by directly
    # interpolating expr_coarse since V_coarse is a subset of V_fine.
    interpolator_fine_to_coarse = Interpolator(TestFunction(V_fine), V_coarse)
    f_fine_on_coarse = assemble(interpolator_fine_to_coarse.interpolate(f_fine))
    assert np.allclose(f_fine_on_coarse.dat.data_ro, f_coarse.dat.data_ro)

    # But adjoint interpolation, which takes us from V_coarse^* to V_fine^*
    # won't exactly reproduce the cofunction of f_coarse, since V_fine^* is a
    # bigger space than V_coarse^*. This only worked before because we could
    # exactly represent the expression in V_coarse.
    cofunction_fine = assemble(inner(f_fine, TestFunction(V_fine)) * dx)
    cofunction_coarse = assemble(inner(f_coarse, TestFunction(V_coarse)) * dx)
    cofunction_coarse_on_fine = assemble(interpolator_fine_to_coarse.interpolate(
        cofunction_coarse, adjoint=True
    ))
    assert not np.allclose(
        cofunction_coarse_on_fine.dat.data_ro, cofunction_fine.dat.data_ro
    )

    # We get a similar result going in the other direction. Forward
    # interpolation from V_coarse to V_fine similarly doesn't reproduce the
    # effect of interpolating expr_fine directly into V_fine
    interpolator_coarse_to_fine = Interpolator(TestFunction(V_coarse), V_fine)
    f_course_on_fine = assemble(interpolator_coarse_to_fine.interpolate(f_coarse))
    assert not np.allclose(f_course_on_fine.dat.data_ro, f_fine.dat.data_ro)

    # But the adjoint operation, which takes us from V_fine^* to
    # V_coarse^* correctly reproduces cofunction_coarse from cofunction_fine
    cofunction_fine_on_coarse = assemble(interpolator_coarse_to_fine.interpolate(
        cofunction_fine, adjoint=True
    ))
    assert not np.allclose(
        cofunction_fine_on_coarse.dat.data_ro, cofunction_coarse.dat.data_ro
    )


def test_interpolate_unitsquare_tfs_shape():
    m_src = UnitSquareMesh(2, 3)
    m_dest = UnitSquareMesh(3, 5, quadrilateral=True)
    V_src = TensorFunctionSpace(m_src, "CG", 3, shape=(1, 2, 3))
    V_dest = TensorFunctionSpace(m_dest, "CG", 4, shape=(1, 2, 3))
    f_src = Function(V_src)
    assemble(interpolate(f_src, V_dest))


def test_interpolate_cross_mesh_not_point_eval():
    m_src = UnitSquareMesh(2, 3)
    m_dest = UnitSquareMesh(3, 5, quadrilateral=True)
    coords = np.array(
        [[0.56, 0.6], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.726, 0.6584]]
    )  # fairly arbitrary
    # add the coordinates of the mesh vertices to test boundaries
    vertices_src = allgather(m_src.comm, m_src.coordinates.dat.data_ro)
    coords = np.concatenate((coords, vertices_src))
    vertices_dest = allgather(m_dest.comm, m_dest.coordinates.dat.data_ro)
    coords = np.concatenate((coords, vertices_dest))
    expr_src = 2 * SpatialCoordinate(m_src)
    expr_dest = 2 * SpatialCoordinate(m_dest)
    expected = 2 * coords
    V_src = FunctionSpace(m_src, "RT", 2)
    V_dest = FunctionSpace(m_dest, "RTCE", 2)
    atol = 1e-8  # default
    # This might not make much mathematical sense, but it should test if we get
    # the not implemented error for non-point evaluation nodes!
    with pytest.raises(NotImplementedError):
        interpolate_function(
            m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest, atol
        )


def get_expected_values(
    m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest
):
    if m_src.name == "src_sphere" and m_dest.name == "dest_sphere":
        # Between immersed manifolds we will often be doing projection so we
        # need a higher tolerance for our tests
        atol = 1e-3
    else:
        atol = 1e-8  # default
    interpolate_function(
        m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest, atol
    )
    interpolate_expression(
        m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest, atol
    )
    interpolator, f_src, f_dest, m_src = interpolator_function(
        m_src,
        m_dest,
        V_src,
        V_dest,
        coords,
        expected,
        expr_src,
        expr_dest,
        atol,
    )
    cofunction_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
    interpolator_function_adjoint(
        interpolator, f_src, cofunction_dest, m_src, m_dest, coords, expected, atol
    )
    interpolator_expression(
        m_src, m_dest, V_src, V_dest, coords, expected, expr_src, atol
    )


def interpolate_function(
    m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest, atol
):
    f_src = Function(V_src).interpolate(expr_src)
    f_dest = assemble(interpolate(f_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)

    f_src = Function(V_src).interpolate(expr_src)
    f_dest = assemble(interpolate(f_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)

    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # works with Function interpolate method
    f_dest = Function(V_dest)
    f_dest.interpolate(f_src)
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # output argument works
    f_dest = Function(V_dest)
    assemble(Interpolate(f_src, V_dest), tensor=f_dest)
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)


def interpolate_expression(
    m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest, atol
):
    f_dest = assemble(interpolate(expr_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)
    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # output argument works for expressions
    f_dest = Function(V_dest)
    assemble(Interpolate(expr_src, V_dest), tensor=f_dest)
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)


def interpolator_function(
    m_src, m_dest, V_src, V_dest, coords, expected, expr_src, expr_dest, atol
):
    f_src = Function(V_src).interpolate(expr_src)

    interpolator = Interpolator(TestFunction(V_src), V_dest)
    assert isinstance(interpolator, Interpolator)
    assert isinstance(interpolator, interpolation.CrossMeshInterpolator)
    f_dest = assemble(interpolator.interpolate(f_src))
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, expected, atol=atol)
    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    with pytest.raises(TypeError):
        # can't interpolate expressions using an interpolator
        assemble(interpolator.interpolate(2 * f_src))
    cofunction_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
    assemble(interpolator.interpolate(2 * cofunction_dest, adjoint=True))

    return interpolator, f_src, f_dest, m_src


def interpolator_function_adjoint(
    interpolator, f_src, cofunction_dest, m_src, m_dest, coords, expected, atol
):
    f_dest = assemble(interpolator.interpolate(f_src))
    cofunction_dest_on_src = assemble(interpolator.interpolate(cofunction_dest, adjoint=True))
    assert cofunction_dest_on_src.function_space().mesh() is m_src
    assert np.isclose(
        assemble(action(cofunction_dest_on_src, f_src)),
        assemble(action(cofunction_dest, f_dest)), atol=atol
    )
    if m_src.name == "src_sphere" and m_dest.name == "dest_sphere" and atol > 1e-8:
        # In the spheresphere case we actually DO get the same cofunction back with an
        # atol of 1e-3
        V_src = f_src.function_space()
        cofunction_src = assemble(inner(f_src, TestFunction(V_src)) * dx)
        assert np.allclose(
            cofunction_dest_on_src.dat.data_ro, cofunction_src.dat.data_ro, atol=atol
        )


def interpolator_expression(
    m_src, m_dest, V_src, V_dest, coords, expected, expr_src, atol
):
    f_src = Function(V_src).interpolate(expr_src)

    interpolator = Interpolator(2 * TestFunction(V_src), V_dest)
    f_dest = assemble(interpolator.interpolate(f_src))
    assert extract_unique_domain(f_dest) is m_dest
    got = f_dest.at(coords)
    assert np.allclose(got, 2 * expected, atol=2 * atol)
    cofunction_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
    cofunction_dest_on_src = assemble(interpolator.interpolate(
        cofunction_dest, adjoint=True
    ))
    assert cofunction_dest_on_src.function_space().mesh() is m_src
    assert np.isclose(
        assemble(action(cofunction_dest_on_src, f_src)),
        assemble(action(cofunction_dest, f_dest)), atol=atol
    )


def test_missing_dofs():
    m_src = UnitSquareMesh(2, 3)
    m_dest = UnitSquareMesh(4, 5)
    m_dest.coordinates.dat.data[:] *= 2
    coords = np.array([[0.5, 0.5], [1.5, 1.5]])
    x, y = SpatialCoordinate(m_src)
    expr = x * y
    V_src = FunctionSpace(m_src, "CG", 2)
    V_dest = FunctionSpace(m_dest, "CG", 3)
    with pytest.raises(DofNotDefinedError):
        Interpolator(TestFunction(V_src), V_dest)
    interpolator = Interpolator(TestFunction(V_src), V_dest, allow_missing_dofs=True)
    f_src = Function(V_src).interpolate(expr)
    f_dest = assemble(interpolator.interpolate(f_src))
    # default value is zero
    assert np.allclose(f_dest.at(coords), np.array([0.25, 0.0]))
    f_dest = Function(V_dest).assign(Constant(1.0))
    # make sure we have actually changed f_dest before checking interpolation
    assert np.allclose(f_dest.at(coords), np.array([1.0, 1.0]))
    assemble(interpolator.interpolate(f_src), tensor=f_dest)
    # assigned value hasn't been changed
    assert np.allclose(f_dest.at(coords), np.array([0.25, 1.0]))
    f_dest = assemble(interpolator.interpolate(f_src, default_missing_val=2.0))
    # should take on the default value
    assert np.allclose(f_dest.at(coords), np.array([0.25, 2.0]))
    f_dest = Function(V_dest).assign(Constant(1.0))
    # make sure we have actually changed f_dest before checking interpolation
    assert np.allclose(f_dest.at(coords), np.array([1.0, 1.0]))
    assemble(interpolator.interpolate(f_src, default_missing_val=2.0), tensor=f_dest)
    assert np.allclose(f_dest.at(coords), np.array([0.25, 2.0]))
    f_dest = Function(V_dest).assign(Constant(1.0))
    # make sure we have actually changed f_dest before checking interpolation
    assert np.allclose(f_dest.at(coords), np.array([1.0, 1.0]))
    assemble(interpolator.interpolate(f_src, default_missing_val=0.0), tensor=f_dest)
    assert np.allclose(f_dest.at(coords), np.array([0.25, 0.0]))

    # Try the other way around so we can check adjoint is unaffected
    m_src = UnitSquareMesh(4, 5)
    m_src.coordinates.dat.data[:] *= 2
    m_dest = UnitSquareMesh(2, 3)
    coords = np.array([[0.5, 0.5], [1.5, 1.5]])
    x, y = SpatialCoordinate(m_src)
    expr = x * y
    V_src = FunctionSpace(m_src, "CG", 3)
    V_dest = FunctionSpace(m_dest, "CG", 2)
    interpolator = Interpolator(TestFunction(V_src), V_dest, allow_missing_dofs=True)
    cofunction_src = assemble(inner(Function(V_dest), TestFunction(V_dest)) * dx)
    cofunction_src.dat.data_wo[:] = 1.0
    cofunction_dest = assemble(interpolator.interpolate(
        cofunction_src, adjoint=True, default_missing_val=2.0
    ))
    assert np.all(cofunction_dest.dat.data_ro != 2.0)


def test_line_integral():
    # start with a simple field exactly represented in the function space
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, "CG", 2)
    x, y = SpatialCoordinate(m)
    f = Function(V).interpolate(x * y)

    # Create a 1D line mesh in 2D from (0, 0) to (1, 1) with 1 cell
    cells = np.asarray([[0, 1]])
    vertex_coords = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    plex = mesh.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
    line = mesh.Mesh(plex, dim=2)
    x, y = SpatialCoordinate(line)
    V_line = FunctionSpace(line, "CG", 2)
    f_line = Function(V_line).interpolate(x * y)
    assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)  # for sanity

    # Create a 1D line around the unit square (2D) with 4 cells
    cells = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]])
    vertex_coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    plex = mesh.plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
    line_square = mesh.Mesh(plex, dim=2)
    x, y = SpatialCoordinate(line_square)
    V_line_square = FunctionSpace(line_square, "CG", 2)
    f_line_square = Function(V_line_square).interpolate(x * y)
    assert np.isclose(assemble(f_line_square * dx), 1.0)  # for sanity

    # See if interpolating onto the lines from a bigger domain gives us the
    # same answer
    f_line.zero()
    assert np.isclose(assemble(f_line * dx), 0)  # sanity again
    f_line.interpolate(f)
    assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)
    f_line_square.zero()
    assert np.isclose(assemble(f_line_square * dx), 0)
    f_line_square.interpolate(f)
    assert np.isclose(assemble(f_line_square * dx), 1.0)


@pytest.mark.parallel
def test_interpolate_cross_mesh_parallel(parameters):
    test_interpolate_cross_mesh(parameters)


@pytest.mark.parallel
def test_interpolate_unitsquare_mixed_parallel():
    test_interpolate_unitsquare_mixed()


@pytest.mark.parallel
def test_missing_dofs_parallel():
    test_missing_dofs()


@pytest.mark.parallel
def test_exact_refinement_parallel():
    test_exact_refinement()


def voting_algorithm_edgecases(nprocs):
    # this triggers lots of cases where the VOM voting algorithm has to deal
    # with points being claimed by multiple ranks: there are cases where each
    # rank will claim another one owns a point, for example, and yet also all
    # claim zero distance to the reference cell!
    s = nprocs
    nx = 2 * s
    mx = 3 * nx
    mh = [UnitCubeMesh(nx, nx, nx),
          UnitCubeMesh(mx, mx, mx)]
    family = "Lagrange"
    degree = 1
    Vc = FunctionSpace(mh[0], family, degree=degree)
    Vf = FunctionSpace(mh[1], family, degree=degree)
    uc = Function(Vc).interpolate(SpatialCoordinate(mh[0])[0])
    uf = Function(Vf).interpolate(uc)
    uf2 = Function(Vf).interpolate(SpatialCoordinate(mh[1])[0])
    assert np.isclose(errornorm(uf, uf2), 0.0)


@pytest.mark.parallel(nprocs=2)
def test_voting_algorithm_edgecases_2_ranks():
    voting_algorithm_edgecases(2)


@pytest.mark.parallel(nprocs=3)
def test_voting_algorithm_edgecases_3_ranks():
    voting_algorithm_edgecases(3)


@pytest.mark.parallel(nprocs=4)
def test_voting_algorithm_edgecases_4_ranks():
    voting_algorithm_edgecases(4)


@pytest.mark.parallel
@pytest.mark.parametrize('periodic', [False, True])
def test_interpolate_cross_mesh_interval(periodic):
    m_src = PeriodicUnitIntervalMesh(3) if periodic else UnitIntervalMesh(3)
    V_src = FunctionSpace(m_src, "CG", 2)
    x_src, = SpatialCoordinate(m_src)
    f_src = Function(V_src).interpolate(-(x_src - .5) ** 2)
    m_dest = PeriodicUnitIntervalMesh(4) if periodic else UnitIntervalMesh(4)
    V_dest = FunctionSpace(m_dest, "CG", 3)
    f_dest = Function(V_dest).interpolate(f_src)
    x_dest, = SpatialCoordinate(m_dest)
    assert abs(assemble((f_dest - (-(x_dest - .5) ** 2)) ** 2 * dx)) < 1.e-16
