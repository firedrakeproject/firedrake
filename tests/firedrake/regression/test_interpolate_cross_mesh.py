from firedrake import *
from firedrake.petsc import DEFAULT_PARTITIONER
from firedrake.ufl_expr import extract_unique_domain
from firedrake.mesh import Mesh, plex_from_cell_list
from firedrake.formmanipulation import split_form
import numpy as np
import pytest
from ufl import product
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
        "unitsquare_from_high_order",
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
        expr_src = product(SpatialCoordinate(m_src))
        expr_dest = product(SpatialCoordinate(m_dest))
        expected = np.prod(coords, axis=-1)
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
        expr_src = product(SpatialCoordinate(m_src))
        expr_dest = product(SpatialCoordinate(m_dest))
        expected = np.prod(coords, axis=-1)
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
        expr_src = product(SpatialCoordinate(m_src))
        expr_dest = product(SpatialCoordinate(m_dest))
        expected = np.prod(coords, axis=-1)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "unitsquare_from_high_order":
        m_low_order, m_dest, coords = unitsquaresetup()
        m_src = make_high_order(m_low_order, 2)
        expr_src = product(SpatialCoordinate(m_src))
        expr_dest = product(SpatialCoordinate(m_dest))
        expected = np.prod(coords, axis=-1)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    elif request.param == "unitsquare_to_high_order":
        m_src, m_low_order, coords = unitsquaresetup()
        m_dest = make_high_order(m_low_order, 2)
        expr_src = product(SpatialCoordinate(m_src))
        expr_dest = product(SpatialCoordinate(m_dest))
        expected = np.prod(coords, axis=-1)
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
        expr_src = sum(SpatialCoordinate(m_src))
        expr_dest = sum(SpatialCoordinate(m_dest))
        expected = sum(coords.T)
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
        V_dest_2 = FunctionSpace(m_dest, "SminusCurl", 2)  # Not point evaluation nodes
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
        expr_src = sum(SpatialCoordinate(m_src))
        expr_dest = sum(SpatialCoordinate(m_dest))
        expected = sum(coords.T)
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
        expr_src = sum(SpatialCoordinate(m_src))
        expr_dest = sum(SpatialCoordinate(m_dest))
        expected = sum(coords.T)
        V_src = FunctionSpace(m_src, "CG", 3)
        V_dest = FunctionSpace(m_dest, "CG", 4)
        V_dest_2 = FunctionSpace(m_dest, "DG", 2)
    else:
        raise ValueError("Unknown mesh pair")

    dest_eval = PointEvaluator(m_dest, coords)
    return m_src, m_dest, dest_eval, expr_src, expr_dest, expected, V_src, V_dest, V_dest_2


@pytest.mark.parallel([1, 3])
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

    expr_1 = sum(SpatialCoordinate(m_src))
    expected_1 = sum(coords.T)
    expr_2 = product(SpatialCoordinate(m_src))
    expected_2 = np.prod(coords, axis=-1)

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
    dest_eval = PointEvaluator(m_dest, coords)
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got[0], expected_1)
    assert np.allclose(got[1], expected_2)

    # adjoint
    cofunc_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
    cofunc_src = assemble(interpolate(TestFunction(V_src), cofunc_dest))
    assert not np.allclose(f_src.dat.data_ro[0], cofunc_src.dat.data_ro[0])
    assert not np.allclose(f_src.dat.data_ro[1], cofunc_src.dat.data_ro[1])

    # Interpolate from non-mixed to mixed
    V_src_2 = VectorFunctionSpace(m_src, "CG", 1)
    assert V_src_2.value_shape == V_src.value_shape
    f_src_2 = Function(V_src_2).interpolate(SpatialCoordinate(m_src))
    result_mixed = assemble(interpolate(f_src_2, V_dest))

    expected_zero_form = 0
    for i in range(len(V_dest)):
        expected = assemble(interpolate(f_src_2[i], V_dest[i]))
        assert np.allclose(result_mixed.dat.data_ro[i], expected.dat.data_ro)

        expected_zero_form += assemble(action(cofunc_dest.subfunctions[i], expected))

    mixed_zero_form = assemble(interpolate(f_src_2, cofunc_dest))
    assert np.isclose(mixed_zero_form, expected_zero_form)


@pytest.mark.parallel([1, 3])
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

    # Build interpolation matrices in both directions
    coarse_to_fine = assemble(interpolate(TrialFunction(V_coarse), V_fine))
    coarse_to_fine_adjoint = assemble(interpolate(TestFunction(V_coarse), TrialFunction(V_fine.dual())))

    # If we now interpolate f_coarse into V_fine we should get a function
    # which has no interpolation error versus f_fine because we were able to
    # exactly represent expr_in_V_coarse in V_coarse and V_coarse is a subset
    # of V_fine
    f_coarse_on_fine = assemble(interpolate(f_coarse, V_fine))
    assert np.allclose(f_coarse_on_fine.dat.data_ro, f_fine.dat.data_ro)
    f_coarse_on_fine_mat = assemble(coarse_to_fine @ f_coarse)
    assert np.allclose(f_coarse_on_fine_mat.dat.data_ro, f_fine.dat.data_ro)

    # Adjoint interpolation takes us from V_fine^* to V_coarse^* so we should
    # also get an exact result here.
    cofunction_coarse = assemble(inner(f_coarse, TestFunction(V_coarse)) * dx)
    cofunction_fine = assemble(inner(f_fine, TestFunction(V_fine)) * dx)
    cofunction_fine_on_coarse = assemble(interpolate(TestFunction(V_coarse), cofunction_fine))
    assert np.allclose(
        cofunction_fine_on_coarse.dat.data_ro, cofunction_coarse.dat.data_ro
    )
    cofunction_fine_on_coarse_mat = assemble(action(coarse_to_fine_adjoint, cofunction_fine))
    assert np.allclose(
        cofunction_fine_on_coarse_mat.dat.data_ro, cofunction_coarse.dat.data_ro
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
    f_fine_on_coarse = assemble(interpolate(f_fine, V_coarse))
    assert np.allclose(f_fine_on_coarse.dat.data_ro, f_coarse.dat.data_ro)

    # But adjoint interpolation, which takes us from V_coarse^* to V_fine^*
    # won't exactly reproduce the cofunction of f_coarse, since V_fine^* is a
    # bigger space than V_coarse^*. This only worked before because we could
    # exactly represent the expression in V_coarse.
    cofunction_fine = assemble(inner(f_fine, TestFunction(V_fine)) * dx)
    cofunction_coarse = assemble(inner(f_coarse, TestFunction(V_coarse)) * dx)
    cofunction_coarse_on_fine = assemble(interpolate(TestFunction(V_fine), cofunction_coarse))
    assert not np.allclose(
        cofunction_coarse_on_fine.dat.data_ro, cofunction_fine.dat.data_ro
    )

    # We get a similar result going in the other direction. Forward
    # interpolation from V_coarse to V_fine similarly doesn't reproduce the
    # effect of interpolating expr_fine directly into V_fine
    f_coarse_on_fine = assemble(interpolate(f_coarse, V_fine))
    assert not np.allclose(f_coarse_on_fine.dat.data_ro, f_fine.dat.data_ro)

    # But the adjoint operation, which takes us from V_fine^* to
    # V_coarse^* correctly reproduces cofunction_coarse from cofunction_fine
    cofunction_fine_on_coarse = assemble(interpolate(TestFunction(V_coarse), cofunction_fine))
    assert not np.allclose(
        cofunction_fine_on_coarse.dat.data_ro, cofunction_coarse.dat.data_ro
    )


@pytest.mark.parametrize("shape,symmetry", [((1, 2, 3), None), ((3, 3), True)])
def test_interpolate_unitsquare_tfs_shape(shape, symmetry):
    m_src = UnitSquareMesh(2, 3)
    m_dest = UnitSquareMesh(3, 5, quadrilateral=True)
    V_src = TensorFunctionSpace(m_src, "CG", 3, shape=shape, symmetry=symmetry)
    V_dest = TensorFunctionSpace(m_dest, "CG", 4, shape=shape, symmetry=symmetry)
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
    dest_eval = PointEvaluator(m_dest, coords)
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
            m_src, m_dest, V_src, V_dest, dest_eval, expected, expr_src, expr_dest, atol
        )


def interpolate_function(
    m_src, m_dest, V_src, V_dest, dest_eval, expected, expr_src, expr_dest, atol
):
    f_dest = Function(V_dest).interpolate(expr_src)
    assert extract_unique_domain(f_dest) is m_dest

    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)

    f_src = Function(V_src).interpolate(expr_src)
    f_dest = assemble(interpolate(f_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)

    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # test Function.interpolate(...)
    f_dest = Function(V_dest)
    f_dest.interpolate(f_src)
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # test assemble(interpolate(Function, ...), tensor=...)
    f_dest = Function(V_dest)
    assemble(interpolate(f_src, V_dest), tensor=f_dest)
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)


def interpolate_expression(
    m_src, m_dest, V_src, V_dest, dest_eval, expected, expr_src, expr_dest, atol
):

    f_dest = assemble(interpolate(expr_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)
    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # test assemble(interpolate(expr, ...), tensor=...)
    f_dest = Function(V_dest)
    assemble(interpolate(expr_src, V_dest), tensor=f_dest)
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)


def interpolate_cofunction(
    m_src, m_dest, V_src, V_dest, dest_eval, expected, expr_src, expr_dest, atol
):
    f_dest = Function(V_dest).interpolate(expr_src)
    assert extract_unique_domain(f_dest) is m_dest

    cofunction_dest = assemble(inner(f_dest, TestFunction(V_dest)) * dx)
    cofunction_dest_on_src = assemble(interpolate(TestFunction(V_src), cofunction_dest))
    assert cofunction_dest_on_src.function_space().mesh() is m_src

    f_src = Function(V_src).interpolate(expr_src)
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


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("space", [0, 1])
@pytest.mark.parametrize("run_test", [interpolate_expression, interpolate_function, interpolate_cofunction])
def test_interpolate_cross_mesh(run_test, space, parameters):
    (
        m_src,
        m_dest,
        dest_eval,
        expr_src,
        expr_dest,
        expected,
        V_src,
        V_dest,
        V_dest_2,
    ) = parameters
    V_dest = (V_dest, V_dest_2)[space]
    if m_src.name == "src_sphere" and m_dest.name == "dest_sphere":
        # Between immersed manifolds we will often be doing projection so we
        # need a higher tolerance for our tests
        atol = 1e-3
    else:
        atol = 1e-8  # default
    run_test(
        m_src, m_dest, V_src, V_dest, dest_eval, expected, expr_src, expr_dest, atol
    )


@pytest.mark.parallel([1, 3])
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
        assemble(interpolate(TrialFunction(V_src), V_dest))
    f_src = Function(V_src).interpolate(expr)
    f_dest = assemble(interpolate(f_src, V_dest, allow_missing_dofs=True))
    dest_eval = PointEvaluator(m_dest, coords)
    # default value is zero
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([0.25, 0.0]))
    f_dest = Function(V_dest).assign(Constant(1.0))
    # make sure we have actually changed f_dest before checking interpolation
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([1.0, 1.0]))
    assemble(interpolate(f_src, V_dest, allow_missing_dofs=True), tensor=f_dest)
    # assigned value hasn't been changed
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([0.25, 1.0]))
    f_dest = assemble(interpolate(f_src, V_dest, allow_missing_dofs=True, default_missing_val=2.0))
    # should take on the default value
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([0.25, 2.0]))
    f_dest = Function(V_dest).assign(Constant(1.0))
    # make sure we have actually changed f_dest before checking interpolation
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([1.0, 1.0]))
    assemble(interpolate(f_src, V_dest, allow_missing_dofs=True, default_missing_val=2.0), tensor=f_dest)
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([0.25, 2.0]))
    f_dest = Function(V_dest).assign(Constant(1.0))
    # make sure we have actually changed f_dest before checking interpolation
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([1.0, 1.0]))
    assemble(interpolate(f_src, V_dest, allow_missing_dofs=True, default_missing_val=0.0), tensor=f_dest)
    assert np.allclose(dest_eval.evaluate(f_dest), np.array([0.25, 0.0]))

    # Try the other way around so we can check adjoint is unaffected
    m_src = UnitSquareMesh(4, 5)
    m_src.coordinates.dat.data[:] *= 2
    m_dest = UnitSquareMesh(2, 3)
    coords = np.array([[0.5, 0.5], [1.5, 1.5]])
    x, y = SpatialCoordinate(m_src)
    expr = x * y
    V_src = FunctionSpace(m_src, "CG", 3)
    V_dest = FunctionSpace(m_dest, "CG", 2)
    cofunction_src = assemble(inner(Function(V_dest), TestFunction(V_dest)) * dx)
    cofunction_src.dat.data_wo[:] = 1.0
    cofunction_dest = assemble(interpolate(
        TestFunction(V_dest), cofunction_src, allow_missing_dofs=True, default_missing_val=2.0
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
    plex = plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
    line = Mesh(plex, dim=2)
    x, y = SpatialCoordinate(line)
    V_line = FunctionSpace(line, "CG", 2)
    f_line = Function(V_line).interpolate(x * y)
    assert np.isclose(assemble(f_line * dx), np.sqrt(2) / 3)  # for sanity

    # Create a 1D line around the unit square (2D) with 4 cells
    cells = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]])
    vertex_coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    plex = plex_from_cell_list(1, cells, vertex_coords, comm=m.comm)
    line_square = Mesh(plex, dim=2)
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


@pytest.mark.parallel([1, 3])
def test_interpolate_matrix_cross_mesh():
    source_mesh = UnitSquareMesh(4, 4)
    target_mesh = UnitSquareMesh(5, 5)
    U = FunctionSpace(source_mesh, "CG", 2)
    V = FunctionSpace(target_mesh, "CG", 3)

    # For comparison later
    x, y = SpatialCoordinate(source_mesh)
    f = Function(U).interpolate(x**2 + y**2)
    x2, y2 = SpatialCoordinate(target_mesh)
    g = Function(V).interpolate(x2**2 + y2**2)

    # We get the VOM at the point evaluation node coords of the target FS
    w = VectorFunctionSpace(target_mesh, V.ufl_element())
    X = assemble(interpolate(target_mesh.coordinates, w))
    vom = VertexOnlyMesh(source_mesh, X.dat.data_ro, redundant=False)
    P0DG = FunctionSpace(vom, "DG", 0)
    # We get the interpolation matrix U -> P0DG which performs point evaluation
    A = assemble(interpolate(TrialFunction(U), P0DG))
    f_at_points = assemble(A @ f)
    f_at_points2 = assemble(interpolate(f, P0DG))
    assert np.allclose(f_at_points.dat.data_ro, f_at_points2.dat.data_ro)
    # To get the points in the correct order in V we interpolate into vom.input_ordering
    # We pass mat_type='aij' which constructs the permutation matrix instead of using SFs
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    B = assemble(interpolate(TrialFunction(P0DG), P0DG_io), mat_type='aij')
    f_at_points_correct_order = assemble(B @ f_at_points)
    f_at_points_correct_order2 = assemble(interpolate(f_at_points, P0DG_io))
    assert np.allclose(f_at_points_correct_order.dat.data_ro, f_at_points_correct_order2.dat.data_ro)

    # f_at_points_correct_order has the correct coefficients of the function in V
    # It is a function in P0DG_io, so we just directly assign it to a function in V
    f_interp = Function(V)
    f_interp.dat.data_wo[:] = f_at_points_correct_order.dat.data_ro[:]
    assert np.allclose(f_interp.dat.data_ro, g.dat.data_ro)

    # Hence interpolation from U to V is the product of the following three matrices:
    # C*B*A
    # A is the interpolation matrix from U to P0DG
    # B is the interpolation matrix from P0DG to vom.input_ordering
    # C is direct assignment to the function in V
    interp_mat = assemble(Action(B, A))
    f_at_points_correct_order3 = assemble(interp_mat @ f)
    f_interp2 = Function(V)
    f_interp2.dat.data_wo[:] = f_at_points_correct_order3.dat.data_ro[:]
    assert np.allclose(f_interp2.dat.data_ro, g.dat.data_ro)

    interp_mat2 = assemble(interpolate(TrialFunction(U), V))
    assert interp_mat2.arguments() == (TestFunction(V.dual()), TrialFunction(U))
    f_interp3 = assemble(interp_mat2 @ f)
    assert f_interp3.function_space() == V
    assert np.allclose(f_interp3.dat.data_ro, g.dat.data_ro)


@pytest.mark.parallel([1, 3])
def test_interpolate_matrix_cross_mesh_adjoint():
    mesh_fine = UnitSquareMesh(4, 4)
    mesh_coarse = UnitSquareMesh(2, 2)

    V_coarse = FunctionSpace(mesh_coarse, "CG", 1)
    V_fine = FunctionSpace(mesh_fine, "CG", 1)

    cofunc_fine = assemble(conj(TestFunction(V_fine)) * dx)

    interp = assemble(interpolate(TestFunction(V_coarse), TrialFunction(V_fine.dual())))
    cofunc_coarse = assemble(Action(interp, cofunc_fine))
    assert interp.arguments() == (TestFunction(V_coarse), TrialFunction(V_fine.dual()))
    assert cofunc_coarse.function_space() == V_coarse.dual()

    # Compare cofunc_fine with direct interpolation
    cofunc_coarse_direct = assemble(conj(TestFunction(V_coarse)) * dx)
    assert np.allclose(cofunc_coarse.dat.data_ro, cofunc_coarse_direct.dat.data_ro)


@pytest.mark.parallel([2, 3, 4])
def test_voting_algorithm_edgecases():
    # this triggers lots of cases where the VOM voting algorithm has to deal
    # with points being claimed by multiple ranks: there are cases where each
    # rank will claim another one owns a point, for example, and yet also all
    # claim zero distance to the reference cell!
    s = COMM_WORLD.size
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


def test_mixed_interpolator_cross_mesh():
    # Tests assembly of mixed interpolator across meshes
    mesh1 = UnitSquareMesh(4, 4)
    mesh2 = UnitSquareMesh(3, 3, quadrilateral=True)
    mesh3 = UnitDiskMesh(2)
    mesh4 = UnitTriangleMesh(3)
    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 2)
    V3 = FunctionSpace(mesh3, "CG", 3)
    V4 = FunctionSpace(mesh4, "CG", 4)

    W = V1 * V2
    U = V3 * V4

    w = TrialFunction(W)
    w0, w1 = split(w)
    expr = as_vector([w0 + w1, w0 + w1])
    mixed_interp = interpolate(expr, U, allow_missing_dofs=True)  # Interpolating from W to U

    # The block matrix structure is
    # | V1 -> V3   V2 -> V3 |
    # | V1 -> V4   V2 -> V4 |

    res = assemble(mixed_interp, mat_type="nest")
    assert isinstance(res, AssembledMatrix)
    assert res.petscmat.type == "nest"

    split_interp = dict(split_form(mixed_interp))

    for i in range(2):
        for j in range(2):
            interp_ij = split_interp[(i, j)]
            assert isinstance(interp_ij, Interpolate)
            res_block = assemble(interpolate(TrialFunction(W.sub(j)), U.sub(i), allow_missing_dofs=True))
            assert np.allclose(res.petscmat.getNestSubMatrix(i, j)[:, :], res_block.petscmat[:, :])
