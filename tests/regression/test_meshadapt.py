from firedrake import *
from firedrake.meshadapt import *
from petsc4py import PETSc
import pytest
import numpy as np


def uniform_mesh(dim, n=5):
    """
    Create a uniform mesh of a specified dimension and size.

    :arg dim: the topological dimension
    :kwarg n: the number of subdivisions in each coordinate
        direction
    """
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")


def uniform_metric(mesh, a=100.0, **kwargs):
    """
    Create a metric which is just the identity
    matrix scaled by `a` at each vertex.

    :arg mesh: the mesh to define the metric upon
    :kwarg a: the scale factor for the identity
    :kwargs: other parameters to pass to the
        :class:`RiemannianMetric` constructor
    """
    dim = mesh.topological_dimension()
    metric = RiemannianMetric(mesh, **kwargs)
    metric.interpolate(a * Identity(dim))
    return metric


def load_mesh(fname):
    """
    Load a mesh in gmsh format.

    :arg fname: file name, without the .msh extension
    """
    from os.path import abspath, join, dirname

    cwd = abspath(dirname(__file__))
    return Mesh(join(cwd, "..", "meshes", fname + ".msh"))


def try_adapt(mesh, metric):
    """
    Attempt to invoke PETSc's mesh adaptation functionality
    and xfail if it is not installed.

    :arg mesh: the current mesh
    :arg metric: the :class:`RiemannianMetric` instance
    :return: the adapted mesh w.r.t. the metric
    """
    try:
        return adapt(mesh, metric)
    except PETSc.Error as exc:
        if exc.ierr == 63:
            pytest.xfail("No mesh adaptation tools are installed")
        else:
            raise Exception(f"PETSc error code {exc.ierr}")


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_enforce_spd(dim):
    """
    Tests for the :meth:`enforce_spd` method:
      * Check that negative eigenvalues are flipped.
      * Check that the minimum magnitude is correctly applied.
    """
    mesh = uniform_mesh(dim)
    metric = uniform_metric(mesh, a=-100.0)
    newmesh = try_adapt(mesh, metric)
    num_vertices = newmesh.coordinates.vector().gather().shape[0]

    hmin = 1.0
    metric.set_parameters({"dm_plex_metric_h_min": hmin})
    assert np.isclose(metric.plex.metricGetMinimumMagnitude(), hmin)
    metric.enforce_spd(restrict_sizes=True)
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape[0] < num_vertices
    expected = uniform_metric(mesh, a=1.0)
    assert np.isclose(errornorm(metric.function, expected.function), 0.0)


@pytest.mark.parallel(nprocs=2)
def test_intersection(dim):
    """
    Test that intersecting two metrics results
    in a metric with the minimal ellipsoid.
    """
    mesh = uniform_mesh(dim)
    metric1 = uniform_metric(mesh, a=100.0)
    metric2 = uniform_metric(mesh, a=25.0)
    metric1.intersect(metric2)
    expected = uniform_metric(mesh, a=100.0)
    assert np.isclose(errornorm(metric1.function, expected.function), 0.0)


def test_no_adapt(dim):
    """
    Test that we can turn off all of Mmg's
    mesh adaptation operations.
    """
    mesh = uniform_mesh(dim)
    dofs = mesh.coordinates.vector().gather().shape
    mp = {
        "dm_plex_metric": {
            "no_insert": None,
            "no_move": None,
            "no_swap": None,
            "no_surf": None,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape == dofs


@pytest.mark.parallel(nprocs=2)
def test_no_adapt_parallel():
    """
    Test that we can turn off all of ParMmg's
    mesh adaptation operations.
    """
    test_no_adapt(3)


@pytest.mark.parallel(nprocs=2)
def test_normalise(dim):
    """
    Test that normalising a metric w.r.t.
    a given metric complexity and the
    normalisation order :math:`p=1` DTRT.
    """
    mesh = uniform_mesh(dim)
    target = 200.0 if dim == 2 else 2500.0
    mp = {
        "dm_plex_metric": {
            "target_complexity": target,
            "normalization_order": 1.0,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    metric.normalise()
    expected = uniform_metric(mesh, a=pow(target, 2.0 / dim))
    assert np.isclose(errornorm(metric.function, expected.function), 0.0)


@pytest.mark.parametrize(
    "meshname",
    [
        "annulus",
        "cell-sets",
        "square_with_embedded_line",
    ],
)
def test_preserve_cell_tags_2d(meshname):
    """
    Test that cell tags are preserved
    after mesh adaptation.
    """
    mesh = load_mesh(meshname)
    metric = uniform_metric(mesh)
    newmesh = try_adapt(mesh, metric)

    tags = set(mesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    newtags = set(newmesh.topology_dm.getLabelIdIS("Cell Sets").indices)
    assert tags == newtags, "Cell tags do not match"

    one = Constant(1.0)
    for tag in tags:
        bnd = assemble(one * dx(tag, domain=mesh))
        newbnd = assemble(one * dx(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Area of region {tag} not preserved"


@pytest.mark.parametrize(
    "meshname",
    [
        "annulus",
        "circle_in_square",
    ],
)
def test_preserve_facet_tags_2d(meshname):
    """
    Test that facet tags are preserved
    after mesh adaptation.
    """
    mesh = load_mesh(meshname)
    metric = uniform_metric(mesh)
    newmesh = try_adapt(mesh, metric)

    newmesh.init()
    tags = set(mesh.exterior_facets.unique_markers)
    newtags = set(newmesh.exterior_facets.unique_markers)
    assert tags == newtags, "Facet tags do not match"

    one = Constant(1.0)
    for tag in tags:
        bnd = assemble(one * ds(tag, domain=mesh))
        newbnd = assemble(one * ds(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Length of arc {tag} not preserved"


def test_adapt_3d():
    """
    Test that we can successfully invoke
    Mmg3d and that it changes the DoF count.
    """
    mesh = uniform_mesh(3)
    dofs = mesh.coordinates.vector().gather().shape
    mp = {
        "dm_plex_metric": {
            "target_complexity": 100.0,
            "normalization_order": 1.0,
        }
    }
    metric = uniform_metric(mesh, metric_parameters=mp)
    newmesh = try_adapt(mesh, metric)
    assert newmesh.coordinates.vector().gather().shape != dofs


@pytest.mark.parallel(nprocs=2)
def test_adapt_parallel_3d():
    """
    Test that we can successfully invoke
    ParMmg and that it changes the DoF count.
    """
    test_adapt_3d()
