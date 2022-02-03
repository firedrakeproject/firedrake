from firedrake import *
from firedrake.meshadapt import *
from petsc4py import PETSc
import pytest
import numpy as np


def uniform_mesh(dim, n=5):
    if dim == 2:
        return UnitSquareMesh(n, n)
    elif dim == 3:
        return UnitCubeMesh(n, n, n)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")


def load_mesh(fname):
    from os.path import abspath, join, dirname

    cwd = abspath(dirname(__file__))
    return Mesh(join(cwd, "..", "meshes", fname + ".msh"))


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.mark.parallel(nprocs=2)
def test_intersection(dim):
    """
    Test that intersecting two metrics results
    in a metric with the minimal ellipsoid.
    """
    Id = Identity(dim)
    mesh = uniform_mesh(dim)
    metric1 = RiemannianMetric(mesh)
    metric1.interpolate(100.0 * Id)
    metric2 = RiemannianMetric(mesh)
    metric2.interpolate(25.0 * Id)
    metric1.intersect(metric2)
    expected = RiemannianMetric(mesh)
    expected.interpolate(100.0 * Id)
    assert np.isclose(errornorm(metric1.function, expected.function), 0.0)


def test_size_restriction(dim):  # FIXME
    """
    Test that enforcing a minimum magnitude
    larger than the domain means that there
    are as few elements as possible.
    """
    Id = Identity(dim)
    mesh = uniform_mesh(dim)
    mp = {"dm_plex_metric_h_min": 2.0}
    metric = RiemannianMetric(mesh, metric_parameters=mp)
    metric.interpolate(100.0 * Id)
    metric.enforce_spd(restrict_sizes=True)
    expected = RiemannianMetric(mesh)
    expected.interpolate(0.25 * Id)
    # assert np.isclose(errornorm(metric.function, expected.function), 0.0)
    try:
        newmesh = adapt(mesh, metric)
    except PETSc.Error as exc:
        if exc.ierr == 63:
            pytest.xfail("No mesh adaptation tools are installed")
        else:
            raise Exception(f"PETSc error code {exc.ierr}")
    assert newmesh.num_cells() < mesh.num_cells()


@pytest.mark.parallel(nprocs=2)
def test_normalise(dim):
    """
    Test that normalising a metric w.r.t.
    a given metric complexity and the
    normalisation order :math:`p=1` DTRT.
    """
    Id = Identity(dim)
    mesh = uniform_mesh(dim)
    target = 200.0 if dim == 2 else 2500.0
    mp = {
        "dm_plex_metric": {
            "target_complexity": target,
            "normalization_order": 1.0,
        }
    }
    metric = RiemannianMetric(mesh, metric_parameters=mp)
    metric.interpolate(100.0 * Id)
    metric.normalise()
    expected = RiemannianMetric(mesh)
    expected.interpolate(pow(target, 2.0 / dim) * Id)
    assert np.isclose(errornorm(metric.function, expected.function), 0.0)


@pytest.mark.parametrize(
    "meshname",
    [
        "annulus",
        "cell-sets",
        "square_with_embedded_line",
    ],
)
def test_preserve_cell_tags(meshname):
    Id = Identity(2)
    mesh = load_mesh(meshname)
    metric = RiemannianMetric(mesh)
    metric.interpolate(100.0 * Id)
    newmesh = adapt(mesh, metric)

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
        "square_with_embedded_line",  # FIXME
    ],
)
def test_preserve_facet_tags(meshname):
    """
    Test that facet tags are preserved
    after mesh adaptation.
    """
    Id = Identity(2)
    mesh = load_mesh(meshname)
    metric = RiemannianMetric(mesh)
    metric.interpolate(100.0 * Id)
    newmesh = adapt(mesh, metric)

    newmesh.init()
    tags = set(mesh.exterior_facets.unique_markers)
    newtags = set(newmesh.exterior_facets.unique_markers)
    assert tags == newtags, "Facet tags do not match"

    one = Constant(1.0)
    for tag in tags:
        bnd = assemble(one * ds(tag, domain=mesh))
        newbnd = assemble(one * ds(tag, domain=newmesh))
        assert np.isclose(bnd, newbnd), f"Length of arc {tag} not preserved"
