from firedrake import *
from firedrake.meshadapt import *
from utility import *
import numpy as np
import pytest


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


@pytest.mark.skipcomplex
def test_project(dim):
    """
    Check that the :meth:`project` method of :class:`MetricBasedAdaptor` works as
    expected.
    """
    mesh = uniform_mesh(dim)
    metric = uniform_metric(mesh, a=100.0)

    # Adapt the mesh
    adaptor = MetricBasedAdaptor(mesh, metric)
    newmesh = adaptor.adapted_mesh

    # Create some fields
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    f1 = interpolate(x[0], V)
    g1 = interpolate(2*x[1], V)

    # Project them onto the new mesh
    f2 = adaptor.project(f1)
    g2 = adaptor.project(g1)
    assert f2.function_space().mesh() == newmesh
    assert g2.function_space().mesh() == newmesh
    assert f1.ufl_element() == f2.ufl_element()
    assert g1.ufl_element() == g2.ufl_element()

    # Check that mass is conserved
    assert np.isclose(assemble(f1*dx), assemble(f2*dx))
    assert np.isclose(assemble(g1*dx), assemble(g2*dx))
