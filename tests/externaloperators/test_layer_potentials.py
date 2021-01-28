"""Test bubble function space"""

import numpy as np
import pytest

from firedrake import MeshHierarchy, norms

# skip testing this module if cannot import pytential
pytential_installed = pytest.importorskip("pytential")

import pyopencl as cl

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from pytools.convergence import EOCRecorder


@pytest.mark.parametrize("fspace_degree", [1, 2, 3])
def test_greens_formula(ctx_factory, fspace_degree):
    # make a computing context
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # We'll use this to test convergence
    eoc_recorder = EOCRecorder()

    # TODO : Pick a mesh
    mesh_hierarchy = MeshHierarchy("TODO")
    for mesh in mesh_hierarchy:
        # NOTE: Assumes mesh is order 1
        cell_size = np.max(mesh.cell_sizes.data.data)
        # TODO : Solve a system
        err = norms.l2_norm(true - comp)
        eoc_recorder.add_data_point(cell_size, err)

    assert(eoc_recorder.order_estimate() >= fspace_degree
           or eoc_recorder.max_error() < 2e-14)
