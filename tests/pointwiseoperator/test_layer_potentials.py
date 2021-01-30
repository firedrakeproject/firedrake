"""Test bubble function space"""

import numpy as np
import pytest

from firedrake import MeshHierarchy, norms, Constant, \
    ln, pi, SpatialCoordinate, sqrt, PytentialLayerOperation, grad, \
    FunctionSpace, VectorFunctionSpace, FacetNormal, inner, assemble, \
    TestFunction, ds
from math import factorial
from warning import warn

# skip testing this module if cannot import pytential
pytential_installed = pytest.importorskip("pytential")

import pyopencl as cl

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

from pytools.convergence import EOCRecorder


# TODO: Actually write this test
@pytest.mark.skip
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


MAX_N = 97


# Need this for true solution to helmholtz-sommerfeld
def hankel_function(expr, n=None):
    """
        Returns a :mod:`firedrake` expression approximation a hankel function
        of the first kind and order 0
        evaluated at :arg:`expr` by using the taylor
        series, expanded out to :arg:`n` terms.
    """
    if n is None:
        warn("Default n to %s, this may cause errors."
             "If it bugs out on you, try setting n to something more reasonable"
             % MAX_N)
        n = MAX_N

    j_0 = 0
    for i in range(n):
        j_0 += (-1)**i * (1 / 4 * expr**2)**i / factorial(i)**2

    g = Constant(0.57721566490153286)
    y_0 = (ln(expr / 2) + g) * j_0
    h_n = 0
    for i in range(n):
        h_n += 1 / (i + 1)
        y_0 += (-1)**(i) * h_n * (expr**2 / 4)**(i+1) / (factorial(i+1))**2
    y_0 *= Constant(2 / pi)

    imag_unit = Constant((np.zeros(1, dtype=np.complex128) + 1j)[0])
    h_0 = j_0 + imag_unit * y_0
    return h_0


# TODO: Actually write this test
@pytest.mark.skip
@pytest.mark.parametrize("fspace_degree", [1, 2, 3],
                         "kappa", [1.0, 2.0])
def test_sommerfeld_helmholtz(ctx_factory, fspace_degree, kappa):
    """
    Solve the Helmholtz equation with a radiating-sommerfeld
    condition
    as in https://arxiv.org/abs/2009.08493
    """
    # make a computing context
    from meshmode.array_context import PyOpenCLArrayContext
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # We'll use this to test convergence
    eoc_recorder = EOCRecorder()

    def get_true_sol_expr(spatial_coord):
        """
        Get the ufl expression for the true solution
        """
        mesh_dim = len(spatial_coord)
        if mesh_dim == 3:
            x, y, z = spatial_coord  # pylint: disable=C0103
            norm = sqrt(x**2 + y**2 + z**2)
            return Constant(1j / (4*pi)) / norm * exp(1j * kappa * norm)

        if mesh_dim == 2:
            x, y = spatial_coord  # pylint: disable=C0103
            return Constant(1j / 4) * hankel_function(kappa * sqrt(x**2 + y**2), n=80)
        raise ValueError("Only meshes of dimension 2, 3 supported")

    # TODO : Pick a mesh
    mesh_hierarchy = MeshHierarchy("TODO")
    # TODO : Put in source/boundary ids
    source_bdy_id = None
    target_bdy_id = None
    for mesh in mesh_hierarchy:
        # Get true solution
        spatial_coord = SpatialCoordinate(mesh)
        true_sol_expr = get_true_sol_expr(spatial_coord)

        # Build function spaces
        cgfspace = FunctionSpace(mesh, "CG", fspace_degree)
        dgfspace = FunctionSpace(mesh, "DG", fspace_degree)
        cgvfspace = VectorFunctionSpace(mesh, "CG", fspace_degree)
        dgvfspace = VectorFunctionSpace(mesh, "DG", fspace_degree)

        # pytential normals point opposite direction of firedrake
        pyt_inner_normal_sign = -1
        ambient_dim = mesh.geometric_dimension()
        # Build rhs pytential operations
        from pytential import sym
        from sumpy.kernel import HelmholtzKernel
        sigma = sym.make_sym_vector("sigma", ambient_dim)
        r"""
        ..math:

        x \in \Sigma

        grad_op(x) =
            \nabla(
                \int_\Gamma(
                    f(y) H_0^{(1)}(\kappa |x - y|)
                )d\gamma(y)
            )
        """
        grad_op = pyt_inner_normal_sign * \
            sym.grad(ambient_dim, sym.S(HelmholtzKernel(ambient_dim),
                                        sym.n_dot(sigma),
                                        k=sym.var("k"), qbx_forced_limit=None))
        r"""
        ..math:

        x \in \Sigma

        op(x) =
            i \kappa \cdot
            \int_\Gamma(
                f(y) H_0^{(1)}(\kappa |x - y|)
            )d\gamma(y)
            )
        """
        op = 1j * sym.var("k") * pyt_inner_normal_sign * \
            sym.S(HelmholtzKernel(ambient_dim),
                  sym.n_dot(sigma),
                  k=sym.var("k"),
                  qbx_forced_limit=None)
        # Compute rhs
        rhs_grad_op = PytentialLayerOperation(grad(true_sol_expr),
                                              function_space=dgvfspace,
                                              operator_data={'op': grad_op,
                                                             'actx': actx,
                                                             'density_name': 'sigma',
                                                             'source_bdy_id': source_bdy_id,
                                                             'target_bdy_id': target_bdy_id,
                                                             'op_kwargs': {'k': 'k'},
                                                             'qbx_order': fspace_degree + 2,
                                                             'fine_order': 4 * fspace_degree,
                                                             'fmm_order': 50,
                                                             })
        rhs_op = PytentialLayerOperation(grad(true_sol_expr),
                                         function_space=dgvfspace,
                                         operator_data={'op': grad_op,
                                                        'actx': actx,
                                                        'density_name': 'sigma',
                                                        'source_bdy_id': source_bdy_id,
                                                        'target_bdy_id': target_bdy_id,
                                                        'op_kwargs': {'k': 'k'},
                                                        'qbx_order': fspace_degree + 2,
                                                        'fine_order': 4 * fspace_degree,
                                                        'fmm_order': 50,
                                                        })
        v = TestFunction(cgfspace)
        rhs_form = inner(inner(grad(true_sol), FacetNormal(mesh)),
                         v) * ds(source_bdy_id) \
            + inner(rhs_op, v) * ds(target_bdy_id) \
            - inner(inner(rhs_grad_op, FacetNormal(mesh)),
                    v) * ds(target_bdy_id)
        rhs = assemble(rhs_form)
        
        # TODO: Build operator

        # Solve
        comp_sol = Function(cgfspace)
        solver_params = {}
        solve(aN + aL == rhs, comp_sol,
              solver_params=solver_params)

        err = norms.l2_norm(true_sol - comp_sol)
        # Record the cell size and error
        # NOTE: Assumes mesh is order 1
        cell_size = np.max(mesh.cell_sizes.data.data)
        eoc_recorder.add_data_point(cell_size, err)

    assert(eoc_recorder.order_estimate() >= fspace_degree
           or eoc_recorder.max_error() < 2e-14)
