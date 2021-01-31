"""Test bubble function space"""

import numpy as np
import pytest

from firedrake import MeshHierarchy, norms, Constant, \
    ln, pi, SpatialCoordinate, sqrt, PytentialLayerOperation, grad, \
    FunctionSpace, VectorFunctionSpace, FacetNormal, inner, assemble, \
    TestFunction, ds, Function, exp, TrialFunction, dx, project, solve, \
    utils, OpenCascadeMeshHierarchy
from math import factorial
from warnings import warn

# skip testing this module if cannot import pytential
pytential = pytest.importorskip("pytential")
# skip testing if opencascade is not installed
STEPControl = pytest.importorskip("OCC.Core.STEPControl")
TopologyUtils = pytest.importorskip("OCC.Extend.TopologyUtils")

import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
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


# Make sure to skip if not in complex mode
@pytest.mark.skipif(not utils.complex_mode, reason="Solves a PDE with complex variables")
# Test following degrees and wave number (kappa)s
@pytest.mark.parametrize("fspace_degree", [1, 2, 3])
@pytest.mark.parametrize("kappa", [1.0])
def test_sommerfeld_helmholtz(ctx_factory, fspace_degree, kappa):
    """
    Solve the Helmholtz equation with a radiating-sommerfeld
    condition
    as in https://arxiv.org/abs/2009.08493
    """
    # make a computing context
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

    # Create mesh and build hierarchy
    mesh_hierarchy = OpenCascadeMeshHierarchy(
        "../meshes/square_without_circle.step",
        element_size=0.5,
        levels=3,
        order=2,
        project_refinements_to_cad=False)
    source_bdy_id = 5  # (inner boundary) the circle
    target_bdy_id = (1, 2, 3, 4)  # (outer boundary) the square
    # Solve for each mesh in hierarchy
    for mesh in mesh_hierarchy:
        # Get true solution
        spatial_coord = SpatialCoordinate(mesh)
        true_sol_expr = get_true_sol_expr(spatial_coord)

        # Build function spaces
        cgfspace = FunctionSpace(mesh, "CG", fspace_degree)
        cgvfspace = VectorFunctionSpace(mesh, "CG", fspace_degree)
        dgfspace = FunctionSpace(mesh, "DG", fspace_degree)
        dgvfspace = VectorFunctionSpace(mesh, "DG", fspace_degree)

        # pytential normals point opposite direction of firedrake
        pyt_inner_normal_sign = -1
        ambient_dim = mesh.geometric_dimension()
        # Build rhs pytential operations
        from pytential import sym
        from sumpy.kernel import HelmholtzKernel
        sigma = sym.make_sym_vector("density", ambient_dim)
        r"""
        ..math:

        x \in \Sigma
        grad_op(x) = \nabla( \int_\Gamma( f(y) H_0^{(1)}(\kappa |x - y|) )d\gamma(y))
        """
        rhs_pyt_grad_layer = pyt_inner_normal_sign * \
            sym.grad(ambient_dim, sym.S(HelmholtzKernel(ambient_dim),
                                        sym.n_dot(sigma),
                                        k=sym.var("k"), qbx_forced_limit=None))
        r"""
        ..math:

        x \in \Sigma
        op(x) = i \kappa \cdot \int_\Gamma( f(y) H_0^{(1)}(\kappa |x - y|) )d\gamma(y)
        """
        rhs_pyt_layer = 1j * sym.var("k") * pyt_inner_normal_sign * \
            sym.S(HelmholtzKernel(ambient_dim),
                  sym.n_dot(sigma),
                  k=sym.var("k"),
                  qbx_forced_limit=None)
        dg_true_sol_grad = Function(dgvfspace).interpolate(grad(true_sol_expr))
        # general operator data settings, missing 'op'
        operator_data = {'actx': actx,
                         'density_name': 'density',
                         'source_bdy_id': source_bdy_id,
                         'target_bdy_id': target_bdy_id,
                         'op_kwargs': {'k': kappa},
                         'qbx_order': fspace_degree + 2,
                         'fine_order': 4 * fspace_degree,
                         'fmm_order': 50,
                         }
        # Bind rhs pytential operations into firedrake
        rhs_grad_layer_operator_data = dict(operator_data)
        rhs_grad_layer_operator_data['op'] = rhs_pyt_grad_layer
        rhs_grad_layer = PytentialLayerOperation(dg_true_sol_grad,
                                                 function_space=cgvfspace,
                                                 operator_data=rhs_grad_layer_operator_data)
        rhs_layer_operator_data = dict(operator_data)
        rhs_layer_operator_data['op'] = rhs_pyt_layer
        rhs_layer = PytentialLayerOperation(dg_true_sol_grad,
                                            function_space=cgfspace,
                                            operator_data=rhs_layer_operator_data)
        # get rhs form
        v = TestFunction(cgfspace)
        rhs_form = inner(inner(grad(true_sol_expr), FacetNormal(mesh)),
                         v) * ds(source_bdy_id) \
            + inner(rhs_layer, v) * ds(target_bdy_id) \
            - inner(inner(rhs_grad_layer, FacetNormal(mesh)),
                    v) * ds(target_bdy_id)
        
        # local helmholtz operator
        r"""
        .. math::

            \langle \nabla u, \nabla v \rangle
            - \kappa^2 \cdot \langle u, v \rangle
            - i \kappa \langle u, v \rangle_\Sigma
        """
        u = TrialFunction(cgfspace)
        aL = inner(grad(u), grad(v)) * dx \
            - Constant(kappa**2) * inner(u, v) * dx \
            - Constant(1j * kappa) * inner(u, v) * ds(target_bdy_id)

        # pytential non-local helmholtz operations
        r"""
        ..math:

        x \in \Sigma
        grad_op(x) = \nabla( \int_\Gamma( u(y) \partial_n H_0^{(1)}(\kappa |x - y|))d\gamma(y) )
        """
        pyt_grad_layer = pyt_inner_normal_sign * sym.grad(
            ambient_dim, sym.D(HelmholtzKernel(ambient_dim),
                               sym.var("density"), k=sym.var("k"),
                               qbx_forced_limit=None))

        r"""
        ..math:

        x \in \Sigma
        op(x) = i \kappa \cdot \int_\Gamma( u(y) \partial_n H_0^{(1)}(\kappa |x - y|) )d\gamma(y)
        """
        pyt_layer = pyt_inner_normal_sign * 1j * sym.var("k") * (
            sym.D(HelmholtzKernel(ambient_dim),
                  sym.var("density"), k=sym.var("k"),
                  qbx_forced_limit=None))
        # rhs pytential operations into firedrake
        grad_layer_operator_data = dict(operator_data)
        grad_layer_operator_data['op'] = pyt_grad_layer
        grad_layer_operator_data['project_to_dg'] = True
        grad_layer = PytentialLayerOperation(u,
                                             function_space=cgvfspace,
                                             operator_data=grad_layer_operator_data)
        layer_operator_data = dict(operator_data)
        layer_operator_data['op'] = pyt_layer
        layer_operator_data['project_to_dg'] = True
        layer = PytentialLayerOperation(u,
                                        function_space=cgfspace,
                                        operator_data=layer_operator_data)

        # non-local helmholtz operator
        r"""
        .. math::

            \langle
                i \kappa \cdot \int_\Gamma( u(y) \partial_n H_0^{(1)}(\kappa |x - y|) )d\gamma(y), v
            \rangle_\Sigma
            - \langle
                n(x) \cdot \nabla( \int_\Gamma( u(y) \partial_n H_0^{(1)}(\kappa |x - y|) )d\gamma(y)), v
            \rangle_\Sigma
        """
        n = FacetNormal(mesh)
        aN = inner(layer, v) * ds(target_bdy_id) - \
            inner(inner(grad_layer, n), v) * ds(target_bdy_id)

        # Solve
        comp_sol = Function(cgfspace)
        solver_params = {'ksp_monitor': None,
                         'mat_type': 'matfree',
                         }
        # make sure to collect petsc errors
        import petsc4py.PETSc
        petsc4py.PETSc.Sys.popErrorHandler()
        solve(aN + aL == rhs_form, u, solver_parameters=solver_params)

        true_sol = Function(cgfspace).interpolate(true_sol_expr)
        err = norms.l2_norm(true_sol - comp_sol)
        # Record the cell size and error
        # NOTE: Assumes mesh is order 1
        cell_size = np.max(mesh.cell_sizes.data.data)
        eoc_recorder.add_data_point(cell_size, err)

    assert(eoc_recorder.order_estimate() >= fspace_degree
           or eoc_recorder.max_error() < 2e-14)


ctx_factory = cl.create_some_context
fspace_degree = 1
kappa = 1.0
test_sommerfeld_helmholtz(ctx_factory, fspace_degree, kappa)
