"""Test bubble function space"""

import numpy as np
import pytest

from firedrake import MeshHierarchy, norms, Constant, \
    ln, pi, SpatialCoordinate, sqrt, grad, \
    FunctionSpace, VectorFunctionSpace, FacetNormal, inner, assemble, \
    TestFunction, ds, Function, exp, TrialFunction, dx, project, solve, \
    utils, OpenCascadeMeshHierarchy, dot, SingleLayerPotential, \
    DoubleLayerPotential, PotentialSourceAndTarget
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
        err = norms.norm(true - comp, norm_type="L2")
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

    def get_true_sol(fspace, kappa, cl_ctx, queue):
        """
        Get the ufl expression for the true solution (3D)
        or a function with the evaluated solution (2D)
        """
        mesh_dim = fspace.mesh().geometric_dimension()
        if mesh_dim == 3:
            spatial_coord = SpatialCoordinate(fspace.mesh())
            x, y, z = spatial_coord  # pylint: disable=C0103
            norm = sqrt(x**2 + y**2 + z**2)
            return Constant(1j / (4*pi)) / norm * exp(1j * kappa * norm)

        if mesh_dim == 2:
            # Evaluate true-sol using sumpy
            from sumpy.p2p import P2P
            from sumpy.kernel import HelmholtzKernel
            # https://github.com/inducer/sumpy/blob/900745184d2618bc27a64c847f247e01c2b90b02/examples/curve-pot.py#L87-L88
            p2p = P2P(cl_ctx, [HelmholtzKernel(dim=2)], exclude_self=False,
                      value_dtypes=np.complex128)
            # source is just (0, 0)
            sources = np.array([[0.0], [0.0]])
            strengths = np.array([[1.0], [1.0]])
            # targets are everywhere
            targets = np.array([Function(fspace).interpolate(x_i).dat.data
                                for x_i in SpatialCoordinate(fspace.mesh())])
            evt, (true_sol_arr,) = p2p(queue, targets, sources, strengths, k=kappa)
            true_sol = Function(fspace)
            true_sol.dat.data[:] = true_sol_arr[:]
            return true_sol
        raise ValueError("Only meshes of dimension 2, 3 supported")

    # Create mesh and build hierarchy
    mesh_hierarchy = OpenCascadeMeshHierarchy(
        "../meshes/square_without_circle.step",
        element_size=0.5,
        levels=3,
        order=2,
        project_refinements_to_cad=False,
        cache=False)
    scatterer_bdy = 5  # (inner boundary) the circle
    truncated_bdy = (1, 2, 3, 4)  # (outer boundary) the square
    # Solve for each mesh in hierarchy
    for h, mesh in zip([0.5 * 2**i for i in range(len(mesh_hierarchy))], mesh_hierarchy):
        # Build function spaces
        V = FunctionSpace(mesh, "CG", fspace_degree)
        Vdg = FunctionSpace(mesh, "DG", fspace_degree)

        # Get true solution
        true_sol = get_true_sol(Vdg, kappa, cl_ctx, queue)

        # {{{ Get Neumann Data

        n = FacetNormal(mesh)
        f_expr = dot(grad(true_sol), n)
        v = TestFunction(Vdg)
        u = TrialFunction(Vdg)
        a = inner(u, v) * ds(scatterer_bdy) + inner(u, v) * dx
        L = inner(f_expr, v) * ds(scatterer_bdy) + inner(Constant(0.0), v) * dx
        from firedrake import DirichletBC
        bc = DirichletBC(Vdg, Constant(0.0), truncated_bdy)
        f = Function(Vdg)
        solve(a == L, f, bcs=[bc])
        f = project(f, V)

        # }}}

        # FIXME: Handle normal signs
        # places has source of inner boundary and target of outer boundary
        from ufl import derivative
        import petsc4py.PETSc
        petsc4py.PETSc.Sys.popErrorHandler()
        places = PotentialSourceAndTarget(mesh,
                                          source_region_dim=1,
                                          source_region_id=scatterer_bdy,
                                          target_region_dim=1,
                                          target_region_id=truncated_bdy)
        # TODO: Fix RHS
        from sumpy.kernel import HelmholtzKernel
        Sf = SingleLayerPotential(f,
                                  HelmholtzKernel(dim=2),
                                  places,
                                  actx=actx,
                                  function_space=V,
                                  op_kwargs={'k': kappa, 'qbx_forced_limit': None})
        v = TestFunction(V)
        rhs = inner(f, v) * ds(scatterer_bdy) + \
            1j * kappa * inner(Sf, v) * ds(truncated_bdy) - \
            inner(dot(grad(Sf), n), v) * ds(truncated_bdy)

        assemble(rhs)
        1/0

        # TODO: Continue fixing test
        
        # local helmholtz operator
        r"""
        .. math::

            \langle \nabla u, \nabla v \rangle
            - \kappa^2 \cdot \langle u, v \rangle
            - i \kappa \langle u, v \rangle_\Sigma
        """
        # local operator as bilinear form
        trial = TrialFunction(cgfspace)
        aL = inner(grad(trial), grad(v)) * dx \
            - Constant(kappa**2) * inner(trial, v) * dx \
            - Constant(1j * kappa) * inner(trial, v) * ds(target_bdy_id)
        # local operator as functional FL(u)
        u = Function(cgfspace, name="u")
        FL = inner(grad(u), grad(v)) * dx \
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
        # pytential operations into firedrake
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
        # Non-local operator as functional of u FN(u)
        n = FacetNormal(mesh)
        FN = inner(layer, v) * ds(target_bdy_id) - \
            inner(dot(grad_layer, n), v) * ds(target_bdy_id)

        from ufl import derivative

        # Solve
        solver_params = {'snes_type': 'ksponly',
                         'ksp_monitor': None,
                         'ksp_rtol': 1e-7,
                         'mat_type': 'matfree',
                         'pmat_type': 'aij',
                         'pc_type': 'lu',
                         }
        # make sure to collect petsc errors
        from ufl import derivative
        import petsc4py.PETSc
        petsc4py.PETSc.Sys.popErrorHandler()
        solve(FN + FL - rhs_form == 0, u,
              Jp=aL,
              solver_parameters=solver_params)

        true_sol = Function(cgfspace).interpolate(true_sol_expr)
        err = norms.norm(true_sol - u, norm_type="L2")
        print("L^2 Error: ", abs(err))
        # Record the cell size and error
        eoc_recorder.add_data_point(h, err)
        # visualize for debugging
        visualize = False
        if visualize:
            from firedrake import trisurf
            import matplotlib.pyplot as plt
            trisurf(true_sol)
            plt.title("True Solution")
            trisurf(u)
            plt.title("Computed Solution")
            plt.show()

    assert(eoc_recorder.order_estimate() >= fspace_degree
           or eoc_recorder.max_error() < 2e-14)


fspace_degree = 1
kappa = 1.0
from pyopencl import create_some_context
test_sommerfeld_helmholtz(create_some_context, fspace_degree, kappa)
