import pytest
from firedrake import *
from firedrake import ufl_expr
import numpy as np


def _rotate_mesh(mesh, theta):
    x, y = SpatialCoordinate(mesh)
    Vc = mesh.coordinates.function_space()
    coords = Function(Vc).interpolate(as_vector([cos(theta) * x - sin(theta) * y,
                                                 sin(theta) * x + cos(theta) * y]))
    return Mesh(coords)


def _poisson_analytical(V, xi, eta, which):
    if which == 'solution':
        #return sin(2 * pi * xi) * sin(2 * pi * eta)
        return cos(2 * pi * xi) * cos(2 * pi * eta)
    elif which == 'force':
        #return 8.0 * pi * pi * sin(2 * pi * xi) * sin(2 * pi * eta)
        return 8.0 * pi * pi * cos(2 * pi * xi) * cos(2 * pi * eta)


def _poisson_get_forms_original(V, f, n):
    u = TrialFunction(V)
    v = TestFunction(V)
    normal = FacetNormal(V.mesh())
    alpha = 100
    h = 1./(2**n)
    a = dot(grad(v), grad(u)) * dx - dot(grad(u), normal) * v * ds - dot(grad(v), normal) * u * ds + alpha / h * u * v * ds
    L = f * v * dx
    return a, L


def _poisson_get_forms_hermite(V, xprime, yprime, g, f):

    # Define op2.subsets to be used when defining filters
    subset_1234 = V.boundary_node_subset((1, 2, 3, 4))
    subset_12 = V.boundary_node_subset((1, 2))
    subset_34 = V.boundary_node_subset((3, 4))
    subset_value = V.node_subset(derivative_order=0)  # subset of value nodes
    subset_deriv = V.node_subset(derivative_order=1)  # subset of derivative nodes

    solver_parameters = {"ksp_rtol": 1.e-16}

    # Define filters
    # -- domain nodes
    g0 = Function(V)
    g0.assign(Constant(1.))
    g0.assign(Constant(0.), subset=subset_1234)
    # -- boundary normal derivative nodes
    g1 = Function(V)
    g1.assign(project(xprime, V, solver_parameters=solver_parameters), subset=subset_12.difference(subset_34).intersection(subset_deriv))
    g1.assign(project(yprime, V, solver_parameters=solver_parameters), subset=subset_34.difference(subset_12).intersection(subset_deriv))
    # -- boundary tangent derivative nodes 
    g2 = Function(V).assign(project(yprime, V, solver_parameters=solver_parameters), subset=subset_12.intersection(subset_deriv))
    g3 = Function(V).assign(project(xprime, V, solver_parameters=solver_parameters), subset=subset_34.intersection(subset_deriv))
    # -- boundary value nodes
    g4 = Function(V).assign(project(Constant(1.), V, solver_parameters=solver_parameters), subset=subset_1234.intersection(subset_value))

    # Filter test function
    u = TrialFunction(V)
    v = TestFunction(V)

    #from firedrake.utils import IntType, RealType, ScalarType

    def normalize_subspace(old_subspace, subdomain):
        domain = ""
        domain = "{[k]: 0 <= k < 3}"
        instructions = """
        <float64> eps = 1e-9
        <float64> norm = 0
        for k
            norm = sqrt(old_subspace[3 * k + 1] * old_subspace[3 * k + 1] + old_subspace[3 * k + 2] * old_subspace[3 * k + 2])
            if norm > eps
                new_subspace[3 * k + 1] = old_subspace[3 * k + 1] / norm
                new_subspace[3 * k + 2] = old_subspace[3 * k + 2] / norm
            end
        end
        """

        V = old_subspace.function_space()
        new_subspace = Function(V)

        par_loop((domain, instructions), ds(subdomain),
                 {"new_subspace": (new_subspace, WRITE),
                  "old_subspace": (old_subspace, READ)},
                 is_loopy_kernel=True)

        return new_subspace

    #print("old_g2:", g2.dat.data)
    g1 = normalize_subspace(g1, (1,2,3,4))
    g2 = normalize_subspace(g2, (1,2,3,4))
    g3 = normalize_subspace(g3, (1,2,3,4))
    #print("new_g1:", g1.dat.data)
    #print("new_g2:", g2.dat.data)
    #print("new_g3:", g3.dat.data)
    #exit(0)



    """
    g5 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset((1,)).intersection(subset_deriv))
    v5 = Masked(v, g5)
    u5 = Masked(u, g5)


    from finat.point_set import PointSet
    from finat.quadrature import QuadratureRule
    point_set = PointSet([[0,], [1,]])
    weights = [1,1]
    quad_rule = QuadratureRule(point_set, weights)

    normal = FacetNormal(V.mesh())
    aa = inner(u - u5, v - v5) * dx + inner(grad(u5), grad(v5)) * ds(1, scheme=quad_rule)
    ff = inner(normal, grad(v5)) * ds(1, scheme=quad_rule)
    u_ = Function(V)
    solve(aa == ff, u_, solver_parameters={"ksp_type": 'preonly', "pc_type": 'lu'})

    #for i in range(u_.dat.data.shape[0]):
    #    print("u_",u_.dat.data[i])
    #for i in range(g1.dat.data.shape[0]):
    #    print("g1",g1.dat.data[i])
    err = u_.dat.data - g1.dat.data
    #for i in range(g1.dat.data.shape[0]):
    #    print("er",err[i])
    print(np.linalg.norm(err))
    """


    V0 = Subspace(V, g0)
    V1 = TransformedSubspace(V, g1)
    V2 = TransformedSubspace(V, g2)
    V3 = TransformedSubspace(V, g3)
    V4 = Subspace(V, g4)

    v0 = Masked(v, V0)
    v1 = Masked(v, V1)
    v2 = Masked(v, V2)
    v3 = Masked(v, V3)
    v4 = Masked(v, V4)
    u0 = Masked(u, V0)
    u1 = Masked(u, V1)
    u2 = Masked(u, V2)
    u3 = Masked(u, V3)
    u4 = Masked(u, V4)

    ub = u4 + u2 + u3
    vb = v4 + v2 + v3

    gb = Masked(g, V4) + Masked(g, V2) + Masked(g, V3)


    # Make sure to project with very small tolerance.
    ud = u - ub
    vd = v - vb

    a0 = inner(grad(ud), grad(vd)) * dx
    a1 = inner(grad(u0+u1), grad(v0+v1)) * dx
    A0 = assemble(a0)
    A1 = assemble(a1)
    #A0[A0<1.e-8]=0
    #print("A0:", A0)
    pA = A1.M.handle - A0.M.handle
    from petsc4py import PETSc
    print("pA:", pA.norm(norm_type=PETSc.NormType.NORM_INFINITY))

    a = inner(grad(ud), grad(vd)) * dx + \
        inner(ub, vb)* ds
    #a = inner(grad(u0 + u1), grad(v0 + v1)) * dx + \
    #    inner(ub, vb) * ds
    L = inner(f, v0 + v1) * dx - inner(grad(gb), grad(v0 + v1)) * dx + inner(gb, vb) * ds


    #f0 = Masked(f, V0)
    #f1 = Masked(f, V1)
    #f2 = Masked(f, V2)
    #f3 = Masked(f, V3)
    #f4 = Masked(f, V4)
    #fb = f4 + f2 + f3
    #fd = f - fb
    #b = fd - (f0 + f1)
    #print("error2:", assemble(b**2 * dx))
    return a, L


def _poisson(n, el_type, degree, perturb):
    mesh = UnitSquareMesh(2**n, 2**n)
    if perturb:
        V = FunctionSpace(mesh, mesh.coordinates.ufl_element())
        eps = Constant(1 / 2**(n+1))

        x, y = SpatialCoordinate(mesh)
        new = Function(V).interpolate(as_vector([x + eps*sin(8*pi*x)*sin(8*pi*y),
                                                 y - eps*sin(8*pi*x)*sin(8*pi*y)]))
        mesh = Mesh(new)

    # Rotate mesh (Currently only works with unrotated mesh)
    theta = pi / 6
    mesh = _rotate_mesh(mesh, theta)

    V = FunctionSpace(mesh, el_type, degree)
    x, y = SpatialCoordinate(mesh)

    # Rotate coordinates
    xprime = cos(theta) * x + sin(theta) * y
    yprime = -sin(theta) * x + cos(theta) * y

    # Define forms
    
    solver_parameters = {"ksp_rtol": 1.e-18}
    f = Function(V).project(_poisson_analytical(V, xprime, yprime, 'force'), solver_parameters=solver_parameters)
    g = Function(V).project(_poisson_analytical(V, xprime, yprime, 'solution'), solver_parameters=solver_parameters)
    #a, L = _poisson_get_forms_original(V, f, n)
    a, L = _poisson_get_forms_hermite(V, xprime, yprime, g, f)
    # Solve
    sol = Function(V)
    solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'preonly', "pc_type": 'lu'})
    #solve(a == L, sol, bcs=[], solver_parameters={"ksp_type": 'gmres',
    #                                              "ksp_rtol": 1.e-12,
    #                                              "ksp_atol": 1.e-12})

    # Postprocess
    sol_exact = _poisson_analytical(V, xprime, yprime, 'solution')
    sol_exact_proj = Function(V).project(_poisson_analytical(V, xprime, yprime, 'solution'), solver_parameters=solver_parameters)
    err = sqrt(assemble(dot(sol - sol_exact, sol - sol_exact) * dx))
    berr = sqrt(assemble(dot(sol - sol_exact_proj, sol - sol_exact_proj) * ds))
    print("error            : ", err)
    print("error on boundary: ", berr)
    """
    # Plot solution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=1, sharex=True, sharey=True)
    #triangles = tripcolor(sol, axes=axes, cmap='coolwarm')
    triangles = tripcolor(Function(V).interpolate(sol_exact), axes=axes, cmap='coolwarm')
    axes.set_aspect("equal")
    axes.set_title("Pressure")
    fig.colorbar(triangles, ax=axes, fraction=0.032, pad=0.02)

    plt.savefig('temphermite.pdf')
    """
    return err, berr


def test_subspace_transformedsubspace_poisson_zany():
    """
    for el, deg, convrate in [('CG', 3, 4),
                              ('CG', 4, 5),
                              ('CG', 5, 6)]:
    for el, deg, convrate in [('Hermite', 3, 3.8),
                              ('Bell', 5, 4.8),
                              ('Argyris', 5, 4.8)]:
        diff = np.array([poisson(i, el, deg, True) for i in range(3, 8)])
        conv = np.log2(diff[:-1] / diff[1:])
        print(conv)
        #assert (np.array(conv) > convrate).all()
    """
    for el, deg, convrate in [('Hermite', 3, 3.8),]:
        errs = []
        for i in range(4, 8):
            err, berr = _poisson(i, el, deg, True)
            errs.append(err)
            assert(berr < 2e-8)
        errs = np.array(errs)
        conv = np.log2(errs[:-1] / errs[1:])
        print(conv)
        assert (np.array(conv) > convrate).all()
