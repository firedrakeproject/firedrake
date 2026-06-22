from firedrake import *
import pytest
import numpy as np


@pytest.fixture(params=(2, 3))
def mh(request):
    dim = request.param
    N_base = 2
    if dim == 2:
        refine = 3
        msh = UnitSquareMesh(N_base, N_base)
    elif dim == 3:
        refine = 2
        msh = UnitCubeMesh(N_base, N_base, N_base)
    else:
        raise ValueError("Unexpected dimension")
    mh = MeshHierarchy(msh, refine)

    V = FunctionSpace(msh, msh.coordinates.ufl_element())
    eps = Constant(1 / 2**(N_base-1))
    x, y, *z = SpatialCoordinate(msh)

    new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                             y - eps*sin(2*pi*x)*sin(2*pi*y), *z]))

    # And propagate to refined meshes
    coords = [new]
    for msh in mh[1:]:
        fine = Function(msh.coordinates.function_space())
        prolong(new, fine)
        coords.append(fine)
        new = fine

    for msh, coord in zip(mh, coords):
        msh.coordinates.assign(coord)
    return mh


def mesh_sizes(mh):
    mesh_size = []
    for msh in mh:
        DG0 = FunctionSpace(msh, "DG", 0)
        h = Function(DG0).interpolate(CellDiameter(msh))
        with h.dat.vec as hvec:
            _, maxh = hvec.max()
        mesh_size.append(maxh)
    return mesh_size


def convergence_orders(error, h):
    return np.diff(np.log2(error)) / np.diff(np.log2(h))


def test_mtw_darcy_convergence(mh):
    sp = {
        "ksp_monitor": None,
        "mat_type": "matfree",
        "pmat_type": "nest",
        "ksp_type": "minres",
        "ksp_norm_type": "preconditioned",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_1_pc_type": "jacobi",
    }
    gamma = Constant(1E4)
    l2_u = []
    l2_p = []
    for msh in mh[1:]:
        x, y, *z = SpatialCoordinate(msh)
        pex = sin(pi * x) * sin(2 * pi * y)
        if z:
            pex *= sin(pi*z[0])

        uex = -grad(pex)
        f = div(uex)

        V = FunctionSpace(msh, "MTW", 1)
        Q = FunctionSpace(msh, "DG", 0)
        Z = V * Q

        u, p = TrialFunctions(Z)
        v, q = TestFunctions(Z)

        a = -inner(u, v) * dx + inner(p, div(v)) * dx + inner(div(u), q) * dx
        L = inner(f, q) * dx

        Jp = inner(u, v)*dx + inner(div(u), gamma*div(v))*dx + inner(p/gamma, q)*dx

        up = Function(Z)

        solve(a == L, up, Jp=Jp, solver_parameters=sp)

        u, p = up.subfunctions
        l2_u.append(errornorm(uex, u))
        l2_p.append(errornorm(pex, p))

    h = mesh_sizes(mh[1:])
    assert min(convergence_orders(l2_u, h)) > 1.75
    assert min(convergence_orders(l2_p, h)) > 0.8


def test_mtw_interior_facet():
    mesh = UnitSquareMesh(4, 4)
    eps = Constant(0.5 / 2**3)

    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                            y - eps*sin(2*pi*x)*sin(2*pi*y)]))

    V = FunctionSpace(mesh, 'Mardal-Tai-Winther', 1)

    uh = Function(V).interpolate(as_vector((x+y, 2*x-y)))

    volume = assemble(div(uh)*dx)

    n = FacetNormal(mesh)
    # Check form
    L = dot(uh, n)*ds + dot(uh('+'), n('+'))*dS + dot(uh('-'), n('-'))*dS
    surface = assemble(L)
    assert abs(volume - surface) < 1E-10

    # Check linear form
    v = TestFunction(V)
    L = inner(n, v)*ds + inner(n('+'), v('+'))*dS + inner(n('-'), v('-'))*dS
    b = assemble(L)

    with b.dat.vec_ro as L_vec:
        with uh.dat.vec_ro as uh_vec:
            surface = L_vec.dot(uh_vec)

    assert abs(volume - surface) < 1E-10

    Q = FunctionSpace(mesh, 'Discontinuous Lagrange', 0)
    q = TestFunction(Q)
    # Check bilinear linear form
    u = TrialFunction(V)
    a = (inner(dot(u, n), q)*ds
         + inner(dot(u('-'), n('-')), q('-'))*dS
         + inner(dot(u('+'), n('+')), q('+'))*dS)
    A = assemble(a).petscmat

    y = A.createVecLeft()
    with uh.dat.vec_ro as uh_vec:
        A.mult(uh_vec, y)
        surface = y.sum()

    assert abs(volume - surface) < 1E-10
