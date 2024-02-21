from firedrake import *
import pytest
import numpy


def outer_jump(v, n):
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))


@pytest.fixture(params=["aij", "nest", "matfree"])
def mat_type(request):
    return request.param


@pytest.fixture(params=[(("RT", 3), ("DG", 2)),
                        (("BDM", 2), ("DG", 1))],
                ids=["RT3-DG2", "BDM2-DG1"])
def element_pair(request):
    return request.param


@pytest.mark.parallel(nprocs=3)
def test_stokes_hdiv_parallel(mat_type, element_pair):
    err_u = []
    err_p = []
    err_div = []
    hdiv, l2 = element_pair
    hdiv_family, degree = hdiv
    for n in [8, 16, 32, 64]:
        mesh = UnitSquareMesh(n, n)

        V = FunctionSpace(mesh, hdiv_family, degree)
        Q = FunctionSpace(mesh, *l2, variant="integral")
        W = V * Q

        x, y = SpatialCoordinate(mesh)
        uxexpr = sin(pi*x)*sin(pi*y)
        uyexpr = cos(pi*x)*cos(pi*y)
        ppexpr = sin(pi*x)*cos(pi*y)
        srcxexpr = 2.0*pi*pi*sin(pi*x)*sin(pi*y) + pi*cos(pi*x)*cos(pi*y)
        srcyexpr = 2.0*pi*pi*cos(pi*x)*cos(pi*y) - pi*sin(pi*x)*sin(pi*y)

        u_exact = as_vector([uxexpr, uyexpr])
        p_exact = ppexpr
        source = as_vector([srcxexpr, srcyexpr])

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        n = FacetNormal(mesh)
        sigma = Constant(degree*(degree+1))
        penalty = sigma * FacetArea(mesh) / CellVolume(mesh)

        # Augmented Lagrangian penalty coefficient
        gamma = Constant(1000)

        # Manually specify integration degree due to non-polynomial
        # source terms.
        qdeg = 2 * degree
        a = (inner(grad(u), grad(v))
             + inner(div(u) * gamma - p, div(v))
             - inner(div(u), q)) * dx(degree=qdeg-2)

        a += (- inner(avg(grad(u)), outer_jump(conj(v), n))
              - inner(outer_jump(conj(u), n), avg(grad(v)))
              + avg(penalty) * inner(outer_jump(u, n), outer_jump(conj(v), n))) * dS(degree=qdeg)

        a += (- inner(grad(u), outer(conj(v), n))
              - inner(outer(conj(u), n), grad(v))
              + 2*penalty * inner(outer(u, n), outer(conj(v), n))) * ds(degree=qdeg)

        L = inner(source, v) * dx(degree=qdeg)
        L += (- inner(outer(conj(u_exact), n), grad(v))
              + 2*penalty * inner(outer(u_exact, n), outer(conj(v), n))) * ds(degree=qdeg)

        # left = 1
        # right = 2
        # bottom = 3
        # top = 4
        bcfunc_left = as_vector([u_exact[0], 0])
        bcfunc_right = as_vector([u_exact[0], 0])
        bcfunc_bottom = as_vector([0, u_exact[1]])
        bcfunc_top = as_vector([0, u_exact[1]])
        bcs = [DirichletBC(W.sub(0), bcfunc_left, 1),
               DirichletBC(W.sub(0), bcfunc_right, 2),
               DirichletBC(W.sub(0), bcfunc_bottom, 3),
               DirichletBC(W.sub(0), bcfunc_top, 4)]

        UP = Function(W)
        # Cannot set the nullspace with constant=True for non-Lagrange pressure elements
        nsp_basis = Function(Q).interpolate(Constant(1))
        nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis([nsp_basis])])

        parameters = {
            "mat_type": mat_type,
            "pmat_type": "matfree",
            "ksp_type": "minres",
            "ksp_norm_type": "preconditioned",
            "ksp_max_it": 10,
            "ksp_atol": "1.e-16",
            "ksp_rtol": "1.e-11",
            "ksp_monitor_true_residual": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                # Avoid MUMPS segfaults
                "assembled_pc_type": "redundant",
                "assembled_redundant_pc_type": "cholesky",
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.MassInvPC",
                "Mp_mat_type": "matfree",
                "Mp_pc_type": "jacobi",
            }
        }

        # Scale for the pressure mass matrix
        mu = 1/gamma
        appctx = {"mu": mu}

        UP.assign(0)
        solve(a == L, UP, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
              appctx=appctx)

        u, p = UP.subfunctions
        u_error = u - u_exact
        p_error = p - p_exact
        l2_norm = lambda z: sqrt(assemble(inner(z, z) * dx(degree=2*qdeg)))
        err_u.append(l2_norm(u_error))
        err_p.append(l2_norm(p_error))
        err_div.append(sqrt(assemble(inner(div(u), div(u)) * dx)))
    err_u = numpy.asarray(err_u)
    err_p = numpy.asarray(err_p)
    err_div = numpy.asarray(err_div)

    assert numpy.allclose(err_div, 0, atol=1e-7, rtol=1e-5)
    assert (numpy.log2(err_u[:-1] / err_u[1:]) > 2.8).all()
    assert (numpy.log2(err_p[:-1] / err_p[1:]) > 1.8).all()
