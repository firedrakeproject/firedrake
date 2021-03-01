from firedrake import *
import pytest
import numpy


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
    for n in [8, 16, 32, 64]:
        mesh = UnitSquareMesh(n, n)

        V = FunctionSpace(mesh, *hdiv)
        Q = FunctionSpace(mesh, *l2)
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
        h = CellSize(mesh)
        sigma = 10.0

        # Manually specify integration degree due to non-polynomial
        # source terms.
        a = (inner(grad(u), grad(v)) - inner(p, div(v)) - inner(div(u), q)) * dx(degree=6)

        a += (- inner(avg(grad(u)), outer(jump(conj(v)), n("+")))
              - inner(outer(jump(conj(u)), n("+")), avg(grad(v)))
              + (sigma/avg(h)) * inner(jump(u), jump(v))) * dS(degree=6)

        a += (- inner(grad(u), outer(conj(v), n))
              - inner(outer(conj(u), n), grad(v))
              + (sigma/h) * inner(u, v)) * ds(degree=6)

        L = (inner(source, v) * dx(degree=6)
             + (sigma/h) * inner(u_exact, v) * ds(degree=6)
             - inner(outer(conj(u_exact), n), grad(v)) * ds(degree=6))

        # left = 1
        # right = 2
        # bottom = 3
        # top = 4
        bcfunc_left = Function(V).project(as_vector([u_exact[0], 0]))
        bcfunc_right = Function(V).project(as_vector([u_exact[0], 0]))
        bcfunc_bottom = Function(V).project(as_vector([0, u_exact[1]]))
        bcfunc_top = Function(V).project(as_vector([0, u_exact[1]]))
        bcs = [DirichletBC(W.sub(0), bcfunc_left, 1),
               DirichletBC(W.sub(0), bcfunc_right, 2),
               DirichletBC(W.sub(0), bcfunc_bottom, 3),
               DirichletBC(W.sub(0), bcfunc_top, 4)]

        UP = Function(W)

        nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

        parameters = {
            "mat_type": mat_type,
            "pmat_type": "matfree",
            "ksp_type": "fgmres",
            "ksp_max_it": "30",
            "ksp_atol": "1.e-16",
            "ksp_rtol": "1.e-11",
            "ksp_monitor_true_residual": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "multiplicative",

            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                # Avoid MUMPS segfaults
                "assembled_pc_type": "redundant",
                "assembled_redundant_pc_type": "lu",
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.MassInvPC",
                # Avoid MUMPS hangs?
                "Mp_ksp_type": "preonly",
                "Mp_pc_type": "redundant",
                "Mp_redundant_pc_type": "lu",
            }
        }

        # Switch sign of pressure mass matrix
        mu = Constant(-1.0)
        appctx = {"mu": mu, "pressure_space": 1}

        UP.assign(0)
        solve(a == L, UP, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
              appctx=appctx)

        u, p = UP.split()
        u_error = u - u_exact
        p_error = p - p_exact
        err_u.append(sqrt(abs(assemble(inner(u_error, u_error) * dx))))
        err_p.append(sqrt(abs(assemble(inner(p_error, p_error) * dx))))
        err_div.append(sqrt(assemble(inner(div(u), div(u)) * dx)))
    err_u = numpy.asarray(err_u)
    err_p = numpy.asarray(err_p)
    err_div = numpy.asarray(err_div)

    assert numpy.allclose(err_div, 0, atol=1e-7, rtol=1e-5)
    assert (numpy.log2(err_u[:-1] / err_u[1:]) > 2.8).all()
    assert (numpy.log2(err_p[:-1] / err_p[1:]) > 1.8).all()
