from firedrake import *


def test_bc_on_sub_sub_domain():

    # Solve a vector poisson problem

    mesh = UnitSquareMesh(50, 50)

    V = VectorFunctionSpace(mesh, "CG", 1)
    VV = MixedFunctionSpace([V, V])

    x, y = SpatialCoordinate(mesh)

    f = Function(V)
    f.interpolate(as_vector([-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2),
                             -8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2)]))

    # interpolate the exact solution on gg[i][j]
    gg = [[None, None], [None, None]]
    for i in [0, 1]:
        for j in [0, 1]:
            gg[i][j] = Function(VV.sub(i).sub(j))
            gg[i][j].interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    uu = Function(VV)
    vv = TestFunction(VV)

    F = 0
    for u, v in zip(split(uu), split(vv)):
        F += (- inner(grad(u), grad(v)) - inner(f, v)) * dx

    bcs = [DirichletBC(VV.sub(0).sub(0), gg[0][0], 1),
           DirichletBC(VV.sub(0).sub(1), gg[0][1], 2),
           DirichletBC(VV.sub(1).sub(0), gg[1][0], 3),
           DirichletBC(VV.sub(1).sub(1), gg[1][1], "on_boundary")]

    parameters = {"mat_type": "nest",
                  "snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "fieldsplit",
                  "pc_fieldsplit_type": "additive",
                  "fieldsplit_ksp_type": "preonly",
                  "fieldsplit_pc_type": "lu"}

    solve(F == 0, uu, bcs=bcs, solver_parameters=parameters)

    # interpolate the exact solution on f
    f.interpolate(as_vector([cos(2 * pi * x) * cos(2 * pi * y),
                             cos(2 * pi * x) * cos(2 * pi * y)]))

    assert sqrt(assemble(dot(uu.subfunctions[0] - f, uu.subfunctions[0] - f) * dx)) < 4.0e-03
    assert sqrt(assemble(dot(uu.subfunctions[1] - f, uu.subfunctions[1] - f) * dx)) < 4.0e-03
