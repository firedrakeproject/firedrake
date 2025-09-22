import pytest
import numpy
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS


class SchurPC(AuxiliaryOperatorPC):
    def form(self, pc, V, U):
        u, p = split(U)
        v, q = split(V)
        A = inner(grad(u), grad(v)) * dx + inner(p, q) * dx
        bcs = (DirichletBC(V.function_space().sub(0), 0, "on_boundary"), )
        return (A, bcs)


@pytest.fixture(params=["flipped", "normal"])
def elim_order(request):
    return request.param


@pytest.fixture
def permute(elim_order):
    if elim_order == "normal":
        return lambda v, q, w, r: (v, q, w, r)
    else:
        return lambda v, q, w, r: (w, r, v, q)


@pytest.fixture
def solver_parameters(permute):
    fields = permute("0", "1", "2", "3")
    # normal order is V*Q*W*R, eliminate W*R onto V*Q
    A = ",".join(fields[2:])
    B = ",".join(fields[:2])
    schurlu = {
        "ksp_type": "gmres",
        "pc_type": "python",
        "pc_python_type": f"{__name__}.SchurPC",
        "aux_pc_mat_type": "aij",
        "aux_pc_type": "lu",
        "aux_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS
    }

    return {
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "snes_view": None,
        "mat_type": "aij",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_0_fields": A,
        "pc_fieldsplit_1_fields": B,
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
        "fieldsplit_1": schurlu,
    }


@pytest.fixture
def mesh():
    return UnitCubeMesh(2, 2, 2)


@pytest.fixture
def Z(mesh, permute):
    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "DG", 1)
    W = FunctionSpace(mesh, "N1div", 1)
    R = FunctionSpace(mesh, "N1curl", 1)
    return MixedFunctionSpace(permute(V, Q, W, R))


@pytest.fixture
def solution(mesh):
    x, y, z = SpatialCoordinate(mesh)

    u_ex = as_vector([sin(x), cos(y), sin(z)])
    p_ex = exp(x)*y
    B_ex = as_vector([exp(x*y), sin(z)*y, cos(z)])
    E_ex = as_vector([sin(x)-y, cos(z*x), sin(x)*y])
    return u_ex, p_ex, B_ex, E_ex


@pytest.fixture
def solver(Z, permute, solution, solver_parameters):
    sol = Function(Z)
    u, p, B, E = permute(*split(sol))
    v, q, C, f = permute(*split(TestFunction(Z)))

    F = (inner(grad(u), grad(v)) * dx
         + inner(p, q) * dx
         + inner(B, C) * dx
         + inner(div(B), div(C)) * dx
         + inner(E, f) * dx
         + inner(curl(E), curl(f)) * dx)

    u_ex, p_ex, B_ex, E_ex = solution
    f1 = -div(grad(u_ex))
    f2 = p_ex
    f3 = B_ex - grad(div(B_ex))
    f4 = E_ex + curl(curl(E_ex))

    F -= inner(f1, v)*dx + inner(f2, q)*dx + inner(f3, C)*dx + inner(f4, f)*dx
    i0, _, i2, i3 = permute(0, 1, 2, 3)
    bcs = [DirichletBC(Z.sub(i0), u_ex, "on_boundary"),
           DirichletBC(Z.sub(i2), B_ex, "on_boundary"),
           DirichletBC(Z.sub(i3), E_ex, "on_boundary")]

    problem = NonlinearVariationalProblem(F, sol, bcs)
    return NonlinearVariationalSolver(problem,
                                      solver_parameters=solver_parameters)


def run(solver, solution, permute):
    u_ex, p_ex, B_ex, E_ex = solution
    solver.solve()
    sol = solver._problem.u
    errors = tuple(errornorm(expect, actual, 'L2') for
                   expect, actual in zip(solution, permute(*sol.subfunctions)))
    diff = numpy.abs(errors - numpy.asarray([0.02551217479,
                                             0.01991075140,
                                             0.22550499155,
                                             0.17968476000]))
    assert all(diff < 1e-7)


def test_fieldsplit_split_reorder_bcs(solver, solution, permute):
    run(solver, solution, permute)


@pytest.mark.parallel(nprocs=2)
def test_fieldsplit_split_reorder_bcs_parallel(solver, solution, permute):
    run(solver, solution, permute)
