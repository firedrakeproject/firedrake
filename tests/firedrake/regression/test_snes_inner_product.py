import numpy as np
import pytest
from firedrake import *


def poisson(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    u = Function(V)
    v = TestFunction(V)
    w = TrialFunction(V)
    x, y = SpatialCoordinate(mesh)
    F = inner((1 + u**2) * grad(u), grad(v)) * dx - inner(x * y, v) * dx
    bcs = DirichletBC(V, x + y, "on_boundary")
    H1 = inner(w, v) * dx + inner(grad(w), grad(v)) * dx
    return F, u, bcs, H1


def dual_norm(F, bcs, M):
    """Return the norm sqrt(r^T M^{-1} r) of the residual r of F, without the rows of the Dirichlet nodes."""
    V = F.arguments()[0].function_space()
    r = assemble(F)
    bcs.zero(r)
    y = Function(V)
    if M is None:
        y.dat.data_wo[:] = r.dat.data_ro
    else:
        solve(M == r, y, bcs=homogenize(bcs), solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    with r.dat.vec_ro as rvec, y.dat.vec_ro as yvec:
        return np.sqrt(rvec.dot(yvec))


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("inner_product", ["l2", "H1", "form", "slate", "matrix", "riesz_map"])
def test_snes_inner_product(inner_product):
    """The residual norms of NGMRES are the dual norms in the inner product from the snes_inner_product option."""
    mesh = UnitSquareMesh(4, 4)
    F, u, bcs, H1 = poisson(mesh)
    V = u.function_space()
    M = None if inner_product == "l2" else H1
    value = {
        "l2": "l2",
        "H1": "H1",
        "form": H1,
        "slate": Tensor(H1),
        "matrix": assemble(H1, bcs=homogenize(bcs)),
        "riesz_map": RieszMap(V, "H1"),
    }[inner_product]

    bcs.apply(u)
    expected = dual_norm(F, bcs, M)

    solver_parameters = {
        "snes_type": "ngmres",
        "snes_max_it": 1,
        "snes_inner_product": value,
        "snes_innerproduct_ksp_type": "preonly",
        "snes_innerproduct_pc_type": "lu",
        "npc_snes_type": "ksponly",
        "npc_ksp_type": "preonly",
        "npc_pc_type": "lu",
    }
    solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, bcs),
                                        solver_parameters=solver_parameters)
    solver.snes.setConvergenceHistory()
    try:
        solver.solve()
    except ConvergenceError:
        pass
    norms = solver.snes.getConvergenceHistory()[0]
    assert np.isclose(norms[0], expected, rtol=1E-12)
    assert (solver.snes.getInnerProductMat()[0] is None) == (M is None)


def test_snes_inner_product_shared_dm():
    """A solver without the snes_inner_product option uses the Euclidean inner product,
    after another solver on the same function space has used snes_inner_product."""
    mesh = UnitSquareMesh(4, 4)
    F, u, bcs, H1 = poisson(mesh)
    solver_parameters = {
        "snes_type": "ngmres",
        "snes_max_it": 1,
        "npc_snes_type": "ksponly",
        "npc_ksp_type": "preonly",
        "npc_pc_type": "lu",
    }
    norms = []
    for inner_product in ["H1", None]:
        parameters = dict(solver_parameters)
        if inner_product is not None:
            parameters["snes_inner_product"] = inner_product
        u.zero()
        bcs.apply(u)
        solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, bcs),
                                            solver_parameters=parameters)
        solver.snes.setConvergenceHistory()
        try:
            solver.solve()
        except ConvergenceError:
            pass
        norms.append(solver.snes.getConvergenceHistory()[0][0])

    u.zero()
    bcs.apply(u)
    assert np.isclose(norms[0], dual_norm(F, bcs, H1), rtol=1E-12)
    assert np.isclose(norms[1], dual_norm(F, bcs, None), rtol=1E-12)
    assert solver.snes.getInnerProductMat()[0] is None


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_snes_inner_product_adapt():
    """After each adaptive refinement, the SNES assembles the inner product on the refined mesh."""
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    mesh = Mesh(geo.GenerateMesh(maxh=0.5))
    MeshHierarchy(mesh)
    F, u, bcs, H1 = poisson(mesh)

    def mark_cells(ctx, current_solution):
        markers = Function(FunctionSpace(current_solution.function_space().mesh(), "DG", 0))
        return markers.assign(1)

    refinements = 2
    solver_parameters = {
        "snes_adapt_sequence": refinements,
        "snes_type": "ngmres",
        "snes_rtol": 1E-8,
        "snes_inner_product": "H1",
        "snes_innerproduct_ksp_type": "preonly",
        "snes_innerproduct_pc_type": "lu",
        "npc_snes_type": "ksponly",
        "npc_ksp_type": "preonly",
        "npc_pc_type": "lu",
    }
    solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, bcs),
                                        solver_parameters=solver_parameters,
                                        marking_callback=mark_cells)
    u_adapted = solver.solve()
    adapted_mesh = u_adapted.function_space().mesh()
    assert adapted_mesh is not mesh

    # The SNES created and assembled the matrix of the inner product with the context on the adapted mesh
    M = solver.snes.getInnerProductMat()[0]
    assert M.getSize() == (u_adapted.function_space().dim(),) * 2
    assert M.handle == solver._ctx._inner_product.petscmat.handle
