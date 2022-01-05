from firedrake import *
import pytest
from firedrake.petsc import PETSc
from mpi4py import MPI
PETSc.Sys.popErrorHandler()

def clean():
    from tempfile import gettempdir
    import os
    os.system('firedrake-clean')
    os.system('rm -r ' + gettempdir() +'/pyop2-cache-*')
    from pyop2.op2 import exit
    exit()
    
def run_gtmg_mixed_poisson():

    n = 3
    L = 3
    m = SquareMesh(n, n, L, quadrilateral=True)
    levels = 2
    basemh = MeshHierarchy(m, levels)
    mh = ExtrudedMeshHierarchy(basemh, L, base_layer=n)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def get_p1_prb_bcs():
        return [DirichletBC(get_p1_space(), Constant(0.0), "on_boundary"),
                DirichletBC(get_p1_space(), Constant(0.0), "top"),
                DirichletBC(get_p1_space(), Constant(0.0), "bottom")]

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return inner(grad(p), grad(q))*dx

    degree = 3
    RT = FiniteElement("RTCF", quadrilateral, degree)
    DG_v = FiniteElement("DG", interval, degree-1)
    DG_h = FiniteElement("DQ", quadrilateral, degree-1)
    CG = FiniteElement("CG", interval, degree)
    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DQ", degree-1)
    W = V * U

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    f = Function(U)
    uex = x[0]*(L-x[0])*x[1]*(L-x[1])*x[2]*(L-x[2])
    f = -div(grad(uex))

    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v))*dx
    L = -inner(f, v)*dx

    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}

    # Test accuracy
    # w = Function(W)
    # params = {'mat_type': 'matfree',
    #           'ksp_type': 'preonly',
    #           'pc_type': 'python',
    #           'pc_python_type': 'firedrake.HybridizationPC',
    #           'hybridization': {'ksp_type': 'cg',
    #                             'mat_type': 'matfree',
    #                             'pc_type': 'python',
    #                             'pc_python_type': 'firedrake.GTMGPC',
    #                             'gt': {'mg_levels': {'ksp_type': 'chebyshev',
    #                                                  'pc_type': 'jacobi',
    #                                                  'ksp_max_it': 3},
    #                                    'mg_coarse': {'ksp_type': 'preonly',
    #                                                  'pc_type': 'mg',
    #                                                  'pc_mg_type': 'full',
    #                                                  'mg_levels': {'ksp_type': 'chebyshev',
    #                                                                'pc_type': 'jacobi',
    #                                                                'ksp_max_it': 3}}}}}

    # solve(a == L, w, solver_parameters=params, appctx=appctx)
    # _, uh = w.split()

    # w_ref = Function(W)
    # ref_params = {'ksp_type': 'gmres',
    #               'pc_type': 'ilu',
    #               "ksp_gmres_restart": 100,
    #               'ksp_rtol': 1.e-12}
    # solve(a == L, w_ref, solver_parameters=ref_params)
    # _, uh_ref = w_ref.split()
    # print("Error of GTMG to LU solution:", errornorm(uh, uh_ref, norm_type="L2"))
    
    # # Analytical solution
    # analytical = Function(U).project(uex)
    # e_analytical = errornorm(analytical, uh, norm_type="L2")
    # print("Error of GTMG to analytical solution:", e_analytical)
    # print("Error of LU to analytical solution:", errornorm(analytical, uh_ref, norm_type="L2"))

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(uh.dat.data, label="uh")
    # plt.plot(analytical.dat.data, label="f")
    # plt.plot(uh_ref.dat.data, label="uh_ref")
    # plt.legend()
    # plt.show()

    # Test that iterative and analytical solution are correct
    # import numpy as np
    # x = SpatialCoordinate(mesh)
    # w_dc = Function(W).assign(w)
    # assert w_dc != w, "Make sure we don't modify w"
    # w_dc.sub(1).project(uex)
    # A = Tensor(a)
    # B = AssembledVector(w_dc)
    # dat1 = assemble(A*B).dat.data[1]
    # dat2 = assemble(-f*v*dx).dat.data[1]
    # assert np.allclose(dat1, dat2), "Analytical solution is not correct"
    # B = AssembledVector(w)
    # dat1 = assemble(A*B).dat.data[1]
    # assert np.allclose(dat1, dat2, rtol=1.e-4), "Iterative solution is not correct"

    # Test that iterative and analytical solution are the same
    # assert np.allclose(w_dc.dat.data[0], w.dat.data[0], rtol=1.e-4), "There is a difference in the solution of the velocity"
    # assert np.allclose(w_dc.dat.data[1], w.dat.data[1], rtol=1.e-4), "There is a difference in the solution of the pressure"
    
    # Time this example 
    base_params = {'mat_type': 'matfree',
                    'ksp_type': 'preonly',
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.HybridizationPC',
                    'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'python',
                                        'pc_python_type': 'firedrake.GTMGPC',
                                        'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                            'pc_type': 'none',
                                                            'ksp_max_it': 3},
                                            'mg_coarse': {'ksp_type': 'preonly',
                                                            'pc_type': 'mg',
                                                            'pc_mg_type': 'full',
                                                            'mg_levels': {'ksp_type': 'chebyshev',
                                                                        'pc_type': 'jacobi',
                                                                        'ksp_max_it': 3}}}}}
    # setup test
    perform_params = {'mat_type': 'matfree',
                     'ksp_type': 'preonly',
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.HybridizationPC',
                     'hybridization': {'ksp_type': 'cg',
                                       'pc_type': 'python',
                                       'mat_type': 'matfree',  # only difference!
                                       'pc_python_type': 'firedrake.GTMGPC',
                                       'localsolve': {'ksp_type': 'preonly',
                                                        'pc_type': 'fieldsplit',
                                                        'pc_fieldsplit_type': 'schur'},
                                       'gt': {'mat_type': 'matfree',  # only difference!
                                              'mg_levels': {'ksp_type': 'chebyshev',
                                                            'pc_type': 'none',
                                                            'ksp_max_it': 3},
                                              'mg_coarse': {'ksp_type': 'preonly',
                                                            "mat_type": "aij",
                                                            'pc_type': 'python',
                                                            'pc_python_type': "firedrake.AssembledPC",
                                                            'assembled_pc_type': 'mg',
                                                            'assembled_pc_mg_type': 'full',
                                                            'assembled_pc_mg_levels': {'ksp_type': 'chebyshev',
                                                                                       'pc_type': 'jacobi',
                                                                                       'ksp_max_it': 3}}}}}
    clean()
    # PETSc.Log.begin()
    # with PETSc.Log.Stage("warmup"):
    #     w = Function(W)
    #     solve(a == L, w, solver_parameters=base_params, appctx=appctx)
    #     trace_solve_time_expl_warm = (mesh.comm.allreduce(PETSc.Log.Event("SCSolve").getPerfInfo()["time"],
    #                                             op=MPI.SUM) / mesh.comm.size)

    # with PETSc.Log.Stage("warmedup"):
    #     w = Function(W)
    #     solve(a == L, w, solver_parameters=base_params, appctx=appctx)
    #     trace_solve_time_expl = (mesh.comm.allreduce(PETSc.Log.Event("SCSolve").getPerfInfo()["time"],
    #                                             op=MPI.SUM) / mesh.comm.size)
    # print("Timing for matrix-explicit GTMG:",trace_solve_time_expl)
    # print("Timing for matrix-explicit GTMG warm:",trace_solve_time_expl_warm)
    clean()
    with PETSc.Log.Stage("warmup2"):
        w = Function(W)
        solve(a == L, w, solver_parameters=perform_params, appctx=appctx)
        trace_solve_time_matf_warm = (mesh.comm.allreduce(PETSc.Log.Event("SCSolve").getPerfInfo()["time"],
                                                op=MPI.SUM) / mesh.comm.size)

    # with PETSc.Log.Stage("warmedup2"):
    #     w = Function(W)
    #     solve(a == L, w, solver_parameters=perform_params, appctx=appctx)
    #     trace_solve_time_matf = (mesh.comm.allreduce(PETSc.Log.Event("SCSolve").getPerfInfo()["time"],
    #                                             op=MPI.SUM) / mesh.comm.size)
    u, p = w.split()
    print("DOFS", u.dof_dset.layout_vec.getSize()+p.dof_dset.layout_vec.getSize())
    
    # print("Timing for matrix-free GTMG:",trace_solve_time_matf)
    print("Timing for matrix-free GTMG warm:",trace_solve_time_matf_warm)
    
    return e_analytical


def run_gtmg_scpc_mixed_poisson():

    m = UnitSquareMesh(10, 10)
    nlevels = 2
    mh = MeshHierarchy(m, nlevels)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def get_p1_prb_bcs():
        return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return inner(grad(p), grad(q))*dx

    degree = 1
    n = FacetNormal(mesh)
    U = FunctionSpace(mesh, "DRT", degree)
    V = FunctionSpace(mesh, "DG", degree - 1)
    T = FunctionSpace(mesh, "DGT", degree - 1)
    W = U * V * T

    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    f = Function(V)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau)*dx - inner(u, div(tau))*dx
         + inner(div(sigma), v)*dx
         + inner(lambdar('+'), jump(tau, n=n))*dS
         # Multiply transmission equation by -1 to ensure
         # SCPC produces the SPD operator after statically
         # condensing
         - inner(jump(sigma, n=n), gammar('+'))*dS)
    L = inner(f, v)*dx

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.SCPC',
              'pc_sc_eliminate_fields': '0, 1',
              'condensed_field': {'ksp_type': 'cg',
                                  'mat_type': 'matfree',
                                  'pc_type': 'python',
                                  'pc_python_type': 'firedrake.GTMGPC',
                                  'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                       'pc_type': 'jacobi',
                                                       'ksp_max_it': 3},
                                         'mg_coarse': {'ksp_type': 'preonly',
                                                       'pc_type': 'mg',
                                                       'pc_mg_type': 'full',
                                                       'mg_levels': {'ksp_type': 'chebyshev',
                                                                     'pc_type': 'jacobi',
                                                                     'ksp_max_it': 3}}}}}
    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}

    bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

    solve(a == L, w, bcs=bcs, solver_parameters=params, appctx=appctx)
    _, uh, _ = w.split()

    # Analytical solution
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    return errornorm(f, uh, norm_type="L2")


@pytest.mark.skipcomplexnoslate
def test_mixed_poisson_gtmg():
    assert run_gtmg_mixed_poisson() < 1e-4


@pytest.mark.skipcomplexnoslate
def test_scpc_mixed_poisson_gtmg():
    assert run_gtmg_scpc_mixed_poisson() < 1e-5

test_mixed_poisson_gtmg()