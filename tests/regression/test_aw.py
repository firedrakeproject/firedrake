from firedrake import *
import pytest
import numpy as np
import scipy
from petsc4py import PETSc
from slepc4py import SLEPc

#convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])
# take the log of 1/the below
relative_magnitudes = lambda x: np.array(x)[1:] / np.array(x)[:-1]
convergence_orders = lambda x: -np.log2(relative_magnitudes(x))

@pytest.fixture(scope='module', params=["conforming", "nonconforming"])
def stress_element(request):
    if request.param == "conforming":
        return FiniteElement("AWc", triangle, 3)
    elif request.param == "nonconforming":
        return FiniteElement("AWnc", triangle, 2)
    else:
        raise ValueError("Unknown family")


def test_aw(stress_element):
    green = '\033[92m'
    white = '\033[0m'
    blue = '\033[94m'

    N_base = 2
    mesh = UnitSquareMesh(N_base, N_base)
    mh = MeshHierarchy(mesh, 4)

    V = FunctionSpace(mesh, mesh.coordinates.ufl_element())

    # Warp the meshes
    eps = Constant(1 / 2**(N_base-1))
    x, y = SpatialCoordinate(mesh)
    new = Function(V).interpolate(as_vector([x + eps*sin(2*pi*x)*sin(2*pi*y),
                                             y - eps*sin(2*pi*x)*sin(2*pi*y)]))
    coords = [new]
    for mesh in mh[1:]:
        fine = Function(mesh.coordinates.function_space())
        prolong(new, fine)
        coords.append(fine)
        new = fine
    for mesh, coord in zip(mh, coords):
        mesh.coordinates.assign(coord)

    nu = Constant(0.25)
    lam = Constant(1)
    mu = lam*(1 - 2*nu)/(2*nu)

    I = Identity(2)

    # Evaluation of a constant compliance tensor
    # (in the homogeneous isotropic case)
    def A(sig):
        return (1/(2*mu))*(sig - nu*tr(sig)*I)

    # Linearised strain rate tensor
    def epsilon(u):
        return sym(grad(u))

    l2_u = []
    l2_sigma = []
    l2_div_sigma = []
    mass_cond = []

    element = MixedElement([stress_element, VectorElement("DG", mesh.ufl_cell(), 1)])
    for msh in mh[1:]:
        x, y = SpatialCoordinate(msh)
        uex = as_vector([sin(pi*x)*sin(pi*y), sin(pi*x)*sin(pi*y)])
        sigex = as_tensor([[cos(pi*x)*cos(3*pi*y), y + 2*cos(pi*x/2)],
                           [y + 2*cos(pi*x/2), -sin(3*pi*x)*cos(2*pi*x)]])

        # Compliance and constraint residuals of the MMS
        comp_r = A(sigex) - epsilon(uex)
        cons_r = div(sigex)

        V = FunctionSpace(msh, element)

        Uh = Function(V)

        (sigh, uh) = split(Uh)
        (tau, v) = TestFunctions(V)

        n = FacetNormal(msh)

        # Hellinger--Reissner residual, incorporating MMS residuals
        F = (
            + inner(A(sigh), tau)*dx
            + inner(uh, div(tau))*dx
            + inner(div(sigh), v)*dx
            - inner(comp_r, tau)*dx
            - inner(cons_r, v)*dx
            - inner(uex, dot(tau, n))*ds
            )  # noqa: E123

        params = {"snes_type": "newtonls",
                  "snes_linesearch_type": "basic",
                  "snes_monitor": None,
                  "mat_type": "aij",
                  "snes_max_it": 10,
                  "snes_lag_jacobian": -2,
                  "snes_lag_preconditioner": -2,
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_shift_type": "inblocks",
                  "snes_rtol": 1e-16,
                  "snes_atol": 1e-25}

        solve(F == 0, Uh, solver_parameters=params)

        error_u = sqrt(assemble(inner(uex - uh, uex - uh)*dx))
        error_sigma = sqrt(assemble(inner(sigh - sigex, sigh - sigex)*dx))
        error_div_sigma = sqrt(assemble(inner(div(sigh - sigex), div(sigh - sigex))*dx))

        l2_u.append(error_u)
        l2_sigma.append(error_sigma)
        l2_div_sigma.append(error_div_sigma)

        Sig = FunctionSpace(msh, stress_element)
        sigh = TrialFunction(Sig)
        tau = TestFunction(Sig)
        mass = inner(sigh, tau)*dx
        #a = derivative(mass, sigh)
        #B = assemble(a, mat_type="aij").M.handle
        #nrow = B.getSize()[0]
        #ai, aj, av = B.getValuesCSR()
        #Asp = scipy.sparse.csr_matrix((av, aj, ai))
        #nnz = Asp.nnz
        #nrows = Asp.shape[0]
        #kappa = np.linalg.cond(Asp.todense())

        #B = assemble(a)
        #kappa = np.linalg.cond(A.array())

        solver_parameters = {
                 "mat_type": "aij",
                 "eps_converged_reason": None,
                 "eps_type": "krylovschur",
                 "eps_nev" : 10,
                 "eps_max_it": 100,
                 "eps_tol" : 1e-15,
                 "eps_target" : 0,
                 "st_type": "sinvert",
                 "st_ksp_type": "preonly",
                 "st_pc_type": "lu",
                 "st_pc_factor_mat_solver_type": "mumps",
                 "st_ksp_max_it": 10,
                 "ds_parallel": "synchronized"
                 }
        opts = PETSc.Options()
        for k in solver_parameters:
            opts[k] = solver_parameters[k]
        M = assemble(mass, mat_type="aij")
        comm = V.mesh().comm
        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(M.M.handle)
        eps.setProblemType(eps.ProblemType.HEP)
        eps.setFromOptions()
        eps.solve()
        min_lam = eps.getEigenvalue(0)
        print(min_lam)
        
        Opts = PETSc.Options()
        solver_parameters["eps_target"] = 300
        for k in solver_parameters:
            Opts[k] = solver_parameters[k]
        Eps = SLEPc.EPS().create(comm=comm)
        Eps.setOperators(M.M.handle)
        Eps.setProblemType(eps.ProblemType.HEP)
        Eps.setFromOptions()
        Eps.solve()
        max_lam = Eps.getEigenvalue(Eps.getConverged() - 1)
        print(max_lam)
        kappa = abs(max_lam / min_lam)
        print(kappa)
        print()

        mass_cond.append(kappa)

    print(relative_magnitudes(mass_cond))
    assert max(relative_magnitudes(mass_cond)) < 1.1

    if stress_element.family().startswith("Conforming"):
        assert min(convergence_orders(l2_u)) > 1.9
        assert min(convergence_orders(l2_sigma)) > 2.9
        assert min(convergence_orders(l2_div_sigma)) > 1.9
    elif stress_element.family().startswith("Nonconforming"):
        assert min(convergence_orders(l2_u)) > 1.9
        assert min(convergence_orders(l2_sigma)) > 1
        assert min(convergence_orders(l2_div_sigma)) > 1.9
    else:
        raise ValueError("Don't know what the convergence should be")

test_aw(FiniteElement("AWc", triangle, 3))
test_aw(FiniteElement("AWnc", triangle, 2))
