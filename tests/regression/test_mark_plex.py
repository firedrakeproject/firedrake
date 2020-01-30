from firedrake import *
from firedrake.cython import dmplex
import numpy


def mark_plex(dm):
    sec = dm.getCoordinateSection()
    coords = dm.getCoordinatesLocal()

    faces = dm.getStratumIS("interior_facets", 1).indices
    for face in faces:
        vertices = dm.vecGetClosure(sec, coords, face).reshape(2, 3)

        if numpy.allclose(vertices[:, 0], 1) and numpy.allclose(vertices[:, 1], 0):
            dm.setLabelValue(dmplex.FACE_SETS_LABEL, face, 3)  # colour 3 on the periodic seam
        else:
            dm.setLabelValue(dmplex.FACE_SETS_LABEL, face, 4)  # colour 4 on non-seam interior facets


def test_run_hcg():
    N = 4
    mesh = PeriodicUnitSquareMesh(N, N, direction="x", mark_plex=mark_plex)

    x = SpatialCoordinate(mesh)

    # symbolic expression for f
    u_e = as_vector([sin(pi*x[0])*sin(pi*x[1]), sin(pi*x[0])*sin(pi*x[1])])
    f = -div(grad(u_e))

    # set up variational problem
    V_element = FiniteElement('CG', mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, VectorElement(BrokenElement(V_element)))
    Qhat = FunctionSpace(mesh, VectorElement(BrokenElement(V_element[facet])))
    Vhat = FunctionSpace(mesh, VectorElement(V_element[facet]))
    W = V * Qhat * Vhat

    z = Function(W)
    (u, qhat, uhat) = split(z)

    J = 0.5 * inner(grad(u), grad(u))*dx - inner(f, u)*dx
    L_match = inner(qhat, uhat - u)
    L = J + (L_match('+') + L_match('-'))*dS(3) + (L_match('+') + L_match('-'))*dS(4) + L_match*ds
    F = derivative(L, z, TestFunction(W))
    bcs = [DirichletBC(W.sub(2), 0, 'on_boundary')]

    # direct solve using LU decomposition
    params = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
    }

    solve(F == 0, z, bcs, solver_parameters=params)
