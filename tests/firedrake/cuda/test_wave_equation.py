from firedrake import *
import os
import numpy as np
import finat
from firedrake.__future__ import interpolate
import pytest


# TODO: add marker for cuda pytestss
def test_kmv_wave_propagation_cuda():
    nested_parameters = {
        "ksp_type": "preonly",
        "pc_type": "jacobi"
    }
    parameters = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.OffloadPC",
        "offload": nested_parameters,
    }


    # Choosing degree
    degree = 4

    # Setting up time variables
    dt = 0.001  # time step in seconds
    final_time = 0.5  # final time in seconds
    total_steps = int(final_time / dt) + 1

    # Setting up mesh parameters
    nx, ny = 10, 10
    mesh = RectangleMesh(nx, ny, 1.0, 1.0)

    # Acquisition geometry
    frequency_peak = 5.0  # The dominant frequency of the Ricker wavelet in Hz.
    offset = 0.2
    source_locations = [(0.5, 0.5)]
    receiver_locations = [(0.5, 0.5 + offset)]

    # Setting up function space
    V = FunctionSpace(mesh, "KMV", degree)

    # Velocity model
    c = Constant(1.5)

    # Ricker wavelet definition
    def ricker_wavelet(t, freq, amp=1.0, delay=0.2, delay_type="time"):
        if delay_type == "multiples_of_minimun":
            time_delay = delay * np.sqrt(6.0) / (np.pi * freq)
        elif delay_type == "time":
            time_delay = delay
        t = t - time_delay
        # t = t - delay / freq
        tt = (np.pi * freq * t) ** 2
        return amp * (1.0 - (2.0) * tt) * np.exp((-1.0) * tt)

    # Using vertex only mesh
    source_mesh = VertexOnlyMesh(mesh, source_locations)
    V_s = FunctionSpace(source_mesh, "DG", 0)
    d_s = Function(V_s)
    d_s.interpolate(1.0)
    source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
    q_s = Cofunction(V.dual()).interpolate(source_cofunction)
    receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
    V_r = FunctionSpace(receiver_mesh, "DG", 0)
    f = Cofunction(V.dual())

    true_data_receivers = []

    # Setting up forward problem
    u = TrialFunction(V)
    v = TestFunction(V)
    u_np1 = Function(V) # timestep n+1
    u_n = Function(V) # timestep n
    u_nm1 = Function(V) # timestep n-1
    # Quadrature rule for lumped mass matrix.
    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
    m = (1 / (c * c))
    time_term =  m * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dx(scheme=quad_rule)
    nf = (1 / c) * ((u_n - u_nm1) / dt) * v * ds
    a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
    F = time_term + a + nf
    lin_var = LinearVariationalProblem(lhs(F), rhs(F) + f, u_np1)
    # Since the linear system matrix is diagonal, the solver parameters are set to construct a solver,
    # which applies a single step of Jacobi preconditioning.

    solver = LinearVariationalSolver(lin_var,solver_parameters=parameters)

    interpolate_receivers = interpolate(u_np1, V_r)

    # Looping in time
    for step in range(total_steps):
        if step % 100 == 0:
            print(f"For time = {step*dt}s")
        f.assign(ricker_wavelet(step * dt, frequency_peak) * q_s)
        solver.solve()
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
        rec_out = assemble(interpolate_receivers)
        true_data_receivers.append(rec_out.dat.data[:])

    rec_matrix = np.matrix(true_data_receivers)

    # Hard coded values from an analytical solution
    # Hard coded values from an analytical solution
    min_value = -0.05708
    max_value = 0.09467
    min_location = 0.2701
    max_location = 0.3528

    correct_min_loc = np.isclose(min_location, np.argmin(rec_matrix)*dt, rtol=1e-2)
    correct_min = np.isclose(min_value, np.min(rec_matrix), rtol=1e-2)
    correct_max = np.isclose(max_value, np.max(rec_matrix), rtol=1e-2)
    correct_max_loc = np.isclose(max_location, np.argmax(rec_matrix)*dt, rtol=1e-2)

    print(f"Correct minimum and its location: {correct_min} and {correct_min_loc}.")
    print(f"Correct maximum and its location: {correct_max} and {correct_max_loc}.")

    print("END", flush=True)
    assert all([correct_min_loc, correct_min, correct_max, correct_max_loc])


if __name__ == "__main__":
    test_kmv_wave_propagation_cuda()
