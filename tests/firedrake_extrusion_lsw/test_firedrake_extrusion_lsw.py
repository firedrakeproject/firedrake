"""Demo of the Shallow Water
"""

# Begin demo
from firedrake import *

power = 5
# Create mesh and define function space
m = UnitSquareMesh(2 ** power, 2 ** power)
layers = 5

# Populate the coordinates of the extruded mesh by providing the
# coordinates as a field.
extrusion_kernel = """
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Only the Z-coord is increased, the others stay the same
    xtr[0][0] = x[0][0];
    xtr[0][1] = x[0][1];
    xtr[0][2] = 0.25*j[0][0];
}"""

mesh = ExtrudedMesh(m, layers, extrusion_kernel)


def lsw_strang(test_mode):
    W = FunctionSpace(mesh, "BDM", 1, vfamily="Lagrange", vdegree=1)
    X = FunctionSpace(mesh, "DG", 0, vfamily="Lagrange", vdegree=1)
    if not test_mode:
        Xplot = FunctionSpace(mesh, "CG", 1, vfamily="Lagrange", vdegree=1)

    # Define starting field
    u_0 = Function(W)
    u_h = Function(W)
    u_1 = Function(W)
    p_0 = Function(X)
    p_1 = Function(X)
    if not test_mode:
        p_plot = Function(Xplot)
    p_0.interpolate(Expression("sin(4*pi*x[0])*sin(2*pi*x[1])"))

    if not test_mode:
        T = 0.5
    else:
        T = 0.01
    t = 0
    dt = 0.0025

    if not test_mode:
        file = File("lsw3d.pvd")
        p_trial = TrialFunction(Xplot)
        p_test = TestFunction(Xplot)
        solve(p_trial * p_test * dx == p_0 * p_test * dx, p_plot)
        file << p_plot, t

    E_0 = assemble(0.5 * p_0 * p_0 * dx + 0.5 * dot(u_0, u_0) * dx)

    while t < T:
        u = TrialFunction(W)
        w = TestFunction(W)
        a_1 = dot(w, u) * dx
        L_1 = dot(w, u_0) * dx + 0.5 * dt * div(w) * p_0 * dx
        solve(a_1 == L_1, u_h)

        p = TrialFunction(X)
        phi = TestFunction(X)
        a_2 = phi * p * dx
        L_2 = phi * p_0 * dx - dt * phi * div(u_h) * dx
        solve(a_2 == L_2, p_1)

        u = TrialFunction(W)
        w = TestFunction(W)
        a_3 = dot(w, u) * dx
        L_3 = dot(w, u_h) * dx + 0.5 * dt * div(w) * p_1 * dx
        solve(a_3 == L_3, u_1)

        u_0.assign(u_1)
        p_0.assign(p_1)
        t += dt

        if not test_mode:
            # project into P1 x P1 for plotting
            p_trial = TrialFunction(Xplot)
            p_test = TestFunction(Xplot)
            solve(p_trial * p_test * dx == p_0 * p_test * dx, p_plot)
            file << p_plot, t
            print t

    E_1 = assemble(0.5 * p_0 * p_0 * dx + 0.5 * dot(u_0, u_0) * dx)
    return np.abs(E_1 - E_0)


def run_test(test_mode=False):
    return [lsw_strang(test_mode)]

if __name__ == "__main__":

    result = run_test()
    for i, res in enumerate(result):
        print "Result for extruded lsw energy error %r: %r" % (i + 1, res)
