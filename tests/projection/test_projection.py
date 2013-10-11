from firedrake import *


def run_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)

    V = FunctionSpace(m, family, degree)

    e = Expression('cos(x[0]*pi*2)*sin(x[1]*pi*2)')

    exact = Function(FunctionSpace(m, 'CG', 5))

    exact.interpolate(e)

    # Solve to machine precision.
    ret = project(e, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble((ret - exact) * (ret - exact) * dx))


def run_convergence_test(degree=1, family='CG'):
    l2_diff = [run_test(x, degree, family) for x in range(3, 8)]
    from math import log
    conv = [log(l2_diff[i] / l2_diff[i + 1], 2)
            for i in range(len(l2_diff) - 1)]

    return np.array(conv)

if __name__ == '__main__':
    print run_convergence_test()
