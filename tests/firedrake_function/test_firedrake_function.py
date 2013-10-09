from firedrake import *

def run_test():

    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(Expression("1"))
    test0 = (f.dat.data_ro == 1.0).all()

    g = Function(f)
    test1 = (g.dat.data_ro == 1.0).all()

    # Check that g is indeed a deep copy
    f.interpolate(Expression("2"))

    test2 = (f.dat.data_ro == 2.0).all()
    test3 = (g.dat.data_ro == 1.0).all()

    return (test0, test1, test2, test3)

if __name__ == "__main__":

    tests = run_test()
    assert all(tests)
    print "Test passed"
