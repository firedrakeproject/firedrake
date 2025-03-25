from firedrake import *
# These Parameter tests are a cut down version of the unit Parameter tests in DOLFIN


def test_simple():

    # Create some parameters
    p0 = Parameters("test")
    p0.add("filename", "foo.txt")
    p0.add("maxiter", 100)
    p0.add("tolerance", 0.001)
    p0.add("monitor_convergence", True)

    # Check values
    assert p0.name() == "test"
    assert p0["filename"] == "foo.txt"
    assert p0["maxiter"] == 100
    assert p0["tolerance"] == 0.001
    assert p0["monitor_convergence"] is True


def test_nested():

    # Create some nested parameters
    p0 = Parameters("test")
    p00 = Parameters("sub0")
    p00.add("filename", "foo.txt")
    p00.add("maxiter", 100)
    p00.add("tolerance", 0.001)
    p00.add("monitor_convergence", True)
    p0.add("foo", "bar")
    p01 = Parameters(p00)
    p01.rename("sub1")
    p0.add(p00)
    p0.add(p01)

    # Check values
    assert p0.name() == "test"
    assert p0["foo"] == "bar"
    assert p0["sub0"]["filename"] == "foo.txt"
    assert p0["sub0"]["maxiter"] == 100
    assert p0["sub0"]["tolerance"] == 0.001
    assert p0["sub0"]["monitor_convergence"] is True
