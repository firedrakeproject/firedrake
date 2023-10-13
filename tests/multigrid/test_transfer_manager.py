import pytest
import numpy
from firedrake import *
from firedrake.mg.ufl_utils import coarsen
from firedrake.utils import complex_mode


@pytest.fixture(scope="module")
def hierarchy():
    mesh = UnitSquareMesh(1, 1)
    return MeshHierarchy(mesh, 1)


@pytest.fixture
def mesh(hierarchy):
    return hierarchy[-1]


transfer = TransferManager()


@pytest.mark.parametrize("sub", (True, False), ids=["Z.sub(0)", "V"])
@pytest.mark.skipcomplexnoslate
def test_transfer_manager_inside_coarsen(sub, mesh):
    V = FunctionSpace(mesh, "N1curl", 2)
    Q = FunctionSpace(mesh, "P", 1)
    Z = V*Q
    x, y = SpatialCoordinate(mesh)

    if sub:
        bc_space = Z.sub(0)
    else:
        bc_space = V
    bcdata = project(as_vector([-y, x]), bc_space)

    bc = DirichletBC(Z.sub(0), bcdata, "on_boundary")

    u = Function(Z)

    v = TestFunction(Z)

    F = inner(u, v)*dx

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(problem)

    with dmhooks.add_hooks(Z.dm, solver, appctx=solver._ctx):
        cctx = coarsen(solver._ctx, coarsen)

    bc, = cctx._problem.bcs
    V = bc.function_space()
    mesh = V.ufl_domain()
    x, y = SpatialCoordinate(mesh)
    expect = project(as_vector([-y, x]), V)
    assert numpy.allclose(bc.function_arg.dat.data_ro, expect.dat.data_ro)


@pytest.mark.parametrize("transfer_op", ("prolong", "restrict", "inject"))
@pytest.mark.parametrize("family", ("CG", "DG", "RT"))
def test_transfer_manager_dat_version_cache(hierarchy, family, transfer_op):
    degree = 1
    Vc = FunctionSpace(hierarchy[0], family, degree)
    Vf = FunctionSpace(hierarchy[1], family, degree)
    if transfer_op == "prolong":
        op = transfer.prolong
        source = Function(Vc)
        target = Function(Vf)
    elif transfer_op == "restrict":
        op = transfer.restrict
        source = Function(Vf)
        target = Function(Vc)
    elif transfer_op == "inject":
        op = transfer.inject
        source = Function(Vf)
        target = Function(Vc)
        if family != "CG" and complex_mode:
            with pytest.raises(NotImplementedError):
                op(source, target)
                return

    # Test that the operator produces an output for an unrecognized input
    source.assign(1)
    op(source, target)
    assert not numpy.allclose(target.dat.data_ro, 0.0)

    # Test no-op for unmodified input
    for k in range(2):
        dat_version = target.dat.dat_version
        op(source, target)
        assert target.dat.dat_version == dat_version

    # Modify the input, test that the output is regenerated
    source.assign(2)
    dat_version = target.dat.dat_version
    op(source, target)
    assert target.dat.dat_version > dat_version

    # Modify the output, test that the output is regenerated
    target.assign(3)
    dat_version = target.dat.dat_version
    op(source, target)
    assert target.dat.dat_version > dat_version

    # Test that the operator produces an output for an unrecognized input
    source = Function(source)
    target = Function(target)
    dat_version = target.dat.dat_version
    op(source, target)
    assert target.dat.dat_version > dat_version

    # Wrap old dats with new functions, test that old dats are still recognized
    dat_version = target.dat.dat_version
    old_dats = (source.dat, target.dat)
    source = Function(source.function_space(), val=source.dat)
    target = Function(target.function_space(), val=target.dat)
    assert (source.dat, target.dat) == old_dats
    assert target.dat.dat_version == dat_version
    op(source, target)
    assert target.dat.dat_version == dat_version
