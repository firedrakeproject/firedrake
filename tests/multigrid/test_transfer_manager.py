import pytest
import numpy
from firedrake import *
from firedrake.mg.ufl_utils import coarsen
from firedrake.utils import complex_mode


@pytest.fixture(scope="module")
def hierarchy():
    mesh = UnitSquareMesh(1, 1)
    return MeshHierarchy(mesh, 2)


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
    mesh = V.mesh()
    x, y = SpatialCoordinate(mesh)
    expect = project(as_vector([-y, x]), V)
    assert numpy.allclose(bc.function_arg.dat.data_ro, expect.dat.data_ro)


@pytest.fixture(params=["CG", "DG", "RT"])
def spaces(hierarchy, request):
    family = request.param
    return tuple(FunctionSpace(mesh, family, 1) for mesh in hierarchy)


@pytest.mark.parametrize("action", ("unmodified", "modify_source", "modify_target", "new_dats", "same_dats"))
@pytest.mark.parametrize("transfer_op", ("prolong", "restrict", "inject"))
def test_transfer_manager_dat_version_cache(action, transfer_op, spaces):
    Vcoarse, Vfine = spaces[0], spaces[-1]
    if transfer_op == "prolong":
        op = transfer.prolong
        Vsource, Vtarget = Vcoarse, Vfine
    elif transfer_op == "restrict":
        op = transfer.restrict
        Vsource, Vtarget = Vfine.dual(), Vcoarse.dual()
    elif transfer_op == "inject":
        op = transfer.inject
        Vsource, Vtarget = Vfine, Vcoarse

    source = Function(Vsource)
    target = Function(Vtarget)
    family = Vsource.ufl_element().family()
    if complex_mode and ((family == "Discontinuous Lagrange" and transfer_op == "inject")
                         or family not in {"Lagrange", "Discontinuous Lagrange"}):
        with pytest.raises(NotImplementedError):
            op(source, target)
        return

    # Test that the operator produces an output for an unrecognized input
    source.dat.data_wo[...] = 1
    op(source, target)
    assert not numpy.allclose(target.dat.data_ro, 0.0)

    if action == "unmodified":
        # Test no-op for unmodified input and outputs
        for k in range(2):
            dat_version = target.dat.dat_version
            op(source, target)
            assert target.dat.dat_version == dat_version

    elif action == "modify_source":
        # Modify the input, test that the output is regenerated
        source.dat.data_wo[...] = 2
        dat_version = target.dat.dat_version
        op(source, target)
        assert target.dat.dat_version > dat_version

    elif action == "modify_target":
        # Modify the output, test that the output is regenerated
        target.dat.data_wo[...] = 3
        dat_version = target.dat.dat_version
        op(source, target)
        assert target.dat.dat_version > dat_version

    elif action == "new_dats":
        # Test that the operator produces an output for an unrecognized input
        source = Function(source)
        target = Function(target)
        dat_version = target.dat.dat_version
        op(source, target)
        assert target.dat.dat_version > dat_version

    elif action == "same_dats":
        # Wrap old dats with new functions, test that old dats are still recognized
        dat_version = target.dat.dat_version
        old_dats = (source.dat, target.dat)
        source = Function(Vsource, val=source.dat)
        target = Function(Vtarget, val=target.dat)
        assert (source.dat, target.dat) == old_dats
        assert target.dat.dat_version == dat_version
        op(source, target)
        assert target.dat.dat_version == dat_version

    else:
        raise ValueError(f"Unrecognized action {action}")
