import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope="module")
def m():
    return UnitIntervalMesh(2)


@pytest.fixture
def f(m):
    cg = FunctionSpace(m, "CG", 1)
    dg = FunctionSpace(m, "DG", 0)

    c = Function(cg)
    d = Function(dg)

    return c, d


@pytest.fixture
def f_mixed(m):
    cg = FunctionSpace(m, "CG", 1)
    dg = FunctionSpace(m, "DG", 0)

    return Function(cg*dg)


@pytest.fixture
def const(m):
    return Constant(1.0, domain=m)


@pytest.fixture
def f_extruded():
    i = UnitIntervalMesh(2)
    m = ExtrudedMesh(i, 2, layer_height=0.1)
    cg = FunctionSpace(m, "CG", 1)
    dg = FunctionSpace(m, "DG", 0)

    c = Function(cg)
    d = Function(dg)

    return c, d


def test_direct_par_loop(f):
    c, _ = f

    domain = ""
    instructions = """
    c[0, 0] = 1
    """
    par_loop((domain, instructions), direct, {'c': (c, WRITE)},
             is_loopy_kernel=True)

    assert np.allclose(c.dat.data, 1.0)


def test_mixed_direct_par_loop(f_mixed):
    with pytest.raises(NotImplementedError):
        domain = ""
        instructions = """
        c[0, 0] = 1
        """
        par_loop((domain, instructions), direct, {'c': (f_mixed, WRITE)},
                 is_loopy_kernel=True)
        assert all(np.allclose(f.dat.data, 1.0) for f in f_mixed.split())


@pytest.mark.parametrize('idx', [0, 1])
def test_mixed_direct_par_loop_components(f_mixed, idx):
    domain = ""
    instructions = """
    c[0, 0] = 1
    """
    par_loop((domain, instructions), direct, {'c': (f_mixed[idx], WRITE)},
             is_loopy_kernel=True)

    assert np.allclose(f_mixed.dat[idx].data, 1.0)


def test_direct_par_loop_read_const(f, const):
    c, _ = f
    const.assign(10.0)

    domain = ""
    instructions = """
    c[0, 0] = constant[0]
    """
    par_loop((domain, instructions), direct, {'c': (c, WRITE), 'constant': (const, READ)},
             is_loopy_kernel=True)

    assert np.allclose(c.dat.data, const.dat.data)


def test_indirect_par_loop_read_const(f, const):
    _, d = f
    const.assign(10.0)

    domain = "{[i]: 0 <= i < d.dofs}"
    instructions = """
    for i
        d[i, 0] = constant[0]
    end
    """
    par_loop((domain, instructions), dx, {'d': (d, WRITE), 'constant': (const, READ)},
             is_loopy_kernel=True)

    assert np.allclose(d.dat.data, const.dat.data)


def test_indirect_par_loop_read_const_mixed(f_mixed, const):
    const.assign(10.0)

    with pytest.raises(NotImplementedError):
        domain = "{[i]: 0 <= i < d.dofs}"
        instructions = """
        for i
            d[i, 0] = constant[0]
        end
        """
        par_loop((domain, instructions), dx, {'d': (f_mixed, WRITE), 'constant': (const, READ)},
                 is_loopy_kernel=True)
        assert all(np.allclose(f.dat.data, const.dat.data) for f in f_mixed.split())


@pytest.mark.parallel(nprocs=2)
def test_dict_order_parallel():
    mesh = UnitIntervalMesh(10)
    d = Function(FunctionSpace(mesh, "DG", 0))
    consts = []
    for i in range(20):
        consts.append(Constant(i, domain=mesh))

    arg = {}
    if mesh.comm.rank == 0:
        arg['d'] = (d, WRITE)

        for i, c in enumerate(consts):
            arg["c%d" % i] = (c, READ)
    else:
        arg['d'] = (d, WRITE)

        for i, c in enumerate(reversed(consts)):
            arg["c%d" % (len(consts) - i - 1)] = (c, READ)

    domain = "{[i]: 0 <= i < d.dofs}"
    instructions = """
    for i
        d[i, 0] = c10[0]
    end
    """
    par_loop((domain, instructions), dx, arg, is_loopy_kernel=True)

    assert np.allclose(d.dat.data, consts[10].dat.data)


@pytest.mark.parametrize('idx', [0, 1])
def test_indirect_par_loop_read_const_mixed_component(f_mixed, const, idx):
    const.assign(10.0)

    domain = "{[i]: 0 <= i < d.dofs}"
    instructions = """
    for i
        d[i, 0] = constant[0]
    end
    """
    par_loop((domain, instructions), dx, {'d': (f_mixed[idx], WRITE), 'constant': (const, READ)},
             is_loopy_kernel=True)

    assert np.allclose(f_mixed.dat[idx].data, const.dat.data)


def test_par_loop_const_write_error(f, const):
    _, d = f
    with pytest.raises(RuntimeError):
        domain = ""
        instructions = """
        c[0] = d[0, 0]
        """
        par_loop((domain, instructions), direct, {'c': (const, WRITE), 'd': (d, READ)},
                 is_loopy_kernel=True)


def test_cg_max_field(f):
    c, d = f
    x = SpatialCoordinate(d.function_space().mesh())
    d.interpolate(x[0])

    domain = "{[i]: 0 <= i < c.dofs}"
    instructions = """
    for i
        c[i, 0] = fmax(c[i, 0], d[0, 0])
    end
    """
    par_loop((domain, instructions), dx, {'c': (c, RW), 'd': (d, READ)},
             is_loopy_kernel=True)

    assert (c.dat.data == [1./4, 3./4, 3./4]).all()


def test_cg_max_field_extruded(f_extruded):
    c, d = f_extruded
    x = SpatialCoordinate(d.function_space().mesh())
    d.interpolate(x[0])

    domain = "{[i]: 0 <= i < c.dofs}"
    instructions = """
    for i
        c[i, 0] = fmax(c[i, 0], d[0, 0])
    end
    """

    par_loop((domain, instructions), dx, {'c': (c, RW), 'd': (d, READ)},
             is_loopy_kernel=True)

    assert (c.dat.data == [1./4, 1./4, 1./4,
                           3./4, 3./4, 3./4,
                           3./4, 3./4, 3./4]).all()


@pytest.mark.parametrize("subdomain", [1, 2])
def test_cell_subdomain(subdomain):
    from os.path import abspath, dirname, join
    mesh = Mesh(join(abspath(dirname(__file__)), "..",
                     "meshes", "cell-sets.msh"))

    V = FunctionSpace(mesh, "DG", 0)
    expect = interpolate(as_ufl(1), V, subset=mesh.cell_subset(subdomain))

    f = Function(V)

    domain = "{[i]: 0 <= i < f.dofs}"
    instructions = """
    for i
        f[i, 0] = 1.0
    end
    """
    par_loop((domain, instructions), dx(subdomain), {'f': (f, WRITE)},
             is_loopy_kernel=True)

    assert np.allclose(f.dat.data, expect.dat.data)


def test_walk_facets_rt():
    m = UnitSquareMesh(3, 3)
    x = SpatialCoordinate(m)
    V = FunctionSpace(m, 'RT', 1)

    f1 = Function(V)
    f2 = Function(V)

    project(as_vector((x[0], x[1])), f1)

    domain = "{[i]: 0 <= i < f1.dofs}"
    instructions = """
    for i
        f2[i, 0] = f1[i, 0]
    end
    """
    par_loop((domain, instructions), dS, {'f1': (f1, READ), 'f2': (f2, WRITE)},
             is_loopy_kernel=True)

    par_loop((domain, instructions), ds, {'f1': (f1, READ), 'f2': (f2, WRITE)},
             is_loopy_kernel=True)

    assert errornorm(f1, f2, degree_rise=0) < 1e-10
