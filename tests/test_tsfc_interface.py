import pytest
from firedrake import *
import loopy


@pytest.fixture(scope='module')
def fs():
    mesh = UnitSquareMesh(1, 1)
    return FunctionSpace(mesh, 'CG', 1)


@pytest.fixture
def mass(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(u, v) * dx


@pytest.fixture
def mixed_mass(fs):
    u, r = TrialFunctions(fs*fs)
    v, s = TestFunctions(fs*fs)
    return (inner(u, v) + inner(r, s)) * dx


@pytest.fixture
def laplace(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(grad(u), grad(v)) * dx


@pytest.fixture
def rhs(fs):
    v = TestFunction(fs)
    g = Function(fs)
    return inner(g, v) * ds


@pytest.fixture
def rhs2(fs):
    v = TestFunction(fs)
    f = Function(fs)
    g = Function(fs)
    return inner(f, v) * dx + inner(g, v) * ds


def test_tsfc_same_form(mass):
    """Compiling the same form twice should load kernels from cache."""
    k1 = tsfc_interface.compile_form(mass, 'mass')
    k2 = tsfc_interface.compile_form(mass, 'mass')

    assert k1 is k2
    assert all(k1_[-1] is k2_[-1] for k1_, k2_ in zip(k1, k2))


def test_tsfc_same_mixed_form(mixed_mass):
    """Compiling a mixed form twice should load kernels from cache."""
    k1 = tsfc_interface.compile_form(mixed_mass, 'mixed_mass')
    k2 = tsfc_interface.compile_form(mixed_mass, 'mixed_mass')

    assert k1 is k2
    assert all(k1_[-1] is k2_[-1] for k1_, k2_ in zip(k1, k2))


def test_tsfc_different_forms(mass, laplace):
    """Compiling different forms should not load kernels from cache."""
    k1, = tsfc_interface.compile_form(mass, 'mass')
    k2, = tsfc_interface.compile_form(laplace, 'mass')

    assert k1[-1] is not k2[-1]


def test_tsfc_different_names(mass):
    """Compiling different forms should not load kernels from cache."""
    k1, = tsfc_interface.compile_form(mass, 'mass')
    k2, = tsfc_interface.compile_form(mass, 'laplace')

    assert k1[-1] is not k2[-1]


def test_tsfc_cell_kernel(mass):
    k = tsfc_interface.compile_form(mass, 'mass')
    assert len(k) == 1 and 'cell_integral' in loopy.generate_code_v2(k[0][1][0].code).device_code()


def test_tsfc_exterior_facet_kernel(rhs):
    k = tsfc_interface.compile_form(rhs, 'rhs')
    assert len(k) == 1 and 'exterior_facet_integral' in loopy.generate_code_v2(k[0][1][0].code).device_code()


def test_tsfc_cell_exterior_facet_kernel(rhs2):
    k = tsfc_interface.compile_form(rhs2, 'rhs2')
    kernel_name = sorted(k_[1][0].name for k_ in k)
    assert len(k) == 2 and 'cell_integral' in kernel_name[0] and \
        'exterior_facet_integral' in kernel_name[1]
