import pytest
from firedrake import *
import os


@pytest.fixture(scope='module')
def fs():
    mesh = UnitSquareMesh(1, 1)
    return FunctionSpace(mesh, 'CG', 1)


@pytest.fixture
def mass(fs):
    u = TestFunction(fs)
    v = TrialFunction(fs)
    return u * v * dx


@pytest.fixture
def laplace(fs):
    u = TestFunction(fs)
    v = TrialFunction(fs)
    return inner(grad(u), grad(v)) * dx


@pytest.fixture
def rhs(fs):
    v = TrialFunction(fs)
    g = Function(fs)
    return g * v * ds


@pytest.fixture
def rhs2(fs):
    v = TrialFunction(fs)
    f = Function(fs)
    g = Function(fs)
    return f * v * dx + g * v * ds


@pytest.fixture
def cache_key(mass):
    return ffc_interface.FFCKernel(mass, 'mass').cache_key


@pytest.mark.xfail("not hasattr(ffc_interface.constants, 'PYOP2_VERSION')")
class TestFFCCache:

    """FFC code generation cache tests."""

    def test_ffc_cache_dir_exists(self):
        """Importing ffc_interface should create FFC Kernel cache dir."""
        assert os.path.exists(ffc_interface.FFCKernel._cachedir)

    def test_ffc_cache_persist_on_disk(self, cache_key):
        """FFCKernel should be persisted on disk."""
        assert os.path.exists(
            os.path.join(ffc_interface.FFCKernel._cachedir, cache_key))

    def test_ffc_cache_read_from_disk(self, cache_key):
        """Loading an FFCKernel from disk should yield the right object."""
        assert ffc_interface.FFCKernel._read_from_disk(
            cache_key).cache_key == cache_key

    def test_ffc_compute_form_data(self, mass):
        """Compiling a form attaches form data."""
        ffc_interface.compile_form(mass, 'mass')

        assert mass.form_data()

    def test_ffc_same_form(self, mass):
        """Compiling the same form twice should load kernels from cache."""
        k1 = ffc_interface.compile_form(mass, 'mass')
        k2 = ffc_interface.compile_form(mass, 'mass')

        assert k1 is k2

    def test_ffc_different_forms(self, mass, laplace):
        """Compiling different forms should not load kernels from cache."""
        k1 = ffc_interface.compile_form(mass, 'mass')
        k2 = ffc_interface.compile_form(laplace, 'mass')

        assert k1 is not k2

    def test_ffc_different_names(self, mass):
        """Compiling different forms should not load kernels from cache."""
        k1 = ffc_interface.compile_form(mass, 'mass')
        k2 = ffc_interface.compile_form(mass, 'laplace')

        assert k1 is not k2

    def test_ffc_cell_kernel(self, mass):
        k = ffc_interface.compile_form(mass, 'mass')
        assert 'cell_integral' in k[0].code and len(k) == 1

    def test_ffc_exterior_facet_kernel(self, rhs):
        k = ffc_interface.compile_form(rhs, 'rhs')
        assert 'exterior_facet_integral' in k[0].code and len(k) == 1

    def test_ffc_cell_exterior_facet_kernel(self, rhs2):
        k = ffc_interface.compile_form(rhs2, 'rhs2')
        assert 'cell_integral' in k[
            0].code and 'exterior_facet_integral' in k[1].code and len(k) == 2

if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
