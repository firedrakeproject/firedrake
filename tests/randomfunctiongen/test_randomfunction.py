from firedrake import *
from firedrake import randomfunctiongen

import pytest

import inspect
import numpy as np
import randomgen


brng_list = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass) if name != 'RandomGenerator']
meth_list = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator) if not name.startswith('_') and name not in ('state', 'poisson_lam_max', 'seed', 'random_integers', 'bytes', 'permutation', 'shuffle', 'dirichlet', 'multinomial', 'multivariate_normal', 'complex_normal')]


@pytest.mark.parametrize("brng", brng_list)
@pytest.mark.parametrize("meth", meth_list)
def test_randomfunc(brng, meth):

    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    seed = 123456789

    # original
    rg_base = getattr(randomgen, brng)(seed=seed).generator

    # Firedrake wrapper
    rg_wrap = getattr(randomfunctiongen, brng)(seed=seed).generator

    if meth == 'beta':
        args = (0.3, 0.5)

    elif meth == 'binomial':
        args = (7, 0.3)

    elif meth == 'chisquare':
        args = (7,)

    elif meth == 'choice':
        args = (1234 * 5678,)

    elif meth == 'exponential':
        args = ()

    elif meth == 'f':
        args = (7, 17)

    elif meth == 'gamma':
        args = (3.14,)

    elif meth == 'geometric':
        args = (3.14,)

    elif meth == 'gumbel':
        args = ()

    elif meth == 'hypergeometric':
        args = (17, 2 * 17, 3 * 17 - 1)

    elif meth == 'laplace':
        args = ()

    elif meth == 'logistic':
        args = ()

    elif meth == 'lognormal':
        args = ()

    elif meth == 'logseries':
        args = (0.3,)

    elif meth == 'negative_binomial':
        args = (7, 0.3)

    elif meth == 'noncentral_chisquare':
        args = (7, 3.14)

    elif meth == 'noncentral_f':
        args = (7, 17, 3.14)

    elif meth == 'normal':
        args = ()

    elif meth == 'pareto':
        args = (3.14,)

    elif meth == 'poisson':
        args = ()

    elif meth == 'power':
        args = (3.14,)

    elif meth == 'rand':
        args = ()

    elif meth == 'randint':
        args = (7,)

    elif meth == 'randn':
        args = ()

    elif meth == 'random_raw':
        args = ()

    elif meth == 'random_sample':
        args = ()

    elif meth == 'random_uintegers':
        args = ()

    elif meth == 'rayleigh':
        args = ()

    elif meth == 'standard_cauchy':
        args = ()

    elif meth == 'standard_exponential':
        args = ()

    elif meth == 'standard_gamma':
        args = (3.14,)

    elif meth == 'standard_normal':
        args = ()

    elif meth == 'standard_t':
        args = (7,)

    elif meth == 'tomaxint':
        args = ()

    elif meth == 'triangular':
        args = (2.71, 3.14, 10.)

    elif meth == 'uniform':
        args = ()

    elif meth == 'vonmises':
        args = (2.71, 3.14)

    elif meth == 'wald':
        args = (2.71, 3.14)

    elif meth == 'weibull':
        args = (3.14,)

    elif meth == 'zipf':
        args = (3.14,)

    for i in range(1, 10):
        f = getattr(rg_wrap, meth)(V, *args)
        with f.dat.vec_ro as v:
            if meth in ('rand', 'randn'):
                assert np.allclose(getattr(rg_base, meth)(v.local_size), v.array[:])
            else:
                kwargs = {'size': (v.local_size,)}
                assert np.allclose(getattr(rg_base, meth)(*args, **kwargs), v.array[:])


@pytest.mark.parallel
def test_randomfunc_parallel_pcg64():

    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    seed = 123456789

    # original
    from mpi4py import MPI
    rg_base = randomgen.PCG64(seed=seed, inc=MPI.COMM_WORLD.rank).generator

    # Firedrake wrapper
    rg_wrap = PCG64(seed=seed).generator

    for i in range(1, 10):
        f = rg_wrap.beta(V, 0.3, 0.5)
        with f.dat.vec_ro as v:
            assert np.allclose(rg_base.beta(0.3, 0.5, size=(v.local_size,)), v.array[:])

    rg_base.seed(seed=12345678910, inc=MPI.COMM_WORLD.rank)
    rg_wrap.seed(seed=12345678910)
    for i in range(1, 10):
        f = rg_wrap.beta(V, 0.3, 0.5)
        with f.dat.vec_ro as v:
            assert np.allclose(rg_base.beta(0.3, 0.5, size=(v.local_size,)), v.array[:])


@pytest.mark.parallel
@pytest.mark.parametrize("brng", ['Philox', 'ThreeFry'])
def test_randomfunc_parallel_philox_threefry(brng):

    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    key_size = 2 if brng == 'Philox' else 4

    # original
    from mpi4py import MPI
    key = np.zeros(key_size, dtype=np.uint64)
    key[0] = MPI.COMM_WORLD.rank
    rg_base = getattr(randomgen, brng)(counter=12345678910, key=key).generator

    # Firedrake wrapper
    rg_wrap = getattr(randomfunctiongen, brng)(counter=12345678910).generator

    for i in range(1, 10):
        f = rg_wrap.beta(V, 0.3, 0.5)
        with f.dat.vec_ro as v:
            assert np.allclose(rg_base.beta(0.3, 0.5, size=(v.local_size,)), v.array[:])

    rg_base.seed(counter=1234567, key=key)
    rg_wrap.seed(counter=1234567)

    for i in range(1, 10):
        f = rg_wrap.beta(V, 0.3, 0.5)
        with f.dat.vec_ro as v:
            assert np.allclose(rg_base.beta(0.3, 0.5, size=(v.local_size,)), v.array[:])
