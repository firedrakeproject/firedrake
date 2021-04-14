from firedrake import *
from firedrake import randomfunctiongen

import pytest

import inspect
import numpy as np
import numpy.random as randomgen


brng_list = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass) if name not in ('BitGenerator', 'Generator', 'RandomState', 'SeedSequence')]
meth_list = [name for name, _ in inspect.getmembers(randomgen.Generator) if not name.startswith('_') and name not in ('bit_generator', 'bytes', 'dirichlet', 'integers', 'multinomial', 'multivariate_hypergeometric', 'multivariate_normal', 'shuffle', 'permutation')]


@pytest.mark.parametrize("brng", brng_list)
@pytest.mark.parametrize("meth", meth_list)
def test_randomfunc(brng, meth):

    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    seed = 123456789
    # Original
    bgen = getattr(randomgen, brng)(seed=seed)
    if brng == 'PCG64':
        state = bgen.state
        state['state'] = {'state': seed, 'inc': V.comm.Get_rank()}
        bgen.state = state
    rg_base = randomgen.Generator(bgen)
    # Firedrake wrapper
    fgen = getattr(randomfunctiongen, brng)(seed=seed)
    if brng == 'PCG64':
        state = fgen.state
        state['state'] = {'state': seed, 'inc': V.comm.Get_rank()}
        fgen.state = state
    rg_wrap = randomfunctiongen.Generator(fgen)

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
        args = (0.5,)
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
    elif meth == 'random':
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
    else:
        raise RuntimeError("Unknown method: add test for %s." % meth)

    for i in range(1, 10):
        f = getattr(rg_wrap, meth)(V, *args)
        with f.dat.vec_ro as v:
            if meth in ('rand', 'randn'):
                assert np.allclose(getattr(rg_base, meth)(v.local_size), v.array[:])
            else:
                kwargs = {'size': (v.local_size, )}
                assert np.allclose(getattr(rg_base, meth)(*args, **kwargs), v.array[:])


@pytest.mark.parallel
def test_randomfunc_parallel_pcg64():
    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    seed = 123456789
    # Original
    bgen = randomgen.PCG64(seed=seed)
    state = bgen.state
    state['state'] = {'state': seed, 'inc': V.comm.Get_rank()}
    bgen.state = state
    rg_base = randomgen.Generator(bgen)
    # Firedrake wrapper
    fgen = randomfunctiongen.PCG64(seed=seed)
    state = fgen.state
    state['state'] = {'state': seed, 'inc': V.comm.Get_rank()}
    fgen.state = state
    rg_wrap = randomfunctiongen.Generator(fgen)
    for i in range(1, 10):
        f = rg_wrap.beta(V, 0.3, 0.5)
        with f.dat.vec_ro as v:
            assert np.allclose(rg_base.beta(0.3, 0.5, size=(v.local_size,)), v.array[:])


@pytest.mark.parallel
def test_randomfunc_parallel_philox():
    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    key_size = 2
    # Original
    key = np.zeros(key_size, dtype=np.uint64)
    key[0] = V.comm.Get_rank()
    rg_base = randomgen.Generator(randomgen.Philox(counter=12345678910, key=key))
    # Firedrake wrapper
    rg_wrap = randomfunctiongen.Generator(randomfunctiongen.Philox(counter=12345678910))
    for i in range(1, 10):
        f = rg_wrap.beta(V, 0.3, 0.5)
        with f.dat.vec_ro as v:
            assert np.allclose(rg_base.beta(0.3, 0.5, size=(v.local_size,)), v.array[:])
