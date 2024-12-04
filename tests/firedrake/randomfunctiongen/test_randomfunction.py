from firedrake import *
from firedrake import randomfunctiongen

import pytest
import numpy as np
import numpy.random as randomgen


@pytest.mark.parametrize("brng", ['MT19937', 'PCG64', 'Philox', 'SFC64'])
@pytest.mark.parametrize("meth_args", [('beta', (0.3, 0.5)),
                                       ('binomial', (7, 0.3)),
                                       ('chisquare', (7,)),
                                       ('choice', (1234 * 5678,)),
                                       ('exponential', ()),
                                       ('f', (7, 17)),
                                       ('gamma', (3.14,)),
                                       ('geometric', (0.5,)),
                                       ('gumbel', ()),
                                       ('hypergeometric', (17, 2 * 17, 3 * 17 - 1)),
                                       ('laplace', ()),
                                       ('logistic', ()),
                                       ('lognormal', ()),
                                       ('logseries', (0.3,)),
                                       ('negative_binomial', (7, 0.3)),
                                       ('noncentral_chisquare', (7, 3.14)),
                                       ('noncentral_f', (7, 17, 3.14)),
                                       ('normal', ()),
                                       ('pareto', (3.14,)),
                                       ('poisson', ()),
                                       ('power', (3.14,)),
                                       ('random', ()),
                                       ('rayleigh', ()),
                                       ('standard_cauchy', ()),
                                       ('standard_exponential', ()),
                                       ('standard_gamma', (3.14,)),
                                       ('standard_normal', ()),
                                       ('standard_t', (7,)),
                                       ('triangular', (2.71, 3.14, 10.)),
                                       ('uniform', ()),
                                       ('vonmises', (2.71, 3.14)),
                                       ('wald', (2.71, 3.14)),
                                       ('weibull', (3.14,)),
                                       ('zipf', (3.14,))])
def test_randomfunc(brng, meth_args):

    meth, args = meth_args

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


@pytest.mark.skip(reason="Require numpy>=1.25.0")
def test_randomfunc_generator_spawn():
    parent = randomfunctiongen.Generator()
    children = parent.spawn(4)
    assert all([isinstance(child, type(parent))] for child in children)
    assert all([isinstance(child.bit_generator, type(parent.bit_generator)) for child in children])
    assert all([child.bit_generator._seed_seq.entropy == parent.bit_generator._seed_seq.entropy for child in children])
    assert all([child.bit_generator._seed_seq.spawn_key == parent.bit_generator._seed_seq.spawn_key + (i, ) for i, child in enumerate(children)])
    assert all([child.bit_generator._seed_seq.pool_size == parent.bit_generator._seed_seq.pool_size for child in children])
