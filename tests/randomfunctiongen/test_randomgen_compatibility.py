from firedrake import randomfunctiongen
import pytest

import numpy as np
import numpy.random as randomgen
import inspect


@pytest.mark.parametrize("brng", ['MT19937', 'PCG64', 'Philox', 'SFC64'])
@pytest.mark.parametrize("meth_args_kwargs", [('beta', (0.3, 0.5), {'size': 17}),
                                              ('binomial', (7, 0.3), {'size': 17}),
                                              ('bytes', (), {}),
                                              ('chisquare', (7,), {'size': 17}),
                                              ('choice', (17,), {'size': 17}),
                                              ('dirichlet', (np.arange(1, 17+1),), {'size': 17}),
                                              ('exponential', (), {'size': 17}),
                                              ('f', (7, 17), {'size': 17}),
                                              ('gamma', (3.14,), {'size': 17}),
                                              ('geometric', (0.5,), {'size': 17}),
                                              ('gumbel', (), {'size': 17}),
                                              ('hypergeometric', (17, 2 * 17, 3 * 17 - 1), {'size': 17}),
                                              ('laplace', (), {'size': 17}),
                                              ('logistic', (), {'size': 17}),
                                              ('lognormal', (), {'size': 17}),
                                              ('logseries', (0.3,), {'size': 17}),
                                              ('multinomial', (17, [0.1, 0.2, 0.3, 0.4]), {'size': 17}),
                                              ('multivariate_hypergeometric', ([16, 8, 4], 6), {'size': 3}),
                                              ('multivariate_normal', ([3.14, 2.71], [[1.00, 0.01], [0.01, 2.00]]), {'size': 17}),
                                              ('negative_binomial', (7, 0.3), {'size': 17}),
                                              ('noncentral_chisquare', (7, 3.14), {'size': 17}),
                                              ('noncentral_f', (7, 17, 3.14), {'size': 17}),
                                              ('normal', (), {'size': 17}),
                                              ('pareto', (3.14,), {'size': 17}),
                                              ('permutation', (10 * 17,), {}),
                                              ('poisson', (), {'size': 17}),
                                              ('power', (3.14,), {'size': 17}),
                                              ('random', (), {'size': 17}),
                                              ('rayleigh', (), {'size': 17}),
                                              ('shuffle', (), {}),
                                              ('standard_cauchy', (), {'size': 17}),
                                              ('standard_exponential', (), {'size': 17}),
                                              ('standard_gamma', (3.14,), {'size': 17}),
                                              ('standard_normal', (), {'size': 17}),
                                              ('standard_t', (7,), {'size': 17}),
                                              ('triangular', (2.71, 3.14, 10.), {'size': 17}),
                                              ('uniform', (), {'size': 17}),
                                              ('vonmises', (2.71, 3.14), {'size': 17}),
                                              ('wald', (2.71, 3.14), {'size': 17}),
                                              ('weibull', (3.14,), {'size': 17}),
                                              ('zipf', (3.14,), {'size': 17})])
def test_randomgen_brng(brng, meth_args_kwargs):

    meth, args, kwargs = meth_args_kwargs

    seed = 123456789
    # Original
    bgen = getattr(randomgen, brng)(seed=seed)
    if brng == 'PCG64':
        state = bgen.state
        state['state'] = {'state': seed, 'inc': 0}
        bgen.state = state
    rg_base = randomgen.Generator(bgen)
    # Firedrake wrapper
    fgen = getattr(randomfunctiongen, brng)(seed=seed)
    if brng == 'PCG64':
        state = fgen.state
        state['state'] = {'state': seed, 'inc': 0}
        fgen.state = state
    rg_wrap = randomfunctiongen.Generator(fgen)

    if meth == 'bytes':
        a0 = getattr(rg_base, meth)(17)
        a1 = getattr(rg_wrap, meth)(17)
        assert a0 == a1
        return
    elif meth == 'shuffle':
        arr0 = np.arange(17)
        arr1 = np.arange(17)
        getattr(rg_base, meth)(arr0)
        getattr(rg_wrap, meth)(arr1)
        assert np.allclose(arr0, arr1)
        return

    for i in range(1, 10):
        assert np.allclose(getattr(rg_base, meth)(*args, **kwargs), getattr(rg_wrap, meth)(*args, **kwargs))


def test_randomgen_known_attributes():
    _found_attributes = [name for name, _ in inspect.getmembers(randomgen) if not name.startswith('_')]
    _found_generator_attributes = [name for name, _ in inspect.getmembers(randomgen.Generator) if not name.startswith('_')]
    A = set(_found_attributes)
    B = set(randomfunctiongen._known_attributes)
    assert A.issubset(B)
    A = set(_found_generator_attributes)
    B = set(randomfunctiongen._known_generator_attributes)
    assert A.issubset(B)
