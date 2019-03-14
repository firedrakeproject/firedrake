from firedrake import randomfunctiongen
import pytest

import inspect
import numpy as np
import randomgen


brng_list = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass) if name != 'RandomGenerator']
meth_list = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator) if not name.startswith('_') and name not in ('state', 'poisson_lam_max', 'seed', 'random_integers')]


@pytest.mark.parametrize("brng", brng_list)
def test_brng(brng):

    seed = 123456789

    # original
    rg_base = getattr(randomgen, brng)(seed=seed).generator

    # Firedrake wrapper
    rg_wrap = getattr(randomfunctiongen, brng)(seed=seed).generator

    for i in range(1, 10):
        for meth in meth_list:

            if meth == 'beta':
                args = (0.3, 0.5)
                kwargs = {'size': i * i}

            elif meth == 'binomial':
                args = (7, 0.3)
                kwargs = {'size': i * i}

            elif meth == 'bytes':
                a0 = getattr(rg_base, meth)(i * i + 1)
                a1 = getattr(rg_wrap, meth)(i * i + 1)
                assert a0 == a1
                continue

            elif meth == 'chisquare':
                args = (7,)
                kwargs = {'size': i * i}

            elif meth == 'choice':
                args = (i * i,)
                kwargs = {'size': i * i}

            elif meth == 'complex_normal':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'dirichlet':
                args = (np.arange(1, i * i+1),)
                kwargs = {'size': i * i}

            elif meth == 'exponential':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'f':
                args = (7, 17)
                kwargs = {'size': i * i}

            elif meth == 'gamma':
                args = (3.14,)
                kwargs = {'size': i * i}

            elif meth == 'geometric':
                args = (3.14,)
                kwargs = {'size': i * i}

            elif meth == 'gumbel':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'hypergeometric':
                args = (i, 2 * i, 3 * i - 1)
                kwargs = {'size': i * i}

            elif meth == 'laplace':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'logistic':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'lognormal':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'logseries':
                args = (0.3,)
                kwargs = {'size': i * i}

            elif meth == 'multinomial':
                args = (i * i, [0.1, 0.2, 0.3, 0.4])
                kwargs = {'size': i * i}

            elif meth == 'multivariate_normal':
                args = ([3.14, 2.71], [[1.00, 0.01], [0.01, 2.00]])
                kwargs = {'size': i * i}

            elif meth == 'negative_binomial':
                args = (7, 0.3)
                kwargs = {'size': i * i}

            elif meth == 'noncentral_chisquare':
                args = (7, 3.14)
                kwargs = {'size': i * i}

            elif meth == 'noncentral_f':
                args = (7, 17, 3.14)
                kwargs = {'size': i * i}

            elif meth == 'normal':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'pareto':
                args = (3.14,)
                kwargs = {'size': i * i}

            elif meth == 'permutation':
                args = (10 * i,)
                kwargs = {}

            elif meth == 'poisson':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'power':
                args = (3.14,)
                kwargs = {'size': i * i}

            elif meth == 'rand':
                args = (i * i,)
                kwargs = {}

            elif meth == 'randint':
                args = (7,)
                kwargs = {'size': i * i}

            elif meth == 'randn':
                args = (i * i,)
                kwargs = {}

            elif meth == 'random_raw':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'random_sample':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'random_uintegers':
                args = (7,)
                kwargs = {}

            elif meth == 'rayleigh':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'shuffle':
                arr0 = np.arange(i * i + 10)
                arr1 = np.arange(i * i + 10)
                getattr(rg_base, meth)(arr0)
                getattr(rg_wrap, meth)(arr1)
                assert np.allclose(arr0, arr1)
                continue

            elif meth == 'standard_cauchy':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'standard_exponential':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'standard_gamma':
                args = (3.14,)
                kwargs = {'size': i * i}

            elif meth == 'standard_normal':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'standard_t':
                args = (7,)
                kwargs = {'size': i * i}

            elif meth == 'tomaxint':
                args = (7,)
                kwargs = {}

            elif meth == 'triangular':
                args = (2.71, 3.14, 10.)
                kwargs = {'size': i * i}

            elif meth == 'uniform':
                args = ()
                kwargs = {'size': i * i}

            elif meth == 'vonmises':
                args = (2.71, 3.14)
                kwargs = {'size': i * i}

            elif meth == 'wald':
                args = (2.71, 3.14)
                kwargs = {'size': i * i}

            elif meth == 'weibull':
                args = (3.14,)
                kwargs = {'size': i * i}

            elif meth == 'zipf':
                args = (3.14,)
                kwargs = {'size': i * i}

            assert np.allclose(getattr(rg_base, meth)(*args, **kwargs), getattr(rg_wrap, meth)(*args, **kwargs))
