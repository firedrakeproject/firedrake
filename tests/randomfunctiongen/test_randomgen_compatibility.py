from firedrake import randomfunctiongen
import pytest

import inspect
import numpy as np
import randomgen


brng_list = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass) if name not in ('RandomGenerator', 'RandomState')]
meth_list = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator) if not name.startswith('_') and name not in ('state', 'poisson_lam_max', 'seed', 'random_integers', 'brng')]


@pytest.mark.parametrize("brng", brng_list)
@pytest.mark.parametrize("meth", meth_list)
def test_randomgen_brng(brng, meth):

    seed = 123456789

    # original
    rg_base = getattr(randomgen, brng)(seed=seed).generator

    # Firedrake wrapper
    rg_wrap = getattr(randomfunctiongen, brng)(seed=seed).generator

    if meth == 'beta':
        args = (0.3, 0.5)
        kwargs = {'size': 17}

    elif meth == 'binomial':
        args = (7, 0.3)
        kwargs = {'size': 17}

    elif meth == 'bytes':
        a0 = getattr(rg_base, meth)(17 + 1)
        a1 = getattr(rg_wrap, meth)(17 + 1)
        assert a0 == a1
        return

    elif meth == 'chisquare':
        args = (7,)
        kwargs = {'size': 17}

    elif meth == 'choice':
        args = (17,)
        kwargs = {'size': 17}

    elif meth == 'complex_normal':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'dirichlet':
        args = (np.arange(1, 17+1),)
        kwargs = {'size': 17}

    elif meth == 'exponential':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'f':
        args = (7, 17)
        kwargs = {'size': 17}

    elif meth == 'gamma':
        args = (3.14,)
        kwargs = {'size': 17}

    elif meth == 'geometric':
        args = (0.5,)
        kwargs = {'size': 17}

    elif meth == 'gumbel':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'hypergeometric':
        args = (17, 2 * 17, 3 * 17 - 1)
        kwargs = {'size': 17}

    elif meth == 'laplace':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'logistic':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'lognormal':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'logseries':
        args = (0.3,)
        kwargs = {'size': 17}

    elif meth == 'multinomial':
        args = (17, [0.1, 0.2, 0.3, 0.4])
        kwargs = {'size': 17}

    elif meth == 'multivariate_normal':
        args = ([3.14, 2.71], [[1.00, 0.01], [0.01, 2.00]])
        kwargs = {'size': 17}

    elif meth == 'negative_binomial':
        args = (7, 0.3)
        kwargs = {'size': 17}

    elif meth == 'noncentral_chisquare':
        args = (7, 3.14)
        kwargs = {'size': 17}

    elif meth == 'noncentral_f':
        args = (7, 17, 3.14)
        kwargs = {'size': 17}

    elif meth == 'normal':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'pareto':
        args = (3.14,)
        kwargs = {'size': 17}

    elif meth == 'permutation':
        args = (10 * 17,)
        kwargs = {}

    elif meth == 'poisson':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'power':
        args = (3.14,)
        kwargs = {'size': 17}

    elif meth == 'rand':
        args = (17,)
        kwargs = {}

    elif meth == 'randint':
        args = (7,)
        kwargs = {'size': 17}

    elif meth == 'randn':
        args = (17,)
        kwargs = {}

    elif meth == 'random_raw':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'random_sample':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'random_uintegers':
        args = (7,)
        kwargs = {}

    elif meth == 'rayleigh':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'shuffle':
        arr0 = np.arange(17 + 10)
        arr1 = np.arange(17 + 10)
        getattr(rg_base, meth)(arr0)
        getattr(rg_wrap, meth)(arr1)
        assert np.allclose(arr0, arr1)
        return

    elif meth == 'standard_cauchy':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'standard_exponential':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'standard_gamma':
        args = (3.14,)
        kwargs = {'size': 17}

    elif meth == 'standard_normal':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'standard_t':
        args = (7,)
        kwargs = {'size': 17}

    elif meth == 'tomaxint':
        args = (7,)
        kwargs = {}

    elif meth == 'triangular':
        args = (2.71, 3.14, 10.)
        kwargs = {'size': 17}

    elif meth == 'uniform':
        args = ()
        kwargs = {'size': 17}

    elif meth == 'vonmises':
        args = (2.71, 3.14)
        kwargs = {'size': 17}

    elif meth == 'wald':
        args = (2.71, 3.14)
        kwargs = {'size': 17}

    elif meth == 'weibull':
        args = (3.14,)
        kwargs = {'size': 17}

    elif meth == 'zipf':
        args = (3.14,)
        kwargs = {'size': 17}

    for i in range(1, 10):
        assert np.allclose(getattr(rg_base, meth)(*args, **kwargs), getattr(rg_wrap, meth)(*args, **kwargs))
