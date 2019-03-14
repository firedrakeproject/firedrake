from firedrake import *
from firedrake import randomfunctiongen
import pytest

import inspect
import numpy as np
import randomgen


brng_list = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass) if name != 'RandomGenerator']
meth_list = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator) if not name.startswith('_') and name not in ('state', 'poisson_lam_max', 'seed', 'random_integers', 'bytes', 'permutation', 'shuffle')]


@pytest.mark.parametrize("brng", brng_list)
def test_func(brng):

    mesh = UnitSquareMesh(10, 10)
    V0 = VectorFunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V1

    seed = 123456789

    # original
    rg_base = getattr(randomgen, brng)(seed=seed).generator

    # Firedrake wrapper
    rg_wrap = getattr(randomfunctiongen, brng)(seed=seed).generator

    for i in range(1, 10):
        for meth in meth_list:

            if meth == 'beta':
                args = (0.3, 0.5)

            elif meth == 'binomial':
                args = (7, 0.3)

            elif meth == 'chisquare':
                args = (7,)

            elif meth == 'choice':
                args = (i * i,)

            elif meth == 'complex_normal':
                continue
                args = ()

            elif meth == 'dirichlet':
                continue
                args = (np.arange(1, i * i+1),)

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
                args = (i, 2 * i, 3 * i - 1)

            elif meth == 'laplace':
                args = ()

            elif meth == 'logistic':
                args = ()

            elif meth == 'lognormal':
                args = ()

            elif meth == 'logseries':
                args = (0.3,)

            elif meth == 'multinomial':
                continue
                args = (i * i, [0.1, 0.2, 0.3, 0.4])

            elif meth == 'multivariate_normal':
                continue
                args = ([3.14, 2.71], [[1.00, 0.01], [0.01, 2.00]])

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
                args = (i * i,)

            elif meth == 'randint':
                args = (7,)

            elif meth == 'randn':
                args = (i * i,)

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

            f = getattr(rg_wrap, meth)(V, *args)
            with f.dat.vec_wo as v:
                if meth in ('rand', 'randn'):
                    assert np.allclose(getattr(rg_base, meth)(v.local_size), v.array[:])
                else:
                    kwargs = {'size': (v.local_size,)}
                    assert np.allclose(getattr(rg_base, meth)(*args, **kwargs), v.array[:])
