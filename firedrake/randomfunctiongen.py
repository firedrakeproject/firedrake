import numpy as np
import inspect
import randomgen

from firedrake.function import Function
from ufl import FunctionSpace


"""

This module wraps 'randomgen' and enables
firedrake users to pass a FunctionSpace
to generate random functions.

Usage:
FunctionSpace, V, has to be passed as
the first argument, e.g.:

test.py:

>>> from firedrake import *
>>> mesh = UnitSquareMesh(2,2)
>>> V = FunctionSpace(mesh, "CG", 1)
>>> pcg = PCG64(seed=123456789)
>>> rg = RandomGenerator(pcg)
>>> f_beta = rg.beta(V, 1.0, 2.0)
>>> print(f_beta.dat.data)
[0.56462514 0.11585311 0.01247943 0.398984   0.19097059 0.5446709
 0.1078666  0.2178807  0.64848515]

In parallel, "mpiexec -n 2 python3 test.py" shows two independent streams:

[0.56462514 0.11585311 0.01247943]
[0.03780042 0.15495691 0.42125296 0.56395561 0.07342598 0.24215732]

"""


__all__ = randomgen.__all__


_class_names = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass)]
_method_names = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator)
                 if not name.startswith('_') and name not in ('state', 'poisson_lam_max')]


def __getattr__(attr):

    if attr == 'RandomGenerator':

        class RandomGenerator(randomgen.RandomGenerator):

            def __init__(self, brng=None):
                super(RandomGenerator, self).__init__(brng=brng)

            def __getattribute__(self, attr):

                # These methods are not to be used with V
                if attr in ('bytes', 'shuffle', 'permutation'):

                    def func(*args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            raise NotImplementedError("RandomGenerator.%s does not take FunctionSpace as argument" % attr)
                        else:
                            return getattr(super(RandomGenerator, self), attr)(*args, **kwargs)
                    return func

                # Arguments for these two are slightly different
                elif attr in ('rand', 'randn'):

                    def func(*args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            # actually seed RNG using V.comm and extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            if V.comm.size > 1:
                                self._basicrng._parallel_seed(V)
                            f = Function(V)
                            # Deal with MixedFunctionSpaces
                            for i in range(len(f.dat)):
                                f.dat[i].data[:] = getattr(self, attr)(*(f.dat[i].data.shape), **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            return getattr(super(RandomGenerator, self), attr)(*args, **kwargs)
                    return func

                # Other methods here
                elif attr in _method_names:

                    def func(*args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            # actually seed RNG using V.comm and extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            self._basicrng._parallel_seed(V)
                            f = Function(V)
                            args = args[1:]
                            # Deal with MixedFunctionSpaces
                            for i in range(len(f.dat)):
                                kwargs['size'] = f.dat[i].data.shape
                                f.dat[i].data[:] = getattr(self, attr)(*args, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            return getattr(super(RandomGenerator, self), attr)(*args, **kwargs)
                    return func

                else:
                    return getattr(super(RandomGenerator, self), attr)

        return RandomGenerator

    elif attr in _class_names:

        class RNGWrapperBase(getattr(randomgen, attr)):

            def __init__(self, *args, **kwargs):
                super(RNGWrapperBase, self).__init__(*args, **kwargs)
                # Save args and kwargs.  We must 'seed' again with
                # appropriate parallel stream ids ('inc' in PCG64;
                # 'key' in Philox/ThreeFry) once V is given and it
                # turns out that V.comm.size > 1.
                self._args = args
                self._kwargs = kwargs
                self._need_parallel_seed = True
                self._generator = None

            def seed(self, *args, **kwargs):
                # If users invoke seed(...) externally, execute
                # parent seed() and store new args and kwargs
                # for parallel generation.
                super(RNGWrapperBase, self).seed(*args, **kwargs)
                self._args = args
                self._kwargs = kwargs
                self._need_parallel_seed = True

            @property
            def generator(self):
                if self._generator is None:
                    from . import RandomGenerator
                    self._generator = RandomGenerator(brng=self)
                return self._generator

            def _parallel_seed(self, V):
                # Actually (re)seed given V when V.comm.size > 1.
                # Use examples in https://bashtage.github.io/randomgen/parallel.html
                # with appropriate changes.
                raise NotImplementedError("Overwrite")

        if attr == 'PCG64':

            class RNGWrapper(RNGWrapperBase):

                def __init__(self, *args, **kwargs):
                    super(RNGWrapper, self).__init__(*args, **kwargs)

                def _parallel_seed(self, V):

                    if not self._need_parallel_seed:
                        return

                    rank = V.comm.rank
                    if 'seed' not in self._kwargs.keys() or self._kwargs['seed'] is None:
                        if rank == 0:
                            # generate a 128bit seed
                            entropy = randomgen.entropy.random_entropy(4)
                            seed = sum([int(entropy[i]) * 2 ** (32 * i) for i in range(4)])
                        else:
                            seed = None
                        # All processes have to have the same seed
                        seed = V.comm.bcast(seed, root=0)
                        self._kwargs['seed'] = seed
                    # Use rank to generate multiple streams.
                    # Always overwrite 'inc'.
                    self._kwargs['inc'] = rank

                    self.seed(*self._args, **self._kwargs)
                    self._need_parallel_seed = False

        elif attr in ('Philox', 'ThreeFry'):

            class RNGWrapper(RNGWrapperBase):

                def __init__(self, *args, **kwargs):
                    super(RNGWrapper, self).__init__(*args, **kwargs)

                def _parallel_seed(self, V):

                    if not self._need_parallel_seed:
                        return

                    rank = V.comm.rank
                    if 'seed' in self._kwargs.keys() and self._kwargs['seed'] is not None:
                        raise TypeError("'seed' should not be used when generating a random function in parallel.  A random 'key' is automatically generated unless specified.")
                    if 'key' not in self._kwargs.keys() or self._kwargs['key'] is None:
                        if rank == 0:
                            key = randomgen.entropy.random_entropy(8)
                            key = key.view(np.uint64)
                            key[0] = 0
                        else:
                            key = None
                        key = V.comm.bcast(key, root=0)
                    else:
                        key = self._kwargs['key']
                    if rank == 0:
                        step = np.zeros(4, dtype=np.uint64)
                        step[0] = 1
                    else:
                        step = None
                    step = V.comm.bcast(step, root=0)
                    # Use rank to generate multiple streams
                    self._kwargs['key'] = key + rank * step

                    self.seed(*self._args, **self._kwargs)
                    self._need_parallel_seed = False

        else:

            class RNGWrapper(RNGWrapperBase):

                def __init__(self, *args, **kwargs):
                    super(RNGWrapper, self).__init__(*args, **kwargs)

                def _parallel_seed(self, V):

                    if V.comm.size == 1 or not self._need_parallel_seed:
                        return

                    raise TypeError("Use 'PCG64', 'Philox', 'ThreeFry' for parallel RNG")

        return RNGWrapper

    elif attr not in ('hypergeometric', 'multinomial', 'random_sample'):
        return getattr(randomgen, attr)

    else:
        return getattr(randomgen.generator, attr)


# __getattr__ on module level only works for 3.7+

import sys

if sys.version_info < (3, 7, 0):
    class Wrapper(object):
        def __getattr__(self, attr):
            return __getattr__(attr)
    sys.modules[__name__] = Wrapper()
