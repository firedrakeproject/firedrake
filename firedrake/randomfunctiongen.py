from mpi4py import MPI

import numpy as np

import inspect

import randomgen
from randomgen import entropy

from firedrake.function import Function
from ufl import FunctionSpace


__all__ = randomgen.__all__

_class_names = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass)]
_module_names = [name for name, _ in inspect.getmembers(randomgen, inspect.ismodule)]



def __getattr__(attr):

    if attr in _class_names:

        class RNGWrapper(getattr(randomgen, attr)):

            def __init__(self, *args, **kwargs):
                self.__class__.__name__ = attr
                super(RNGWrapper, self).__init__(*args, **kwargs)
                # At this point we cannot fully initialise the class
                # as we need V.comm.rank to set streams, so
                # remember the args and kwargs for later use.
                self._args = args
                self._kwargs = kwargs
                self._need_parallel_seed = True

            # If users invoke seed(...) externally, remember
            # args and kwargs. It has to be reseeded once
            # the function space V is given and it turns out
            # that V.comm.size > 1.
            def seed(self, *args, **kwargs):
                super(RNGWrapper, self).seed(*args, **kwargs)
                self._args = args
                self._kwargs = kwargs
                self._need_parallel_seed = True

            # Actually seed/reseed given V if V.comm.size > 1
            def _parallel_seed(self, V):
                if self.__class__.__name__ not in ('PCG64', 'Philox', 'ThreeFry'):
                    raise TypeError("Use 'PCG64', 'Philox', 'ThreeFry' for parallel RNG")
                elif self.__class__.__name__ == 'PCG64':
                    nproc = V.comm.size
                    rank = V.comm.rank
                    if 'seed' not in self._kwargs.keys() or self._kwargs['seed'] is None:
                        if rank == 0:
                            # generate a 128bit seed
                            entrpy = entropy.random_entropy(4)
                            seed = sum([int(entrpy[i]) * 2 ** (32 * i) for i in range(4)])
                        else:
                            seed = None
                        # All processes have to have the same seed
                        seed = V.comm.bcast(seed, root=0)
                        self._kwargs['seed'] = seed
                    # always overwrite random stream id, 'inc', for parallel
                    self._kwargs['inc'] = rank

                super(RNGWrapper, self).seed(*self._args, **self._kwargs)

                self._need_parallel_seed = False

        return RNGWrapper


_method_names = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator)
                  if not name.startswith('_') and name not in ('state', 'poisson_lam_max')]

class RandomGenerator(randomgen.RandomGenerator):

    def __init__(self, brng=None):
        super(RandomGenerator, self).__init__(brng=brng)

    def __getattribute__(self, attr):

        if attr in ('rand', 'randn'):

            def func(*args, **kwargs):
                if len(args) > 0 and isinstance(args[0], FunctionSpace):
                    # actually seed RNG using V.comm and extract size from V
                    if 'size' in kwargs.keys():
                        raise RuntimeError("Only one of V or size can be specified")
                    V = args[0]
                    if V.comm.size > 1 and self._basicrng._need_parallel_seed:
                        self._basicrng._parallel_seed(V)
                    f = Function(V)
                    f.dat.data[:] = getattr(super(RandomGenerator, self), attr)(*(f.dat.data.shape), **kwargs)
                    return f
                else:
                    # forward to the original implementation
                    return getattr(super(RandomGenerator, self), attr)(*args, **kwargs)
            return func

        elif attr in _method_names:

            def func(*args, **kwargs):
                if len(args) > 0 and isinstance(args[0], FunctionSpace):
                    # actually seed RNG using V.comm and extract size from V
                    if 'size' in kwargs.keys():
                        raise RuntimeError("Only one of V or size can be specified")
                    V = args[0]
                    if V.comm.size > 1 and self._basicrng._need_parallel_seed:
                        self._basicrng._parallel_seed(V)
                    f = Function(V)
                    args = args[1:]
                    kwargs['size'] = f.dat.data.shape
                    f.dat.data[:] = getattr(super(RandomGenerator, self), attr)(*args, **kwargs)
                    return f
                else:
                    # forward to the original implementation
                    return getattr(super(RandomGenerator, self), attr)(*args, **kwargs)
            return func
        else:
            return getattr(super(RandomGenerator, self), attr)

