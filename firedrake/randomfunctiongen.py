"""

This module wraps `randomgen <https://pypi.org/project/randomgen/>`__ and enables
firedrake users to pass a FunctionSpace
to generate random functions.

Usage:
FunctionSpace, V, has to be passed as
the first argument, e.g.:

Example
_______

>>> from firedrake import *
>>> mesh = UnitSquareMesh(2,2)
>>> V = FunctionSpace(mesh, "CG", 1)
>>> pcg = PCG64(seed=123456789)
>>> rg = RandomGenerator(pcg)
>>> f_beta = rg.beta(V, 1.0, 2.0)
>>> print(f_beta.dat.data)
[0.56462514 0.11585311 0.01247943 0.398984   0.19097059 0.5446709
 0.1078666  0.2178807  0.64848515]





"""


import numpy as np
import inspect
import randomgen

from firedrake.function import Function
from ufl import FunctionSpace


__all__ = [item for item in randomgen.__all__ if item not in ('hypergeometric', 'multinomial', 'random_sample')]


_class_names = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass)]
_method_names = [name for name, _ in inspect.getmembers(randomgen.RandomGenerator)
                 if not name.startswith('_') and name not in ('state', 'poisson_lam_max')]


def __getattr__(module_attr):

    # Reformat the original documentation
    def _clean_doc_string(strng):

        # remove redundant '\n' in math mode
        st = ""
        _in_math_mode = False
        for s in strng.splitlines():
            if '.. math::' in s:
                _in_math_mode = True
            st += s + '\n' if not _in_math_mode else s
            if _in_math_mode and s == '':
                st += '\n\n'
                _in_math_mode = False

        strng = st.replace('\n\n', '\n')
        strng = st.replace('optional\n', 'optional. \n')

        st = ""
        s_ = ''
        for s in strng.splitlines():
            s = s.lstrip()
            if '----' in s:
                st += s_ + s
                s_ = ''
            else:
                st += s_ + '\n'
                s_ = s
        st += s_

        st = st.replace('\\P', 'P')
        st = st.replace('Returns-------\nout :', ':returns:')
        st = st.replace('Returns-------\nsamples :', ':returns:')
        st = st.replace('Returns-------\nZ :', ':returns:')
        st = st.replace('Raises------\nValueError', '\n:raises ValueError:')
        st = st.replace('Raises-------\nValueError', '\n:raises ValueError:')

        strng = st
        st = ""
        for s in strng.splitlines():
            if 'from randomgen ' in s:
                continue
            if 'd0, d1, ..., dn :' in s:
                st += ':arg d0, d1, ..., dn' + s[16:]
                continue
            elif ' ' in s and s.find(' ') != len(s)-1:
                n = s.find(' ')
                if s[n+1] == ':' and (n < len(s) - 2 and s[n+2] == ' '):
                    param_name = s[:n]
                    if param_name not in ('where', 'the', 'of', 'standard_normal') and 'scipy.stats' not in param_name and 'numpy.random' not in param_name:
                            st += ':arg ' + param_name + s[n+1:]
                            continue
            st += s + '\n'

        st = st.replace('randomgen', 'randomfunctiongen')

        strng = st
        st = ""
        sp = ' ' * 4
        # Remove redundant '\n's
        _in_list = False
        name=''
        for s in strng.splitlines():
            if _in_list:
                if '----' in s:
                    st += '\n'
                    s_ = s.replace('-', '').replace('\n', '')
                    s__ = s.replace(s_, '').replace('\n', '')
                    _in_list = False
                    if s_ in ('Parameters', 'Returns', 'References'):
                        _in_list = True
                    if s_ == 'See Also':
                        st += '\n' + sp * 2 + '**' + s_ + '**\n'
                    #elif s_ == 'Raises':
                    #    st += '\n' + sp * 2 + '**' + s_ + '** '
                    elif s_ == 'Notes':
                        st += '\n' + sp * 2 + '**' + s_ + '**\n'
                    elif s_ == 'References':
                        st += '\n' + sp * 2 + '**' + s_ + '**\n'
                    elif s_ == 'Examples':
                        st += '\n' + sp * 2 + '**' + s_ + '**'+ '\n' + '\n'
                    continue
                st += '\n' + sp * 2 + s if (':arg' in s) or (':returns:' in s) or ('.. [' in s) else ' ' + s
                continue

            # Insert doc for Firedrake wrapper
            if "(d0, d1, ..., dn, dtype='d')" in s:
                name = s[:s.find('(')]
                st += sp * 2 + name + '(*args, **kwargs)\n\n'
                s = '*' + name + '* ' + s[len(name):]
                st += sp * 2 + s.replace('(', '(*').replace('d0, d1, ..., dn', 'V').replace(')', '*)') + '\n\n'
                st += sp * 2 + 'Generate a function f = Function(V), internally call the original method *' + name + '* with given arguments, and return f.\n\n'
                st += sp * 2 + ':arg V: :class:`.FunctionSpace`\n\n'
                st += sp * 2 + ':returns: :class:`.Function`\n\n'
                st += sp * 2 + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif 'size=None' in s:
                name = s[:s.find('(')]
                st += sp * 2 + name + '(*args, **kwargs)\n\n'
                s = '*' + name + '* ' + s[len(name):]
                st += sp * 2 + s.replace('(', '(*V, ').replace(', size=None', '').replace(')', '*)') + '\n\n'
                st += sp * 2 + 'Generate a :class:`.Function` f = Function(V), randomise it by calling the original method *' + name + '* (...) with given arguments, and return f.\n\n'
                st += sp * 2 + ':arg V: :class:`.FunctionSpace`\n\n'
                st += sp * 2 + ':returns: :class:`.Function`\n\n'
                st += sp * 2 + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif '----' in s:
                st += '\n'
                s_ = s.replace('-', '').replace('\n', '')
                s__ = s.replace(s_, '').replace('\n', '')
                if s_ in ('Parameters', 'Returns', 'References'):
                    _in_list = True
                if s_ == 'See Also':
                    st += '\n' + sp * 2 + '**' + s_ + '**\n'
                #elif s_ == 'Raises':
                #    st += '\n' + sp * 2 + '**' + s_ + '** '
                elif s_ == 'Notes':
                    st += '\n' + sp * 2 + '**' + s_ + '**\n'
                elif s_ == 'References':
                    st += '\n' + sp * 2 + '**' + s_ + '**\n'
                elif s_ == 'Examples':
                    st += '\n' + sp * 2 + '**' + s_ + '**'+ '\n' + '\n'
            elif '.. math::' in s:
                st += '\n' + sp * 2 + s + '\n\n'
            else:
                st += sp * 2 + s + '\n'

        st = st.replace(' Drawn samples from the', '. Drawn samples from the')

        return st

    if module_attr == 'RandomGenerator':

        _Base = getattr(randomgen, module_attr)

        _dict = {}

        # Use decorator to add doc strings to
        # auto generated functions
        def add_doc_string(doc_string):

            def f(func):
                func.__doc__ = _clean_doc_string(doc_string)
                return func

            return f

        for class_attr, _ in inspect.getmembers(randomgen.__getattribute__(module_attr)):

            # These methods are not to be used with V
            if class_attr in ('bytes', 'shuffle', 'permutation'):

                # class_attr is mutable, so we have to wrap func with
                # another function to lock the value of class_attr
                def funcgen(c_a):

                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            raise NotImplementedError("%s.%s does not take FunctionSpace as argument" % (module_attr, c_a))
                        else:
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)
                    return func

                _dict[class_attr] = funcgen(class_attr)

            # Arguments for these two are slightly different
            elif class_attr in ('rand', 'randn'):

                # Here, too, wrap func with funcgen.
                def funcgen(c_a):

                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            # actually seed RNG using V.comm and extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            if V.comm.size > 1:
                                self._basicrng._parallel_seed(V)
                            f = Function(V)
                            with f.dat.vec_wo as v:
                                v.array[:] = self.__getattribute__(c_a)(v.local_size, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            #return super(_Wrapper, self).__getattribute__(c_a)(*args, **kwargs)
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)
                    return func

                _dict[class_attr] = funcgen(class_attr)

            # Other methods here
            elif class_attr in _method_names:

                # Here, too, wrap func with funcgen.
                def funcgen(c_a):

                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            # actually seed RNG using V.comm and extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            self._basicrng._parallel_seed(V)
                            f = Function(V)
                            args = args[1:]
                            with f.dat.vec_wo as v:
                                kwargs['size'] = (v.local_size,)
                                v.array[:] = self.__getattribute__(c_a)(*args, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)
                    return func

                _dict[class_attr] = funcgen(class_attr)

        _dict["__doc__"] = _Base.__doc__
        _Wrapper = type(module_attr, (_Base,), _dict)

        return _Wrapper

    elif module_attr in _class_names:

        _Base = getattr(randomgen, module_attr)

        def __init__(self, *args, **kwargs):
            super(_Wrapper, self).__init__(*args, **kwargs)
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
            super(_Wrapper, self).seed(*args, **kwargs)
            self._args = args
            self._kwargs = kwargs
            self._need_parallel_seed = True

        @property
        def generator(self):
            if self._generator is None:
                from . import RandomGenerator
                self._generator = RandomGenerator(brng=self)
            return self._generator

        if module_attr == 'PCG64':

            # Actually (re)seed given V when V.comm.size > 1.
            # Use examples in https://bashtage.github.io/randomgen/parallel.html
            # with appropriate changes.
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

        elif module_attr in ('Philox', 'ThreeFry'):

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

            def _parallel_seed(self, V):

                raise TypeError("Use 'PCG64', 'Philox', 'ThreeFry' for parallel RNG")

        _dict = {"__init__": __init__,
                 "seed": seed,
                 "generator": generator,
                 "_parallel_seed": _parallel_seed,
                 "__doc__": _clean_doc_string(getattr(randomgen, module_attr).__doc__)}

        _Wrapper = type(module_attr, (_Base,), _dict)

        return _Wrapper

    else:

        return getattr(randomgen, module_attr)


# __getattr__ on module level only works for 3.7+

import sys

if sys.version_info < (3, 7, 0):
    class Wrapper(object):
        def __getattr__(self, attr):
            return __getattr__(attr)
    sys.modules[__name__] = Wrapper()
