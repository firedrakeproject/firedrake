"""

This module wraps `randomgen <https://pypi.org/project/randomgen/>`__
and enables users to generate a randomised :class:`.Function`
from a :class:`.FunctionSpace`.
This module inherits all attributes from `randomgen <https://pypi.org/project/randomgen/>`__.

"""
from mpi4py import MPI

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
    def _reformat_doc(strng):

        # Reformat code examples
        st = ""
        flag = False
        strng = strng.replace('rs[i].jump(i)', '...     rs[i].jump(i)')
        strng = strng.replace('... ', '>>> ')
        for s in strng.splitlines():
            if flag and not ('>>>' in s or s.lstrip() == '' or s.lstrip()[0] == '#'):
                s = '>>> #' + s
            st += s + '\n'
            flag = '>>>' in s

        # Reformat the body
        strng = st
        st = ""
        for s in strng.splitlines():
            if 'from randomgen ' not in s:
                st += s.lstrip() + '\n'
        st = st.replace('randomgen', 'randomfunctiongen')
        st = st.replace('Parameters\n----------\n', '')
        st = st.replace('Returns\n-------\nout :', ':returns:')
        st = st.replace('Returns\n-------\nsamples :', ':returns:')
        st = st.replace('Returns\n-------\nZ :', ':returns:')
        st = st.replace('Raises\n-------\nValueError', '\n:raises ValueError:')
        st = st.replace('Raises\n------\nValueError', '\n:raises ValueError:')
        st = st.replace('Examples\n--------', '**Examples**\n')
        st = st.replace('Notes\n-----', '**Notes**\n')
        st = st.replace('See Also\n--------', '**See Also**\n')
        st = st.replace('References\n----------', '**References**')
        st = st.replace('\\P', 'P')
        st = st.replace('htm\n', 'html\n')
        st = st.replace('\n# ', '\n>>> # ')
        st = st.replace(':\n\n>>> ', '::\n\n    ')
        st = st.replace('.\n\n>>> ', '::\n\n    ')
        st = st.replace('\n\n>>> ', '::\n\n    ')
        st = st.replace('\n>>> ', '\n    ')

        # Convert some_par : -> :arg some_par:
        strng = st
        st = ""
        for s in strng.splitlines():
            if 'd0, d1, ..., dn :' in s:
                st += ':arg d0, d1, ..., dn' + s[16:] + '\n'
                continue
            elif ' ' in s and s.find(' ') != len(s)-1:
                n = s.find(' ')
                if s[n+1] == ':' and (n < len(s) - 2 and s[n+2] == ' '):
                    param_name = s[:n]
                    if param_name not in ('where', 'the', 'of', 'standard_normal') and 'scipy.stats' not in param_name and 'numpy.random' not in param_name:
                        st += ':arg ' + param_name + s[n+1:] + '\n'
                        continue
            st += s + '\n'

        # Remove redundant '\n' characters
        strng = st
        st = ""
        _in_block = False
        for s in strng.splitlines():
            if ':arg' in s or ':returns:' in s or '.. [' in s or '.. math::' in s:
                st += '\n' + s
                _in_block = True
                continue
            if _in_block:
                if s == '':
                    _in_block = False
                    st += '\n\n'
                else:
                    st += '. ' + s if s[0].isupper() and st[-1] != '.' else ' ' + s
            else:
                st += s + '\n'

        # Insert Firedrake wrapper doc and apply correct indentations
        strng = st
        st = ""
        sp = ' ' * 8
        for s in strng.splitlines():
            if "(d0, d1, ..., dn, dtype='d')" in s:
                name = s[:s.find('(')]
                st += sp + name + '(*args, **kwargs)\n\n'
                s = '*' + name + '* ' + s[len(name):]
                st += sp + s.replace('(', '(*').replace('d0, d1, ..., dn', 'V').replace(')', '*)') + '\n\n'
                st += sp + 'Generate a function :math:`f` = Function(V), internally call the original method *' + name + '* with given arguments, and return :math:`f`.\n\n'
                st += sp + ':arg V: :class:`.FunctionSpace`\n\n'
                st += sp + ':returns: :class:`.Function`\n\n'
                st += sp + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif 'size=None' in s:
                name = s[:s.find('(')]
                st += sp + name + '(*args, **kwargs)\n\n'
                s = '*' + name + '* ' + s[len(name):]
                st += sp + s.replace('(', '(*V, ').replace(', size=None', '').replace(')', '*)') + '\n\n'
                st += sp + 'Generate a :class:`.Function` f = Function(V), randomise it by calling the original method *' + name + '* (...) with given arguments, and return f.\n\n'
                st += sp + ':arg V: :class:`.FunctionSpace`\n\n'
                st += sp + ':returns: :class:`.Function`\n\n'
                st += sp + "The original documentation is found at `<https://bashtage.github.io/randomgen/generated/randomgen.legacy.legacy.LegacyGenerator." + name + ".html>`__, which is reproduced below with appropriate changes.\n\n"
                st += sp + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif '.. math::' in s:
                st += '\n' + sp + s + '\n\n'
            else:
                st += sp + s + '\n'

        return st

    if module_attr == 'RandomGenerator':

        _Base = getattr(randomgen, module_attr)

        _dict = {}

        _dict["__doc__"] = ("\n"
                            "    Container for the Basic Random Number Generators.\n"
                            "\n"
                            "    Users can pass to many of the available distribution methods\n"
                            "    a :class:`.FunctionSpace` as the first argument to obtain a randomised :class:`.Function`.\n"
                            "\n"
                            "    .. note ::\n"
                            "        FunctionSpace, V, has to be passed as\n"
                            "        the first argument.\n"
                            "\n"
                            "    **Example**::\n"
                            "\n"
                            "        from firedrake import *\n"
                            "        mesh = UnitSquareMesh(2,2)\n"
                            "        V = FunctionSpace(mesh, 'CG', 1)\n"
                            "        pcg = PCG64(seed=123456789)\n"
                            "        rg = RandomGenerator(pcg)\n"
                            "        f_beta = rg.beta(V, 1.0, 2.0)\n"
                            "        print(f_beta.dat.data)\n"
                            "        # produces:\n"
                            "        # [0.56462514 0.11585311 0.01247943 0.398984 0.19097059 0.5446709 0.1078666 0.2178807 0.64848515]\n"
                            "\n")

        # Use decorator to add doc strings to
        # auto generated methods
        def add_doc_string(doc_string):

            def f(func):
                func.__doc__ = _reformat_doc(doc_string)
                return func

            return f

        # To have Sphinx generate docs, make the following methods "static"
        for class_attr, _ in inspect.getmembers(getattr(randomgen, module_attr)):

            # These methods are not to be used with V
            if class_attr in ('bytes', 'shuffle', 'permutation', 'dirichlet', 'multinomial', 'multivariate_normal', 'complex_normal'):

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
                            # Extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            f = Function(V)
                            with f.dat.vec_wo as v:
                                v.array[:] = getattr(self, c_a)(v.local_size, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
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
                            # Extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            f = Function(V)
                            args = args[1:]
                            with f.dat.vec_wo as v:
                                kwargs['size'] = (v.local_size,)
                                v.array[:] = getattr(self, c_a)(*args, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)

                    return func

                _dict[class_attr] = funcgen(class_attr)

        _Wrapper = type(module_attr, (_Base,), _dict)

        return _Wrapper

    elif module_attr in _class_names:

        _Base = getattr(randomgen, module_attr)

        def __init__(self, *args, **kwargs):
            self._comm = kwargs.pop('comm', MPI.COMM_WORLD)
            # Remember args, kwargs as these are changed in super().__init__
            _args = args
            _kwargs = kwargs
            self._initialized = False
            super(_Wrapper, self).__init__(*args, **kwargs)
            self._initialized = True
            self.seed(*_args, **_kwargs)
            self._generator = None

        @property
        def generator(self):
            if self._generator is None:
                from . import RandomGenerator
                self._generator = RandomGenerator(brng=self)
            return self._generator

        if module_attr == 'PCG64':

            # Use examples in https://bashtage.github.io/randomgen/parallel.html
            # with appropriate changes.
            def seed(self, seed=None, inc=None):

                if self._comm.Get_size() == 1:
                    super(_Wrapper, self).seed(seed=seed, inc=0 if inc is None else inc)
                else:
                    rank = self._comm.Get_rank()
                    if seed is None:
                        if rank == 0:
                            # generate a 128bit seed
                            entropy = randomgen.entropy.random_entropy(4)
                            seed = sum([int(entropy[i]) * 2 ** (32 * i) for i in range(4)])
                        else:
                            seed = None
                        # All processes have to have the same seed
                        seed = self._comm.bcast(seed, root=0)
                    # Use rank to generate multiple streams.
                    # If 'inc' is to be passed, it is users' responsibility
                    # to provide an appropriate value.
                    inc = inc or rank
                    super(_Wrapper, self).seed(seed=seed, inc=inc)

        elif module_attr in ('Philox', 'ThreeFry'):

            def seed(self, seed=None, counter=None, key=None):

                if self._comm.Get_size() > 1:
                    rank = self._comm.Get_rank()
                    if seed is not None:
                        raise TypeError("'seed' should not be used when using 'Philox'/'ThreeFry' in parallel.  A random 'key' is automatically generated and used unless specified.")
                    # if 'key' is to be passed, it is users' responsibility
                    # to provide an appropriate one
                    if key is None:
                        # Use rank to generate multiple streams
                        key = np.zeros(2 if module_attr == 'Philox' else 4, dtype=np.uint64)
                        key[0] = rank

                super(_Wrapper, self).seed(seed=seed, counter=counter, key=key)

        else:

            def seed(self, *args, **kwargs):

                if self._comm.Get_size() > 1:
                    raise TypeError("Use 'PCG64', 'Philox', 'ThreeFry' for parallel RNG")

                super(_Wrapper, self).seed(*args, **kwargs)

        _dict = {"__init__": __init__,
                 "seed": seed,
                 "generator": generator,
                 "__doc__": _reformat_doc(getattr(randomgen, module_attr).__doc__)}

        _Wrapper = type(module_attr, (_Base,), _dict)

        return _Wrapper

    else:

        return getattr(randomgen, module_attr)


# __getattr__ on module level only works for 3.7+

import sys

if sys.version_info < (3, 7, 0):
    class Wrapper(object):
        __all__ = __all__

        def __getattr__(self, attr):
            return __getattr__(attr)

    sys.modules[__name__] = Wrapper()
