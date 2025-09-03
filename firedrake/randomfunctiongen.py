"""
Overview
========

This module wraps `numpy.random <https://numpy.org/doc/stable/reference/random/index.html>`__,
and enables users to generate a randomised :class:`.Function` from a :class:`.WithGeometry` or :class:`.FiredrakeDualSpace`.
This module inherits almost all attributes from `numpy.random <https://numpy.org/doc/stable/reference/random/index.html>`__ with the following changes:

Generator
---------

A :class:`.Generator` wraps `numpy.random.Generator <https://numpy.org/doc/stable/reference/random/generator.html>`__.
:class:`.Generator` inherits almost all distribution methods from `numpy.random.Generator <https://numpy.org/doc/stable/reference/random/generator.html>`__,
and they can be used to generate a randomised :class:`.Function` by passing a :class:`.WithGeometry` or :class:`.FiredrakeDualSpace` as the first argument.

Example:

.. code-block:: python3

    from firedrake import *

    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    pcg = PCG64(seed=123456789)
    rg = Generator(pcg)
    f_beta = rg.beta(V, 1.0, 2.0)
    print(f_beta.dat.data)
    # prints:
    # [0.0075147 0.40893448 0.18390776 0.46192167 0.20055854 0.02231147 0.47424777 0.24177973 0.55937075]

BitGenerator
------------

A ``.BitGenerator`` is the base class for bit generators; see `numpy.random.BitGenerator <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.BitGenerator.html#numpy.random.BitGenerator>`__.
A ``.BitGenerator`` takes an additional keyword argument ``comm`` (defaulting to ``COMM_WORLD``).
If ``comm.Get_rank() > 1``, ``.PCG64``, ``.PCG64DXSM``, or ``.Philox`` should be used, as these bit generators are known to be parallel-safe.

PCG64
~~~~~

``.PCG64`` wraps `numpy.random.PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`__.
If ``seed`` keyword is not provided by the user, it is set using `numpy.random.SeedSequence <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html>`__.
To make ``.PCG64`` automatically generate multiple streams in parallel, Firedrake preprocesses the ``seed`` as the following before
passing it to `numpy.random.PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`__:

.. code-block:: python3

    rank = comm.Get_rank()
    size = comm.Get_size()
    sg = numpy.random.SeedSequence(seed)
    seed = sg.spawn(size)[rank]

.. note::

    ``inc`` is no longer a valid keyword for ``.PCG64`` constructor. However, one can reset the ``state`` after construction as:

    .. code-block:: python3

        pcg = PCG64()
        state = pcg.state
        state['state'] = {'state': seed, 'inc': inc}
        pcg.state = state

PCG64DXSM
~~~~~~~~~

``.PCG64DXSM`` wraps `numpy.random.PCG64DXSM <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html>`__.
If ``seed`` keyword is not provided by the user, it is set using `numpy.random.SeedSequence <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html>`__.
To make ``.PCG64DXSM`` automatically generate multiple streams in parallel, Firedrake preprocesses the ``seed`` as the following before
passing it to `numpy.random.PCG64DXSM <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html>`__:

.. code-block:: python3

    rank = comm.Get_rank()
    size = comm.Get_size()
    sg = numpy.random.SeedSequence(seed)
    seed = sg.spawn(size)[rank]

.. note::

    ``inc`` is no longer a valid keyword for ``.PCG64DXSM`` constructor. However, one can reset the ``state`` after construction as:

    .. code-block:: python3

        pcg = PCG64DXSM()
        state = pcg.state
        state['state'] = {'state': seed, 'inc': inc}
        pcg.state = state

Philox
~~~~~~

``.Philox`` wraps `numpy.random.Philox <https://numpy.org/doc/stable/reference/random/bit_generators/philox.html>`__.
If the ``key`` keyword is not provided by the user, ``.Philox`` computes a default key as:

.. code-block:: python3

    key = np.zeros(2, dtype=np.uint64)
    key[0] = comm.Get_rank()

"""

import inspect
import numpy as np
import numpy.random as randomgen

from firedrake.function import Function
from pyop2.mpi import COMM_WORLD
from ufl.functionspace import BaseFunctionSpace

_deprecated_attributes = ['RandomGenerator', ]

__all__ = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass)] + _deprecated_attributes

# >>> [name for name, _ in inspect.getmembers(numpy.random) if not name.startswith('_')]
_known_attributes = ['BitGenerator', 'Generator', 'MT19937', 'PCG64', 'PCG64DXSM', 'Philox', 'RandomState', 'SFC64', 'SeedSequence', 'beta', 'binomial', 'bit_generator', 'bytes', 'chisquare', 'choice', 'default_rng', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'get_bit_generator', 'get_state', 'gumbel', 'hypergeometric', 'laplace', 'logistic', 'lognormal', 'logseries', 'mtrand', 'multinomial', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power', 'rand', 'randint', 'randn', 'random', 'random_integers', 'random_sample', 'ranf', 'rayleigh', 'sample', 'seed', 'set_bit_generator', 'set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'test', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf']
# >>> [name for name, _ in inspect.getmembers(numpy.random.Generator) if not name.startswith('_')]
_known_generator_attributes = ['beta', 'binomial', 'bit_generator', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'gumbel', 'hypergeometric', 'integers', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_hypergeometric', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'permuted', 'poisson', 'power', 'random', 'rayleigh', 'shuffle', 'spawn', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf']


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
                st += sp + "The original documentation is found at `<https://numpy.org/doc/stable/reference/random/generated/numpy.random." + name + ".html>`__, which is reproduced below with appropriate changes.\n\n"
                st += sp + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif '.. math::' in s:
                st += '\n' + sp + s + '\n\n'
            else:
                st += sp + s + '\n'

        return st
    if module_attr == 'Generator':
        _Base = getattr(randomgen, module_attr)
        _dict = {}
        _dict["__doc__"] = ("\n"
                            "    Container for the Basic Random Number Generators.\n"
                            "\n"
                            "    The original documentation is found at `<https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`__, which is reproduced below with appropriate changes.\n"
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
                            "        rg = Generator(pcg)\n"
                            "        f_beta = rg.beta(V, 1.0, 2.0)\n"
                            "        print(f_beta.dat.data)\n"
                            "        # prints:\n"
                            "        # [0.0075147 0.40893448 0.18390776 0.46192167 0.20055854 0.02231147 0.47424777 0.24177973 0.55937075]\n"
                            "\n")

        def __init__(self, bit_generator=None):
            if bit_generator is None:
                from firedrake.randomfunctiongen import PCG64
                bit_generator = PCG64()
            super(_Wrapper, self).__init__(bit_generator)
        _dict['__init__'] = __init__

        # Use decorator to add doc strings to
        # auto generated methods
        def add_doc_string(doc_string):
            def f(func):
                func.__doc__ = _reformat_doc(doc_string)
                return func
            return f

        # To have Sphinx generate docs, make the following methods "static"
        for class_attr, _ in inspect.getmembers(randomgen.Generator):
            if class_attr.startswith('_'):
                continue
            elif class_attr in ['bit_generator', 'spawn']:
                continue
            elif class_attr in ['bytes', 'dirichlet', 'integers', 'multinomial', 'multivariate_hypergeometric', 'multivariate_normal', 'shuffle', 'permutation', 'permuted']:
                # These methods are not to be used with V.
                # class_attr is mutable, so we have to wrap func with
                # another function to lock the value of class_attr
                def funcgen(c_a):

                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], BaseFunctionSpace):
                            raise NotImplementedError("%s.%s does not take FunctionSpace as argument" % (module_attr, c_a))
                        else:
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)

                    return func

                _dict[class_attr] = funcgen(class_attr)
            # Other methods here
            elif class_attr in _known_generator_attributes:
                # Here, too, wrap func with funcgen.
                def funcgen(c_a):
                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], BaseFunctionSpace):
                            # Extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            f = Function(V)
                            args = args[1:]
                            with f.vec_wo as v:
                                kwargs['size'] = (v.local_size,)
                                v.array[:] = getattr(self, c_a)(*args, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)
                    return func
                _dict[class_attr] = funcgen(class_attr)
            else:
                def funcgen(c_a):
                    def func(self, *args, **kwargs):
                        raise NotImplementedError(f"Firedrake has not yet wrapped numpy.random.{c_a}")
                    return func
                _dict[class_attr] = funcgen(class_attr)
        _Wrapper = type(module_attr, (_Base,), _dict)
        return _Wrapper
    elif module_attr == "RandomGenerator":
        from firedrake.randomfunctiongen import Generator
        return Generator
    elif module_attr in ['MT19937', 'Philox', 'PCG64', 'PCG64DXSM', 'SFC64']:
        _Base = getattr(randomgen, module_attr)

        def __init__(self, *args, **kwargs):
            _kwargs = kwargs.copy()
            self._comm = _kwargs.pop('comm', COMM_WORLD)
            if self._comm.Get_size() > 1 and module_attr not in ['PCG64', 'PCG64DXSM', 'Philox']:
                raise TypeError("Use 'PCG64', 'PCG64DXSM', or 'Philox', for parallel RNG")
            self._init(*args, **_kwargs)

        def seed(self, *args, **kwargs):
            raise AttributeError("`seed` method is not available in `numpy.random`; if reseeding, create a new bit generator with the new seed.")

        if module_attr in ('PCG64', 'PCG64DXSM'):
            def _init(self, *args, **kwargs):
                if 'inc' in kwargs:
                    raise RuntimeError("'inc' is no longer a valid keyword; see <https://www.firedrakeproject.org/firedrake.html#module-firedrake.randomfunctiongen>")
                rank = self._comm.Get_rank()
                size = self._comm.Get_size()
                _kwargs = kwargs.copy()
                seed = _kwargs.get("seed")
                if seed is None:
                    if rank == 0:
                        # generate a 128bit seed
                        seed = randomgen.SeedSequence().entropy
                    else:
                        seed = None
                    seed = self._comm.bcast(seed, root=0)
                if isinstance(seed, randomgen.SeedSequence):
                    # We assume that the user has generated
                    # a parallel-safe SeedSequence.
                    pass
                else:
                    # Create multiple streams
                    sg = randomgen.SeedSequence(seed)
                    _kwargs["seed"] = sg.spawn(size)[rank]
                super(_Wrapper, self).__init__(*args, **_kwargs)
        elif module_attr == 'Philox':
            def _init(self, *args, **kwargs):
                seed = kwargs.get("seed")
                # counter = kwargs.get("counter")
                key = kwargs.get("key")
                if self._comm.Get_size() > 1:
                    rank = self._comm.Get_rank()
                    if seed is not None:
                        raise TypeError("'seed' should not be used when using 'Philox' in parallel.  A random 'key' is automatically generated and used unless specified.")
                    # if 'key' is to be passed, it is users' responsibility
                    # to provide an appropriate one
                    if key is None:
                        # Use rank to generate multiple streams
                        key = np.zeros(2, dtype=np.uint64)
                        key[0] = rank
                _kwargs = kwargs.copy()
                _kwargs["key"] = key
                super(_Wrapper, self).__init__(*args, **_kwargs)
        else:
            def _init(self, *args, **kwargs):
                super(_Wrapper, self).__init__(*args, **kwargs)
        _dict = {"__init__": __init__,
                 "_init": _init,
                 "seed": seed,
                 "__doc__": _reformat_doc(getattr(randomgen, module_attr).__doc__)}
        _Wrapper = type(module_attr, (_Base,), _dict)
        return _Wrapper
    elif module_attr == 'default_rng':
        from firedrake.randomfunctiongen import Generator, PCG64, SeedSequence

        def _wrapper(seed=None):
            if seed is None or \
               isinstance(seed, int) or \
               isinstance(seed, SeedSequence):
                return Generator(PCG64(seed=seed))
            else:
                raise ValueError("Firedrake wrapper of numpy.random.%s only takes seed of type {None, int, SeedSequence}." % module_attr)
        return _wrapper
    elif module_attr in ['BitGenerator', 'RandomState', 'bit_generator', 'get_state', 'mtrand', 'seed', 'set_state', 'test']:
        def _wrapper(*args, **kwargs):
            raise NotImplementedError("numpy.random.%s is not wrapped in Firedrake. Consider using numpy.random.%s directly." % (module_attr, module_attr))
        return _wrapper
    elif module_attr == 'SeedSequence':
        return getattr(randomgen, module_attr)
    elif not module_attr.startswith('_'):
        # module_attr not in _known_attributes + _deprecated_attributes

        def _wrapper(*args, **kwargs):
            raise NotImplementedError("Firedrake has not yet wrapped numpy.random.%s." % module_attr)
        return _wrapper
