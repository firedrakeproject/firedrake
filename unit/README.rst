Auto-parametrization of test cases
==================================

Passing the parameter ``backend`` to any test case will auto-parametrise
that test case for all selected backends. By default all backends from
the ``backends`` dict in the ``backends`` module are selected. Backends
for which the dependencies are not installed are thereby automatically
skipped. Tests execution is grouped per backend and ``op2.init()`` and
``op2.exit()`` for a backend are only called once per test session.

Not passing the parameter ``backend`` to a test case will cause it to
run before the first backend is initialized, which is mostly not what
you want.

**Note:** The parameter order matters in some cases: If your test uses a
funcarg parameter, which creates any OP2 resources and hence requires a
backend to be initialized, it is imperative that ``backend`` is the
*first* parameter to the test function.

Selecting for which backend to run the test session
---------------------------------------------------

The default backends can be overridden by passing the
`--backend=<backend_string>` parameter on test invocation. Passing it
multiple times runs the tests for all the given backends.

Skipping backends on a per-test basis
-------------------------------------

To skip a particular backend in a test case, pass the
``skip_<backend_string>`` parameter to the test function, where
``<backend_string>`` is any valid backend string.

Skipping backends on a module or class basis
--------------------------------------------

You can supply a list of backends to skip for all tests in a given
module or class with the ``skip_backends`` attribute in the module or
class scope::

    # module test_foo.py

    # All tests in this module will not run for the CUDA backend
    skip_backends = ['cuda']

    class TestFoo: # All tests in this class will not run for the CUDA
    and OpenCL # backends skip_backends = ['opencl']

Selecting backends on a module or class basis
---------------------------------------------

You can supply a list of backends for which to run all tests in a given
module or class with the ``backends`` attribute in the module or class
scope::

    # module test_foo.py

    # All tests in this module will only run for the CUDA and OpenCL #
    backens backends = ['cuda', 'opencl']

    class TestFoo: # All tests in this class will only run for the CUDA
    backend backends = ['sequential', 'cuda']

This set of backends to run for will be further restricted by the
backends selected via command line parameters if applicable.
