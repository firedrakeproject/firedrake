pyop2 user documentation
========================

:mod:`pyop2` Package
--------------------

.. automodule:: pyop2
    :members:
    :show-inheritance:
    :inherited-members:

    Initialization and finalization
    ...............................

    .. autofunction:: init
    .. autofunction:: exit

    Data structures
    ...............

    .. autoclass:: Set
       :inherited-members:
    .. autoclass:: ExtrudedSet
       :inherited-members:
    .. autoclass:: Subset
       :inherited-members:
    .. autoclass:: MixedSet
       :inherited-members:
    .. autoclass:: DataSet
       :inherited-members:
    .. autoclass:: MixedDataSet
       :inherited-members:
    .. autoclass:: Map
       :inherited-members:
    .. autoclass:: MixedMap
       :inherited-members:
    .. autoclass:: Sparsity
       :inherited-members:

    .. autoclass:: Const
       :inherited-members:
    .. autoclass:: Global
       :inherited-members:
    .. autoclass:: Dat
       :inherited-members:
    .. autoclass:: MixedDat
       :inherited-members:
    .. autoclass:: Mat
       :inherited-members:

    Parallel loops, kernels and linear solves
    .........................................

    .. autofunction:: par_loop
    .. autofunction:: solve

    .. autoclass:: Kernel
       :inherited-members:
    .. autoclass:: Solver
       :inherited-members:

    .. autodata:: i
    .. autodata:: READ
    .. autodata:: WRITE
    .. autodata:: RW
    .. autodata:: INC
    .. autodata:: MIN
    .. autodata:: MAX
