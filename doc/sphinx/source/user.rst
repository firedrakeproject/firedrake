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

    Parallel loops and linear solves
    ................................

    .. autofunction:: par_loop
    .. autofunction:: solve

    Data structures
    ...............

    .. autoclass:: Set
       :inherited-members:
    .. autoclass:: DataSet
       :inherited-members:
    .. autoclass:: Map
       :inherited-members:
    .. autoclass:: Sparsity
       :inherited-members:

    .. autoclass:: Const
       :inherited-members:
    .. autoclass:: Global
       :inherited-members:
    .. autoclass:: Dat
       :inherited-members:
    .. autoclass:: Mat
       :inherited-members:

    Kernels
    .......

    .. autoclass:: Kernel
       :inherited-members:

    .. autodata:: i
    .. autodata:: READ
    .. autodata:: WRITE
    .. autodata:: RW
    .. autodata:: INC
    .. autodata:: MIN
    .. autodata:: MAX
