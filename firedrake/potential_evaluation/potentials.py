from firedrake.functionspaceimpl import WithGeometry


def SingleLayerPotential(density, kernel, potential_src_and_tgt, **kwargs):
    """
    Evaluate the single layer potential of the density
    function convoluted against the kernel on the source
    at each target point.

    :arg density: The :mod:`firedrake` density function
    :arg kernel: the :class:`sumpy.kernel.Kernel`
    :arg potential_src_and_tgt: A
        :class:`firedrake.potential_evaluation.PotentialSourceAndTarget`

    :kwarg function_space: If density is a firedrake function,
        this is optional. Otherwise it must be supplied.
        If supplied and density is a function, they must
        be the same object.
    :kwarg cl_ctx: (Optional) a :class:`pyopencl.Context` used
        to create a command queue.
        At most one of *cl_ctx*, *queue*, *actx* should be included.
        If none is included, a *cl_ctx* will be created.
    :kwarg queue: (Optional) a :class:`pyopencl.CommandQueue` used
        to create an array context.
        At most one of *cl_ctx*, *queue*, *actx* should be included.
        If none is included, a *cl_ctx* will be created.
    :kwarg actx: (Optional) a :class:`meshmode.array_context.PyOpenCLArrayContext`
        used for :mod:`pytential` and :mod:`meshmode` computations.
        At most one of *cl_ctx*, *queue*, *actx* should be included.
        If none is included, a *cl_ctx* will be created.
    :kwarg op_kwargs: (Optional)  kwargs passed to the construction
        of the unbound pytential operator (e.g. {'k': 0.5} for
        a :class:`sumpy.kernel.HelmholtzKernel` must be passed
        to :func:`pytential.symbolic.primitives.S`),
        DEFAULT:
        * {'qbx_forced_limit': None}
    :kwarg qbx_kwargs: (Optional) *None*, or a dict with arguments to pass to
        :class:`pytential.qbx.QBXLayerPotentialSource`.
        DEFAULTS:
            - fine_order: 4 8 function space degree
            - qbx_order: function space degree
            - fmm_order: *None*
            - fmm_level_to_order: A
                :class:`sumpy.level_to_order.FMMLibExpansionOrderFinder` set
                to provide tolerance of double machine epsilon
                2**-53 with an extra order of 1.
                This allows for taking 1 derivative, but should be
                increased for taking more derivatives.
            - fmm_backend: "fmmlib"
    :kwarg meshmode_connection_kwargs: (Optional)
        *None*, or a dict with arguments to pass to
        :func:`meshmode.interop.firedrake.build_connection_from_firedrake`
    :kwarg warn_if_cg: (Optional) Pass *False* to suppress the warning if
        a "CG" space is used. *True* by default
    """
    # TODO: Make fmm_level_to_order depend on number of derivatives somehow?
    from pytential.symbolic.primitives import S
    return _layer_potential(S, density, kernel, potential_src_and_tgt, **kwargs)


def DoubleLayerPotential(density, kernel, potential_src_and_tgt, **kwargs):
    """
    Same as SingleLayerPotential, but evaluates the double layer potential
    instead of the single
    """
    from pytential.symbolic.primitives import D
    return _layer_potential(D, density, kernel, potential_src_and_tgt, **kwargs)


def _layer_potential(layer_potential_sym,
                     density, kernel, potential_src_and_tgt, **kwargs):
    """
    Build a layer potential. For usage, see
    :func:`SingleLayerPotential` or :func:`DoubleLayerPotential`.
    """
    kwargs = _validate_layer_potential_args_and_set_defaults(
        density, kernel, potential_src_and_tgt, **kwargs)
    # build our unbound operator
    assert 'op_kwargs' in kwargs  # make sure we got op kwargs
    op_kwargs = kwargs['op_kwargs']
    from pytential import sym
    unbound_op = layer_potential_sym(kernel,
                                     sym.var("density"),
                                     **op_kwargs)
    # make sure we got an actx during validation
    assert 'actx' in kwargs
    actx = kwargs['actx']
    # make sure we got a function_space during validation
    assert 'function_space' in kwargs
    function_space = kwargs['function_space']
    # extract Pytential operation kwargs
    pyt_op_kwargs = _extract_pytential_operation_kwargs(**kwargs)
    # now return the pytential operation as an external operator
    from firedrake.potential_evaluation.pytential import PytentialOperation
    return PytentialOperation(actx, density, unbound_op, "density",
                              potential_src_and_tgt,
                              **pyt_op_kwargs)


def _validate_layer_potential_args_and_set_defaults(density,
                                                    kernel,
                                                    places,
                                                    **kwargs):
    """
    Validate the arguments for single/double layer potential.

    Returns dictionary updated with default arguments
    """
    # validate density function space
    if hasattr(density, 'function_space'):
        if not isinstance(density.function_space(), WithGeometry):
            raise TypeError("density.function_space() must be of type "
                            f"WithGeometry, not {type(density.function_space())}")
        function_space = kwargs.get('function_space', None)
        if function_space is not None:
            if function_space is not density.function_space():
                raise ValueError("density.function_space() and function_space"
                                 " must be the same object")
        else:
            kwargs['function_space'] = density.function_space()
    else:
        function_space = kwargs.get('function_space', None)
        if function_space is None:
            raise ValueError("density has no function_space method, so"
                             "function_space kwarg must be supplied")
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be of type "
                            f"WithGeometry, not {type(function_space)}")
    function_space = kwargs['function_space']

    # validate kernel
    from sumpy.kernel import Kernel
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a sumpy.kernel.Kernel, not of "
                        f"type '{type(kernel)}'.")
    # validate src and tgts
    from firedrake.potential_evaluation import PotentialSourceAndTarget
    if not isinstance(places, PotentialSourceAndTarget):
        raise TypeError("potential_src_and_tgt must be a sumpy.kernel.Kernel, "
                        f"not of type '{type(places)}'.")
    # Make sure src is of right dimension
    mesh_gdim = places.mesh.geometric_dimension()
    mesh_tdim = places.mesh.topological_dimension()
    src_tdim = places.get_source_dimension()
    # sanity check
    assert mesh_gdim - mesh_tdim in [0, 1]
    assert mesh_gdim - src_tdim in [0, 1]
    assert mesh_tdim in (mesh_gdim, src_tdim)
    # now do the real user-input check
    if mesh_gdim - src_tdim != 1:
        raise ValueError("source of a layer potential must have co-dimension 1,"
                         f" not {mesh_gdim - src_tdim}.")

    # Make sure all kwargs are recognized
    allowed_kwargs = ("cl_ctx", "queue", "actx", "op_kwargs", "qbx_kwargs",
                      "meshmode_connection_kwargs", "function_space")
    for key in kwargs:
        if key not in allowed_kwargs:
            raise ValueError(f"Unrecognized kwarg {key}")

    # Now handle pyopencl computing contexts and build
    # a PyOpenCLArrayContext
    from meshmode.array_context import PyOpenCLArrayContext
    cl_ctx = None
    queue = None
    actx = None
    if 'cl_ctx' in kwargs:
        if 'actx' in kwargs or 'queue' in kwargs:
            raise ValueError("At most one of 'actx', 'queue', 'cl_ctx' should "
                             "be supplied")
        cl_ctx = kwargs['cl_ctx']
        from pyopencl import Context
        if not isinstance(cl_ctx, Context):
            raise TypeError("cl_ctx must be of type Context, not "
                            f"{type(cl_ctx)}")
        queue = None
        actx = None
    elif 'queue' in kwargs:
        if 'actx' in kwargs:
            raise ValueError("At most one of 'actx', 'queue' should "
                             "be supplied")
        queue = kwargs['queue']
        from pyopencl import CommandQueue
        if not isinstance(queue, CommandQueue):
            raise TypeError("queue must be of type CommandQueue, not "
                            f"{type(queue)}")
        actx = None
    elif 'actx' in kwargs:
        actx = kwargs['actx']
        if not isinstance(actx, PyOpenCLArrayContext):
            raise TypeError("actx must be of type PyOpenCLArrayContext, not "
                            f"{type(actx)}")

    # now make sure we actually get an actx in kwargs
    if actx is None:
        if queue is None:
            if cl_ctx is None:
                from pyopencl import create_some_context
                cl_ctx = create_some_context()
            from pyopencl import CommandQueue
            queue = CommandQueue(cl_ctx)
        actx = PyOpenCLArrayContext(queue)
        kwargs['actx'] = actx

    # Set qbx_kwargs defaults
    if 'qbx_kwargs' not in kwargs:
        degree = function_space.ufl_element().degree()
        from sumpy.expansion.level_to_order import FMMLibExpansionOrderFinder
        qbx_kwargs = {'fine_order': 4 * degree,
                      'qbx_order': degree,
                      'fmm_order': None,
                      'fmm_level_to_order':
                          FMMLibExpansionOrderFinder(2**-53, extra_order=1),
                      'fmm_backend': "fmmlib",
                      }
        kwargs['qbx_kwargs'] = qbx_kwargs

    # Set op_kwargs defaults
    if 'op_kwargs' not in kwargs:
        kwargs['op_kwargs'] = {'qbx_forced_limit': None}

    return kwargs


def _extract_pytential_operation_kwargs(**kwargs):
    """
    Extract kwargs to be passed to :func:`PytentialOperation`
    """
    pyt_op_kwargs = {}
    pyt_op_possible_keywords = ("warn_if_cg",
                                "meshmode_connection_kwargs",
                                "qbx_kwargs",
                                "function_space")
    for key in pyt_op_possible_keywords:
        if key in kwargs:
            pyt_op_kwargs[key] = kwargs[key]

    return pyt_op_kwargs


def VolumePotential(density, kernel, potential_src_and_tgt, **kwargs):
    raise NotImplementedError
