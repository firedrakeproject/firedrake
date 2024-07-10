from contextlib import contextmanager

import firedrake


def isconstant(expr):
    """Check whether expression is constant type.
    In firedrake this is a function in the real space
    Ie: `firedrake.Function(FunctionSpace(mesh, "R"))`"""
    if isinstance(expr, firedrake.Constant):
        raise ValueError("Firedrake Constant requires a domain to work with pyadjoint")
    return isinstance(expr, firedrake.Function) and expr.ufl_element().family() == "Real"


@contextmanager
def restored_outputs(*X, restore=None):
    """Construct a context manager which can be used to temporarily restore
    block variable outputs to saved values.

    Parameters
    ----------
    X : tuple[BlockVariable]
        Block variables to temporarily restore.
    restore : callable
        Can be used to exclude variables. Only inputs for which
        `restore(x.output)` is true have their outputs temporarily restored.

    Returns
    -------

    The context manager.

    Notes
    -----

    A forward operation is allowed to modify the original variable, e.g. in

    .. code-block:: python3

        solve(inner(trial, test) * dx
              == inner(x * x, test) * dx,
              x)

    `x` has two versions: the input and the output. Reverse-over-forward AD
    requires that we use the symbolic representation `x`, but with input value
    `x.block_variable.saved_output`. A context manager can be used to
    temporarily restore the value of `x` so that we can perform and annotate
    a tangent-linear operation,

    .. code-block:: python3

        with restored_outputs(x):
            # The value of x is now x.block_variable.saved_output
            solve(inner(trial, test) * dx
                  == 2 * inner(x * x.block_variable.tlm_value, test) * dx,
                  x.block_variable.tlm_value)
        # The value of x is again the output from the forward solve(...)
    """

    if restore is None:
        def restore(x):
            return True

    X = tuple(x for x in X if restore(x.output))
    X_old = tuple(x.output._ad_copy(x) for x in X)
    for x in X:
        # Ideally would use a generic _ad_assign here
        x.output.assign(x.output.block_variable.saved_output)
    try:
        yield
    finally:
        for x, x_old in zip(X, X_old):
            # Ideally would use a generic _ad_assign here
            x.output.assign(x_old)
