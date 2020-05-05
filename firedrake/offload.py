from contextlib import contextmanager


@contextmanager
def offloading(backend):
    """
    Changes to the compute backend to ``backend`` within the context block.

    This would ensure that all the operations (for ex. assemble, interpolation,
    etc) would be executed on ``backend``. Any backend specific object
    instantiated in the context will be allocated on ``backend``.
    """
    from pyop2 import op2
    preoffloading_backend = op2.compute_backend

    op2.compute_backend = backend
    yield

    op2.compute_backend = preoffloading_backend
    return
