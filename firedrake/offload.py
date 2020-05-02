from contextlib import contextmanager


@contextmanager
def offloading(backend):
    from pyop2 import op2
    preoffloading_backend = op2.compute_backend

    op2.compute_backend = backend
    yield

    op2.compute_backend = preoffloading_backend
    return
