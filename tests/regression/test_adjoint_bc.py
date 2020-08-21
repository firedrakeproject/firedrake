import gc
from firedrake import *


def howmany(cls):
    n = 0
    for x in gc.get_objects():
        try:
            if isinstance(x, cls):
                n += 1
        except (ReferenceError, AttributeError):
            pass
    return n


def test_bcs_collected_when_not_annotating():
    mesh = UnitTriangleMesh()

    V = FunctionSpace(mesh, "DG", 0)

    u = Function(V)

    def run(u, n):
        for _ in range(n):
            bc = DirichletBC(u.function_space(), 0, "on_boundary")
            bc.apply(u)

    before = howmany(DirichletBC)
    run(u, 100)
    after = howmany(DirichletBC)
    assert before == after
