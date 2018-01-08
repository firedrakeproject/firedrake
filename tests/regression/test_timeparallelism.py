from firedrake import *
import pytest

@pytest.mark.parallel(nprocs=6)
def test_time_allreduce():
    subcomms = Subcommunicators(3,2)

    mesh = UnitSquareMesh(20,20,comm=subcomms.space_comm)

    x,y = mesh.coordinates

    V = FunctionSpace(mesh,"CG",1)
    u_correct = Function(V)
    u = Function(V)
    usum = Function(V)

    u_correct.interpolate(sin(pi*x)*cos(pi*y) + sin(2*pi*x)*cos(2*pi*y) + sin(3*pi*x)*cos(3*pi*y))
    q = Constant(subcomms.time_rank+1)
    u.interpolate(sin(q*pi*x)*cos(q*pi*y))
    subcomms.time_allreduce(u, usum)

    assert(assemble((u_correct-u)**2*dx) < 1.0e-4)

if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
