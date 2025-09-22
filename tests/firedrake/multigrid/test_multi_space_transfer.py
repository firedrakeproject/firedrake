from firedrake import *
import numpy
import pytest


@pytest.mark.parametrize("typ",
                         ["inject", "prolong"])
def test_transfer_scalar_vector(typ):
    mesh = UnitSquareMesh(1, 1)

    mh = MeshHierarchy(mesh, 1)

    coarse, fine = mh

    for space, val in [(FunctionSpace, 1), (VectorFunctionSpace, [2, 1])]:
        Vc = space(coarse, "CG", 1)
        Vf = space(fine, "CG", 1)

        if typ == "inject":
            Vdonor = Function(Vf)
            Vtarget = Function(Vc)
            transfer = inject
        else:
            Vdonor = Function(Vc)
            Vtarget = Function(Vf)
            transfer = prolong
        donor = Function(Vdonor)
        target = Function(Vtarget)
        donor.assign(Constant(val))
        transfer(donor, target)
        assert numpy.allclose(target.dat.data_ro, val)
