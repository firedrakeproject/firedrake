from firedrake import *
import pytest


@pytest.mark.parametrize("transfer",
                         [prolong, inject, restrict])
def test_transfer_invalid_level_combo(transfer):
    m = UnitIntervalMesh(10)
    mh = MeshHierarchy(m, 2)
    Vcoarse = FunctionSpace(mh[0], "DG", 0)
    Vfine = FunctionSpace(mh[-1], "DG", 0)
    if transfer == restrict:
        Vcoarse, Vfine = Vcoarse.dual(), Vfine.dual()
    if transfer == prolong:
        source, target = Function(Vfine), Function(Vcoarse)
    else:
        source, target = Function(Vcoarse), Function(Vfine)
    with pytest.raises(ValueError):
        transfer(source, target)


@pytest.mark.parametrize("transfer",
                         [prolong, inject, restrict])
def test_transfer_invalid_type(transfer):
    m = UnitIntervalMesh(10)
    mh = MeshHierarchy(m, 2)
    Vcoarse = FunctionSpace(mh[0], "DG", 0)
    Vfine = FunctionSpace(mh[-1], "DG", 0)
    if transfer != restrict:
        Vcoarse, Vfine = Vcoarse.dual(), Vfine.dual()
    if transfer == prolong:
        source, target = Function(Vcoarse), Function(Vfine)
    else:
        source, target = Function(Vfine), Function(Vcoarse)
    with pytest.raises(TypeError):
        transfer(source, target)
