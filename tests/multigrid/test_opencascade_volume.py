from firedrake import *

import pytest
import os


@pytest.fixture(scope='module', params=[("cylinder.step", 20), ("cylinder.step", (15, 30)), ("cylinder.step", "Mesh.CharacteristicLengthFactor=2;\n "), ("cylinder.step", "Characteristic Length{2}=10;\nCharacteristic Length {1}=5;"), ("t_twist.step", 3)])
def stepdata(request):
    (stepfile, h) = request.param
    curpath = os.path.dirname(os.path.realpath(__file__))
    return (os.path.abspath(os.path.join(curpath, os.path.pardir, "meshes", stepfile)), h)


@pytest.fixture(scope='module', params=[1, 2])
def order(request):
    return request.param


def get_volume(stepfile):

    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties

    step_reader = STEPControl_Reader()
    step_reader.ReadFile(stepfile)
    step_reader.TransferRoot()
    shape = step_reader.Shape()
    prop = GProp_GProps()
    brepgprop_VolumeProperties(shape, prop, 1e-5)
    return prop.Mass()


def compute_err(mh, v_true):
    err = []

    for m in mh:
        v_approx = assemble(Constant(1)*dx(domain=m))
        err.append(abs(v_true - v_approx) / v_true)

    return err


@pytest.mark.parallel(nprocs=2)
def test_volume(stepdata):

    (stepfile, h) = stepdata
    try:
        mh = OpenCascadeMeshHierarchy(stepfile, element_size=h, levels=3, cache=False, verbose=False)
        v_true = get_volume(stepfile)
    except ImportError:
        pytest.skip(reason="OpenCascade unavailable, skipping test")

    print("True volume for %s: %s" % (os.path.basename(stepfile), v_true))
    err = compute_err(mh, v_true)
    print("Volume errors: %s" % err)

    for pair in zip(err, err[1:]):
        assert pair[0] > pair[1]


@pytest.mark.parallel(nprocs=2)
def test_area(order):
    curpath = os.path.dirname(os.path.realpath(__file__))
    stepfile = os.path.abspath(os.path.join(curpath, os.path.pardir, "meshes", "disk.step"))
    h = 0.5
    try:
        mh = OpenCascadeMeshHierarchy(stepfile, element_size=h, levels=3, cache=False, verbose=False, order=order)
    except ImportError:
        pytest.skip(reason="OpenCascade unavailable, skipping test")
    from math import pi
    a_true = pi/4
    print("True ara for %s: %s" % (os.path.basename(stepfile), a_true))
    err = compute_err(mh, a_true)
    print("Area errors: %s" % err)
    for pair in zip(err, err[1:]):
        assert pair[0] > pair[1]
