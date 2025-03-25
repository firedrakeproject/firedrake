import numpy
import pytest

from ufl import (Mesh, FunctionSpace,
                 Coefficient, TestFunction, interval, indices, dx)
from finat.ufl import VectorElement, TensorElement
from ufl.classes import IndexSum, Product, MultiIndex

from tsfc import compile_form


def count_flops(n):
    mesh = Mesh(VectorElement('CG', interval, 1))
    tfs = FunctionSpace(mesh, TensorElement('DG', interval, 1, shape=(n, n)))
    vfs = FunctionSpace(mesh, VectorElement('DG', interval, 1, dim=n))

    ensemble_f = Coefficient(vfs)
    ensemble2_f = Coefficient(vfs)
    phi = TestFunction(tfs)

    i, j = indices(2)
    nc = 42  # magic number
    L = ((IndexSum(IndexSum(Product(nc * phi[i, j], Product(ensemble_f[i], ensemble_f[i])),
                            MultiIndex((i,))), MultiIndex((j,))) * dx)
         + (IndexSum(IndexSum(Product(nc * phi[i, j], Product(ensemble2_f[j], ensemble2_f[j])),
                              MultiIndex((i,))), MultiIndex((j,))) * dx)
         - (IndexSum(IndexSum(2 * nc * Product(phi[i, j], Product(ensemble_f[i], ensemble2_f[j])),
                              MultiIndex((i,))), MultiIndex((j,))) * dx))

    kernel, = compile_form(L, parameters=dict(mode='spectral'))
    return kernel.flop_count


def test_convergence():
    ns = [10, 20, 40, 80, 100]
    flops = [count_flops(n) for n in ns]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(ns))
    assert (rates < 2).all()  # only quadratic operation count, not more


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
