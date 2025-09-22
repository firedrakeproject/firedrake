"""
Tests impero flop counts against loopy.
"""
import pytest
import numpy
import loopy
from tsfc import compile_form
from ufl import (FunctionSpace, Mesh, TestFunction,
                 TrialFunction, dx, grad, inner,
                 interval, triangle, quadrilateral,
                 TensorProductCell)
from finat.ufl import FiniteElement, VectorElement
from tsfc.parameters import target


def count_loopy_flops(kernel):
    name = kernel.name
    program = kernel.ast
    program = program.with_kernel(
        program[name].copy(
            target=target,
            silenced_warnings=["insn_count_subgroups_upper_bound",
                               "get_x_map_guessing_subgroup_size"])
    )
    op_map = loopy.get_op_map(program
                              .with_entrypoints(kernel.name),
                              subgroup_size=1)
    return op_map.filter_by(name=['add', 'sub', 'mul', 'div',
                                  'func:abs'],
                            dtype=[float]).eval_and_sum({})


@pytest.fixture(params=[interval, triangle, quadrilateral,
                        TensorProductCell(triangle, interval)],
                ids=lambda cell: cell.cellname())
def cell(request):
    return request.param


@pytest.fixture(params=[{"mode": "vanilla"},
                        {"mode": "spectral"}],
                ids=["vanilla", "spectral"])
def parameters(request):
    return request.param


def test_flop_count(cell, parameters):
    mesh = Mesh(VectorElement("P", cell, 1))
    loopy_flops = []
    new_flops = []
    for k in range(1, 5):
        V = FunctionSpace(mesh, FiniteElement("P", cell, k))
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(u, v)*dx + inner(grad(u), grad(v))*dx
        kernel, = compile_form(a, prefix="form",
                               parameters=parameters)
        # Record new flops here, and compare asymptotics and
        # approximate order of magnitude.
        new_flops.append(kernel.flop_count)
        loopy_flops.append(count_loopy_flops(kernel))

    new_flops = numpy.asarray(new_flops)
    loopy_flops = numpy.asarray(loopy_flops)

    assert all(new_flops == loopy_flops)
