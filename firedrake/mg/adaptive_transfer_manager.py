"""
This module contains the AdaptiveTransferManager used to perform
transfer operations on AdaptiveMeshHierarchies
"""
from firedrake.mg.embedded import TransferManager
from firedrake.ufl_expr import action, TrialFunction
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.interpolation import interpolate
from firedrake.preconditioners.bddc import is_lagrange
from finat.quadrature import QuadratureRule
from functools import partial

import numpy


__all__ = ("AdaptiveTransferManager",)


class AdaptiveTransferManager(TransferManager):
    """
    TransferManager for adaptively refined mesh hierarchies
    """
    def __init__(self, *, native_transfers=None, use_averaging=True):
        super().__init__(native_transfers=native_transfers, use_averaging=use_averaging)
        self.cache = {}

    def get_operators(self, Vc, Vf):
        key = (Vc, Vf)
        try:
            return self.cache[key]
        except KeyError:
            ops = get_mg_interpolator(Vc, Vf)
            return self.cache.setdefault(key, ops)

    def forward(self, uc, uf):
        from firedrake.assemble import assemble
        Vc = uc.function_space()
        Vf = uf.function_space()
        ops = self.get_operators(Vc, Vf)

        expr = uc
        for op in ops:
            expr = action(op, expr)
        return assemble(expr, tensor=uf)

    def adjoint(self, uf, uc):
        from firedrake.assemble import assemble
        Vc = uc.function_space().dual()
        Vf = uf.function_space().dual()
        ops = self.get_operators(Vc, Vf)

        expr = uf
        for op in reversed(ops):
            expr = action(expr, op)
        return assemble(expr, tensor=uc)

    def prolong(self, uf, uc):
        return self.forward(uf, uc)

    def inject(self, uc, uf):
        return self.forward(uc, uf)

    def restrict(self, uc, uf):
        return self.adjoint(uc, uf)


def make_quadrature_space(V):
    fe = V.finat_element
    _, ps = fe.dual_basis
    wts = numpy.full(len(ps.points), numpy.nan)
    scheme = QuadratureRule(ps, wts, ref_el=fe.cell)
    if V.value_shape == ():
        make_space = FunctionSpace
    else:
        make_space = partial(TensorFunctionSpace, shape=V.value_shape)
    return make_space(V.mesh(), "Quadrature", degree=fe.degree, quad_scheme=scheme)


def get_mg_interpolator(V1, V2):
    from firedrake.assemble import assemble
    if is_lagrange(V2.finat_element):
        spaces = (V1, V2)
    else:
        Q2 = make_quadrature_space(V2)
        spaces = (V1, Q2, V2)

    ops = []
    for i in range(len(spaces)-1):
        Vsrc = spaces[i]
        Vdest = spaces[i+1]
        Iexpr = interpolate(TrialFunction(Vsrc), Vdest)
        op = assemble(Iexpr, mat_type="aij")
        ops.append(op)
    return ops
