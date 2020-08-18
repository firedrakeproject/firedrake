import numpy as np
import ufl

import firedrake
from firedrake import functionspaceimpl
from firedrake.function import Function, CoordinatelessFunction
from firedrake.constant import Constant
#from firedrake.utils import IntType, RealType, ScalarType

from pyop2 import op2
from pyop2.datatypes import ScalarType, IntType, as_ctypes
from pyop2.utils import as_tuple

from finat.point_set import PointSet
from finat.quadrature import QuadratureRule

import functools

__all__ = ['ScalarSubspace', 'RotatedSubspace', 'Subspaces',
           'BoundarySubspace', 'BCRotatedSubspace']


class Subspace(ufl.Subspace):
    r"""Wrapper base for `ufl.Subspace`.

    :arg function_space: The :class:`~.functionspaceimpl.WithGeometry`.
    :arg val: The subspace values that are multiplied to basis functions.
    :arg subdomain: The subdomain(s) on which values are set.
    The constructor mimics that of :class:`~DirichletBC`.
    """
    def __init__(self, function_space, val=None, subdomain=None, name=None, dtype=ScalarType):
        V = function_space
        if isinstance(V, Function):
            V = V.function_space()
        elif not isinstance(V, functionspaceimpl.WithGeometry):
            raise NotImplementedError("Can't make a Subspace defined on a "
                                      + str(type(function_space)))
        if subdomain:
            if not val:
                raise RuntimeError("Must provide val if providing subdomain.")
            if not isinstance(subdomain, op2.Subset):
               # Turn subdomain into op2.Subset.
               subdomain = V.boundary_node_subset(subdomain) 
            val = Function(V).assign(val, subset=subdomain)
            self._data = val.topological
        else:
            if isinstance(val, (Function, CoordinatelessFunction)):
                val = val.topological
                if val.function_space() != V.topological:
                    raise ValueError("Function values have wrong function space.")
                self._data = val
            else:
                self._data = CoordinatelessFunction(V.topological,
                                                    val=val, name=name, dtype=dtype)
        self._function_space = V
        super().__init__(V)

    def __getattr__(self, name):
        val = getattr(self._data, name)
        setattr(self, name, val)
        return val

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            that this :class:`Subspace` is a subspace of.
        """
        return self._function_space

    #@utils.cached_property
    @property
    def complement(self):
        return ComplementSubspace(self)


class ScalarSubspace(Subspace):
    def __init__(self, function_space, val=None, subdomain=None, name=None, dtype=ScalarType):
        super().__init__(function_space, val=val, subdomain=subdomain, name=name, dtype=dtype)


class RotatedSubspace(Subspace):
    def __init__(self, function_space, val=None, subdomain=None, name=None, dtype=ScalarType):
        super().__init__(function_space, val=val, subdomain=subdomain, name=name, dtype=dtype)


class Subspaces(object):
    r"""Bag of :class:`.Subspace`s.

    :arg subspaces: :class:`.Subspace` objects.
    """

    def __init__(self, *subspaces):
        self._components = tuple(subspaces)

    def __iter__(self):
        return iter(self._components)

    #@utils.cached_property
    @property
    def complement(self):
        return ComplementSubspace(self)


class ComplementSubspace(object):
    r"""Complement of :class:`.Subspace` or :class:`.Subspaces`."""

    def __init__(self, subspace):
        if not isinstance(subspace, (Subspace, Subspaces)):
            raise TypeError("Expecting `Subspace` or `Subspaces`,"
                            " not %s." % subspace.__class__.__name__)
        self._subspace = subspace

    #@utils.cached_property
    @property
    def complement(self):
        return self._subspace


def BoundarySubspace(V, subdomain):
    r"""Return Subspace associated with boundaries.

    :arg V: The function space.
    :arg subdomain: The subdomain.
    """
    subdomain = as_tuple(subdomain)
    if not isinstance(V, functionspaceimpl.WithGeometry):
        raise TypeError("V must be `functionspaceimpl.WithGeometry`, not %s." % V.__class__.__name__ )
    tV = V.topological
    if type(tV) == functionspaceimpl.MixedFunctionSpace:
        g = {False: {'scalar': None, 'rot': None},  # primary subspaces (_compl = False)
             True: {'scalar': None, 'rot': None}}  # complement subspaces (_compl = True)
        for i, Vsub in enumerate(V):
            ff, _compl = _boundary_subspace_functions(Vsub, subdomain)
            for (typ, f) in zip(('scalar', 'rot'), ff):
                if f:
                    if not g[_compl][typ]:
                        g[_compl][typ] = Function(V)
                    g[_compl][typ].assign(f)
        ss = []
        if (typ, cls) in zip(('scalar', 'rot'), (ScalarSubspace, RotatedSubspace)):
            if g[False][typ]:
                ss.append(cls(g[False][typ]))
            if g[True][typ]:
                ss.append(cls(g[True][typ]).complement)
        return Subspaces(*ss)
    else:
        ff, _complement = _boundary_subspace_functions(V, subdomain)
        # Reconstruct the parent WithGeometry
        # TODO: When submesh lands, just use W = V.parent.
        indices = []
        while tV.parent:
            indices.append(tV.index if tV.index is not None else tV.component)
            tV = tV.parent
        if len(indices) == 0:
            W = V
        else:
            W = functionspaceimpl.WithGeometry(tV, V.mesh())
        gg = [None, None]
        for i, f in enumerate(ff):
            if f:
                gg[i] = Function(W)
                gsub = gg[i]
                for ix in reversed(indices):
                    gsub = gsub.sub(ix)
                gsub.assign(f)
        if gg[0] and gg[1]:
            ss = Subspaces(ScalarSubspace(W, val=gg[0]), RotatedSubspace(W, val=gg[1]))
        elif gg[0]:
            ss = ScalarSubspace(W, val=gg[0])
        elif gg[1]:
            ss = RotatedSubspace(W, val=gg[1])
        else:
            raise NotImplementedError("Implement EmptySubspace?")

        if _complement:
            return ss.complement
        else:
            return ss


def _boundary_subspace_functions(V, subdomain):
    #from firedrake import TestFunction, TrialFunction, Masked, FacetNormal, inner, dx, grad, ds, solve, par_loop
    from firedrake import FacetNormal, inner, dx, grad, ds, solve, par_loop, dot, as_tensor
    #from firedrake.parloops import par_loop
    #from firedrake import solve
    # Define op2.subsets to be used when defining filters
    if V.ufl_element().family() == 'Hermite':
        assert V.ufl_element().degree() == 3

        v = firedrake.TestFunction(V)
        u = firedrake.TrialFunction(V)

        subset_value = V.node_subset(derivative_order=0)  # subset of value nodes
        subset_deriv = V.node_subset(derivative_order=1)  # subset of derivative nodes

        corner_list = []
        nsubdomain = len(subdomain)
        for i in range(nsubdomain):
            for j in range(i + 1, nsubdomain):
                a = V.boundary_node_subset(subdomain[i])
                b = V.boundary_node_subset(subdomain[j])
                corner_list.append(a.intersection(b))
        corners = functools.reduce(lambda a, b: a.union(b), corner_list)
        g1 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain).difference(corners).intersection(subset_deriv))
        v1 = firedrake.Masked(v, g1)
        u1 = firedrake.Masked(u, g1)
        quad_rule_boun = QuadratureRule(PointSet([[0, ], [1, ]]), [0.5, 0.5])

        normal = FacetNormal(V.mesh())

        """
        aa = inner(u - u1, v - v1) * dx + inner(grad(u1), grad(v1)) * ds(subdomain, scheme=quad_rule_boun)
        ff = inner(normal, grad(v1)) * ds(subdomain, scheme=quad_rule_boun)
        s0 = Function(V)
        s1 = Function(V)
        solve(aa == ff, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s1 = _normalise_subspace(s1, subdomain)
        s0.assign(Constant(1.), subset=V.node_set.difference(V.boundary_node_subset(subdomain)))
        return (s0, s1), True
        """
        tangent = dot(as_tensor([[0., 1.], [-1., 0.]]), normal)
        aa = inner(u - u1, v - v1) * dx + inner(grad(u1), grad(v1)) * ds(subdomain, scheme=quad_rule_boun)
        ff = inner(tangent, grad(v1)) * ds(subdomain, scheme=quad_rule_boun)
        s0 = Function(V)
        s1 = Function(V)
        solve(aa == ff, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s1 = _normalise_subspace(s1, subdomain)
        s0.assign(Constant(1.), subset=V.boundary_node_subset(subdomain).difference(corners).intersection(subset_value))
        s0.assign(Constant(1.), subset=corners)
        return (s0, s1), False
    else:
        f0 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        return (f0, None), False


def BCRotatedSubspace():
    pass


def _normalise_subspace(old_subspace, subdomain):
    from firedrake import par_loop, ds, WRITE, READ
    domain = "{[k]: 0 <= k < 3}"
    instructions = """
    <float64> eps = 1e-9
    <float64> norm = 0
    for k
        norm = sqrt(old_subspace[3 * k + 1] * old_subspace[3 * k + 1] + old_subspace[3 * k + 2] * old_subspace[3 * k + 2])
        if norm > eps
            new_subspace[3 * k + 1] = old_subspace[3 * k + 1] / norm
            new_subspace[3 * k + 2] = old_subspace[3 * k + 2] / norm
        end
    end
    """
    V = old_subspace.function_space()
    new_subspace = Function(V)
    par_loop((domain, instructions), ds(subdomain),
             {"new_subspace": (new_subspace, WRITE),
              "old_subspace": (old_subspace, READ)},
             is_loopy_kernel=True)
    return new_subspace
