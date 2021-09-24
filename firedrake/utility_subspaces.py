import functools

import firedrake
from firedrake import functionspaceimpl
from firedrake.function import Function 
from firedrake.constant import Constant
from firedrake.subspace import ScalarSubspace, RotatedSubspace, Subspaces, DirectSumSubspace

from pyop2.datatypes import ScalarType
from pyop2.utils import as_tuple

from finat.point_set import PointSet
from finat.quadrature import QuadratureRule

import ufl


__all__ = ['BoundarySubspace', 'BoundaryComponentSubspace']


def _BoundarySubspace(V, subdomain, constructor, extra_tuple=None):
    r"""Return Subspace required to constrain ALL DoFs in `subdomain`.

    :arg V: The function space.
    :arg subdomain: The subdomain.
    :arg extra_tuple:
    """
    if not isinstance(V, functionspaceimpl.WithGeometry):
        raise TypeError("V must be `functionspaceimpl.WithGeometry`, not %s." % type(V))
    if isinstance(subdomain, str):
        subdomain = (subdomain, )
    else:
        subdomain = as_tuple(subdomain)
    if extra_tuple is not None:
        if not isinstance(extra_tuple, (tuple, list)):
            _ = tuple(extra_tuple for _ in subdomain)
            extra_tuple = _
        assert len(extra_tuple) == len(subdomain)
    tV = V.topological
    if type(tV) == functionspaceimpl.MixedFunctionSpace:
        W = V
        Wsub_tuple = tuple(W)
        indices_tuple = tuple((i, ) for i, _ in enumerate(W))
        if extra_tuple is not None:
            extra_tuple = zip(*tuple(extra.split() for extra in extra_tuple))
    else:
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
        Wsub_tuple = (V, )
        indices_tuple = (indices, )
    if extra_tuple == None:
        extra_tuple = tuple(None for _ in Wsub_tuple)
    else:
        extra_tuple = (extra_tuple, )
    gg = {}
    for Wsub, indices, extra in zip(Wsub_tuple, indices_tuple, extra_tuple):
        if extra is None:
            ff = constructor(Wsub, subdomain)
        else:
            ff = constructor(Wsub, subdomain, extra)
        for f in ff:
            g = gg.setdefault(f.__class__.__name__, type(f)(W))
            gsub = g
            for ix in reversed(indices):
                gsub = gsub.sub(ix)
            f.dat.copy(gsub.dat)
    #return Subspaces(*tuple(g for _, g in gg.items()))
    return DirectSumSubspace(*tuple(g for _, g in gg.items()))


def BoundarySubspace(V, subdomain):
    return _BoundarySubspace(V, subdomain, _boundary_subspace_functions)


def BoundaryComponentSubspace(V, subdomain, thetas):
    return _BoundarySubspace(V, subdomain, _boundary_component_subspace_functions, extra_tuple=thetas)


def _boundary_subspace_functions(V, subdomain):
    from firedrake import TestFunction, TrialFunction, Projected, \
                          FacetNormal, FacetArea, \
                          grad, dot, inner, dx, ds, as_tensor, solve
    mesh = V.mesh()
    elem = V.ufl_element()
    if elem.family() == 'Hermite':
        assert elem.degree() == 3

        subset_all = V.boundary_node_subset(subdomain)
        subset_value = V.node_subset(derivative_order=0)  # subset of value nodes
        subset_deriv = V.node_subset(derivative_order=1)  # subset of derivative nodes
        subset_corners = V.subdomain_intersection_subset(subdomain)
        g_ = Function(V).assign(Constant(1.), subset=subset_all.difference(subset_corners).intersection(subset_deriv))
        s_ = ScalarSubspace(V, val=g_)
        quad_rule_ds = QuadratureRule(PointSet([[0, ], [1, ]]), [0.5, 0.5])
        farea = FacetArea(mesh)
        normal = FacetNormal(mesh)
        tangent = dot(as_tensor([[0., 1.], [-1., 0.]]), normal)
        v = TestFunction(V)
        u = TrialFunction(V)
        v_ = Projected(v, s_)
        u_ = Projected(u, s_)
        a = inner(u - u_, v - v_) * dx + farea * inner(grad(u_), grad(v_)) * ds(subdomain, scheme=quad_rule_ds)
        L = farea * inner(tangent, grad(v_)) * ds(subdomain, scheme=quad_rule_ds)
        s0 = Function(V)
        s1 = Function(V)
        solve(a == L, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s1 = _normalise_subspace_hermite(s1, subdomain)
        s0.assign(Constant(1.), subset=subset_all.difference(subset_corners).intersection(subset_value).union(subset_corners))
        return Subspaces(ScalarSubspace(V, s0), RotatedSubspace(V, s1))
    elif V.ufl_element().family() == 'Morley':
        raise NotImplementedError("Morley not implemented.")
    elif V.ufl_element().family() == 'Argyris':
        raise NotImplementedError("Argyris not implemented.")
    elif V.ufl_element().family() == 'Bell':
        raise NotImplementedError("Bell not implemented.")
    else:
        f0 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        return Subspaces(ScalarSubspace(V, f0), )


def _boundary_component_subspace_functions(V, subdomain, thetas):
    from firedrake import TestFunction, TrialFunction, Projected, \
                          FacetNormal, FacetArea, \
                          grad, dot, inner, dx, ds, as_tensor, solve
    mesh = V.mesh()
    elem = V.ufl_element()
    tdim = mesh.topology.topology_dm.getDimension()
    gdim = mesh.topology.topology_dm.getCoordinateDim()
    if tdim != gdim:
        raise NotImplementedError("Currently not implemented for immersed manifolds.")
    if tdim != 2:
        raise NotImplementedError("Currently only implemented for dim == 2.")
    if isinstance(elem, ufl.VectorElement) and elem.sub_elements()[0].family() == "Lagrange":
        if elem.reference_value_shape()[0] != tdim:
            raise NotImplementedError("Currently only implemented for vector dim == topological dim.")
        order = V.finat_element.fiat_equivalent.get_order()
        entity_dofs = V.finat_element.entity_dofs()
        if len(entity_dofs[0]) == 3:  # Simplex
            nodes = entity_dofs[0][0] + entity_dofs[1][2] + entity_dofs[0][1]
        elif len(entity_dofs[0]) == 4:  # Quad
            nodes = [(order + 1) * i for i in range(order + 1)]
        points = [key[0] for node in nodes for key in V.finat_element.fiat_equivalent.dual_basis()[node].get_point_dict()]
        subset_all = V.boundary_node_subset(subdomain)
        subset_corners = V.subdomain_intersection_subset(subdomain)
        g_ = Function(V).assign(Constant(1.), subset=subset_all.difference(subset_corners))
        s_ = ScalarSubspace(V, g_)
        v = firedrake.TestFunction(V)
        u = firedrake.TrialFunction(V)
        v_ = Projected(v, s_)
        u_ = Projected(u, s_)
        quad_rule_ds = QuadratureRule(PointSet([[p, ] for p in points]), [1. for _ in range(order + 1)])
        a = inner(u - u_, v - v_) * dx + inner(u_, v_) * ds(subdomain, scheme=quad_rule_ds)
        L = functools.reduce(lambda b, c: b + c, [inner(theta, v_) * ds(sd, scheme=quad_rule_ds)
                                                  for sd, theta in zip(subdomain, thetas)])
        s1 = Function(V)
        solve(a == L, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s0 = Function(V)
        s0.assign(Constant(1.), subset=subset_corners)
        #return Subspaces(RotatedSubspace(V, s1))
        return Subspaces(ScalarSubspace(V, s0), RotatedSubspace(V, s1))
    else:
        raise NotImplementedError("Currently only implemented for vector Lagrange element.")


def _normalise_subspace_hermite(old_subspace, subdomain):
    from firedrake import ds, par_loop, WRITE, READ
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
