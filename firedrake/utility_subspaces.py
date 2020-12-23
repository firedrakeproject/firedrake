import functools

import firedrake
from firedrake import functionspaceimpl
from firedrake.function import Function 
from firedrake.constant import Constant
from firedrake.subspace import ScalarSubspace, RotatedSubspace, Subspaces

from pyop2.datatypes import ScalarType
from pyop2.utils import as_tuple

from finat.point_set import PointSet
from finat.quadrature import QuadratureRule


__all__ = ['BoundarySubspace', 'BoundaryComponentSubspace']


def BoundarySubspace(V, subdomain):
    r"""Return Subspace required to constrain ALL DoFs in `subdomain`.

    :arg V: The function space.
    :arg subdomain: The subdomain.
    """
    if not isinstance(V, functionspaceimpl.WithGeometry):
        raise TypeError("V must be `functionspaceimpl.WithGeometry`, not %s." % V.__class__.__name__ )
    if isinstance(subdomain, str):
        subdomain = (subdomain, )
    else:
        subdomain = as_tuple(subdomain)
    tV = V.topological
    if type(tV) == functionspaceimpl.MixedFunctionSpace:
        #f = {'scalar': None, 'rotated': None}
        f = {}
        for i, Vi in enumerate(V):
            fis = _boundary_subspace_functions(Vi, subdomain)
            for fi in fis:
                s = f.setdefault(fi.__class__.__name__, type(fi)(V))
                fi.dat.copy(s.sub(i).dat)
        return Subspaces(*tuple(s for _, s in f.items()))
    else:
        ff = _boundary_subspace_functions(V, subdomain)
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
        gg = {}
        for f in ff:
            g = gg.setdefault(f.__class__.__name__, type(f)(W))
            gsub = g
            for ix in reversed(indices):
                gsub = gsub.sub(ix)
            f.dat.copy(gsub.dat)
        return Subspaces(*tuple(g for _, g in gg.items()))


def BoundaryComponentSubspace(V, subdomain, thetas):
    r"""Return Subspace required to constrain DoF in `subdomain` in thetas-directions.

    :arg V: The function space.
    :arg subdomain: The subdomain.
    :arg thetas: directions
    """
    subdomain = as_tuple(subdomain)
    if not isinstance(thetas, (tuple, list)):
        theta = thetas
        thetas = tuple(theta for _ in subdomain)
    assert len(thetas) == len(subdomain)
    if not isinstance(V, functionspaceimpl.WithGeometry):
        raise TypeError("V must be `functionspaceimpl.WithGeometry`, not %s." % V.__class__.__name__ )
    tV = V.topological
    if type(tV) == functionspaceimpl.MixedFunctionSpace:
        raise NotImplementedError("MixedFunctionSpace not implemented yet.")
    else:
        elem = V.ufl_element()
        shape = elem.value_shape()
        if shape == ():
            # Scalar element
            raise TypeError("Can not rotate Scalar element.")
        elif len(shape) == 1:
            # Vector element
            pass
        else:
            # Tensor element
            raise NotImplementedError("TensorElement not implemented yet.")

        ff = _boundary_component_subspace_functions(V, subdomain, thetas)
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
                g = Function(W)
                gsub = g
                for ix in reversed(indices):
                    gsub = gsub.sub(ix)
                gsub.assign(f)
                gg[i] = g
        if gg[0] and gg[1]:
            ss = Subspaces(ScalarSubspace(W, val=gg[0]), RotatedSubspace(W, val=gg[1]))
        elif gg[0]:
            ss = ScalarSubspace(W, val=gg[0])
        elif gg[1]:
            ss = RotatedSubspace(W, val=gg[1])
        else:
            raise NotImplementedError("Implement EmptySubspace?")
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
        corners = functools.reduce(lambda a, b: a.union(b), corner_list) if corner_list else V.boundary_node_empty_subset()
        g1 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain).difference(corners).intersection(subset_deriv))
        _subspace = ScalarSubspace(V, val=g1)
        v1 = firedrake.Projected(v, _subspace)
        u1 = firedrake.Projected(u, _subspace)
        quad_rule_boun = QuadratureRule(PointSet([[0, ], [1, ]]), [0.5, 0.5])

        normal = FacetNormal(V.mesh())
        tangent = dot(as_tensor([[0., 1.], [-1., 0.]]), normal)
        aa = inner(u - u1, v - v1) * dx + firedrake.FacetArea(V.mesh()) * inner(grad(u1), grad(v1)) * ds(subdomain, scheme=quad_rule_boun)
        ff = firedrake.FacetArea(V.mesh()) * inner(tangent, grad(v1)) * ds(subdomain, scheme=quad_rule_boun)
        s0 = Function(V)
        s1 = Function(V)
        solve(aa == ff, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s1 = _normalise_subspace(s1, subdomain)
        s0.assign(Constant(1.), subset=V.boundary_node_subset(subdomain).difference(corners).intersection(subset_value))
        s0.assign(Constant(1.), subset=corners)
        #return s0, s1
        return Subspaces(ScalarSubspace(V, s0), RotatedSubspace(V, s1))
    elif V.ufl_element().family() == 'Morley':
        raise NotImplementedError("Morley not implemented.")
    elif V.ufl_element().family() == 'Argyris':
        raise NotImplementedError("Argyris not implemented.")
    elif V.ufl_element().family() == 'Bell':
        raise NotImplementedError("Bell not implemented.")
    else:
        f0 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        #return f0, None
        return Subspaces(ScalarSubspace(V, f0), )


def _boundary_component_subspace_functions(V, subdomain, thetas):
    from firedrake import FacetNormal, inner, dx, grad, ds, solve, par_loop, dot, as_tensor
    if True:

        v = firedrake.TestFunction(V)
        u = firedrake.TrialFunction(V)

        g1 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        ss0 = ScalarSubspace(V, g1)
        ss1 = ScalarSubspace(V, g1)
        v1 = firedrake.Projected(v, ss0)
        u1 = firedrake.Projected(u, ss1)
        #subs = ScalarSubspace(V, g1)
        #v1 = firedrake.Projected(v, subs)
        #u1 = firedrake.Projected(u, subs)
        quad_rule_boun = QuadratureRule(PointSet([[0, ], [0.5, ], [1, ]]), [0.25, 0.50, 0.25])

        normal = FacetNormal(V.mesh())

        aa = inner(u - u1, v - v1) * dx + inner(u1, v1) * ds(subdomain, scheme=quad_rule_boun)
        ff = inner(thetas[0], v1) * ds(subdomain, scheme=quad_rule_boun)
        s1 = Function(V)
        solve(aa == ff, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s1 = _normalise_subspace2(s1, subdomain)
        return None, s1
    else:
        f0 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        return f0, None


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


def _normalise_subspace2(old_subspace, subdomain):
    from firedrake import par_loop, ds, WRITE, READ
    domain = "{[k]: 0 <= k < 6}"
    instructions = """
    <float64> eps = 1e-9
    <float64> norm = 0
    for k
        norm = sqrt(old_subspace[k, 0] * old_subspace[k, 0] + old_subspace[k, 1] * old_subspace[k, 1])
        if norm > eps
            new_subspace[k, 0] = old_subspace[k, 0] / norm
            new_subspace[k, 1] = old_subspace[k, 1] / norm
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


