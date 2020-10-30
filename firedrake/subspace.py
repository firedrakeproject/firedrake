import functools
import numpy as np

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
from tsfc.finatinterface import create_element

import gem


__all__ = ['ScalarSubspace', 'RotatedSubspace', 'Subspaces',
           'BoundarySubspace', 'BoundaryComponentSubspace']


class Subspace(object):
    r"""Wrapper base for Firedrake subspaces.

    :arg function_space: The :class:`~.functionspaceimpl.WithGeometry`.
    :arg val: The subspace values that are multiplied to basis functions.
    :arg subdomain: The subdomain(s) on which values are set.
    The constructor mimics that of :class:`~DirichletBC`.
    """

    _globalcount = 0

    def __init__(self, function_space, val=None, subdomain=None, name=None, dtype=ScalarType, count=None):

        self._count = count or Subspace._globalcount
        if self._count >= Subspace._globalcount:
            Subspace._globalcount = self._count + 1

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
        self.parent = None
        self.index = None

        self._repr = "Subspace(%s, %s)" % (repr(self._function_space), repr(self._count))

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            that this :class:`Subspace` is a subspace of.
        """
        return self._function_space

    def ufl_element(self):
        return self.function_space().ufl_element()

    def __getattr__(self, name):
        val = getattr(self._data, name)
        setattr(self, name, val)
        return val
    
    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "s_%s" % count
        else:
            return "s_{%s}" % count

    def __repr__(self):
        return self._repr

    #@utils.cached_property
    @property
    def complement(self):
        return ComplementSubspace(self)

    @staticmethod
    def transform_matrix(elem, expression, dtype):
        r"""Construct transformation matrix.

        :arg elem: UFL element: `self.ufl_function_space().ufl_element`
            or its subelement (in case of `MixedElement`).
        :arg expression: GEM expression representing local subspace data array
            associated with elem.
        :arg dtype: data type (= KernelBuilder.scalar_type).

        Classical implementation of functions/function spaces.
        Linear combination of basis:
        
        u = \sum [ u_i * \phi_i ]
              i
        
        u     : function
        u_i   : ith coefficient
        \phi_i: ith basis
        """
        raise NotImplementedError("Must implement `transform_matrix` method.")


class IndexedSubspace(object):
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def function_space(self):
        return self.parent.function_space().split()[self.index]

    def ufl_element(self):
        return self.function_space().ufl_element()

    def transform_matrix(self, elem, expression, dtype):
        return self.parent.transform_matrix(elem, expression, dtype)

    def __eq__(self, other):
        return self.parent is other.parent and \
               self.index == other.index
    
    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "%s[%s]" % (self.parent, self.index)

    def __repr__(self):
        return "IndexedSubspace(%s, %s)" % (repr(self.parent), repr(self.index))


class ScalarSubspace(Subspace):
    def __init__(self, V, val=None, subdomain=None, name=None, dtype=ScalarType):
        Subspace.__init__(self, V, val=val, subdomain=subdomain, name=name, dtype=dtype)

    @staticmethod
    def transform_matrix(elem, expression, dtype):
        r"""Basic subspace.

        Linear combination of weighted basis:

        u = \sum [ u_i * (w_i * \phi_i) ]
              i

        u     : function
        u_i   : ith coefficient
        \phi_i: ith basis
        w_i   : ith weight (stored in the subspace object)
                w_i = 0 to deselect the associated basis.
                w_i = 1 to select.
        """
        shape = expression.shape
        ii = tuple(gem.Index(extent=extent) for extent in shape)
        jj = tuple(gem.Index(extent=extent) for extent in shape)
        eye = gem.Literal(1)
        for i, j in zip(ii, jj):
            #eye = gem.Product(eye, gem.Delta(i, j))
            eye = gem.Product(eye, gem.Indexed(gem.Identity(i.extent), (i, j)))
        mat = gem.ComponentTensor(gem.Product(eye, expression[ii]), ii + jj)
        return mat


class RotatedSubspace(Subspace):
    def __init__(self, V, val=None, subdomain=None, name=None, dtype=ScalarType):
        Subspace.__init__(self, V, val=val, subdomain=subdomain, name=name, dtype=dtype)

    @staticmethod
    def transform_matrix(elem, expression, dtype):
        r"""Rotation subspace.

        u = \sum [ u_i * \sum [ \psi(e)_i * \sum [ \psi(e)_k * \phi(e)_k ] ] ]
              i            e                  k

        u       : function
        u_i     : ith coefficient
        \phi(e) : basis vector whose elements not associated with
                  entity e are set zero.
        \psi(e) : rotation vector whose elements not associated with
                  entity e are set zero.
        """
        shape = expression.shape
        finat_element = create_element(elem)
        if len(shape) == 1:
            entity_dofs = finat_element.entity_dofs()
        else:
            entity_dofs = finat_element.base_element.entity_dofs()
        ii = tuple(gem.Index(extent=extent) for extent in shape)
        jj = tuple(gem.Index(extent=extent) for extent in shape)
        comp = gem.Zero()
        for dim in entity_dofs:
            for _, dofs in entity_dofs[dim].items():
                if len(dofs) == 0 or (len(dofs) == 1 and len(shape) == 1):
                    continue
                ind = np.zeros(shape, dtype=dtype)
                for dof in dofs:
                    for ndind in np.ndindex(shape[1:]):
                        ind[(dof, ) + ndind] = 1.
                comp = gem.Sum(comp, gem.Product(gem.Product(gem.Literal(ind)[ii], expression[ii]),
                                                 gem.Product(gem.Literal(ind)[jj], expression[jj])))
        mat = gem.ComponentTensor(comp, ii + jj)
        return mat


class Subspaces(object):
    r"""Bag of :class:`.Subspace`s.

    :arg subspaces: :class:`.Subspace` objects.
    """

    def __init__(self, *subspaces):
        self._components = tuple(subspaces)

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    #@utils.cached_property
    @property
    def components(self):
        return self._components

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
    r"""Return Subspace required to constrain ALL DoFs in `subdomain`.

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
                    g[_compl][typ].sub(i).assign(f)
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

        if _complement:
            return ss.complement
        else:
            return ss


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

        ff, _complement = _boundary_component_subspace_functions(V, subdomain, thetas)
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
        corners = functools.reduce(lambda a, b: a.union(b), corner_list) if corner_list else V.boundary_node_empty_subset()
        g1 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain).difference(corners).intersection(subset_deriv))
        v1 = firedrake.Projected(v, ScalarSubspace(V, val=g1))
        u1 = firedrake.Projected(u, ScalarSubspace(V, val=g1))
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
    elif V.ufl_element().family() == 'Morley':
        raise NotImplementedError("Morley not implemented.")
    elif V.ufl_element().family() == 'Argyris':
        raise NotImplementedError("Argyris not implemented.")
    elif V.ufl_element().family() == 'Bell':
        raise NotImplementedError("Bell not implemented.")
    else:
        f0 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        return (f0, None), False


def _boundary_component_subspace_functions(V, subdomain, thetas):
    #from firedrake import TestFunction, TrialFunction, Masked, FacetNormal, inner, dx, grad, ds, solve, par_loop
    from firedrake import FacetNormal, inner, dx, grad, ds, solve, par_loop, dot, as_tensor
    #from firedrake.parloops import par_loop
    #from firedrake import solve
    # Define op2.subsets to be used when defining filters
    if True:

        v = firedrake.TestFunction(V)
        u = firedrake.TrialFunction(V)

        #corner_list = []
        #nsubdomain = len(subdomain)
        #for i in range(nsubdomain):
        #    for j in range(i + 1, nsubdomain):
        #        a = V.boundary_node_subset(subdomain[i])
        #        b = V.boundary_node_subset(subdomain[j])
        #        corner_list.append(a.intersection(b))
        #corners = functools.reduce(lambda a, b: a.union(b), corner_list)
        #g1 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain).difference(corners).intersection(subset_deriv))
        g1 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        v1 = firedrake.Projected(v, ScalarSubspace(V, g1))
        u1 = firedrake.Projected(u, ScalarSubspace(V, g1))
        #quad_rule_boun = QuadratureRule(PointSet([[0, ], [1, ]]), [0.5, 0.5])
        quad_rule_boun = QuadratureRule(PointSet([[0, ], [0.5, ], [1, ]]), [0.25, 0.50, 0.25])

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
        #tangent = dot(as_tensor([[0., 1.], [-1., 0.]]), normal)
        aa = inner(u - u1, v - v1) * dx + inner(u1, v1) * ds(subdomain, scheme=quad_rule_boun)
        ff = inner(thetas[0], v1) * ds(subdomain, scheme=quad_rule_boun)
        s1 = Function(V)
        solve(aa == ff, s1, solver_parameters={"ksp_type": 'cg', "ksp_rtol": 1.e-16})
        s1 = _normalise_subspace2(s1, subdomain)
        return (None, s1), False
    else:
        f0 = Function(V).assign(Constant(1.), subset=V.boundary_node_subset(subdomain))
        return (f0, None), False


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


def make_subspace_numbers_and_parts(subspaces, original_subspaces):
    # -- subspace_numbers_: which subspaces are used in this TSFCIntegralData.
    # -- subspace_parts_  : which components are used if mixed (otherwise None).
    subspaces_and_parts_dict = {}
    for subspace in subspaces:
        if subspace.parent:
            subspaces_and_parts_dict.setdefault(subspace.parent, set()).update((subspace.index, ))
        else:
            subspaces_and_parts_dict[subspace] = None
    subspace_numbers = []
    subspace_parts = []
    for i, subspace in enumerate(original_subspaces):
        if subspace in subspaces_and_parts_dict:
            subspace_numbers.append(i)
            parts = subspaces_and_parts_dict[subspace]
            if parts is None:
                subspace_parts.append(None)
            else:
                parts = sorted(parts)
                subspace_parts.append(parts)
    subspaces = sort_indexed_subspaces(subspaces)
    return subspaces, subspace_numbers, subspace_parts


def sort_indexed_subspaces(subspaces):
    return sorted(subspaces, key=lambda s: (s.parent.count() if s.parent else s.count(), 
                                            -1 if s.index is None else s.index))
