"""
This module implements the user-visible API for constructing
:class:`FunctionSpace` and :class:`MixedFunctionSpace` objects.  The
API is functional, rather than object-based, to allow for simple
backwards-compatibility and argument checking and dispatch.
"""
from __future__ import absolute_import

import ufl

from pyop2.utils import flatten

from firedrake import functionspaceimpl as impl


__all__ = ("MixedFunctionSpace", "FunctionSpace",
           "VectorFunctionSpace", "TensorFunctionSpace")


def make_scalar_element(mesh, family, degree, vfamily, vdegree):
    """Build a scalar :class:`ufl.FiniteElement`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the element is returned immediately.

    .. note::

       As a side effect, this function finalises the initialisation of
       the provided mesh, by calling :meth:`.MeshTopology.init` (or
       :meth:`.MeshGeometry.init` as appropriate.
    """
    mesh.init()
    if isinstance(family, ufl.FiniteElementBase):
        return family

    topology = mesh.topology
    cell = topology.ufl_cell()

    if isinstance(cell, ufl.TensorProductCell) \
       and vfamily is not None and vdegree is not None:
        la = ufl.FiniteElement(family,
                               cell=cell.sub_cells()[0],
                               degree=degree)
        # If second element was passed in, use it
        lb = ufl.FiniteElement(vfamily,
                               cell=ufl.interval,
                               degree=vdegree)
        # Now make the TensorProductElement
        return ufl.TensorProductElement(la, lb)
    else:
        return ufl.FiniteElement(family, cell=cell, degree=degree)


def FunctionSpace(mesh, family, degree=None, name=None, vfamily=None,
                  vdegree=None):
    """Create a :class:`.FunctionSpace`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg name: An optional name for the function space.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is returned.
    """
    element = make_scalar_element(mesh, family, degree, vfamily, vdegree)

    # Support FunctionSpace(mesh, MixedElement)
    if type(element) is ufl.MixedElement:
        return MixedFunctionSpace(element, mesh=mesh, name=name)

    # Otherwise, build the FunctionSpace.
    topology = mesh.topology
    new = impl.FunctionSpace(topology, element, name=name)
    if mesh is not topology:
        return impl.WithGeometry(new, mesh)
    else:
        return new


def VectorFunctionSpace(mesh, family, degree=None, dim=None,
                        name=None, vfamily=None, vdegree=None):
    """Create a rank-1 :class:`.FunctionSpace`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg dim: An optional number of degrees of freedom per function
       space node (defaults to the geometric dimension of the mesh).
    :arg name: An optional name for the function space.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is
    returned.  In this case, the provided element must have an empty
    :meth:`ufl.FiniteElementBase.value_shape`.

    If you have an existing :class:`ufl.VectorElement`, you should
    pass it to :func:`FunctionSpace`.
    """
    sub_element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    assert sub_element.value_shape() == ()
    dim = dim or mesh.ufl_cell().geometric_dimension()
    element = ufl.VectorElement(sub_element, dim=dim)
    return FunctionSpace(mesh, element, name=name)


def TensorFunctionSpace(mesh, family, degree=None, shape=None,
                        symmetry=None, name=None, vfamily=None,
                        vdegree=None):
    """Create a rank-2 :class:`.FunctionSpace`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg shape: An optional shape for the tensor-valued degrees of
       freedom at each function space node (defaults to a square
       tensor using the geometric dimension of the mesh).
    :arg symmetry: Optional symmetries in the tensor value.
    :arg name: An optional name for the function space.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is
    returned.  In this case, the provided element must have an empty
    :meth:`ufl.FiniteElementBase.value_shape`.

    If you have an existing :class:`ufl.TensorElement`, you should
    pass it to :func:`FunctionSpace`.
    """
    sub_element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    assert sub_element.value_shape() == ()
    shape = shape or (mesh.ufl_cell().geometric_dimension(),) * 2
    element = ufl.TensorElement(sub_element, shape=shape, symmetry=symmetry)
    return FunctionSpace(mesh, element, name=name)


def MixedFunctionSpace(spaces, name=None, mesh=None):
    """Create a :class:`.MixedFunctionSpace`.

    :arg spaces: An iterable of constituent spaces, or a
        :class:`ufl.MixedElement`.
    :arg name: An optional name for the mixed function space.
    :arg mesh: An optional mesh.  Must be provided if spaces is a
        :class:`ufl.MixedElement`, ignored otherwise.
    """
    if isinstance(spaces, ufl.FiniteElementBase):
        # Build the spaces if we got a mixed element
        assert type(spaces) is ufl.MixedElement and mesh is not None
        sub_elements = []

        def rec(eles):
            for ele in eles:
                if ele.num_sub_elements() > 0:
                    rec(ele.sub_elements())
                else:
                    sub_elements.append(ele)
        rec(spaces.sub_elements())
        spaces = [FunctionSpace(mesh, element) for element in sub_elements]

    # Check that function spaces are on the same mesh
    meshes = [space.mesh() for space in spaces]
    for i in xrange(1, len(meshes)):
        if meshes[i] is not meshes[0]:
            raise ValueError("All function spaces must be defined on the same mesh!")

    # Select mesh
    mesh = meshes[0]
    # Get topological spaces
    spaces = tuple(s.topological for s in flatten(spaces))
    # Error checking
    for space in spaces:
        if type(space) is impl.FunctionSpace:
            continue
        elif type(space) is impl.ProxyFunctionSpace:
            if space.component is not None:
                raise ValueError("Can't pass a %s %s" % (space.typ, type(space)))
            continue
        else:
            raise ValueError("Can't make mixed space with %s" % type(space))

    new = impl.MixedFunctionSpace(spaces, name=name)
    if mesh is not mesh.topology:
        return impl.WithGeometry(new, mesh)
    return new
