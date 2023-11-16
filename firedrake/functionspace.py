"""
This module implements the user-visible API for constructing
:class:`.FunctionSpace` and :class:`.MixedFunctionSpace` objects.  The
API is functional, rather than object-based, to allow for simple
backwards-compatibility, argument checking, and dispatch.
"""
import ufl
import finat.ufl

from pyop2.utils import flatten

from firedrake import functionspaceimpl as impl
from firedrake.petsc import PETSc

import numbers


__all__ = ("MixedFunctionSpace", "FunctionSpace",
           "VectorFunctionSpace", "TensorFunctionSpace")


@PETSc.Log.EventDecorator()
def make_scalar_element(mesh, family, degree, vfamily, vdegree):
    """Build a scalar :class:`finat.ufl.FiniteElement`.

    :arg mesh: The mesh to determine the cell from.
    :arg family: The finite element family.
    :arg degree: The degree of the finite element.
    :arg vfamily: The finite element in the vertical dimension
        (extruded meshes only).
    :arg vdegree: The degree of the element in the vertical dimension
        (extruded meshes only).

    The ``family`` argument may be an existing
    :class:`finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the element is returned immediately.

    .. note::

       As a side effect, this function finalises the initialisation of
       the provided mesh, by calling :meth:`.AbstractMeshTopology.init` (or
       :meth:`.MeshGeometry.init`) as appropriate.
    """
    mesh.init()
    topology = mesh.topology
    cell = topology.ufl_cell()
    if isinstance(family, finat.ufl.FiniteElementBase):
        return family.reconstruct(cell=cell)

    if isinstance(cell, ufl.TensorProductCell) \
       and vfamily is not None and vdegree is not None:
        la = finat.ufl.FiniteElement(family,
                                     cell=cell.sub_cells()[0],
                                     degree=degree)
        # If second element was passed in, use it
        lb = finat.ufl.FiniteElement(vfamily,
                                     cell=ufl.interval,
                                     degree=vdegree)
        # Now make the TensorProductElement
        return finat.ufl.TensorProductElement(la, lb)
    else:
        return finat.ufl.FiniteElement(family, cell=cell, degree=degree)


def check_element(element, top=True):
    """Run some checks on the provided element.

    The :class:`~finat.ufl.VectorElement` and
    :class:`~finat.ufl.TensorElement` modifiers must be "outermost"
    for function space construction to work, excepting that they
    should not wrap a :class:`~finat.ufl.MixedElement`.  Similarly,
    a base :class:`~finat.ufl.MixedElement` must be outermost (it
    can contain :class:`~finat.ufl.MixedElement` instances, provided
    they satisfy the other rules). This function checks that.

    :arg element: The :class:`UFL element
        <finat.ufl.FiniteElementBase>` to check.
    :kwarg top: Are we at the top element (in which case the modifier
        is legal).
    :returns: ``None`` if the element is legal.
    :raises ValueError: if the element is illegal.

    """
    if type(element) in (finat.ufl.BrokenElement, finat.ufl.RestrictedElement,
                         finat.ufl.HDivElement, finat.ufl.HCurlElement):
        inner = (element._element, )
    elif type(element) is finat.ufl.EnrichedElement:
        inner = element._elements
    elif type(element) is finat.ufl.TensorProductElement:
        inner = element.sub_elements
    elif isinstance(element, finat.ufl.MixedElement):
        if not top:
            raise ValueError("%s modifier must be outermost" % type(element))
        else:
            inner = element.sub_elements
    else:
        return
    for e in inner:
        check_element(e, top=False)


@PETSc.Log.EventDecorator("CreateFunctionSpace")
def FunctionSpace(mesh, family, degree=None, name=None, vfamily=None,
                  vdegree=None):
    """Create a `.FunctionSpace`.

    Parameters
    ----------
    mesh:
        The mesh to determine the cell from.
    family:
        The finite element family.
    degree:
        The degree of the finite element.
    name:
        An optional name for the function space.
    vfamily:
        The finite element in the vertical dimension (extruded meshes
        only).
    vdegree:
        The degree of the element in the vertical dimension (extruded
        meshes only).

    Notes
    -----
    The ``family`` argument may be an existing
    `finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate `.FunctionSpace` is returned.

    """
    element = make_scalar_element(mesh, family, degree, vfamily, vdegree)

    # Support FunctionSpace(mesh, MixedElement)
    if type(element) is finat.ufl.MixedElement:
        return MixedFunctionSpace(element, mesh=mesh, name=name)
    if mesh.ufl_cell().cellname() == "hexahedron" and \
       element.family() not in ["Q", "DQ"]:
        raise NotImplementedError("Currently can only use 'Q' and/or 'DQ' elements on hexahedral meshes")
    # Check that any Vector/Tensor/Mixed modifiers are outermost.
    check_element(element)

    # Otherwise, build the FunctionSpace.
    topology = mesh.topology
    if element.family() == "Real":
        new = impl.RealFunctionSpace(topology, element, name=name)
    else:
        new = impl.FunctionSpace(topology, element, name=name)
    if mesh is not topology:
        return impl.WithGeometry.create(new, mesh)
    else:
        return new


@PETSc.Log.EventDecorator()
def DualSpace(mesh, family, degree=None, name=None, vfamily=None,
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
    :class:`finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is returned.
    """
    element = make_scalar_element(mesh, family, degree, vfamily, vdegree)

    # Support FunctionSpace(mesh, MixedElement)
    if type(element) is finat.ufl.MixedElement:
        return MixedFunctionSpace(element, mesh=mesh, name=name)

    # Check that any Vector/Tensor/Mixed modifiers are outermost.
    check_element(element)

    # Otherwise, build the FunctionSpace.
    topology = mesh.topology
    if element.family() == "Real":
        new = impl.RealFunctionSpace(topology, element, name=name)
    else:
        new = impl.FunctionSpace(topology, element, name=name)
    if mesh is not topology:
        return impl.FiredrakeDualSpace.create(new, mesh)
    else:
        return new


@PETSc.Log.EventDecorator()
def VectorFunctionSpace(mesh, family, degree=None, dim=None,
                        name=None, vfamily=None, vdegree=None):
    """Create a rank-1 `.FunctionSpace`.

    Parameters
    ----------
    mesh:
        The mesh to determine the cell from.
    family:
        The finite element family.
    degree:
        The degree of the finite element.
    dim:
        An optional number of degrees of freedom per function space
        node (defaults to the geometric dimension of the mesh).
    name:
        An optional name for the function space.
    vfamily:
        The finite element in the vertical dimension (extruded meshes
        only).
    vdegree:
        The degree of the element in the vertical dimension (extruded
        meshes only).

    Notes
    -----
    The ``family`` argument may be an existing
    `finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate `.FunctionSpace` is returned.  In
    this case, the provided element must have an empty
    `finat.ufl.FiniteElementBase.value_shape`.

    The element that you provide need be a scalar element (with empty
    ``value_shape``), however, it should not be an existing
    `~finat.ufl.VectorElement`.  If you already have an existing
    `~finat.ufl.VectorElement`, you should pass it to `FunctionSpace`
    directly instead.

    """
    sub_element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    if dim is None:
        dim = mesh.ufl_cell().geometric_dimension()
    if not isinstance(dim, numbers.Integral) and dim > 0:
        raise ValueError(f"Can't make VectorFunctionSpace with dim={dim}")
    element = finat.ufl.VectorElement(sub_element, dim=dim)
    return FunctionSpace(mesh, element, name=name)


@PETSc.Log.EventDecorator()
def TensorFunctionSpace(mesh, family, degree=None, shape=None,
                        symmetry=None, name=None, vfamily=None,
                        vdegree=None):
    """Create a rank-2 FunctionSpace.

    Parameters
    ----------
    mesh:
        The mesh to determine the cell from.
    family:
        The finite element family.
    degree:
        The degree of the finite element.
    shape:
        An optional shape for the tensor-valued degrees of freedom at
        each function space node (defaults to a square tensor using the
        geometric dimension of the mesh).
    symmetry:
        Optional symmetries in the tensor value.
    name:
        An optional name for the function space.
    vfamily:
        The finite element in the vertical dimension (extruded meshes
        only).
    vdegree:
        The degree of the element in the vertical dimension (extruded
        meshes only).

    Notes
    -----
    The ``family`` argument may be an existing
    `~finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate `FunctionSpace` is
    returned.  In this case, the provided element must have an empty
    `finat.ufl.FiniteElementBase.value_shape`.

    The element that you provide must be a scalar element (with empty
    ``value_shape``).  If you already have an existing
    `~finat.ufl.TensorElement`, you should pass it to `FunctionSpace`
    directly instead.

    """
    sub_element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    shape = shape or (mesh.ufl_cell().geometric_dimension(),) * 2
    element = finat.ufl.TensorElement(sub_element, shape=shape, symmetry=symmetry)
    return FunctionSpace(mesh, element, name=name)


@PETSc.Log.EventDecorator()
def MixedFunctionSpace(spaces, name=None, mesh=None):
    """Create a MixedFunctionSpace.

    Parameters
    ----------

    spaces:
        An iterable of constituent spaces, or a
        finat.ufl.mixedelement.MixedElement.
    name:
        An optional name for the mixed function space.
    mesh:
        An optional mesh.  Must be provided if spaces is a
        finat.ufl.mixedelement.MixedElement, ignored otherwise.

    """
    if isinstance(spaces, finat.ufl.FiniteElementBase):
        # Build the spaces if we got a mixed element
        assert type(spaces) is finat.ufl.MixedElement and mesh is not None
        sub_elements = []

        def rec(eles):
            for ele in eles:
                # Only want to recurse into MixedElements
                if type(ele) is finat.ufl.MixedElement:
                    rec(ele.sub_elements)
                else:
                    sub_elements.append(ele)
        rec(spaces.sub_elements)
        spaces = [FunctionSpace(mesh, element) for element in sub_elements]

    # Check that function spaces are on the same mesh
    meshes = [space.mesh() for space in spaces]
    for i in range(1, len(meshes)):
        if meshes[i] is not meshes[0]:
            raise ValueError("All function spaces must be defined on the same mesh!")

    # Select mesh
    mesh = meshes[0]
    # Get topological spaces
    spaces = tuple(s.topological for s in flatten(spaces))
    # Error checking
    for space in spaces:
        if type(space) in (impl.FunctionSpace, impl.RealFunctionSpace):
            continue
        elif type(space) is impl.ProxyFunctionSpace:
            if space.component is not None:
                raise ValueError("Can't make mixed space with %s" % space)
            continue
        else:
            raise ValueError("Can't make mixed space with %s" % type(space))

    new = impl.MixedFunctionSpace(spaces, name=name)
    if mesh is not mesh.topology:
        return impl.WithGeometry.create(new, mesh)
    return new
