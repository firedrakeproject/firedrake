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


@PETSc.Log.EventDecorator("CreateFunctionSpace")
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
    :class:`finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is returned.
    """
    element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    return impl.WithGeometry.make_function_space(mesh, element, name=name)


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
    return impl.FiredrakeDualSpace.make_function_space(mesh, element, name=name)


@PETSc.Log.EventDecorator()
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
    :class:`finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is
    returned.  In this case, the provided element must have an empty
    :meth:`finat.ufl.FiniteElementBase.value_shape`.

    .. note::

       The element that you provide need be a scalar element (with
       empty ``value_shape``), however, it should not be an existing
       :class:`~finat.ufl.VectorElement`.  If you already have an
       existing :class:`~finat.ufl.VectorElement`, you should pass
       it to :func:`FunctionSpace` directly instead.

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
    :class:`~finat.ufl.FiniteElementBase`, in which case all other arguments
    are ignored and the appropriate :class:`.FunctionSpace` is
    returned.  In this case, the provided element must have an empty
    :meth:`~finat.ufl.FiniteElementBase.value_shape`.

    .. note::

       The element that you provide must be a scalar element (with
       empty ``value_shape``).  If you already have an existing
       :class:`~finat.ufl.TensorElement`, you should pass it to
       :func:`FunctionSpace` directly instead.
    """
    sub_element = make_scalar_element(mesh, family, degree, vfamily, vdegree)
    shape = shape or (mesh.ufl_cell().geometric_dimension(),) * 2
    element = finat.ufl.TensorElement(sub_element, shape=shape, symmetry=symmetry)
    return FunctionSpace(mesh, element, name=name)


@PETSc.Log.EventDecorator()
def MixedFunctionSpace(spaces, name=None, mesh=None):
    """Create a :class:`.MixedFunctionSpace`.

    :arg spaces: An iterable of constituent spaces, or a
        :class:`~finat.ufl.MixedElement`.
    :arg name: An optional name for the mixed function space.
    :arg mesh: An optional mesh.  Must be provided if spaces is a
        :class:`~finat.ufl.MixedElement`, ignored otherwise.
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

    # Check that function spaces are all primal or all dual
    try:
        cls, = set(type(s) for s in spaces)
    except ValueError:
        raise ValueError("All function spaces must be either primal or dual!")

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
        return cls.create(new, mesh)
    return new
