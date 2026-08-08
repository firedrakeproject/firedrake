import functools
import operator

import numpy as np
from functools import cached_property

from pyadjoint.tape import annotate_tape
from pyop2 import op2
import pytools
import finat.ufl
from ufl.algorithms import extract_coefficients
from ufl.cell import TensorProductCell
from ufl.constantvalue import as_ufl
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain

from firedrake.cofunction import Cofunction
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.halo import _get_mtype
from firedrake.petsc import PETSc
from firedrake.utils import IntType, ScalarType, split_by

from mpi4py import MPI


def _submesh_point_sf(target_mesh, source_mesh):
    """Find the point SF relating a submesh to its parent.

    Parameters
    ----------
    target_mesh : AbstractMeshTopology
        The mesh being assigned to.
    source_mesh : AbstractMeshTopology
        The mesh being assigned from.

    Returns
    -------
    tuple
        The `PETSc.SF` mapping the points of the parent mesh (roots) to the
        points of the submesh (leaves), and whether ``target_mesh`` is the
        submesh. Both are `None` if neither mesh is a submesh of the other.
        When the two share their distribution, the SF is the local subpoint
        map rather than one of the meshes' own ``submesh_point_sf``.

    """
    from firedrake.mesh import _make_submesh_point_sf

    if target_mesh.submesh_parent is source_mesh:
        submesh, target_is_submesh = target_mesh, True
    elif source_mesh.submesh_parent is target_mesh:
        submesh, target_is_submesh = source_mesh, False
    else:
        return None, None
    point_sf = submesh.submesh_point_sf
    if point_sf is None:
        # The two share their distribution, so their points are related
        # locally by the subpoint IS.
        point_sf = _make_submesh_point_sf(submesh.submesh_parent.topology_dm,
                                          submesh.topology_dm)
    return point_sf, target_is_submesh


def _make_section_sf(point_sf, root_V, leaf_V):
    """Expand a point SF into an SF relating the nodes of two function spaces.

    Parameters
    ----------
    point_sf : PETSc.SF
        SF mapping the points of the mesh of ``root_V`` (roots) to the points
        of the mesh of ``leaf_V`` (leaves).
    root_V : firedrake.functionspaceimpl.WithGeometry
        Function space holding the root data.
    leaf_V : firedrake.functionspaceimpl.WithGeometry
        Function space holding the leaf data.

    Returns
    -------
    tuple
        The `PETSc.SF` mapping the nodes of ``root_V`` to the nodes of
        ``leaf_V``, and the boolean array telling which nodes of ``root_V``
        have a counterpart in ``leaf_V``.

    """
    cache = leaf_V.mesh().topology._shared_data_cache["submesh_section_sf"]
    key = (root_V, leaf_V)
    try:
        return cache[key]
    except KeyError:
        root_section = root_V.dm.getSection()
        leaf_section = leaf_V.dm.getSection()
        # `distributeSection` overwrites the section it is handed, so let it
        # build its own and only keep the root offsets it broadcasts.
        remote_offsets, distributed_section = point_sf.distributeSection(root_section)
        if distributed_section.getChart() != leaf_section.getChart():
            raise RuntimeError("Point SF does not cover the nodes of the leaf function space")
        section_sf = point_sf.createSectionSF(root_section, remote_offsets, leaf_section)
        # A submesh only covers part of its parent, so not every root node
        # is reduced into.
        return cache.setdefault(key, (section_sf, section_sf.computeDegree() > 0))


def _identity_point_sf(mesh):
    """Create the `PETSc.SF` mapping the points of a mesh onto themselves.

    Two function spaces on the same mesh are related by their `PETSc.Section`
    alone, which the section SF machinery expresses as the identity on points.

    Parameters
    ----------
    mesh : firedrake.mesh.AbstractMeshTopology
        The mesh.

    Returns
    -------
    PETSc.SF
        SF whose roots and leaves are both the points of ``mesh``.

    """
    plex = mesh.topology_dm
    pStart, pEnd = plex.getChart()
    remote = np.empty((pEnd - pStart, 2), dtype=IntType)
    remote[:, 0] = plex.comm.rank
    remote[:, 1] = np.arange(pEnd - pStart, dtype=IntType)
    point_sf = PETSc.SF().create(comm=plex.comm)
    point_sf.setGraph(pEnd - pStart, None, remote)
    return point_sf


def _entity_dof_counts(element):
    """Count the nodes an element places on each entity of the reference cell.

    Parameters
    ----------
    element : finat.ufl.finiteelementbase.FiniteElementBase
        The UFL element.

    Returns
    -------
    dict
        The number of nodes on each ``(dimension, entity)`` of the reference cell.

    """
    from firedrake.functionspacedata import create_element

    entity_dofs = create_element(element).entity_dofs()
    return {(dim, entity): len(dofs)
            for dim, entities in entity_dofs.items()
            for entity, dofs in entities.items()}


def _unrestricted(element):
    """Strip the topological restrictions off an element.

    Parameters
    ----------
    element : finat.ufl.finiteelementbase.FiniteElementBase
        The UFL element.

    Returns
    -------
    finat.ufl.finiteelementbase.FiniteElementBase
        The element whose nodes ``element`` selects from.

    """
    while isinstance(element, finat.ufl.RestrictedElement):
        element = element.sub_element()
    return element


def _same_element_on_either_cell(target, source):
    """Whether two elements differ only in the cell they are defined on.

    Parameters
    ----------
    target : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned to.
    source : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned from.

    Returns
    -------
    bool
        Whether one element is the other, carried onto a different cell.

    """
    if target.cell == source.cell:
        return target == source
    # A family that the lower cell does not name, such as "Q" on an interval,
    # raises rather than compares. The reconstruction must therefore go up,
    # onto the cell of higher dimension.
    low, high = sorted((target, source), key=lambda e: e.cell.topological_dimension)
    try:
        return low.reconstruct(cell=high.cell) == high
    except (ValueError, KeyError):
        return False


def _dimension_dof_counts(element):
    """Count the nodes an element places on each dimension of the reference cell.

    Parameters
    ----------
    element : finat.ufl.finiteelementbase.FiniteElementBase
        The UFL element.

    Returns
    -------
    dict or None
        The number of nodes on an entity of each dimension. `None` if some
        dimension holds entities that carry different numbers of nodes, which
        leaves the dimension alone unable to say how many a point carries.

    """
    counts = {}
    for (dim, _), nodes in _entity_dof_counts(element).items():
        if counts.setdefault(dim, nodes) != nodes:
            return None
    return counts


def _compatible_elements(target, source):
    """Whether functions in two elements share a node layout.

    Two elements are compatible under two conditions. They must restrict a
    common element. They must also place the same number of nodes on every
    entity of the reference cell, up to the entities one of them drops.

    Their nodes are then the same functionals, entity by entity. The
    `PETSc.Section` of either function space therefore describes both, and
    data moves between them without reference to the cell node maps.

    Parameters
    ----------
    target : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned to.
    source : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned from.

    Returns
    -------
    bool
        Whether the two elements share a node layout.

    """
    if target == source:
        return True
    blocked_types = (finat.ufl.VectorElement, finat.ufl.TensorElement)
    if isinstance(target, blocked_types) or isinstance(source, blocked_types):
        return (type(target) is type(source)
                and target.num_sub_elements == source.num_sub_elements
                and target.reference_value_shape == source.reference_value_shape
                and _compatible_elements(target.sub_elements[0], source.sub_elements[0]))
    # Equal node counts on an entity do not by themselves make two nodes the
    # same functional, so the elements must restrict a common element.
    if not _same_element_on_either_cell(_unrestricted(target), _unrestricted(source)):
        return False
    if target.cell != source.cell:
        # The two cells have different entities, so the entity numbers can not
        # be compared. A submesh of lower dimension holds the entities of its
        # parent up to its own dimension. The point SF pairs them off by
        # dimension. The counts must therefore agree dimension by dimension.
        target_counts = _dimension_dof_counts(target)
        source_counts = _dimension_dof_counts(source)
        if target_counts is None or source_counts is None:
            return False
        shared_dimensions = min(len(target_counts), len(source_counts))
        return all(target_counts[dim] == source_counts[dim]
                   for dim in range(shared_dimensions))
    if isinstance(target.cell, TensorProductCell):
        # A base mesh point of an extruded mesh carries the nodes of a whole
        # column of entities. A restriction may drop only some of them. The
        # Section counts the nodes on a point without saying which.
        raise NotImplementedError(
            "Assigning between an element and its restriction is not "
            "implemented on extruded meshes"
        )
    target_counts = _entity_dof_counts(target)
    source_counts = _entity_dof_counts(source)
    if target_counts.keys() != source_counts.keys():
        return False
    # An entity either carries the same nodes in both elements, or is dropped
    # by one of them. A partial overlap has no entity-wise correspondence.
    # The two Sections therefore cannot express it.
    return all(target_nodes == source_counts[entity]
               or target_nodes == 0 or source_counts[entity] == 0
               for entity, target_nodes in target_counts.items())


def _node_subset(V, cell_subset):
    """Find the nodes of the cells of a subset.

    A node on the boundary of the subset belongs to cells outside it too, and
    is included. The subset selects the cells whose nodes are assigned. It
    does not select the nodes that no other cell shares.

    Parameters
    ----------
    V : firedrake.functionspaceimpl.WithGeometry
        Function space whose nodes are selected.
    cell_subset : pyop2.types.set.Subset
        Subset of the cells of the mesh of ``V``.

    Returns
    -------
    pyop2.types.set.Subset
        The nodes of ``V`` on the cells of ``cell_subset``.

    """
    if V.extruded:
        raise NotImplementedError(
            "Assigning over a subset of the cells is not implemented on "
            "extruded meshes"
        )
    node_map = V.cell_node_map()
    if node_map is None:
        raise ValueError(f"Function space ({V}) has no nodes on the cells")
    # A node is on the subset for every rank that shares it, as soon as it is
    # on the subset for one of them. The cells known to a single rank do not
    # say this. A rank owning the node need not own, or even halo, a cell of
    # the subset that carries it.
    marker = op2.Dat(V.node_set, dtype=IntType)
    marker.data_wo_with_halos[np.unique(node_map.values_with_halo[cell_subset.indices])] = 1
    marker.local_to_global_begin(op2.MAX)
    marker.local_to_global_end(op2.MAX)
    marker.global_to_local_begin(op2.READ)
    marker.global_to_local_end(op2.READ)
    nodes, = np.nonzero(marker.data_ro_with_halos)
    return op2.Subset(V.node_set, nodes)


def _assigned_nodes(V, subset):
    """Find the nodes of a space that an assignment writes to.

    Parameters
    ----------
    V : firedrake.functionspaceimpl.WithGeometry
        Function space of the function being assigned to.
    subset : pyop2.types.set.Set or pyop2.types.set.Subset or None
        The set to assign over. This is the node set of ``V``, the cell set of
        the mesh of ``V``, a `pyop2.types.set.Subset` of either, or `None`.

    Returns
    -------
    pyop2.types.set.Subset or None
        The nodes of ``V`` to assign, or `None` for all of them.

    Raises
    ------
    ValueError
        If the subset belongs to neither the nodes of ``V`` nor the cells of
        the mesh of ``V``.

    """
    all_nodes = V.node_set
    all_cells = V.mesh().cell_set
    if subset is None or subset is all_nodes or subset is all_cells:
        return None
    superset = getattr(subset, "superset", None)
    if superset is all_nodes:
        return subset
    if superset is all_cells:
        return _node_subset(V, subset)
    raise ValueError(f"subset ({subset}) is neither a subset of the nodes of "
                     f"the function space ({all_nodes}) nor of the cells of "
                     f"its mesh ({all_cells})")


def _target_is_leaf(target, source):
    """Whether the target element's nodes are the subset of the two.

    Data travels root to leaf by broadcast, which requires every leaf node to
    have a counterpart, and leaf to root by reduction, which does not. The
    target can therefore be the leaf, unless it carries nodes on an entity
    that the source drops. Take it to be the leaf whenever possible, which
    keeps the halo of the assignee up to date.

    Parameters
    ----------
    target : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned to.
    source : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned from.

    Returns
    -------
    bool
        Whether every node of ``target`` has a counterpart in ``source``.

    """
    target_counts = _entity_dof_counts(target)
    source_counts = _entity_dof_counts(source)
    return not any(source_counts[e] == 0 < target_counts[e] for e in target_counts)


def _relate_to_target(target_mesh, target_element, source_V):
    """Find the section SF relating a source function space to the assignee.

    Parameters
    ----------
    target_mesh : AbstractMeshTopology
        Mesh of the function being assigned to.
    target_element : finat.ufl.finiteelementbase.FiniteElementBase
        Element of the function being assigned to.
    source_V : firedrake.functionspaceimpl.WithGeometry
        Function space of the function being assigned from.

    Returns
    -------
    tuple
        The `PETSc.SF` relating the points of ``target_mesh`` to those of
        ``source_V``'s mesh, and whether the assignee is the leaf.

    """
    source_mesh = source_V.mesh().topology
    if target_mesh is source_mesh:
        return _identity_point_sf(target_mesh), _target_is_leaf(target_element, source_V.ufl_element())
    point_sf, target_is_leaf = _submesh_point_sf(target_mesh, source_mesh)
    if point_sf is None:
        raise NotImplementedError(
            "Can only assign between a redistributed mesh and its parent"
        )
    return point_sf, target_is_leaf


def _isconstant(expr):
    return isinstance(expr, Constant) or \
        (isinstance(expr, (Function, Cofunction)) and expr.ufl_element().family() == "Real")


def _isfunction(expr):
    return isinstance(expr, (Function, Cofunction)) and expr.ufl_element().family() != "Real"


class CoefficientCollector(MultiFunction):
    """Multifunction used for converting an expression into a weighted sum of coefficients.

    Calling ``map_expr_dag(CoefficientCollector(), expr)`` will return a tuple whose entries
    are of the form ``(coefficient, weight)``. Expressions that cannot be expressed as a
    weighted sum will raise an exception.

    Note: As well as being simple weighted sums (e.g. ``u.assign(2*v1 + 3*v2)``), one can
    also assign constant expressions of the appropriate shape (e.g. ``u.assign(1.0)`` or
    ``u.assign(2*v + 3)``). Therefore the returned tuple must be split since ``coefficient``
    may be either a :class:`firedrake.constant.Constant` or :class:`firedrake.function.Function`.
    """

    def product(self, o, a, b):
        scalars, vectors = split_by(self._is_scalar_equiv, [a, b])
        # Case 1: scalar * scalar
        if len(scalars) == 2:
            # Compress the first argument (arbitrary)
            scalar, vector = scalars
        # Case 2: scalar * vector
        elif len(scalars) == 1:
            scalar, = scalars
            vector, = vectors
        # Case 3: vector * vector (invalid)
        else:
            raise ValueError("Expressions containing the product of two vector-valued "
                             "subexpressions cannot be used for assignment. Consider using "
                             "interpolate instead.")
        scaling = self._as_scalar(scalar)
        return tuple((coeff, weight*scaling) for coeff, weight in vector)

    def division(self, o, a, b):
        # Division is only valid if b (the divisor) is a scalar
        if self._is_scalar_equiv(b):
            divisor = self._as_scalar(b)
            return tuple((coeff, weight/divisor) for coeff, weight in a)
        else:
            raise ValueError("Expressions involving division by a vector-valued subexpression "
                             "cannot be used for assignment. Consider using interpolate instead.")

    def sum(self, o, a, b):
        # Note: a and b are tuples of (coefficient, weight) so addition is concatenation
        return a + b

    def power(self, o, a, b):
        # Only valid if a and b are scalars
        return ((Constant(self._as_scalar(a) ** self._as_scalar(b)), 1),)

    def abs(self, o, a):
        # Only valid if a is a scalar
        return ((Constant(abs(self._as_scalar(a))), 1),)

    def _scalar(self, o):
        return ((Constant(o), 1),)

    int_value = _scalar
    float_value = _scalar
    complex_value = _scalar
    zero = _scalar

    def multi_index(self, o):
        pass

    def indexed(self, o, a, _):
        return a

    def component_tensor(self, o, a, _):
        return a

    def coefficient(self, o):
        return ((o, 1),)

    def cofunction(self, o):
        return ((o, 1),)

    def constant_value(self, o):
        return ((o, 1),)

    def expr(self, o, *operands):
        raise NotImplementedError(f"Handler not defined for {type(o)}")

    def _is_scalar_equiv(self, weighted_coefficients):
        """Return ``True`` if the sequence of ``(coefficient, weight)`` can be compressed to
        a single scalar value.

        This is only true when all coefficients are :class:`firedrake.Constant` or
        are :class:`firedrake.Function` and ``c.ufl_element().family() == "Real"``
        in both cases ``c.dat.dim`` must have shape ``(1,)``.
        """
        return all(_isconstant(c) and c.dat.dim == (1,) for (c, _) in weighted_coefficients)

    def _as_scalar(self, weighted_coefficients):
        """Compress a sequence of ``(coefficient, weight)`` tuples to a single scalar value.

        This is necessary because we do not know a priori whether a :class:`firedrake.Constant`
        is going to be used as a scale factor (e.g. ``u.assign(Constant(2)*v)``), or as a
        constant to be added (e.g. ``u.assign(2*v + Constant(3))``). Therefore we only
        compress to a scalar when we know it is required (e.g. inside a product with a
        :class:`~.firedrake.function.Function`).
        """
        return pytools.one(
            functools.reduce(operator.add, (c.dat.data_ro*w for c, w in weighted_coefficients))
        )


class Assigner:
    """Class performing pointwise assignment of an expression to a function or a cofunction.

    Parameters
    ----------
    assignee : firedrake.function.Function or firedrake.cofunction.Cofunction
        Function or Cofunction being assigned to.
    expression : ufl.core.expr.Expr or ufl.form.BaseForm
        Expression to be assigned.
    subset : pyop2.types.set.Set or pyop2.types.set.Subset or pyop2.types.set.MixedSet
        Subset to apply the assignment over.

    """
    symbol = "="

    _coefficient_collector = CoefficientCollector()

    def __init__(self, assignee, expression, subset=None):
        expression = as_ufl(expression)
        source_meshes = set()
        for coeff in extract_coefficients(expression):
            if isinstance(coeff, (Function, Cofunction)) and coeff.ufl_element().family() != "Real":
                if not _compatible_elements(assignee.ufl_element(), coeff.ufl_element()):
                    raise ValueError("All functions in the expression must have an "
                                     "element compatible with that of the assignee")
                source_meshes.add(extract_unique_domain(coeff, expand_mesh_sequence=False))
        if len(source_meshes) == 0:
            pass
        elif len(source_meshes) == 1:
            target_mesh = extract_unique_domain(assignee, expand_mesh_sequence=False)
            source_mesh, = source_meshes
            if target_mesh == source_mesh:
                pass
            elif target_mesh.submesh_youngest_common_ancestor(source_mesh) is None:
                raise ValueError(
                    "All functions in the expression must be defined on a single domain "
                    "that is in the same submesh family as domain of the assignee"
                )
        else:
            raise ValueError(
                "All functions in the expression must be defined on a single domain"
            )
        if subset is None:
            subset = tuple(None for _ in assignee.function_space())
        if len(subset) != len(assignee.function_space()):
            raise ValueError(f"Provided subset ({subset}) incompatible with assignee ({assignee})")
        if type(assignee.ufl_element()) == finat.ufl.MixedElement:
            for subs, el in zip(subset, assignee.function_space().ufl_element().sub_elements):
                if subs is not None and el.family() == "Real":
                    raise ValueError(
                        "Subset is not a valid argument for assigning to a mixed "
                        "element including a real element"
                    )
        self._assignee = assignee
        self._expression = expression
        self._subset = subset

    def __str__(self):
        return f"{self._assignee} {self.symbol} {self._expression}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self._assignee!r}, {self._expression!r})"

    @PETSc.Log.EventDecorator()
    def assign(self, allow_missing_dofs=False):
        """Perform the assignment.

        Parameters
        ----------
        allow_missing_dofs : bool
            Permit assignment between objects with mismatching nodes. If `True` then
            assignee nodes with no matching assigner nodes are ignored.

        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit Assigner objects is not supported yet. "
                "Use Function.assign instead."
            )
        # To minimize communication during assignment we perform a number of tricks:
        # * If we are not assigning to a subset then we can always write to the
        #   halo. The validity of the original assignee dat halo does not matter
        #   since we are overwriting it entirely.
        # * We can also write to the halo if we are assigning to a subset provided
        #   that the assignee halo is not dirty to start with.
        # * If we are assigning to a subset where the assignee dat has a dirty halo,
        #   then we should only write to the owned values. There is no point in
        #   writing to the halo since a full halo exchange is still required.
        # * If any of the functions in the expression do not have valid halos then
        #   we only write to the owned values in the assignee. Otherwise we might
        #   end up doing a lot of halo exchanges for the expression just to avoid
        #   a single halo exchange for the assignee.
        # * If we do write to the halo then the resulting halo will never be dirty.
        # If mixed, loop over individual components
        for lhs_func, subset, *funcs in zip(self._assignee.subfunctions, self._subset, *(f.subfunctions for f in self._functions)):
            target_mesh = extract_unique_domain(lhs_func)
            target_V = lhs_func.function_space()
            subset = _assigned_nodes(target_V, subset)
            source_meshes = set(extract_unique_domain(f) for f in funcs)
            if len(source_meshes) == 0:
                # Assign constants only.
                single_mesh_assign = True
            elif len(source_meshes) == 1:
                source_mesh, = source_meshes
                if target_mesh is source_mesh:
                    # Two distinct spaces on one mesh lay their nodes out
                    # differently, even when they share an element. Their
                    # Sections relate them, as they relate spaces on different
                    # meshes.
                    single_mesh_assign = all(f.function_space() == lhs_func.function_space()
                                             for f in funcs)
                else:
                    # Assign (co)functions between a submesh and the parent or between two submeshes.
                    single_mesh_assign = False
            else:
                raise ValueError("All functions in the expression must be defined on a single domain")
            if single_mesh_assign:
                self._assign_single_mesh(lhs_func, subset, funcs, operator)
            else:
                self._assign_multi_mesh(lhs_func, subset, funcs, operator, allow_missing_dofs)

    def _assign_single_mesh(self, lhs_func, subset, funcs, operator):
        assign_to_halos = all(f.dat.halo_valid for f in funcs) and (lhs_func.dat.halo_valid or subset is None)
        if assign_to_halos:
            indices = operator.attrgetter("indices")
            data_ro = operator.attrgetter("data_ro_with_halos")
        else:
            indices = operator.attrgetter("owned_indices")
            data_ro = operator.attrgetter("data_ro")
        subset_indices = Ellipsis if subset is None else indices(subset)

        func_data = np.array([data_ro(f.dat)[subset_indices] for f in funcs])
        rvalue = self._compute_rvalue(func_data)
        self._assign_single_dat(lhs_func.dat, subset_indices, rvalue, assign_to_halos)
        if assign_to_halos:
            lhs_func.dat.halo_valid = True

    def _assign_multi_mesh(self, lhs_func, subset, funcs, operator, allow_missing_dofs):
        target_mesh = extract_unique_domain(lhs_func).topology
        source_spaces = set(f.function_space() for f in funcs)
        if len(source_spaces) > 1:
            # Every function is compatible with the assignee, which is checked
            # at construction time, but not necessarily with one another. Each
            # is therefore related to the assignee by its own section SF, and
            # mapped into its layout before they are combined.
            self._assign_multi_space(lhs_func, subset, funcs, allow_missing_dofs)
            return
        source_V, = source_spaces
        source_mesh = source_V.mesh().topology
        # Spaces on meshes that share their distribution are related by their
        # entity maps. Elements that lay the nodes out differently are the
        # exception: there, only the Sections relate them.
        same_element = source_V.ufl_element() == lhs_func.ufl_element()
        if target_mesh is not source_mesh and same_element and target_mesh.submesh_shares_distribution(source_mesh):
            self._assign_submesh(lhs_func, subset, funcs, operator, allow_missing_dofs)
            return
        point_sf, target_is_leaf = _relate_to_target(target_mesh, lhs_func.ufl_element(), source_V)
        self._assign_via_sections(lhs_func, subset, funcs, point_sf,
                                  target_is_leaf, allow_missing_dofs)

    def _assign_via_sections(self, lhs_func, subset, funcs, point_sf,
                             target_is_leaf, allow_missing_dofs):
        """Assign between (co)functions whose nodes are related by a section SF.

        The nodes the two spaces have in common correspond one to one. The
        expression is evaluated in the source layout. One communication then
        moves the result into the target layout.

        ``target_is_leaf`` says which of the two carries the subset of the
        nodes. Data travels from root to leaf by broadcast, and from leaf to
        root by reduction. Only a reduction can miss nodes.
        """
        target_V = lhs_func.function_space()
        source_V, = set(f.function_space() for f in funcs)
        func_data = np.array([f.dat.data_ro_with_halos for f in funcs])
        source_data = self._compute_rvalue(func_data)
        target_data, covered, assign_to_halos = self._transfer_via_section(
            lhs_func, source_V, source_data, point_sf, target_is_leaf)
        indices = self._covered_indices(target_V, subset, covered, assign_to_halos, allow_missing_dofs)
        self._assign_single_dat(lhs_func.dat, indices, target_data[indices], assign_to_halos)
        lhs_func.dat.halo_valid = assign_to_halos

    def _assign_multi_space(self, lhs_func, subset, funcs, allow_missing_dofs):
        """Assign an expression combining functions from more than one
        function space, each compatible with the assignee's element but not
        necessarily with one another's.

        Every function is moved, unweighted, into the assignee's node layout
        by its own section SF. The weighted combination then happens in that
        common layout, exactly as it would on a single mesh.
        """
        target_mesh = extract_unique_domain(lhs_func).topology
        target_V = lhs_func.function_space()
        rows = []
        covered = None
        assign_to_halos = True
        for f in funcs:
            source_V = f.function_space()
            point_sf, target_is_leaf = _relate_to_target(target_mesh, lhs_func.ufl_element(), source_V)
            data, cov, halo_ok = self._transfer_via_section(
                lhs_func, source_V, f.dat.data_ro_with_halos, point_sf, target_is_leaf)
            rows.append(data)
            covered = cov if covered is None else (covered | cov)
            assign_to_halos = assign_to_halos and halo_ok
        target_data = self._compute_rvalue(np.array(rows))
        indices = self._covered_indices(target_V, subset, covered, assign_to_halos, allow_missing_dofs)
        self._assign_single_dat(lhs_func.dat, indices, target_data[indices], assign_to_halos)
        lhs_func.dat.halo_valid = assign_to_halos

    def _transfer_via_section(self, lhs_func, source_V, source_data, point_sf, target_is_leaf):
        """Move data from a source layout into the assignee's, via a section SF.

        Parameters
        ----------
        lhs_func : firedrake.function.Function or firedrake.cofunction.Cofunction
            The function being assigned to; only its type and function space
            are used.
        source_V : firedrake.functionspaceimpl.WithGeometry
            Function space ``source_data`` is laid out in.
        source_data : numpy.ndarray
            Data in ``source_V``'s with-halo layout, already combined if it
            comes from more than one function.
        point_sf : PETSc.SF
            SF relating the assignee's mesh to ``source_V``'s, as returned by
            `_relate_to_target`.
        target_is_leaf : bool
            Whether the assignee carries the subset of the nodes.

        Returns
        -------
        tuple
            The data moved into the assignee's with-halo layout, the boolean
            array of which of those nodes received data, and whether the
            halo of the result can be trusted.

        """
        target_V = lhs_func.function_space()
        if target_is_leaf:
            root_V, leaf_V = source_V, target_V
        else:
            root_V, leaf_V = target_V, source_V
        section_sf, covered_roots = _make_section_sf(point_sf, root_V, leaf_V)

        source_buffer = Function(source_V)
        target_buffer = Function(target_V)
        source_buffer.dat.data_wo_with_halos[...] = source_data
        mtype, _ = _get_mtype(source_buffer.dat)
        source_buffer_data = source_buffer.dat.data_ro_with_halos
        target_buffer_data = target_buffer.dat.data_wo_with_halos
        if target_is_leaf:
            section_sf.bcastBegin(mtype, source_buffer_data, target_buffer_data, MPI.REPLACE)
            section_sf.bcastEnd(mtype, source_buffer_data, target_buffer_data, MPI.REPLACE)
            # Every node of a submesh, including its halo, has a counterpart
            # in the parent.
            covered = np.ones(target_buffer_data.shape[0], dtype=bool)
            halos_valid = True
        else:
            section_sf.reduceBegin(mtype, source_buffer_data, target_buffer_data, MPI.REPLACE)
            section_sf.reduceEnd(mtype, source_buffer_data, target_buffer_data, MPI.REPLACE)
            # Only the owned parent nodes that the source covers have been
            # reduced into; the parent halo never is.
            covered = np.zeros(target_buffer_data.shape[0], dtype=bool)
            covered[:target_V.dof_dset.size] = covered_roots[:target_V.dof_dset.size]
            halos_valid = False
        return target_buffer.dat.data_ro_with_halos, covered, halos_valid

    def _covered_indices(self, target_V, subset, covered, assign_to_halos, allow_missing_dofs):
        """Find the indices of the assignee's nodes that received data.

        Parameters
        ----------
        target_V : firedrake.functionspaceimpl.WithGeometry
            Function space of the function being assigned to.
        subset : pyop2.types.set.Subset or None
            Subset of the assignee's node set to restrict the assignment to.
        covered : numpy.ndarray
            Boolean array, in the assignee's with-halo layout, of which
            nodes received data.
        assign_to_halos : bool
            Whether ``covered`` (and the halo of the assignee) can be trusted.
        allow_missing_dofs : bool
            Permit assignee nodes with no matching data, subject to
            ``subset``, rather than raising.

        Returns
        -------
        numpy.ndarray or Ellipsis
            The indices, in the assignee's with-halo layout, to assign to.

        """
        if assign_to_halos:
            return Ellipsis if subset is None else subset.indices
        owned = covered[:target_V.dof_dset.size]
        comm = target_V.mesh().comm
        if not comm.allreduce(bool(owned.all()), op=MPI.LAND) and not allow_missing_dofs:
            raise ValueError("Found assignee nodes with no matching assigner "
                             "nodes: run with `allow_missing_dofs=True`")
        indices, = np.nonzero(owned)
        if subset is not None:
            indices = np.intersect1d(indices, subset.owned_indices)
        return indices

    def _assign_submesh(self, lhs_func, subset, funcs, operator, allow_missing_dofs):
        target_mesh = extract_unique_domain(lhs_func)
        target_V = lhs_func.function_space()
        source_V, = set(f.function_space() for f in funcs)
        composed_map = source_V.topological.entity_node_map(target_mesh.topology, "cell", "everywhere", None)
        indices_active = composed_map.indices_active_with_halo
        indices_active_all = indices_active.all()
        indices_active_all = target_mesh.comm.allreduce(indices_active_all, op=MPI.LAND)
        if subset is None:
            if not indices_active_all and not allow_missing_dofs:
                raise ValueError("Found assignee nodes with no matching assigner nodes: run with `allow_missing_dofs=True`")
            subset_indices_target = target_V.cell_node_map().values_with_halo[indices_active, :].flatten()
            subset_indices_source = composed_map.values_with_halo[indices_active, :].flatten()
        else:
            subset_indices_target, perm, _ = np.intersect1d(
                target_V.cell_node_map().values_with_halo[indices_active, :].flatten(),
                subset.indices,
                return_indices=True,
            )
            if len(subset.indices) > len(subset_indices_target) and not allow_missing_dofs:
                raise ValueError("Found assignee nodes with no matching assigner nodes: run with `allow_missing_dofs=True`")
            subset_indices_source = composed_map.values_with_halo[indices_active, :].flatten()[perm]
        # Use buffer array to make sure that owned DoFs are updated upon assigning.
        # The following example illustrates the issue that a naive assignment would cause.
        #
        # Consider the following target/source meshes distributed over 2 processes
        # with no partition overlap:
        #
        #                0----0----0----1----1
        #                |         |         |
        # target         0    0    0    1    1
        # (parent mesh)  |         |         |
        #                0----0----0----1----1  (owning ranks are shown)
        #
        #                          1----1----1
        #                          |         |
        # source                   1    1    1
        # (submesh)                |         |
        #                          1----1----1  (owning ranks are shown)
        #
        # Consider CG1 functions f (on parent) and fsub (on submesh). By a naive
        # f.assign(fsub, subset=...), the DoFs shared by rank 0 and rank 1 would
        # only be updated on rank 1, which sees those DoFs as ghost, and those
        # updated values on rank 1 would be overridden by the old values on rank 0
        # upon a halo exchange.
        #
        # TODO: Use work array for buffer?
        buffer = type(lhs_func)(target_V)
        finfo = np.finfo(lhs_func.dat.dtype)
        buffer.dat._data[:] = finfo.max
        func_data = np.array([f.dat.data_ro_with_halos[subset_indices_source] for f in funcs])
        rvalue = self._compute_rvalue(func_data)
        self._assign_single_dat(buffer.dat, subset_indices_target, rvalue, True)
        # Make all owned DoFs up-to-date; ghost DoFs may or may not be up-to-date after this.
        buffer.dat.local_to_global_begin(op2.MIN)
        buffer.dat.local_to_global_end(op2.MIN)
        indices = np.where(buffer.dat.data_ro_with_halos < finfo.max * 0.999999999999)
        lhs_func.dat.data_wo_with_halos[indices] = buffer.dat.data_ro_with_halos[indices]

    @cached_property
    def _constants(self):
        return tuple(c for (c, _) in self._weighted_coefficients if _isconstant(c))

    @cached_property
    def _constant_weights(self):
        return tuple(w for (c, w) in self._weighted_coefficients if _isconstant(c))

    @cached_property
    def _functions(self):
        return tuple(c for (c, _) in self._weighted_coefficients if _isfunction(c))

    @cached_property
    def _function_weights(self):
        return tuple(w for (c, w) in self._weighted_coefficients if _isfunction(c))

    def _assign_single_dat(self, lhs_dat, indices, rvalue, assign_to_halos):
        if assign_to_halos:
            lhs_dat.data_wo_with_halos[indices] = rvalue
        else:
            lhs_dat.data_wo[indices] = rvalue

    def _compute_rvalue(self, func_data):
        # There are two components to the rvalue: weighted functions (in the same function space),
        # and constants (e.g. u.assign(2*v + 3)).
        func_rvalue = (func_data.T @ self._function_weights).T
        const_data = np.array([c.dat.data_ro for c in self._constants], dtype=ScalarType)
        const_rvalue = const_data.T @ self._constant_weights
        return func_rvalue + const_rvalue

    @cached_property
    def _weighted_coefficients(self):
        # TODO: It would be nice to stash this on the expression so we can avoid extra
        # traversals for non-persistent Assigner objects, but expressions do not currently
        # have caches attached to them.
        return map_expr_dag(self._coefficient_collector, self._expression)


class IAddAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__iadd__``."""
    symbol = "+="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if assign_to_halos:
            lhs.data_with_halos[indices] += rvalue
        else:
            lhs.data[indices] += rvalue


class ISubAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__isub__``."""
    symbol = "-="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if assign_to_halos:
            lhs.data_with_halos[indices] -= rvalue
        else:
            lhs.data[indices] -= rvalue


class IMulAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__imul__``."""
    symbol = "*="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if self._functions:
            raise ValueError("Only multiplication by scalars is supported")

        if assign_to_halos:
            lhs.data_with_halos[indices] *= rvalue
        else:
            lhs.data[indices] *= rvalue


class IDivAssigner(Assigner):
    """Assigner class for ``firedrake.function.Function.__itruediv__``."""
    symbol = "/="

    def _assign_single_dat(self, lhs, indices, rvalue, assign_to_halos):
        if self._functions:
            raise ValueError("Only division by scalars is supported")

        if assign_to_halos:
            lhs.data_with_halos[indices] /= rvalue
        else:
            lhs.data[indices] /= rvalue
