import numpy
from functools import partial, singledispatch
import os
import tempfile

import FIAT
import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.signature import compute_expression_signature
from ufl.domain import extract_unique_domain

from pyop2 import op2
from pyop2.caching import disk_cached

from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc import compile_expression_dual_evaluation
from tsfc.ufl_utils import extract_firedrake_constants

import gem
import finat

import firedrake
from firedrake import tsfc_interface, utils
from firedrake.adjoint_utils import annotate_interpolate
from firedrake.petsc import PETSc
from firedrake.halo import _get_mtype as get_dat_mpi_type
from mpi4py import MPI

from pyadjoint import stop_annotating

__all__ = ("interpolate", "Interpolator")


# Current behaviour of interpolation in Firedrake:
# - v.interpolate(expr),
#   interpolate(expr, v),
#   Interpolator(expr, v).interpolate(),
#   v = interpolate(expr, V) and
#   Interpolator(expr, V).interpolate(v)
#   - Works with UFL expressions which contain no UFL Arguments. The
#     expression can contain functions (UFL Coefficients) from other
#     function spaces which will be interpolated into V.
#   - Either operates on a function v in V (UFL Coefficient) or outputs a
#     function in V.
#   - Maths: v = A(expr) where A : W_0 x ... x W_n-1 -> V
#   - NOTE: this will seem to work on assembled 1-forms (cofunctions) but
#     is mathematical nonsense due to the absence of UFL Cofunctions in
#     Firedrake. See
#     https://github.com/firedrakeproject/firedrake/issues/3017
# - B = Interpolator(expr_1_argument, V)
#   - creates the linear interpolation operator B : W -> V where the UFL
#     Argument is linear in the expression and is in W. The UFL Argument must
#     be number 0 (i.e. TestFunction(W) rather than TrialFunction(W)).
#   - The rest of the expression, including any functions (UFL
#     Coefficients), are already interpolated into V and are encorporated
#     in the operator.
#   - NOTE: Nonlinear Arguments are currently allowed in the expression and
#     shouldn't be. See
#     https://github.com/firedrakeproject/firedrake/issues/3018
# - w = B.interpolate(v)
#   - v is a function in V (NOT an expression).
#   - w is a function in W.
#   - Maths: v = Bw
# - v_star = B.interpolate(w_star, transpose = True)
#   - w_star is a cofunction in W^* (such as an assembled 1-form).
#   - v_star is a cofunction in V^*.
#   - Maths: v^* = B^* w^*


@PETSc.Log.EventDecorator()
def interpolate(expr, V, subset=None, access=op2.WRITE, ad_block_tag=None):
    """Interpolate an expression onto a new function in V.

    :arg expr: a UFL expression.
    :arg V: the :class:`.FunctionSpace` to interpolate into (or else
        an existing :class:`.Function`).
    :kwarg subset: An optional :class:`pyop2.types.set.Subset` to apply the
        interpolation over.
    :kwarg access: The access descriptor for combining updates to shared dofs.
    :kwarg ad_block_tag: string for tagging the resulting block on the Pyadjoint tape
    :returns: a new :class:`.Function` in the space ``V`` (or ``V`` if
        it was a Function).

    .. note::

       If you use an access descriptor other than ``WRITE``, the
       behaviour of interpolation is changes if interpolating into a
       function space, or an existing function. If the former, then
       the newly allocated function will be initialised with
       appropriate values (e.g. for MIN access, it will be initialised
       with MAX_FLOAT). On the other hand, if you provide a function,
       then it is assumed that its values should take part in the
       reduction (hence using MIN will compute the MIN between the
       existing values and any new values).

    .. note::

       If you find interpolating the same expression again and again
       (for example in a time loop) you may find you get better
       performance by using an :class:`Interpolator` instead.

    """
    return Interpolator(expr, V, subset=subset, access=access).interpolate(ad_block_tag=ad_block_tag)


class Interpolator(object):
    """A reusable interpolation object.

    :arg expr: The expression to interpolate.
    :arg V: The :class:`.FunctionSpace` or :class:`.Function` to
        interpolate into.
    :kwarg subset: An optional :class:`pyop2.types.set.Subset` to apply the
        interpolation over.
    :kwarg freeze_expr: Set to True to prevent the expression being
        re-evaluated on each call.

    This object can be used to carry out the same interpolation
    multiple times (for example in a timestepping loop).

    .. note::

       The :class:`Interpolator` holds a reference to the provided
       arguments (such that they won't be collected until the
       :class:`Interpolator` is also collected).

    """
    def __init__(self, expr, V, subset=None, freeze_expr=False, access=op2.WRITE, bcs=None):
        try:
            self.callable, arguments = make_interpolator(expr, V, subset, access, bcs=bcs)
        except FIAT.hdiv_trace.TraceError:
            raise NotImplementedError("Can't interpolate onto traces sorry")
        self.arguments = arguments
        self.nargs = len(arguments)
        self.freeze_expr = freeze_expr
        self.expr = expr
        self.V = V
        self.bcs = bcs

    @PETSc.Log.EventDecorator()
    @annotate_interpolate
    def interpolate(self, *function, output=None, transpose=False):
        """Compute the interpolation.

        :arg function: If the expression being interpolated contains an
            :class:`ufl.Argument`, then the :class:`.Function` value to
            interpolate.
        :kwarg output: Optional. A :class:`.Function` to contain the output.
        :kwarg transpose: Set to true to apply the transpose (adjoint) of the
              interpolation operator.
        :returns: The resulting interpolated :class:`.Function`.
        """
        if transpose and not self.nargs:
            raise ValueError("Can currently only apply transpose interpolation with arguments.")
        if self.nargs != len(function):
            raise ValueError("Passed %d Functions to interpolate, expected %d"
                             % (len(function), self.nargs))
        try:
            assembled_interpolator = self.frozen_assembled_interpolator
            copy_required = True
        except AttributeError:
            assembled_interpolator = self.callable()
            copy_required = False  # Return the original
            if self.freeze_expr:
                if self.nargs:
                    # Interpolation operator
                    self.frozen_assembled_interpolator = assembled_interpolator
                else:
                    # Interpolation action
                    self.frozen_assembled_interpolator = assembled_interpolator.copy()

        if self.nargs:
            function, = function
            if not hasattr(function, "dat"):
                raise ValueError("The expression had arguments: we therefore need to be given a Function (not an expression) to interpolate!")
            if transpose:
                mul = assembled_interpolator.handle.multTranspose
                V = self.arguments[0].function_space()
            else:
                mul = assembled_interpolator.handle.mult
                V = self.V
            result = output or firedrake.Function(V)
            with function.dat.vec_ro as x, result.dat.vec_wo as out:
                mul(x, out)
            return result

        else:
            if output:
                output.assign(assembled_interpolator)
                return output
            if isinstance(self.V, firedrake.Function):
                if copy_required:
                    self.V.assign(assembled_interpolator)
                return self.V
            else:
                if copy_required:
                    return assembled_interpolator.copy()
                else:
                    return assembled_interpolator


@PETSc.Log.EventDecorator()
def make_interpolator(expr, V, subset, access, bcs=None):
    assert isinstance(expr, ufl.classes.Expr)

    arguments = extract_arguments(expr)
    target_mesh = V.ufl_domain()
    if len(arguments) == 0:
        source_mesh = extract_unique_domain(expr) or target_mesh
        vom_onto_other_vom = (
            isinstance(target_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology)
            and isinstance(source_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology)
            and target_mesh is not source_mesh
        )
        if isinstance(V, firedrake.Function):
            f = V
            V = f.function_space()
        else:
            f = firedrake.Function(V)
            if access in {firedrake.MIN, firedrake.MAX}:
                finfo = numpy.finfo(f.dat.dtype)
                if access == firedrake.MIN:
                    val = firedrake.Constant(finfo.max)
                else:
                    val = firedrake.Constant(finfo.min)
                f.assign(val)
        tensor = f.dat
    elif len(arguments) == 1:
        if isinstance(V, firedrake.Function):
            raise ValueError("Cannot interpolate an expression with an argument into a Function")
        argfs = arguments[0].function_space()
        source_mesh = argfs.mesh()
        argfs_map = argfs.cell_node_map()
        vom_onto_other_vom = (
            isinstance(target_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology)
            and isinstance(source_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology)
            and target_mesh is not source_mesh
        )
        if target_mesh is not source_mesh and not vom_onto_other_vom:
            if not isinstance(target_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology):
                raise NotImplementedError("Can only interpolate onto a Vertex Only Mesh")
            if target_mesh.geometric_dimension() != source_mesh.geometric_dimension():
                raise ValueError("Cannot interpolate onto a mesh of a different geometric dimension")
            if not hasattr(target_mesh, "_parent_mesh") or target_mesh._parent_mesh is not source_mesh:
                raise ValueError("Can only interpolate across meshes where the source mesh is the parent of the target")
            if argfs_map:
                # Since the par_loop is over the target mesh cells we need to
                # compose a map that takes us from target mesh cells to the
                # function space nodes on the source mesh. NOTE: argfs_map is
                # allowed to be None when interpolating from a Real space, even
                # in the trans-mesh case.
                if source_mesh.extruded:
                    # ExtrudedSet cannot be a map target so we need to build
                    # this ourselves
                    argfs_map = vom_cell_parent_node_map_extruded(target_mesh, argfs_map)
                else:
                    argfs_map = compose_map_and_cache(target_mesh.cell_parent_cell_map, argfs_map)
        if vom_onto_other_vom:
            # We make our own linear operator for this case using PETSc SFs
            tensor = None
        else:
            sparsity = op2.Sparsity((V.dof_dset, argfs.dof_dset),
                                    ((V.cell_node_map(), argfs_map),),
                                    name="%s_%s_sparsity" % (V.name, argfs.name),
                                    nest=False,
                                    block_sparse=True)
            tensor = op2.Mat(sparsity)
        f = tensor
    else:
        raise ValueError("Cannot interpolate an expression with %d arguments" % len(arguments))

    if vom_onto_other_vom:
        # To interpolate between vertex-only meshes we use a PETSc SF
        wrapper = VomOntoVomWrapper(V, source_mesh, target_mesh, expr, arguments)
        # NOTE: get_dat_mpi_type ensures we get the correct MPI type for the
        # data, including the correct data size and dimensional information
        # (so for vector function spaces in 2 dimensions we might need a
        # concatenation of 2 MPI.DOUBLE types when we are in real mode)
        if tensor is not None:
            # Callable will do interpolation into tensor (which is a Dat) when
            # it is called.
            wrapper.mpi_type, _ = get_dat_mpi_type(tensor)
            assert not len(arguments)
            callable = partial(wrapper.forward_operation, tensor)
        else:
            assert len(arguments) == 1
            assert tensor is None
            # we know we will be outputting either a function or a cofunction,
            # both of which will use a dat as a data carrier. At present, the
            # data type does not depend on function space dimension, so we can
            # safely use the argument function space. NOTE: If this changes
            # after cofunctions are fully implemented, this will need to be
            # reconsidered.
            temp_source_func = firedrake.Function(argfs)
            wrapper.mpi_type, _ = get_dat_mpi_type(temp_source_func.dat)

            # Leave wrapper inside a callable so we can access the handle
            # property (which is pretending to be a petsc mat)
            def callable():
                return wrapper

        return callable, arguments
    else:
        # Make sure we have an expression of the right length i.e. a value for
        # each component in the value shape of each function space
        dims = [numpy.prod(fs.ufl_element().value_shape(), dtype=int)
                for fs in V]
        loops = []
        if numpy.prod(expr.ufl_shape, dtype=int) != sum(dims):
            raise RuntimeError('Expression of length %d required, got length %d'
                               % (sum(dims), numpy.prod(expr.ufl_shape, dtype=int)))

        if len(V) > 1:
            raise NotImplementedError(
                "UFL expressions for mixed functions are not yet supported.")
        loops.extend(_interpolator(V, tensor, expr, subset, arguments, access, bcs=bcs))

        if bcs and len(arguments) == 0:
            loops.extend([partial(bc.apply, f) for bc in bcs])

        def callable(loops, f):
            for l in loops:
                l()
            return f

        return partial(callable, loops, f), arguments


@utils.known_pyop2_safe
def _interpolator(V, tensor, expr, subset, arguments, access, bcs=None):
    try:
        expr = ufl.as_ufl(expr)
    except ValueError:
        raise ValueError("Expecting to interpolate a UFL expression")
    try:
        to_element = create_element(V.ufl_element())
    except KeyError:
        # FInAT only elements
        raise NotImplementedError("Don't know how to create FIAT element for %s" % V.ufl_element())

    if access is op2.READ:
        raise ValueError("Can't have READ access for output function")

    if len(expr.ufl_shape) != len(V.ufl_element().value_shape()):
        raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                           % (len(expr.ufl_shape), len(V.ufl_element().value_shape())))

    if expr.ufl_shape != V.ufl_element().value_shape():
        raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                           % (expr.ufl_shape, V.ufl_element().value_shape()))

    # NOTE: The par_loop is always over the target mesh cells.
    target_mesh = V.ufl_domain()
    source_mesh = extract_unique_domain(expr) or target_mesh

    if target_mesh is not source_mesh:
        if not isinstance(target_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology):
            raise NotImplementedError("Can only interpolate onto a Vertex Only Mesh")
        if target_mesh.geometric_dimension() != source_mesh.geometric_dimension():
            raise ValueError("Cannot interpolate onto a mesh of a different geometric dimension")
        if not hasattr(target_mesh, "_parent_mesh") or target_mesh._parent_mesh is not source_mesh:
            raise ValueError("Can only interpolate across meshes where the source mesh is the parent of the target")
        # For trans-mesh interpolation we use a FInAT QuadratureElement as the
        # (base) target element with runtime point set expressions as their
        # quadrature rule point set and weights from their dual basis.
        # NOTE: This setup is useful for thinking about future design - in the
        # future this `rebuild` function can be absorbed into FInAT as a
        # transformer that eats an element and gives you an equivalent (which
        # may or may not be a QuadratureElement) that lets you do run time
        # tabulation. Alternatively (and this all depends on future design
        # decision about FInAT how dual evaluation should work) the
        # to_element's dual basis (which look rather like quadrature rules) can
        # have their pointset(s) directly replaced with run-time tabulated
        # equivalent(s) (i.e. finat.point_set.UnknownPointSet(s))
        rt_var_name = 'rt_X'
        to_element = rebuild(to_element, expr, rt_var_name)

    cell_set = target_mesh.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset

    parameters = {}
    parameters['scalar_type'] = utils.ScalarType

    # We need to pass both the ufl element and the finat element
    # because the finat elements might not have the right mapping
    # (e.g. L2 Piola, or tensor element with symmetries)
    # FIXME: for the runtime unknown point set (for cross-mesh
    # interpolation) we have to pass the finat element we construct
    # here. Ideally we would only pass the UFL element through.
    kernel = compile_expression(cell_set.comm, expr, to_element, V.ufl_element(),
                                domain=source_mesh, parameters=parameters,
                                log=PETSc.Log.isActive())
    ast = kernel.ast
    oriented = kernel.oriented
    needs_cell_sizes = kernel.needs_cell_sizes
    coefficient_numbers = kernel.coefficient_numbers
    needs_external_coords = kernel.needs_external_coords
    name = kernel.name
    kernel = op2.Kernel(ast, name, requires_zeroed_output_arguments=True,
                        flop_count=kernel.flop_count, events=(kernel.event,))
    parloop_args = [kernel, cell_set]

    coefficients = tsfc_interface.extract_numbered_coefficients(expr, coefficient_numbers)
    if needs_external_coords:
        coefficients = [source_mesh.coordinates] + coefficients

    if target_mesh is not source_mesh:
        # NOTE: TSFC will sometimes drop run-time arguments in generated
        # kernels if they are deemed not-necessary.
        # FIXME: Checking for argument name in the inner kernel to decide
        # whether to add an extra coefficient is a stopgap until
        # compile_expression_dual_evaluation
        #   (a) outputs a coefficient map to indicate argument ordering in
        #       parloops as `compile_form` does and
        #   (b) allows the dual evaluation related coefficients to be supplied to
        #       them rather than having to be added post-hoc (likely by
        #       replacing `to_element` with a CoFunction/CoArgument as the
        #       target `dual` which would contain `dual` related
        #       coefficient(s))
        if rt_var_name in [arg.name for arg in kernel.code[name].args]:
            # Add the coordinates of the target mesh quadrature points in the
            # source mesh's reference cell as an extra argument for the inner
            # loop. (With a vertex only mesh this is a single point for each
            # vertex cell.)
            coefficients.append(target_mesh.reference_coordinates)

    if tensor in set((c.dat for c in coefficients)):
        output = tensor
        tensor = op2.Dat(tensor.dataset)
        if access is not op2.WRITE:
            copyin = (partial(output.copy, tensor), )
        else:
            copyin = ()
        copyout = (partial(tensor.copy, output), )
    else:
        copyin = ()
        copyout = ()
    if isinstance(tensor, op2.Global):
        parloop_args.append(tensor(access))
    elif isinstance(tensor, op2.Dat):
        parloop_args.append(tensor(access, V.cell_node_map()))
    else:
        assert access == op2.WRITE  # Other access descriptors not done for Matrices.
        rows_map = V.cell_node_map()
        Vcol = arguments[0].function_space()
        columns_map = Vcol.cell_node_map()
        if target_mesh is not source_mesh:
            # Since the par_loop is over the target mesh cells we need to
            # compose a map that takes us from target mesh cells to the
            # function space nodes on the source mesh.
            if source_mesh.extruded:
                # ExtrudedSet cannot be a map target so we need to build
                # this ourselves
                columns_map = vom_cell_parent_node_map_extruded(target_mesh, columns_map)
            else:
                columns_map = compose_map_and_cache(target_mesh.cell_parent_cell_map,
                                                    columns_map)
        lgmaps = None
        if bcs:
            bc_rows = [bc for bc in bcs if bc.function_space() == V]
            bc_cols = [bc for bc in bcs if bc.function_space() == Vcol]
            lgmaps = [(V.local_to_global_map(bc_rows), Vcol.local_to_global_map(bc_cols))]
        parloop_args.append(tensor(op2.WRITE, (rows_map, columns_map), lgmaps=lgmaps))
    if oriented:
        co = target_mesh.cell_orientations()
        parloop_args.append(co.dat(op2.READ, co.cell_node_map()))
    if needs_cell_sizes:
        cs = target_mesh.cell_sizes
        parloop_args.append(cs.dat(op2.READ, cs.cell_node_map()))
    for coefficient in coefficients:
        coeff_mesh = extract_unique_domain(coefficient)
        if coeff_mesh is target_mesh or not coeff_mesh:
            # NOTE: coeff_mesh is None is allowed e.g. when interpolating from
            # a Real space
            m_ = coefficient.cell_node_map()
        elif coeff_mesh is source_mesh:
            if coefficient.cell_node_map():
                # Since the par_loop is over the target mesh cells we need to
                # compose a map that takes us from target mesh cells to the
                # function space nodes on the source mesh.
                if source_mesh.extruded:
                    # ExtrudedSet cannot be a map target so we need to build
                    # this ourselves
                    m_ = vom_cell_parent_node_map_extruded(target_mesh, coefficient.cell_node_map())
                else:
                    m_ = compose_map_and_cache(target_mesh.cell_parent_cell_map, coefficient.cell_node_map())
            else:
                # m_ is allowed to be None when interpolating from a Real space,
                # even in the trans-mesh case.
                m_ = coefficient.cell_node_map()
        else:
            raise ValueError("Have coefficient with unexpected mesh")
        parloop_args.append(coefficient.dat(op2.READ, m_))

    for const in extract_firedrake_constants(expr):
        parloop_args.append(const.dat(op2.READ))

    parloop = op2.ParLoop(*parloop_args)
    parloop_compute_callable = parloop.compute
    if isinstance(tensor, op2.Mat):
        return parloop_compute_callable, tensor.assemble
    else:
        return copyin + (parloop_compute_callable, ) + copyout


try:
    _expr_cachedir = os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"]
except KeyError:
    _expr_cachedir = os.path.join(tempfile.gettempdir(),
                                  f"firedrake-tsfc-expression-kernel-cache-uid{os.getuid()}")


def _compile_expression_key(comm, expr, to_element, ufl_element, domain, parameters, log):
    """Generate a cache key suitable for :func:`tsfc.compile_expression_dual_evaluation`."""
    # Since the caching is collective, this function must return a 2-tuple of
    # the form (comm, key) where comm is the communicator the cache is collective over.
    # FIXME FInAT elements are not safely hashable so we ignore them here
    key = hash_expr(expr), hash(ufl_element), utils.tuplify(parameters), log
    return comm, key


@disk_cached({}, _expr_cachedir, key=_compile_expression_key, collective=True)
def compile_expression(comm, *args, **kwargs):
    return compile_expression_dual_evaluation(*args, **kwargs)


@singledispatch
def rebuild(element, expr, rt_var_name):
    raise NotImplementedError(f"Cross mesh interpolation not implemented for a {element} element.")


@rebuild.register(finat.DiscontinuousLagrange)
def rebuild_dg(element, expr, rt_var_name):
    # To tabulate on the given element (which is on a different mesh to the
    # expression) we must do so at runtime. We therefore create a quadrature
    # element with runtime points to evaluate for each point in the element's
    # dual basis. This exists on the same reference cell as the input element
    # and we can interpolate onto it before mapping the result back onto the
    # target space.
    expr_tdim = extract_unique_domain(expr).topological_dimension()
    # Need point evaluations and matching weights from dual basis.
    # This could use FIAT's dual basis as below:
    # num_points = sum(len(dual.get_point_dict()) for dual in element.fiat_equivalent.dual_basis())
    # weights = []
    # for dual in element.fiat_equivalent.dual_basis():
    #     pts = dual.get_point_dict().keys()
    #     for p in pts:
    #         for w, _ in dual.get_point_dict()[p]:
    #             weights.append(w)
    # assert len(weights) == num_points
    # but for now we just fix the values to what we know works:
    if element.degree != 0 or not isinstance(element.cell, FIAT.reference_element.Point):
        raise NotImplementedError("Cross mesh interpolation only implemented for P0DG on vertex cells.")
    num_points = 1
    weights = [1.]*num_points
    # gem.Variable name starting with rt_ forces TSFC runtime tabulation
    assert rt_var_name.startswith("rt_")
    runtime_points_expr = gem.Variable(rt_var_name, (num_points, expr_tdim))
    rule_pointset = finat.point_set.UnknownPointSet(runtime_points_expr)
    try:
        expr_fiat_cell = as_fiat_cell(expr.ufl_element().cell())
    except AttributeError:
        # expression must be pure function of spatial coordinates so
        # domain has correct ufl cell
        expr_fiat_cell = as_fiat_cell(extract_unique_domain(expr).ufl_cell())
    rule = finat.quadrature.QuadratureRule(rule_pointset, weights=weights)
    return finat.QuadratureElement(expr_fiat_cell, rule)


@rebuild.register(finat.TensorFiniteElement)
def rebuild_te(element, expr, rt_var_name):
    return finat.TensorFiniteElement(rebuild(element.base_element,
                                             expr, rt_var_name),
                                     element._shape,
                                     transpose=element._transpose)


def compose_map_and_cache(map1, map2):
    """
    Retrieve a :class:`pyop2.ComposedMap` map from the cache of map1
    using map2 as the cache key. The composed map maps from the iterset
    of map1 to the toset of map2. Makes :class:`pyop2.ComposedMap` and
    caches the result on map1 if the composed map is not found.

    :arg map1: The map with the desired iterset from which the result is
        retrieved or cached
    :arg map2: The map with the desired toset

    :returns:  The composed map
    """
    cache_key = hash((map2, "composed"))
    try:
        cmap = map1._cache[cache_key]
    except KeyError:
        # Real function space case separately
        cmap = None if map2 is None else op2.ComposedMap(map2, map1)
        map1._cache[cache_key] = cmap
    return cmap


def vom_cell_parent_node_map_extruded(vertex_only_mesh, extruded_cell_node_map):
    """Build a map from the cells of a vertex only mesh to the nodes of the
    nodes on the source mesh where the source mesh is extruded.

    Parameters
    ----------
    vertex_only_mesh : :class:`mesh.MeshGeometry`
        The ``mesh.VertexOnlyMesh`` whose cells we iterate over.
    extruded_cell_node_map : :class:`pyop2.Map`
        The cell node map of the function space on the extruded mesh within
        which the ``mesh.VertexOnlyMesh`` is immersed.

    Returns
    -------
    :class:`pyop2.Map`
        The map from the cells of the vertex only mesh to the nodes of the
        source mesh's cell node map. The map iterset is the
        ``vertex_only_mesh.cell_set`` and the map toset is the
        ``extruded_cell_node_map.toset``.

    Notes
    -----

    For an extruded mesh the cell node map is a map from a
    :class:`pyop2.ExtrudedSet` (the cells of the extruded mesh) to a
    :class:`pyop2.Set` (the nodes of the extruded mesh).

    Take for example

    ``mx = ExtrudedMesh(UnitIntervalMesh(2), 3)`` with
    ``mx.layers = 4``

    which looks like

    .. code-block:: text

        -------------------layer 4-------------------
        | parent_cell_num =  2 | parent_cell_num =  5 |
        |                      |                      |
        | extrusion_height = 2 | extrusion_height = 2 |
        -------------------layer 3-------------------
        | parent_cell_num =  1 | parent_cell_num =  4 |
        |                      |                      |
        | extrusion_height = 1 | extrusion_height = 1 |
        -------------------layer 2-------------------
        | parent_cell_num =  0 | parent_cell_num =  3 |
        |                      |                      |
        | extrusion_height = 0 | extrusion_height = 0 |
        -------------------layer 1-------------------
          base_cell_num = 0      base_cell_num = 1


    If we declare ``FunctionSpace(mx, "CG", 2)`` then the node numbering (i.e.
    Degree of Freedom/DoF numbering) is

    .. code-block:: text

        6 ---------13----------20---------27---------34
        |                       |                     |
        5          12          19         26         33
        |                       |                     |
        4 ---------11----------18---------25---------32
        |                       |                     |
        3          10          17         24         31
        |                       |                     |
        2 ---------9-----------16---------23---------30
        |                       |                     |
        1          8           15         22         29
        |                       |                     |
        0 ---------7-----------14---------21---------28
          base_cell_num = 0       base_cell_num = 1

    Cell node map values for an extruded mesh are indexed by the base cell
    number (rows) and the degree of freedom (DoF) index (columns). So
    ``extruded_cell_node_map.values[0] = [14, 15, 16,  0,  1,  2,  7,  8,  9]``
    are all the DoF/node numbers for the ``base_cell_num = 0``.
    Similarly
    ``extruded_cell_node_map.values[1] = [28, 29, 30, 21, 22, 23, 14, 15, 16]``
    contain all 9 of the DoFs for ``base_cell_num = 1``.
    To get the DoFs/nodes for the rest of the  cells we need to include the
    ``extruded_cell_node_map.offset``, which tells us how far each cell's
    DoFs/nodes are translated up from the first layer to the second, and
    multiply these by the the given ``extrusion_height``. So in our example
    ``extruded_cell_node_map.offset = [2, 2, 2, 2, 2, 2, 2, 2, 2]`` (we index
    this with the DoF/node index - it's an array because each DoF/node in the
    extruded mesh cell, in principal, can be offset upwards by a different
    amount).
    For ``base_cell_num = 0`` with ``extrusion_height = 1``
    (``parent_cell_num = 1``) we add ``1*2 = 2`` to each of the DoFs/nodes in
    ``extruded_cell_node_map.values[0]`` to get
    ``extruded_cell_node_map.values[0] + 1 * extruded_cell_node_map.offset[0] =
    [16, 17, 18,  2,  3,  4,  9, 10, 11]`` where ``0`` is the DoF/node index.

    For each cell (vertex) of a vertex only mesh immersed in a parent
    extruded mesh, we can can get the corresponding ``base_cell_num`` and
    ``extrusion_height`` of the parent extruded mesh. Armed with this
    information we use the above to work out the corresponding DoFs/nodes on
    the parent extruded mesh.

    """
    if not isinstance(vertex_only_mesh.topology, firedrake.mesh.VertexOnlyMeshTopology):
        raise TypeError("The input mesh must be a VertexOnlyMesh")
    cnm = extruded_cell_node_map
    vmx = vertex_only_mesh
    dofs_per_target_cell = cnm.arity
    base_cells = vmx.cell_parent_base_cell_list
    heights = vmx.cell_parent_extrusion_height_list
    assert cnm.values_with_halo.shape[1] == dofs_per_target_cell
    assert len(cnm.offset) == dofs_per_target_cell
    target_cell_parent_node_list = [
        cnm.values_with_halo[base_cell, :] + height * cnm.offset[:]
        for base_cell, height in zip(base_cells, heights)
    ]
    return op2.Map(
        vmx.cell_set, cnm.toset, dofs_per_target_cell, target_cell_parent_node_list
    )


class GlobalWrapper(object):
    """Wrapper object that fakes a Global to behave like a Function."""
    def __init__(self, glob):
        self.dat = glob
        self.cell_node_map = lambda *arguments: None
        self.ufl_domain = lambda: None


def hash_expr(expr):
    """Return a numbering-invariant hash of a UFL expression.

    :arg expr: A UFL expression.
    :returns: A numbering-invariant hash for the expression.
    """
    domain_numbering = {d: i for i, d in enumerate(ufl.domain.extract_domains(expr))}
    coefficient_numbering = {c: i for i, c in enumerate(extract_coefficients(expr))}
    constant_numbering = {c: i for i, c in enumerate(extract_firedrake_constants(expr))}
    return compute_expression_signature(
        expr, {**domain_numbering, **coefficient_numbering, **constant_numbering}
    )


class VomOntoVomWrapper(object):
    """Utility class for interpolating from one ``VertexOnlyMesh`` to it's
    intput ordering ``VertexOnlyMesh``, or vice versa.

    Parameters
    ----------
    V : `.FunctionSpace`
        The P0DG function space (which may be vector or tensor valued) on the
        source vertex-only mesh.
    source_vom : `.VertexOnlyMesh`
        The vertex-only mesh we interpolate from.
    target_vom : `.VertexOnlyMesh`
        The vertex-only mesh we interpolate to.
    expr : `ufl.Expr`
        The expression to interpolate. If ``arguments`` is not empty, those
        arguments must be present within it.
    arguments : list of `ufl.Argument`
        The arguments in the expression. These are not extracted from expr here
        since, where we use this, we already have them.
    """

    def __init__(self, V, source_vom, target_vom, expr, arguments):
        reduce = False
        if source_vom.input_ordering is target_vom:
            reduce = True
            original_vom = source_vom
        elif target_vom.input_ordering is source_vom:
            original_vom = target_vom
        else:
            raise ValueError(
                "The target vom and source vom must be linked by input ordering!"
            )
        self.V = V
        self.source_vom = source_vom
        self.expr = expr
        self.arguments = arguments
        self.reduce = reduce
        # note that interpolation doesn't include halo cells
        self.handle = VomOntoVomDummyMat(
            original_vom.input_ordering_without_halos_sf, reduce, V, source_vom, expr, arguments
        )

    @property
    def mpi_type(self):
        """
        The MPI type to use for the PETSc SF.

        Should correspond to the underlying data type of the PETSc Vec.
        """
        return self.handle.mpi_type

    @mpi_type.setter
    def mpi_type(self, val):
        self.handle.mpi_type = val

    def forward_operation(self, target_dat):
        coeff = self.handle.expr_as_coeff()
        with coeff.dat.vec_ro as coeff_vec, target_dat.vec_wo as target_vec:
            self.handle.mult(coeff_vec, target_vec)


class VomOntoVomDummyMat(object):
    """Dummy object to stand in for a PETSc ``Mat`` when we are interpolating
    between vertex-only meshes.

    Parameters
    ----------
    sf: PETSc.sf
        The PETSc Star Forest (SF) to use for the operation
    forward_reduce : bool
        If ``True``, the action of the operator (accessed via the `mult`
        method) is to perform a SF reduce from the source vec to the target
        vec, whilst the adjoint action (accessed via the `multTranspose`
        method) is to perform a SF broadcast from the source vec to the target
        vec. If ``False``, the opposite is true.
    V : `.FunctionSpace`
        The P0DG function space (which may be vector or tensor valued) on the
        source vertex-only mesh.
    source_vom : `.VertexOnlyMesh`
        The vertex-only mesh we interpolate from.
    expr : `ufl.Expr`
        The expression to interpolate. If ``arguments`` is not empty, those
        arguments must be present within it.
    arguments : list of `ufl.Argument`
        The arguments in the expression.
    """

    def __init__(self, sf, forward_reduce, V, source_vom, expr, arguments):
        self.sf = sf
        self.forward_reduce = forward_reduce
        self.V = V
        self.source_vom = source_vom
        self.expr = expr
        self.arguments = arguments

    @property
    def mpi_type(self):
        """
        The MPI type to use for the PETSc SF.

        Should correspond to the underlying data type of the PETSc Vec.
        """
        return self._mpi_type

    @mpi_type.setter
    def mpi_type(self, val):
        self._mpi_type = val

    def expr_as_coeff(self, source_vec=None):
        """
        Return a coefficient that corresponds to the expression used at
        construction, where the expression has been interpolated into the P0DG
        function space on the source vertex-only mesh.

        Will fail if there are no arguments.
        """
        # Since we always output a coefficient when we don't have arguments in
        # the expression, we should evaluate the expression on the source mesh
        # so its dat can be sent to the target mesh.
        with stop_annotating():
            element = self.V.ufl_element()  # Could be vector/tensor valued
            P0DG = firedrake.FunctionSpace(self.source_vom, element)
            # if we have any arguments in the expression we need to replace
            # them with equivalent coefficients now
            coeff_expr = self.expr
            if len(self.arguments):
                if len(self.arguments) > 1:
                    raise NotImplementedError(
                        "Can only interpolate expressions with one argument!"
                    )
                if source_vec is None:
                    raise ValueError("Need to provide a source dat for the argument!")
                arg = self.arguments[0]
                arg_coeff = firedrake.Function(arg.function_space())
                arg_coeff.dat.data_wo[:] = source_vec.getArray().reshape(
                    arg_coeff.dat.data_wo.shape
                )
                coeff_expr = ufl.replace(self.expr, {arg: arg_coeff})
            coeff = firedrake.Function(P0DG).interpolate(coeff_expr)
        return coeff

    def reduce(self, source_vec, target_vec):
        source_arr = source_vec.getArray()
        target_arr = target_vec.getArray()
        self.sf.reduceBegin(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )
        self.sf.reduceEnd(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )

    def broadcast(self, source_vec, target_vec):
        source_arr = source_vec.getArray()
        target_arr = target_vec.getArray()
        self.sf.bcastBegin(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )
        self.sf.bcastEnd(
            self.mpi_type,
            source_arr,
            target_arr,
            MPI.REPLACE,
        )

    def mult(self, source_vec, target_vec):
        # need to evaluate expression before doing mult
        coeff = self.expr_as_coeff(source_vec)
        with coeff.dat.vec_ro as coeff_vec:
            if self.forward_reduce:
                self.reduce(coeff_vec, target_vec)
            else:
                self.broadcast(coeff_vec, target_vec)

    def multTranspose(self, source_vec, target_vec):
        # can only do transpose if our expression exclusively contains a
        # single argument, making the application of the adjoint operator
        # straightforward (haven't worked out how to do this otherwise!)
        if not len(self.arguments) == 1:
            raise NotImplementedError(
                "Can only apply transpose to expressions with one argument!"
            )
        if self.arguments[0] is not self.expr:
            raise NotImplementedError(
                "Can only apply transpose to expressions consisting of a single argument at the moment."
            )
        if self.forward_reduce:
            self.broadcast(source_vec, target_vec)
        else:
            self.reduce(source_vec, target_vec)
