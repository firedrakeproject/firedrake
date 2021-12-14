import numpy
from functools import partial, singledispatch

import FIAT
import ufl
from ufl.algorithms import extract_arguments

from pyop2 import op2

from tsfc.finatinterface import create_element, as_fiat_cell
from tsfc import compile_expression_dual_evaluation

import gem
import finat

import firedrake
from firedrake import utils
from firedrake.adjoint import annotate_interpolate
from firedrake.petsc import PETSc

__all__ = ("interpolate", "Interpolator")


@PETSc.Log.EventDecorator()
def interpolate(expr, V, subset=None, access=op2.WRITE, ad_block_tag=None):
    """Interpolate an expression onto a new function in V.

    :arg expr: a UFL expression.
    :arg V: the :class:`.FunctionSpace` to interpolate into (or else
        an existing :class:`.Function`).
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
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
    :kwarg subset: An optional :class:`pyop2.Subset` to apply the
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
    def __init__(self, expr, V, subset=None, freeze_expr=False, access=op2.WRITE):
        try:
            self.callable, arguments = make_interpolator(expr, V, subset, access)
        except FIAT.hdiv_trace.TraceError:
            raise NotImplementedError("Can't interpolate onto traces sorry")
        self.arguments = arguments
        self.nargs = len(arguments)
        self.freeze_expr = freeze_expr
        self.expr = expr
        self.V = V

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
def make_interpolator(expr, V, subset, access):
    assert isinstance(expr, ufl.classes.Expr)

    arguments = extract_arguments(expr)
    if len(arguments) == 0:
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
        target_mesh = V.ufl_domain()
        source_mesh = argfs.mesh()
        argfs_map = argfs.cell_node_map()
        if target_mesh is not source_mesh:
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
                argfs_map = compose_map_and_cache(target_mesh.cell_parent_cell_map, argfs_map)
        sparsity = op2.Sparsity((V.dof_dset, argfs.dof_dset),
                                ((V.cell_node_map(), argfs_map),),
                                name="%s_%s_sparsity" % (V.name, argfs.name),
                                nest=False,
                                block_sparse=True)
        tensor = op2.Mat(sparsity)
        f = tensor
    else:
        raise ValueError("Cannot interpolate an expression with %d arguments" % len(arguments))

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
    loops.extend(_interpolator(V, tensor, expr, subset, arguments, access))

    def callable(loops, f):
        for l in loops:
            l()
        return f

    return partial(callable, loops, f), arguments


@utils.known_pyop2_safe
def _interpolator(V, tensor, expr, subset, arguments, access):
    try:
        expr = ufl.as_ufl(expr)
    except ufl.UFLException:
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
    source_mesh = expr.ufl_domain() or target_mesh

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

    parameters = {}
    parameters['scalar_type'] = utils.ScalarType

    # We need to pass both the ufl element and the finat element
    # because the finat elements might not have the right mapping
    # (e.g. L2 Piola, or tensor element with symmetries)
    # FIXME: for the runtime unknown point set (for cross-mesh
    # interpolation) we have to pass the finat element we construct
    # here. Ideally we would only pass the UFL element through.
    kernel = compile_expression_dual_evaluation(expr, to_element,
                                                V.ufl_element(),
                                                domain=source_mesh,
                                                parameters=parameters)
    ast = kernel.ast
    oriented = kernel.oriented
    needs_cell_sizes = kernel.needs_cell_sizes
    coefficients = kernel.coefficients
    first_coeff_fake_coords = kernel.first_coefficient_fake_coords
    name = kernel.name
    kernel = op2.Kernel(ast, name, requires_zeroed_output_arguments=True,
                        flop_count=kernel.flop_count)
    cell_set = target_mesh.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset
    parloop_args = [kernel, cell_set]

    if first_coeff_fake_coords:
        # Replace with real source mesh coordinates
        coefficients[0] = source_mesh.coordinates

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
        columns_map = arguments[0].function_space().cell_node_map()
        if target_mesh is not source_mesh:
            # Since the par_loop is over the target mesh cells we need to
            # compose a map that takes us from target mesh cells to the
            # function space nodes on the source mesh.
            columns_map = compose_map_and_cache(target_mesh.cell_parent_cell_map,
                                                columns_map)
        parloop_args.append(tensor(op2.WRITE, (rows_map, columns_map)))
    if oriented:
        co = target_mesh.cell_orientations()
        parloop_args.append(co.dat(op2.READ, co.cell_node_map()))
    if needs_cell_sizes:
        cs = target_mesh.cell_sizes
        parloop_args.append(cs.dat(op2.READ, cs.cell_node_map()))
    for coefficient in coefficients:
        coeff_mesh = coefficient.ufl_domain()
        if coeff_mesh is target_mesh or not coeff_mesh:
            # NOTE: coeff_mesh is None is allowed e.g. when interpolating from
            # a Real space
            m_ = coefficient.cell_node_map()
        elif coeff_mesh is source_mesh:
            if coefficient.cell_node_map():
                # Since the par_loop is over the target mesh cells we need to
                # compose a map that takes us from target mesh cells to the
                # function space nodes on the source mesh.
                m_ = compose_map_and_cache(target_mesh.cell_parent_cell_map, coefficient.cell_node_map())
            else:
                # m_ is allowed to be None when interpolating from a Real space,
                # even in the trans-mesh case.
                m_ = coefficient.cell_node_map()
        else:
            raise ValueError("Have coefficient with unexpected mesh")
        parloop_args.append(coefficient.dat(op2.READ, m_))

    parloop = op2.ParLoop(*parloop_args)
    parloop_compute_callable = parloop.compute
    if isinstance(tensor, op2.Mat):
        return parloop_compute_callable, tensor.assemble
    else:
        return copyin + (parloop_compute_callable, ) + copyout


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
    expr_tdim = expr.ufl_domain().topological_dimension()
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
        expr_fiat_cell = as_fiat_cell(expr.ufl_domain().ufl_cell())
    rule = finat.quadrature.QuadratureRule(rule_pointset, weights=weights)
    return finat.QuadratureElement(expr_fiat_cell, rule)


@rebuild.register(finat.TensorFiniteElement)
def rebuild_te(element, expr, rt_var_name):
    return finat.TensorFiniteElement(rebuild(element.base_element,
                                             expr, rt_var_name),
                                     element._shape,
                                     transpose=element._transpose)


def composed_map(map1, map2):
    """
    Manually build a :class:`PyOP2.Map` from the iterset of map1 to the
    toset of map2.

    :arg map1: The map with the desired iterset
    :arg map2: The map with the desired toset

    :returns:  The composed map

    Requires that `map1.toset == map2.iterset`.
    Only currently implemented for `map1.arity == 1`
    """
    if map2 is None:
        # Real function space case
        return None
    if map1.toset != map2.iterset:
        raise ValueError("Cannot compose a map where the intermediate sets do not match!")
    if map1.arity != 1:
        raise NotImplementedError("Can only currently build composed maps where map1.arity == 1")
    iterset = map1.iterset
    toset = map2.toset
    arity = map2.arity
    values = map2.values[map1.values].reshape(iterset.size, arity)
    assert values.shape == (iterset.size, arity)
    return op2.Map(iterset, toset, arity, values)


def compose_map_and_cache(map1, map2):
    """
    Retrieve a composed :class:`PyOP2.Map` map from the cache of map1
    using map2 as the cache key. The composed map maps from the iterset
    of map1 to the toset of map2. Calls `composed_map` and caches the
    result on map1 if the composed map is not found.

    :arg map1: The map with the desired iterset from which the result is
        retrieved or cached
    :arg map2: The map with the desired toset

    :returns:  The composed map

    See also `composed_map`.
    """
    cache_key = hash((map2, "composed"))
    try:
        cmap = map1._cache[cache_key]
    except KeyError:
        cmap = composed_map(map1, map2)
        map1._cache[cache_key] = cmap
    return cmap


class GlobalWrapper(object):
    """Wrapper object that fakes a Global to behave like a Function."""
    def __init__(self, glob):
        self.dat = glob
        self.cell_node_map = lambda *arguments: None
        self.ufl_domain = lambda: None
