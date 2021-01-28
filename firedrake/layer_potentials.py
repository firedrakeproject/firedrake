import numpy as np
from firedrake.pointwise_operators import AbstractExternalOperator
from firedrake.utils import cached_property

from firedrake import Function, FunctionSpace, interpolate, Interpolator, \
    SpatialCoordinate, VectorFunctionSpace

from ufl.algorithms.apply_derivatives import VariableRuleset
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.core.multiindex import indices
from ufl.tensors import as_tensor

from pyop2.datatypes import ScalarType


__all__ = ("DoubleLayerPotential", "PytentialLayerOperation",
           "SingleLayerPotential", "VolumePotential",)


# TODO: Provide better description of FMM Kwargs
class PytentialLayerOperation(AbstractExternalOperator):
    r"""
    Evaluates a pytential layer operation on a 2 or 3D mesh
    (with geometric dim == topological dim) which has
    a source and target of co-dimension 1 (the source
    and target must be external boundaries in the mesh).

    IMPORTANT: The pytential normal direction is OPPOSITE that
               of the firedrake normal direction.
               This is not automatically accounted for.
               (Although in :class:`SingleLayerPotential`
                and :class:`DoubleLayerPotential` it is automatically
                accounted for)

    :kwarg operator_data: A map with required keys

        * 'op': The pytential operation (see, for example,
                :class:`SingleLayerPotential` or :class:`DoubleLayerPotential`).
                Must be made up of :class:`pytential.sym.primitives`
                as described in the *expr* argument to
                :function:`pytential.bind`. This is not validated.
        * 'op_shape': The shape of the output of this operator
                      (e.g. (,) for scalar, (3,) for a vector in 3-space,
                       (2,) for a vector in 2-space, etc.)
        * 'density_name': The name of the density function in the
                          pytential operation *op* (see, for example,
                          :class:`SingleLayerPotential` or
                          :class:`DoubleLayerPotential`).
        * 'actx': a :mod:`meshmode` :class:`meshmode.array_context.ArrayContext`
        * 'source_bdy_id' A boundary id representing the source
        * 'target_bdy_id' A boundary id representing the target

        And optional keys

        * 'op_kwargs': (optional) keyword arguments to be passed to the
                       evaluator of the pytential operation. For instance,
                       if a :class:`sumpy.kernel.HelmholtzKernel` is being
                       evaluated with a kappa of 0.5, you might pass
                       `operator_data['op_kwargs'] = {'k': 0.5}`.
                       DO NOT include the density functions in these kwargs,
                       it will be automatically inserted.
        * 'grp_factory': (optional) An interpolatory group factory
            inheriting from :class:`meshmode.discretization.ElementGroupFactory`
            to be used in the intermediate :mod:`meshmode` representation
        * 'qbx_order': As described in :class:`pytential.qbx.QBXLayerPotentialSource`
        * 'fine_order': As described in :class:`pytential.qbx.QBXLayerPotentialSource`
        * 'fmm_order': As described in :class:`pytential.qbx.QBXLayerPotentialSource`

    """

    _external_operator_type = 'GLOBAL'

    def __init__(self, *operands, **kwargs):
        self.operands = operands
        self.kwargs = kwargs

        AbstractExternalOperator.__init__(self, operands[0], **kwargs)

        # Validate input
        print(len(self.operands), self.derivatives)
        # assert self.derivatives == (0,), \
        #     "Derivatives of single layer potential not currently supported: " + str(self.derivatives)
        # from firedrake import Function
        #
        # if not isinstance(operand, Function):
        #     raise TypeError(":arg:`operand` must be of type firedrake.Function"
        #                     ", not %s." % type(operand))
        # if operand.function_space().shape != tuple():
        #     raise ValueError(":arg:`operand` must be a function with shape (),"
        #                      " not %s." % operand.function_space().shape)
        operator_data = kwargs["operator_data"]
        assert isinstance(operator_data, dict)
        required_keys = ('op', 'op_shape', 'density_name',
                         'actx', 'source_bdy_id', 'target_bdy_id')
        optional_keys = ('op_kwargs',
                         'grp_factory', 'qbx_order', 'fmm_order', 'fine_order',)
        permissible_keys = required_keys + optional_keys
        if not all(key in operator_data for key in required_keys):
            raise ValueError("operator_data is missing one of the required "
                             "keys: %s" % required_keys)
        if not all(key in permissible_keys for key in operator_data):
            raise ValueError("operator_data contains an unexpected key. All "
                             "keys must be one of %s." % (permissible_keys,))

    @cached_property
    def _evaluator(self):
        return PytentialLayerOperationEvaluator(self,
                                                *self.operands,
                                                **self.kwargs)

    def _evaluate(self):
        1/0

    def _compute_derivatives(self):
        1/0

    def _evaluate_action(self, *args):
        return self._evaluator._evaluate_action()


class SingleLayerPotential(PytentialLayerOperation):
    r"""
    Layer potential which evaluates to

    .. math::

         f(x)|_{x\in\Gamma} = \int_\Om K(x-y) op(y) \,dx

    where \Gamma is the target boundary id and \Om is the
    source bdy id
    as described in :class:`~firedrake.layer_potentials.LayerPotential`,
    and K is a :class:`sumpy.kernel.Kernel`.
    The function space must have scalar shape.

    :kwarg operator_data: A map as described in
                          :class:`~firedrake.layer_potentials.LayerPotential`
                          except that

        * 'kernel' must be included, and map to a value of type
          :class:`sumpy.kernel.Kernel`
        * 'op' must not be included
        * 'op_shape' must not be included
        * 'density_name' must not be included
    """
    def __init__(self):
        # TODO
        pass


class DoubleLayerPotential(PytentialLayerOperation):
    r"""
    Layer potential which evaluates to

    .. math::

         f(x)|_{x\in\Gamma} = \int_\Om \partial_n K(x-y) op(y) \,dx

    where \Gamma is the target boundary id and \Om is the
    source bdy id
    as described in :class:`~firedrake.layer_potentials.LayerPotential`,
    and K is a :class:`sumpy.kernel.Kernel`.
    The function space must have scalar shape.

    :kwarg operator_data: A map as described in
                          :class:`~firedrake.layer_potentials.LayerPotential`
                          except that

        * 'kernel' must be included, and map to a value of type
          :class:`sumpy.kernel.Kernel`
        * 'op' must not be included
        * 'op_shape' must not be included
        * 'density_name' must not be included
    """
    @cached_property
    def __init__(self):
        # TODO:
        pass


def _get_target_points_and_indices(fspace, boundary_ids):
    """
    Get  the points from the function space which lie on the given boundary
    id as a pytential PointsTarget, and their indices into the
    firedrake function

    :arg fspace: The function space
    :arg boundary_ids: the boundary ids (an int or tuple of ints,
                                         not validated)
    :return: (target_indices, target_points)
    """
    if isinstance(boundary_ids, int):
        boundary_ids = tuple(boundary_ids)
    target_indices = set()
    for marker in boundary_ids:
        target_indices |= set(
            fspace.boundary_nodes(marker, 'topological'))
    target_indices = np.array(list(target_indices), dtype=np.int32)

    target_indices = np.array(target_indices, dtype=np.int32)
    # Get coordinates of nodes
    coords = SpatialCoordinate(fspace.mesh())
    function_space_dim = VectorFunctionSpace(
        fspace.mesh(),
        fspace.ufl_element().family(),
        degree=fspace.ufl_element().degree())

    coords = Function(function_space_dim).interpolate(coords)
    coords = np.real(coords.dat.data)

    target_pts = coords[target_indices]
    # change from [nnodes][ambient_dim] to [ambient_dim][nnodes]
    target_pts = np.transpose(target_pts).copy()
    from pytential.target import PointsTarget
    return (target_indices, PointsTarget(target_pts))


class PytentialLayerOperationEvaluator:
    def __init__(self, vp, *operands, **kwargs):
        self.vp = vp
        operand = operands[0]

        function_space = kwargs["function_space"]
        # Make sure function space is of the appropriate
        # family, and mesh lives in the appropriate dimensions
        if function_space.ufl_element().family() != 'Discontinuous Lagrange':
            raise TypeError("function_space family must be "
                            "'Discontinuous Lagrange', not %s" %
                            function_space.ufl_element().family())
        valid_geo_dims = [2, 3]
        mesh = function_space.mesh()
        if mesh.geometric_dimension() not in valid_geo_dims:
            raise ValueError("function_space.mesh().geometric_dimension() "
                             "%s is not in %s" %
                             (mesh.geometric_dimension(), valid_geo_dims))
        if mesh.geometric_dimension() != mesh.topological_dimension():
            raise ValueError("function_space.mesh().topological_dimension() of "
                             "%s does not equal "
                             "function_space.mesh().geometric_dimension() of %s"
                             % (mesh.topolocial_dimension(),
                                mesh.geometric_dimension()))

        # Create an interpolator from the original operand so that
        # we can re-interpolate it into the space at each evaluation
        # in case it has changed (e.g. cost functional in PDE constrained optimization.

        # Get operator data
        operator_data = kwargs["operator_data"]
        # get op and op-shape
        op = operator_data["op"]
        op_shape = operator_data["op_shape"]
        # Validate operator-shape
        if not isinstance(op_shape, tuple):
            raise TypeError("operator_data['op_shape'] must be of type "
                            "tuple, not %s." % type(op_shape))
        for dim in op_shape:
            if not isinstance(dim, int):
                raise TypeError("'%s' in operator_data['op_shape'] is of "
                                " non-integer type %s." % (dim, type(dim)))
            if dim <= 0:
                raise ValueError("'%s' in operator_data['op_shape'] is "
                                 "not positive." % dim)
        # Get density-name and validate
        density_name = operator_data["density_name"]
        if not isinstance(density_name, str):
            raise TypeError("operator_data['density_name'] must be of type "
                            " str, not '%s'." % type(density_name))
        # Get op kwargs and validate
        op_kwargs = operator_data.get('op_kwargs', {})
        if not isinstance(op_kwargs, dict):
            raise TypeError("operator_data['op_kwargs'] must be of type "
                            " dict, not '%s'." % type(op_kwargs))
        for k in op_kwargs.keys():
            if not isinstance(k, str):
                raise TypeError("Key '%s' in operator_data['op_kwargs'] must "
                                " be of type str, not '%s'." % type(k))
        """
        # Validate kernel is a sumpy kernel
        kernel = operator_data['kernel']
        from sumpy.kernel import Kernel
        if not isinstance(kernel, Kernel):
            raise TypeError("operator_data['kernel'] must be of type "
                            "sumpy.kernel.Kernel, not %s." % type(kernel))
        """

        # Validate actx type
        actx = operator_data['actx']
        from meshmode.array_context import PyOpenCLArrayContext
        if not isinstance(actx, PyOpenCLArrayContext):
            raise TypeError("operator_data['actx'] must be of type "
                            "PyOpenCLArrayContext, not %s." % type(actx))

        source_bdy_id = operator_data["source_bdy_id"]
        target_bdy_id = operator_data["target_bdy_id"]
        # Make sure bdy ids are appropriate types
        if not isinstance(source_bdy_id, int):
            raise TypeError("operator_data['source_bdy_id'] must be of type int,"
                            " not type %s" % type(source_bdy_id))
        if isinstance(target_bdy_id, int):
            target_bdy_id = tuple(target_bdy_id)
        if not isinstance(target_bdy_id, tuple):
            raise TypeError("operator_data['target_bdy_id'] must be an int "
                            " or a tuple of ints, not of type %s" %
                            type(target_bdy_id))
        for bdy_id in target_bdy_id:
            if not isinstance(bdy_id, int):
                raise TypeError("non-integer value '%s' found in "
                                "operator_data['target_bdy_id']" % bdy_id)
        # Make sure bdy ids are actually boundary ids
        valid_ids = function_space.mesh().exterior_facets.unique_markers
        if not set(valid_ids) >= set(target_bdy_id):
            raise ValueError("Invalid target bdy id(s): %s." %
                             set(target_bdy_id) - set(valid_ids))
        if source_bdy_id not in valid_ids:
            raise ValueError("Invalid source bdy id: %s" % source_bdy_id)
        # Make sure bdy ids are disjoint
        if source_bdy_id in target_bdy_id:
            raise NotImplementedError("source and target boundaries must be "
                                      "disjoint")

        degree = function_space.ufl_element().degree()
        # Get group factory, if any
        grp_factory = operator_data.get('grp_factory', None)
        # validate grp_factory
        from meshmode.discretization.poly_element import ElementGroupFactory
        if grp_factory is not None:
            if not isinstance(grp_factory, ElementGroupFactory):
                raise TypeError("operator_data['grp_factory'] must be *None*"
                                " or of type ElementGroupFactory, not %s." %
                                type(grp_factory))
        # Set defaults for qbx kwargs
        qbx_order = kwargs.get('qbx_order', degree+2)
        fine_order = kwargs.get('fine_order', 4 * degree)
        fmm_order = kwargs.get('fmm_order', 6)
        # Validate qbx kwargs
        for var, name in zip([qbx_order, fine_order, fmm_order],
                             ['qbx_order', 'fine_order', 'fmm_order']):
            if not isinstance(var, int):
                raise TypeError("operator_data['%s'] must be of type int, "
                                "not %s." % (name, type(var)))
            if not var > 0:
                raise ValueError("operator_data['%s'] = %s is not positive."
                                 % (name, var))

        qbx_kwargs = {'qbx_order': qbx_order,
                      'fine_order': fine_order,
                      'fmm_order': fmm_order,
                      'fmm_backend': 'fmmlib',
                      }
        # }}}

        # Build connection into meshmode
        from meshmode.discretization.poly_element import \
            PolynomialRecursiveNodesElementGroupFactory
        from meshmode.interop.firedrake import build_connection_from_firedrake
        if grp_factory is None:
            grp_factory = PolynomialRecursiveNodesElementGroupFactory(degree)
        meshmode_connection = build_connection_from_firedrake(
            actx, function_space, grp_factory=grp_factory,
            restrict_to_boundary=source_bdy_id)

        # build connection meshmode near src boundary -> src boundary inside meshmode
        from meshmode.discretization.connection import make_face_restriction
        src_bdy_connection = make_face_restriction(actx,
                                                   meshmode_connection.discr,
                                                   grp_factory,
                                                   source_bdy_id)

        # Build QBX
        from pytential.qbx import QBXLayerPotentialSource
        qbx = QBXLayerPotentialSource(src_bdy_connection.to_discr, **qbx_kwargs)
        # Get target, and store the firedrake indices of the points
        target_indices, target = _get_target_points_and_indices(function_space,
                                                                target_bdy_id)

        # Bind pytential operator
        from pytential import bind
        self.pyt_op = bind((qbx, target), op)
        self.density_name = density_name
        self.op_kwargs = op_kwargs

        # Store attributes that we may need later
        # self.ufl_operands = operands
        self.actx = actx
        self.op_shape = op_shape
        self.meshmode_connection = meshmode_connection
        self.src_bdy_connection = src_bdy_connection
        self.target_indices = target_indices
        # so we don't have to keep making a fd function during conversion
        # from meshmode. AbstractExternalOperator s aren't functions,
        # so can't use *self*
        self.fd_pot = Function(function_space)
        # initialize to zero so that only has values on target boundary
        self.fd_pot.dat.data[:] = 0.0


    def _evaluate(self):
        operand, = self.vp.ufl_operands
        operand_discrete = interpolate(operand, self.vp.function_space())

        # pass operand into a meshmode DOFArray
        from meshmode.dof_array import flatten
        meshmode_src_vals = \
            self.meshmode_connection.from_firedrake(operand_discrete, actx=self.actx)
        # pass operand onto boundary
        meshmode_src_vals_on_bdy = self.src_bdy_connection(meshmode_src_vals)

        # Evaluate pytential potential
        self.op_kwargs[self.density_name] = meshmode_src_vals_on_bdy
        # FIXME : Make sure this conversion works for non-scalar shapes!!!
        self.fd_pot.dat.data[self.target_indices] = self.pyt_op(**self.op_kwargs)
        # Store in vp
        self.vp.dat.data[:] = self.fd_pot.dat.data[:]
        # Return evaluated potential
        return self.fd_pot

    def _compute_derivatives(self):
        # TODO : Support derivatives
        return self._evaluate()

    def _evaluate_action(self, *args):
        # From tests/pointwiseoperator/test_point_expr.py
        # https://github.com/firedrakeproject/firedrake/blob/c0d9b592f587fa8c7437f690da7a6595f6804c1b/tests/pointwiseoperator/test_point_expr.py  # noqa
        if len(args) == 0:
            # Evaluate the operator
            return self._evaluate()

        # Evaluate the Jacobian/Hessian action
        operands = self.ufl_operands
        operator = self._compute_derivatives()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            var = VariableRuleset(self.ufl_operands[0])
            expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)

        for arg in args:
            mi = indices(len(expr.ufl_shape))
            aa = mi
            bb = mi[-len(arg.ufl_shape):]
            expr = arg[bb] * expr[aa]
            mi_tensor = tuple(e for e in mi if not (e in aa and e in bb))
            if len(expr.ufl_free_indices):
                expr = as_tensor(expr, mi_tensor)
        return self.interpolate(expr)


class VolumePotential(AbstractExternalOperator):
    r"""
    Evaluates to

    .. math::

         f(x) = \int_\Om K(x-y) op(y) \,dx

    where op is a function-type living in *function_space*, \Om is the
    3-D mesh on which *function_space* lives, and K
    a :mod:`sumpy` kernel.

    :arg operator_data: A map with required keys

        * 'kernel': a :class:`sumpy.kernel.Kernel` (*K* in the math above)
        * 'kernel_type': A string describing the kernel, e.g. ``"Laplace"``
        * 'cl_ctx': a :mod:`pyopencl` computing context
        * 'queue': a :mod:`pyopencl` command queue compatible with the
                   computing context
        * 'nlevels': The desired number of levels to be used by
                      :mod:`volumential`
        * 'm_order': multipole order used by the expansion
        * 'dataset_filename': the filename to pass to
            :class:`volumential.table_manager.NearFieldInteractionTableManager`

        And optional keys

        * 'grp_factory': (optional) An interpolatory group factory
            inheriting from :class:`meshmode.discretization.ElementGroupFactory`
            to be used in the intermediate :mod:`meshmode` representation
        * 'q_order': (optional) The desired :mod:`volumential` quadrature
                     order, defaults to *function_space*'s degree
        * 'force_direct_evaluation': (optional) As in
                     :func:`volumential.volume_fmm.drive_volume_fmm`.
                     Defaults to *False*
        * 'fmm_kwargs': (optional) A dictionary of kwargs
                 to pass to :func:`volumential.volume_fmm.drive_fmm`.
        * 'root_extent': (optional) the root extent to pass to
            :class:`volumential.table_manager.NearFieldInteractionTableManager`
            (defaults to 1)
        * 'table_compute_method': (optional) used by
            :class:`volumential.table_manager.NearFieldInteractionTableManager`
            (defaults to "DrosteSum")
        * 'table_kwargs': (optional) passed to
            :class:`volumential.table_manager.NearFieldInteractionTableManager`
            's *get_table* method

    """

    _external_operator_type = 'GLOBAL'

    def __init__(self, *operands, **kwargs):
        self.operands = operands
        self.kwargs = kwargs

        AbstractExternalOperator.__init__(self, operands[0], **kwargs)

        # Validate input
        print(len(self.operands), self.derivatives)
        # assert self.derivatives == (0,), \
        #     "Derivatives of volume potential not currently supported: " + str(self.derivatives)
        # from firedrake import Function
        #
        # if not isinstance(operand, Function):
        #     raise TypeError(":arg:`operand` must be of type firedrake.Function"
        #                     ", not %s." % type(operand))
        # if operand.function_space().shape != tuple():
        #     raise ValueError(":arg:`operand` must be a function with shape (),"
        #                      " not %s." % operand.function_space().shape)
        operator_data = kwargs["operator_data"]
        assert isinstance(operator_data, dict)
        required_keys = ('kernel', 'kernel_type', 'cl_ctx', 'queue', 'nlevels',
                         'm_order', 'dataset_filename')
        optional_keys = ('grp_factory', 'q_order', 'force_direct_evaluation',
                         'fmm_kwargs', 'root_extent',
                         'table_compute_method', 'table_kwargs',
                         )
        permissible_keys = required_keys + optional_keys
        if not all(key in operator_data for key in required_keys):
            raise ValueError("operator_data is missing one of the required "
                             "keys: %s" % required_keys)
        if not all(key in permissible_keys for key in operator_data):
            raise ValueError("operator_data contains an unexpected key. All "
                             "keys must be one of %s." % (permissible_keys,))

    @cached_property
    def _evaluator(self):
        return VolumePotentialEvaluator(self, *self.operands, **self.kwargs)

    def _evaluate(self):
        1/0

    def _compute_derivatives(self):
        1/0

    def _evaluate_action(self, *args):
        return self._evaluator._evaluate_action()


class VolumePotentialEvaluator:
    def __init__(self, vp, *operands, **kwargs):
        self.vp = vp
        operand = operands[0]

        function_space = kwargs["function_space"]

        # Create an interpolator from the original operand so that
        # we can re-interpolate it into the space at each evaluation
        # in case it has changed (e.g. cost functional in PDE constrained optimization.

        operator_data = kwargs["operator_data"]
        kernel = operator_data['kernel']
        kernel_type = operator_data['kernel_type']
        cl_ctx = operator_data['cl_ctx']
        queue = operator_data['queue']
        nlevels = operator_data['nlevels']
        m_order = operator_data['m_order']
        dataset_filename = operator_data['dataset_filename']
        degree = function_space.ufl_element().degree()
        grp_factory = operator_data.get('grp_factory', None)
        q_order = operator_data.get('q_order', degree)
        force_direct_evaluation = operator_data.get('force_direct_evaluation',
                                                    False)
        fmm_kwargs = operator_data.get('fmm_kwargs', {})
        root_extent = operator_data.get("root_extent", 1)
        table_compute_method = operator_data.get('table_compute_method',
                                                 'DrosteSum')
        table_kwargs = operator_data.get('table_kwargs', {})

        from sumpy.kernel import Kernel
        if not isinstance(kernel, Kernel):
            raise TypeError("operator_data['kernel'] must be of type "
                            "sumpy.kernel.Kernel, not %s." % type(kernel))
        if not isinstance(nlevels, int):
            raise TypeError("operator_data['nlevels'] must be of type int, "
                            "not %s." % type(nlevels))
        if not isinstance(m_order, int):
            raise TypeError("operator_data['m_order'] must be of type int, "
                            "not %s." % type(m_order))
        if not isinstance(q_order, int):
            raise TypeError("operator_data['q_order'] must be of type int, "
                            "not %s." % type(q_order))

        # Build connection into meshmode
        from meshmode.interop.firedrake import build_connection_from_firedrake
        from meshmode.array_context import PyOpenCLArrayContext
        actx = PyOpenCLArrayContext(queue)
        meshmode_connection = build_connection_from_firedrake(
            actx, function_space, grp_factory=grp_factory)

        # Build connection from meshmode into volumential
        # (following https://gitlab.tiker.net/xywei/volumential/-/blob/fe2c3e7af355d5c527060e783237c124c95397b5/test/test_interpolation.py#L72 ) # noqa : E501
        from volumential.geometry import (
            BoundingBoxFactory, BoxFMMGeometryFactory)
        from volumential.interpolation import ElementsToSourcesLookupBuilder
        dim = function_space.mesh().geometric_dimension()
        bbox_fac = BoundingBoxFactory(dim=dim)
        bboxfmm_fac = BoxFMMGeometryFactory(
            cl_ctx, dim=dim, order=q_order,
            nlevels=nlevels, bbox_getter=bbox_fac,
            expand_to_hold_mesh=meshmode_connection.discr.mesh,
            mesh_padding_factor=0.0)
        boxgeo = bboxfmm_fac(queue)
        elt_to_src_lookup_fac = ElementsToSourcesLookupBuilder(
            cl_ctx, tree=boxgeo.tree, discr=meshmode_connection.discr)
        elt_to_src_lookup, evt = elt_to_src_lookup_fac(queue)

        # Build connection from volumential into meshmode
        from volumential.interpolation import LeavesToNodesLookupBuilder
        leaves_to_node_lookup_fac = LeavesToNodesLookupBuilder(
            cl_ctx, trav=boxgeo.trav, discr=meshmode_connection.discr)
        leaves_to_node_lookup, evt = leaves_to_node_lookup_fac(queue)

        # Create near-field table in volumential
        from volumential.table_manager import NearFieldInteractionTableManager
        table_manager = NearFieldInteractionTableManager(
            dataset_filename, root_extent=root_extent, queue=queue)

        nftable, _ = table_manager.get_table(
            dim,
            kernel_type,
            q_order,
            compute_method=table_compute_method,
            queue=queue,
            **table_kwargs)

        # Create kernel wrangler in volumential
        from sumpy.expansion import DefaultExpansionFactory
        out_kernels = [kernel]
        expn_factory = DefaultExpansionFactory()
        local_expn_class = expn_factory.get_local_expansion_class(kernel)
        mpole_expn_class = expn_factory.get_multipole_expansion_class(kernel)

        from functools import partial
        from volumential.expansion_wrangler_fpnd import (
            FPNDExpansionWrangler, FPNDExpansionWranglerCodeContainer)
        wrangler_cc = FPNDExpansionWranglerCodeContainer(
            cl_ctx,
            partial(mpole_expn_class, kernel),
            partial(local_expn_class, kernel),
            out_kernels)
        wrangler = FPNDExpansionWrangler(
            code_container=wrangler_cc,
            queue=queue,
            tree=boxgeo.tree,
            near_field_table=nftable,
            dtype=ScalarType,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: m_order,
            quad_order=q_order)

        # Store attributes that we may need later
        #self.ufl_operands = operands
        self.actx = actx
        self.queue = queue
        self.sumpy_kernel = kernel
        self.meshmode_connection = meshmode_connection
        self.volumential_boxgeo = boxgeo
        self.to_volumential_lookup = elt_to_src_lookup
        self.from_volumential_lookup = leaves_to_node_lookup
        self.force_direct_evaluation = force_direct_evaluation
        self.fmm_kwargs = fmm_kwargs
        self.expansion_wrangler = wrangler
        # so we don't have to keep making a fd function during conversion
        # from meshmode. AbstractExternalOperator s aren't functions,
        # so can't use *self*
        self.fd_pot = Function(function_space)

    def _evaluate(self):
        operand, = self.vp.ufl_operands
        operand_discrete = interpolate(operand, self.vp.function_space())

        # pass operand into a meshmode DOFArray
        from meshmode.dof_array import flatten
        meshmode_src_vals = \
            self.meshmode_connection.from_firedrake(operand_discrete, actx=self.actx)
        meshmode_src_vals = flatten(meshmode_src_vals)

        if 1:
            alpha = 40
            dim = 2

        if 1:
            mm_discr = self.meshmode_connection.discr
            from meshmode.dof_array import thaw, flat_norm
            x = flatten(thaw(self.actx, mm_discr.nodes()[0]))
            y = flatten(thaw(self.actx, mm_discr.nodes()[1]))
            import pyopencl.clmath as clmath
            import pyopencl.array as cla
            norm2 = (x-0.5)**2 + (y-0.5)**2
            ref_src = -(4 * alpha ** 2 * norm2 - 2 * dim * alpha) * clmath.exp(
                    -alpha * norm2)
            denom = np.max(np.abs(ref_src.get()))
            err = np.abs((ref_src - meshmode_src_vals).get())
            print("Fdrake -> meshmode error:", np.max(err)/denom)

        # pass flattened operand into volumential interpolator
        from volumential.interpolation import interpolate_from_meshmode
        volumential_src_vals = \
            interpolate_from_meshmode(self.queue,
                                      meshmode_src_vals,
                                      self.to_volumential_lookup,
                                      order="user")  # user order is more intuitive

        if 1:
            # check source density against known value
            # x = self.volumential_boxgeo.tree.sources[0].with_queue(self.queue)
            # y = self.volumential_boxgeo.tree.sources[1].with_queue(self.queue)
            x, y = self.volumential_boxgeo.nodes
            import pyopencl.clmath as clmath
            import pyopencl.array as cla
            norm2 = (x-0.5)**2 + (y-0.5)**2
            ref_src = -(4 * alpha ** 2 * norm2 - 2 * dim * alpha) * clmath.exp(
                    -alpha * norm2)
            denom = np.max(np.abs(ref_src.get()))
            err = np.abs((ref_src-volumential_src_vals).get())
            print("Fdrake -> volumential error:", np.max(err)/denom)

        # evaluate volume fmm
        from volumential.volume_fmm import drive_volume_fmm
        pot, = drive_volume_fmm(
            self.volumential_boxgeo.traversal,
            self.expansion_wrangler,
            volumential_src_vals * self.volumential_boxgeo.weights,
            volumential_src_vals,
            direct_evaluation=self.force_direct_evaluation,
            reorder_sources=True,
            **self.fmm_kwargs)

        if 1:
            # check potential against known value
            ref_pot = clmath.exp(-alpha * norm2)
            denom = np.max(np.abs(ref_pot.get()))
            err = np.abs((ref_pot-pot).get())
            print("volumential potential error:", np.max(err)/denom)

            # evaluate volume fmm with ref_src
            pot_w_ref_src, = drive_volume_fmm(
                self.volumential_boxgeo.traversal,
                self.expansion_wrangler,
                ref_src * self.volumential_boxgeo.weights,
                ref_src,
                direct_evaluation=self.force_direct_evaluation,
                reorder_sources=True,
                **self.fmm_kwargs)

            err = np.abs((ref_pot-pot_w_ref_src).get())
            print("volumential potential w/t ref src error:", np.max(err)/denom)

        # pass volumential back to meshmode DOFArray
        from volumential.interpolation import interpolate_to_meshmode
        from meshmode.dof_array import unflatten
        user_order = self.fmm_kwargs.get("reorder_potentials", True)
        if user_order:
            order = "user"
        else:
            order = "tree"
        meshmode_pot_vals = interpolate_to_meshmode(self.queue,
                                                    pot,
                                                    self.from_volumential_lookup,
                                                    order=order)
        meshmode_pot_vals = unflatten(self.actx,
                                      self.meshmode_connection.discr,
                                      meshmode_pot_vals)
        # get meshmode data back as firedrake function
        self.meshmode_connection.from_meshmode(
            meshmode_pot_vals,
            out=self.fd_pot)
        self.vp.dat.data[:] = self.fd_pot.dat.data[:]

        return self.fd_pot

    def _compute_derivatives(self):
        # TODO : Support derivatives
        return self._evaluate()

    def _evaluate_action(self, *args):
        # From tests/pointwiseoperator/test_point_expr.py
        # https://github.com/firedrakeproject/firedrake/blob/c0d9b592f587fa8c7437f690da7a6595f6804c1b/tests/pointwiseoperator/test_point_expr.py  # noqa
        if len(args) == 0:
            # Evaluate the operator
            return self._evaluate()

        # Evaluate the Jacobian/Hessian action
        operands = self.ufl_operands
        operator = self._compute_derivatives()
        expr = as_ufl(operator(*operands))
        if expr.ufl_shape == () and expr != 0:
            var = VariableRuleset(self.ufl_operands[0])
            expr = expr*var._Id
        elif expr == 0:
            return self.assign(expr)

        for arg in args:
            mi = indices(len(expr.ufl_shape))
            aa = mi
            bb = mi[-len(arg.ufl_shape):]
            expr = arg[bb] * expr[aa]
            mi_tensor = tuple(e for e in mi if not (e in aa and e in bb))
            if len(expr.ufl_free_indices):
                expr = as_tensor(expr, mi_tensor)
        return self.interpolate(expr)

