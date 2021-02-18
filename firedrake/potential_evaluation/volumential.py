from firedrake.potential_evaluation import PotentialEvaluationLibraryConnection
from firedrake.potential_evaluation.pytential import \
    MeshmodeConnection


class VolumentialConnection(PotentialEvaluationLibraryConnection):
    """
    Build a connection into :mod:`volumential`
    """
    def __init__(self,
                 potential_source_and_target,
                 function_space,
                 cl_ctx,
                 queue,
                 box_fmm_geometry_kwargs,
                 reorder_potentials,
                 meshmode_connection_kwargs,
                 warn_if_cg=True):
        """
        For other args see :class:`PotentialEvaluationLibraryConnection`.
        The *potential_source_and_target* must have a 3D
        source, 3D target, and 3D mesh of co-dimension 0.

        :arg cl_ctx: A :class:`pyopencl.Context`
        :arg queue: A :class:`pyopencl.CommandQueue`
        :arg box_fmm_geometry_kwargs: kwargs passed to
            :class:`volumential.geometry.BoxFMMGeometryFactory`.
            Must include the key *'nlevels'*
        :arg reorder_potentials: *True* iff the volumential FMM
            is reordering potentials. *False* otherwise.
        :arg meshmode_connection_kwargs: A dict passes as kwargs
            to :func:`meshmode.interop.build_connection_from_firedrake`.
            :mod:`meshmode` is used an intermediate level between
            :mod:`firedrake` and :mod:`volumential`.
        """
        # super
        PotentialEvaluationLibraryConnection.__init__(
            self,
            function_space,
            potential_source_and_target,
            warn_if_cg=warn_if_cg)

        # validate that sources, target, and mesh are of right dimension
        if potential_source_and_target.get_source_dimension() != 3:
            raise ValueError("potential source must have dimension 3")
        if potential_source_and_target.get_target_dimension() != 3:
            raise ValueError("potential target must have dimension 3")
        if potential_source_and_target.mesh().geometric_dimension() != 3:
            raise ValueError("mesh must have geometric dimension 3")
        if potential_source_and_target.mesh().topological_dimension() != 3:
            raise ValueError("mesh must have topological dimension 3")

        # FIXME : allow subdomain regions
        if potential_source_and_target.get_source_id() != "everywhere":
            raise NotImplementedError("subdomain sources not implemented")
        if potential_source_and_target.get_target_id() != "everywhere":
            raise NotImplementedError("subdomain targets not implemented")

        # validate cl_ctx argument
        from pyopencl import Context
        if not isinstance(cl_ctx, Context):
            raise TypeError("cl_ctx must be of type Context, not "
                            "%s." % type(cl_ctx))

        # validate queue argument
        from pyopencl import CommandQueue
        if not isinstance(queue, CommandQueue):
            raise TypeError("queue must be of type CommandQueue, not "
                            "%s." % type(queue))

        # validate reorder_potentials
        if not isinstance(reorder_potentials, bool):
            raise TypeError("reorder_potentials must be of type bool, not "
                            "%s." % type(reorder_potentials))

        # validate box_fmm_geometry_kwargs
        if not isinstance(box_fmm_geometry_kwargs, dict):
            raise TypeError("box_fmm_geometry_kwargs must be of "
                            "type dict, not '%s'." %
                            type(box_fmm_geometry_kwargs))
        if 'nlevels' not in box_fmm_geometry_kwargs:
            raise ValueError("box_fmm_geometry_kwargs missing required keyword"
                             " 'nlevels'.")

        # validate meshmode_connection_kwargs
        if meshmode_connection_kwargs is None:
            meshmode_connection_kwargs = {}
        if not isinstance(meshmode_connection_kwargs, dict):
            raise TypeError("meshmode_connection_kwargs must be *None* or of "
                            "type dict, not '%s'." %
                            type(meshmode_connection_kwargs))

        # Build intermediate connection into meshmode
        from meshmode.interop.firedrake import build_connection_from_firedrake
        from meshmode.array_context import PyOpenCLArrayContext
        actx = PyOpenCLArrayContext(queue)
        meshmode_connection = MeshmodeConnection(
                function_space,
                potential_source_and_target,
                actx,
                warn_if_cg=warn_if_cg,
                meshmode_connection_kwargs=meshmode_connection_kwargs)

        # Build connection from meshmode into volumential
        # (following https://gitlab.tiker.net/xywei/volumential/-/blob/fe2c3e7af355d5c527060e783237c124c95397b5/test/test_interpolation.py#L72 ) # noqa : E501
        from volumential.geometry import (
            BoundingBoxFactory, BoxFMMGeometryFactory)
        from volumential.interpolation import ElementsToSourcesLookupBuilder
        dim = function_space.mesh().geometric_dimension()
        bboxfmm_fac = BoxFMMGeometryFactory(cl_ctx, dim=dim,
                                            **box_fmm_geometry_kwargs)
        boxgeo = bboxfmm_fac(queue)
        elt_to_src_lookup_fac = ElementsToSourcesLookupBuilder(
            cl_ctx, tree=boxgeo.tree, discr=meshmode_connection.discr)
        elt_to_src_lookup, evt = elt_to_src_lookup_fac(queue)

        # Build connection from volumential into meshmode
        from volumential.interpolation import LeavesToNodesLookupBuilder
        leaves_to_node_lookup_fac = LeavesToNodesLookupBuilder(
            cl_ctx, trav=boxgeo.trav, discr=meshmode_connection.discr)
        leaves_to_node_lookup, evt = leaves_to_node_lookup_fac(queue)

        # Figure out whether or not conversion from volumential -> meshmode
        # should user order or tree order
        order = "user" if reorder_potentials else "tree"

        # Store maps for conversion to/from volumential
        self.meshmode_connection = meshmode_connection
        self.to_volumential_lookup = elt_to_src_lookup
        self.from_volumential_lookup = leaves_to_node_lookup
        self.from_volumential_order = order
        # store necessary computing contextx
        self.queue = queue
        self.actx = actx

    def from_firedrake(self, density):
        # pass operand into a meshmode DOFArray
        from meshmode.dof_array import flatten
        meshmode_density = self.meshmode_connection.from_firedrake(density)
        meshmode_density = flatten(meshmode_density)

        # pass flattened operand into volumential interpolator
        from volumential.interpolation import interpolate_from_meshmode
        volumential_density = \
            interpolate_from_meshmode(self.queue,
                                      meshmode_density,
                                      self.to_volumential_lookup,
                                      order="user")  # user order is more intuitive
        return volumential_density

    def to_firedrake(self, evaluated_potential, out=None):
        # pass volumential back to meshmode DOFArray
        from volumential.interpolation import interpolate_to_meshmode
        from meshmode.dof_array import unflatten
        meshmode_pot_vals = \
            interpolate_to_meshmode(self.queue,
                                    evaluated_potential,
                                    self.from_volumential_lookup,
                                    order=self.from_volumential_order)
        meshmode_pot_vals = unflatten(self.actx,
                                      self.meshmode_connection.discr,
                                      meshmode_pot_vals)
        # get meshmode data back as firedrake function
        return self.meshmode_connection.to_firedrake(meshmode_pot_vals, out=self)

def VolumentialOperation(density, unbound_op, potential_src_and_tgt,
                         function_space=None):
    pass


# TODO: Merge this into VolumentialOperation function
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


        # Build a broken version of input function space
        fs = kwargs["function_space"]
        fsdg = FunctionSpace(fs.mesh(), "DG", fs.ufl_element().degree())

        self.evaluator_kwargs = kwargs.copy()
        self.evaluator_kwargs["function_space"] = fsdg

        # Validate input
        # print(len(self.operands), self.derivatives)
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
        return VolumePotentialEvaluator(self, *self.operands, **self.evaluator_kwargs)

    def _evaluate(self, continuity_tolerance=None):
        1/0

    def _compute_derivatives(self, continuity_tolerance=None):
        1/0

    def _evaluate_action(self, *args, continuity_tolerance=None):
        return self._evaluator._evaluate_action(continuity_tolerance=continuity_tolerance)

    def _evaluate_adjoint_action(self, *args, continuity_tolerance=None):
        return self._evaluator._evaluate_action(continuity_tolerance=continuity_tolerance)

    def evaluate_adj_component_control(self, x, idx):
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        dN = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        return dN._evaluate_adjoint_action((x.function,)).vector()

    def evaluate_adj_component(self, x, idx):
        print(x, type(x))
        1/0


class VolumePotentialEvaluator:
    def __init__(self, vp, *operands, **kwargs):
        self.vp = vp
        operand = operands[0]

        cg_function_space = vp.function_space()
        function_space = kwargs["function_space"]
        self.dg_function_space = function_space

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
        # self.ufl_operands = operands
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
        self.fd_pot = Function(self.dg_function_space)
        self.fd_pot_cg = Function(vp.function_space())

    def _evaluate(self, continuity_tolerance=None):
        """
        :arg continuity_tolerance: If *None* then ignored. Otherwise, a floating
            point value. If the function space associated to this operator
            is a continuous one and *continuity_tolerance* is a float, it is
            verified that,
            during the conversion of the evaluated potential
            from meshmode's discontinuous function space representation into
            firedrake's continuous one, the potential's value at
            a firedrake node is within *continuity_tolerance* of its
            value at any duplicated firedrake node
        """
        operand, = self.vp.ufl_operands
        operand_discrete = interpolate(operand, self.dg_function_space)

        # pass operand into a meshmode DOFArray
        from meshmode.dof_array import flatten
        meshmode_src_vals = \
            self.meshmode_connection.from_firedrake(operand_discrete, actx=self.actx)
        meshmode_src_vals = flatten(meshmode_src_vals)

        # if 1:
        #     alpha = 40
        #     dim = 2

        # if 1:
        #     mm_discr = self.meshmode_connection.discr
        #     from meshmode.dof_array import thaw, flat_norm
        #     x = flatten(thaw(self.actx, mm_discr.nodes()[0]))
        #     y = flatten(thaw(self.actx, mm_discr.nodes()[1]))
        #     import pyopencl.clmath as clmath
        #     import pyopencl.array as cla
        #     norm2 = (x-0.5)**2 + (y-0.5)**2
        #     ref_src = -(4 * alpha ** 2 * norm2 - 2 * dim * alpha) * clmath.exp(
        #             -alpha * norm2)
        #     denom = np.max(np.abs(ref_src.get()))
        #     err = np.abs((ref_src - meshmode_src_vals).get())
        #     print("Fdrake -> meshmode error:", np.max(err)/denom)

        # pass flattened operand into volumential interpolator
        from volumential.interpolation import interpolate_from_meshmode
        volumential_src_vals = \
            interpolate_from_meshmode(self.queue,
                                      meshmode_src_vals,
                                      self.to_volumential_lookup,
                                      order="user")  # user order is more intuitive

        # if 1:
        #     # check source density against known value
        #     # x = self.volumential_boxgeo.tree.sources[0].with_queue(self.queue)
        #     # y = self.volumential_boxgeo.tree.sources[1].with_queue(self.queue)
        #     x, y = self.volumential_boxgeo.nodes
        #     import pyopencl.clmath as clmath
        #     import pyopencl.array as cla
        #     norm2 = (x-0.5)**2 + (y-0.5)**2
        #     ref_src = -(4 * alpha ** 2 * norm2 - 2 * dim * alpha) * clmath.exp(
        #             -alpha * norm2)
        #     denom = np.max(np.abs(ref_src.get()))
        #     err = np.abs((ref_src-volumential_src_vals).get())
        #     print("Fdrake -> volumential error:", np.max(err)/denom)

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

        # if 1:
        #     # check potential against known value
        #     ref_pot = clmath.exp(-alpha * norm2)
        #     denom = np.max(np.abs(ref_pot.get()))
        #     err = np.abs((ref_pot-pot).get())
        #     print("volumential potential error:", np.max(err)/denom)

        #     # evaluate volume fmm with ref_src
        #     pot_w_ref_src, = drive_volume_fmm(
        #         self.volumential_boxgeo.traversal,
        #         self.expansion_wrangler,
        #         ref_src * self.volumential_boxgeo.weights,
        #         ref_src,
        #         direct_evaluation=self.force_direct_evaluation,
        #         reorder_sources=True,
        #         **self.fmm_kwargs)

        #     err = np.abs((ref_pot-pot_w_ref_src).get())
        #     print("volumential potential w/t ref src error:", np.max(err)/denom)

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

        self.fd_pot_cg = project(self.fd_pot, self.vp.function_space())

        self.vp.dat.data[:] = self.fd_pot_cg.dat.data[:]

        return self.fd_pot_cg

    def _compute_derivatives(self, continuity_tolerance=None):
        # TODO : Support derivatives
        return self._evaluate(continuity_tolerance=continuity_tolerance)

    def _evaluate_action(self, *args, continuity_tolerance=None):
        # From tests/pointwiseoperator/test_point_expr.py
        # https://github.com/firedrakeproject/firedrake/blob/c0d9b592f587fa8c7437f690da7a6595f6804c1b/tests/pointwiseoperator/test_point_expr.py  # noqa
        if len(args) == 0:
            # Evaluate the operator
            return self._evaluate(continuity_tolerance=continuity_tolerance)

        # Evaluate the Jacobian/Hessian action
        operands = self.ufl_operands
        operator = self._compute_derivatives(continuity_tolerance=continuity_tolerance)
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
