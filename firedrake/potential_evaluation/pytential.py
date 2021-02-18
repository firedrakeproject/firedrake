from firedrake.potential_evaluation import \
    Potential, PotentialEvaluationLibraryConnection
from firedrake import project


class MeshmodeConnection(PotentialEvaluationLibraryConnection):
    """
    Build a connection into :mod:`meshmode`
    """
    def __init__(self, function_space, potential_source_and_target, actx,
                 warn_if_cg=True,
                 meshmode_connection_kwargs=None):
        """
        For other args see :class:`PotentialEvaluationLibraryConnection`

        :arg actx: a :class:`meshmode.array_context.PyOpenCLArrayContext`
           used for :func:`meshmode.interop.build_connection_from_firedrake`
           and for conversion by the
           :class:`meshmode.interop.FiredrakeConnection`.
        :arg meshmode_connection_kwargs: A dict passed as kwargs
            to :func:`meshmode.interop.build_connection_from_firedrake`.
            Must not include 'restrict_to_boundary'
        """
        PotentialEvaluationLibraryConnection.__init__(
            self,
            function_space,
            potential_source_and_target,
            warn_if_cg=warn_if_cg)

        # FIXME : allow subdomain regions
        tdim = potential_source_and_target.mesh.topological_dimension()
        if potential_source_and_target.get_source_dim() != tdim:
            if potential_source_and_target.get_source_id() != "everywhere":
                raise NotImplementedError("subdomain sources not implemented")
        if potential_source_and_target.get_target_dim() != tdim:
            if potential_source_and_target.get_target_id() != "everywhere":
                raise NotImplementedError("subdomain targets not implemented")

        # validate actx
        from meshmode.array_context import PyOpenCLArrayContext
        if not isinstance(actx, PyOpenCLArrayContext):
            raise TypeError("actx must be of type PyOpenCLArrayContext, not "
                            "%s." % type(actx))

        # validate meshmode_connection_kwargs
        if meshmode_connection_kwargs is None:
            meshmode_connection_kwargs = {}
        if not isinstance(meshmode_connection_kwargs, dict):
            raise TypeError("meshmode_connection_kwargs must be *None* or of "
                            "type dict, not '%s'." %
                            type(meshmode_connection_kwargs))
        if 'restrict_to_boundary' in meshmode_connection_kwargs:
            raise ValueError("'restrict_to_boundary' must not be set by "
                             "meshmode_connection_kwargs")

        # build a meshmode connection for the source
        src_bdy = None
        if potential_source_and_target.get_source_dim() != tdim:
            # sanity check
            assert potential_source_and_target.get_source_dim() == tdim - 1
            src_bdy = potential_source_and_target.get_source_id()

        from meshmode.interop.firedrake import build_connection_from_firedrake
        src_conn = build_connection_from_firedrake(actx,
                                                   self.dg_function_space,
                                                   restrict_to_boundary=src_bdy,
                                                   **meshmode_connection_kwargs)
        # If source is a boundary, build a connection to restrict it to the
        # boundary
        restrict_conn = None
        if src_bdy is not None:
            # convert "everywhere" to meshmode BTAG_ALL
            meshmode_src_bdy = src_bdy
            if meshmode_src_bdy == "everywhere":
                from meshmode.mesh import BTAG_ALL
                meshmode_src_bdy = BTAG_ALL
            # get group factory
            order = self.dg_function_space.degree()
            from meshmode.discretization.poly_element import \
                PolynomialRecursiveNodesGroupFactory
            default_grp_factory = \
                PolynomialRecursiveNodesGroupFactory(order, 'lgl')
            grp_factory = meshmode_connection_kwargs.get('grp_factory',
                                                         default_grp_factory)
            from meshmode.discretization.connection import make_face_restriction
            restrict_conn = make_face_restriction(actx,
                                                  src_conn.discr,
                                                  grp_factory,
                                                  meshmode_src_bdy)

        # Build a meshmode connection for the target
        tgt_bdy = None
        if potential_source_and_target.get_target_dim() != tdim:
            # sanity check
            assert potential_source_and_target.get_target_dim() == tdim - 1
            tgt_bdy = potential_source_and_target.get_target_id()

        # Can we re-use the source connection?
        src_dim = potential_source_and_target.get_source_dim()
        tgt_dim = potential_source_and_target.get_target_dim()
        if tgt_bdy == src_bdy and src_dim == tgt_dim:
            tgt_conn = src_conn
        else:
            # If not, build a new one
            tgt_conn = build_connection_from_firedrake(
                actx,
                self.dg_function_space,
                restrict_to_boundary=tgt_bdy,
                **meshmode_connection_kwargs)

        # store computing context
        self.actx = actx
        # store connections
        self.source_to_meshmode_connection = src_conn
        self.restrict_source_to_boundary = restrict_conn
        self.target_to_meshmode_connection = tgt_conn

    def from_firedrake(self, density):
        # make sure we are in the dg function space
        if not self.is_dg:
            density = project(density, self.dg_function_space)
        # Convert to meshmode.
        density = \
            self.source_to_meshmode_connection.from_firedrake(density,
                                                              actx=self.actx)
        # restrict to boundary if necessary
        if self.restrict_source_to_boundary is not None:
            density = self.restrict_source_to_boundary(density)
        return density

    def to_firedrake(self, evaluated_potential, out=None):
        # if we are dg, it's simple
        if self.is_dg:
            return self.target_to_meshmode_connection.from_meshmode(
                evaluated_potential,
                out=out)
        else:
            # Otherwise, we have to project back to our function space
            pot = \
                self.target_to_meshmode_connection.from_meshmode(evaluated_potential)
            pot = project(evaluated_potential, self.function_space)
            if out is not None:
                out.dat.data[:] = pot.dat.data[:]
                pot = out
            return pot

    def get_source_discretization(self):
        """
        Get the :class:`meshmode.discretization.Discretization`
        of the source
        """
        if self.restrict_source_to_boundary is None:
            return self.source_to_meshmode_connection.discr
        return self.restrict_source_to_boundary.to_discr

    def get_target_discretization(self):
        """
        Get the :class:`meshmode.discretization.Discretization`
        of the target
        """
        return self.target_to_meshmode_connection.discr


def PytentialOperation(actx,
                       density,
                       unbound_op,
                       density_name,
                       potential_src_and_tgt,
                       **kwargs):
    """
    :arg actx: A :class:`meshmode.dof_array.PyOpenCLArrayContext`
    :arg density: the :mod:`firedrake` density function
    :arg unbound_op: A :mod:`pytential` operation which has not been
        bound (e.g. a :mod:`pymbolic` expression)
    :arg density_name: A string, the name of the density function
        in the unbound_op
    :arg potential_src_and_tgt: A :class:`PotentialSourceAndTarget`

    :kwarg warn_if_cg: Pass *False* to suppress the warning if
        a "CG" space is used
    :kwarg meshmode_connection_kwargs: *None*, or
        a dict with arguments to pass to
        :func:`meshmode.interop.firedrake.build_connection_from_firedrake`
    :kwarg qbx_kwargs: *None*, or a dict with arguments to pass to
        :class:`pytential.qbx.QBXLayerPotentialSource`.
    :kwarg op_kwargs: kwargs passed to the invocation of the
        bound pytential operator (e.g. {'k': 0.5} for
        a :class:`sumpy.kernel.HelmholtzKernel`).
        Must not include *density_name*

    Remaining kwargs are passed to Potential.__init__
    """
    # make sure density name is a string
    if not isinstance(density_name, str):
        raise TypeError("density_name must be of type str, not '%s'." %
                        density_name)

    # get kwargs to build connection
    warn_if_cg = kwargs.pop('warn_if_cg', True)
    meshmode_connection_kwargs = kwargs.pop('meshmode_connection_kwargs', None)

    # Build meshmode connection
    meshmode_connection = MeshmodeConnection(
        density.function_space(),
        potential_src_and_tgt,
        actx,
        warn_if_cg=warn_if_cg,
        meshmode_connection_kwargs=meshmode_connection_kwargs)

    # Build QBX
    src_discr = meshmode_connection.get_source_discretization()
    qbx_kwargs = kwargs.pop('qbx_kwargs', None)
    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(src_discr, **qbx_kwargs)

    # Bind pytential operator
    tgt_discr = meshmode_connection.get_target_discretization()
    from pytential import bind
    pyt_op = bind((qbx, tgt_discr), unbound_op)

    # Get operator kwargs
    op_kwargs = kwargs.pop('op_kwargs', {})
    if density_name in op_kwargs:
        raise ValueError(f"density_name '{density_name}' should not be included"
                         " in op_kwargs.")

    # build bound operator that just takes density as argument
    def bound_op_with_kwargs(density_arg):
        op_kwargs[density_name] = density_arg
        return pyt_op(**op_kwargs)

    # Now build and return Potential object
    return Potential(density,
                     connection=meshmode_connection,
                     potential_operator=bound_op_with_kwargs,
                     **kwargs)
