import numpy as np
import firedrake.function

from firedrake.functionspaceimpl import WithGeometry
from firedrake.mesh import MeshGeometry
from firedrake.pointwise_operators import AbstractExternalOperator
from firedrake.utils import cached_property

from firedrake import Function, FunctionSpace, interpolate, Interpolator, \
    SpatialCoordinate, VectorFunctionSpace, TensorFunctionSpace, project

from ufl.algorithms.apply_derivatives import VariableRuleset
from ufl.algorithms import extract_coefficients
from ufl.constantvalue import as_ufl
from ufl.core.multiindex import indices
from ufl.tensors import as_tensor

from pyop2.datatypes import ScalarType
from warnings import warn


from firedrake.potential_evaluation.potentials import \
    SingleLayerPotential, DoubleLayerPotential, VolumePotential
__all__ = ("SingleLayerPotential",
           "DoubleLayerPotential",
           "VolumePotential",
           "PotentialSourceAndTarget")


class Potential(AbstractExternalOperator):
    r"""
    This is a function which represents a potential
    computed on a firedrake function.

    For a firedrake function :math:`u:\Omega\to\mathbb C^n`
    with :math:`\Omega \subseteq \mathbb R^m`,
    the potential :math:`P(u):\Omega \to\mathbb C^n` is a
    convolution of :math:`u` against some kernel function.
    More concretely, given a source region :math:`\Gamma \subseteq \Omega`
    a target region :math:`\Sigma \subseteq \Omega`, and
    a kernel function :math:`K`, we define

    .. math::

        P(u)(x) = \begin{cases}
            \int_\Gamma K(x, y) u(y) \,dy & x \in \Sigma
            0 & x \notin \Sigma
        \end{cases}
    """
    def __init__(self, density, **kwargs):
        """
        :arg density: A :mod:`firedrake` :class:`firedrake.function.Function`
            or UFL expression which represents the density :math:`u`

        :kwarg operator_data: Should have keys

            :kwarg connection: A :class:`PotentialEvaluationLibraryConnection`
            :kwarg potential_operator: The external potential evaluation library
                                     bound operator.
        :kwarg function_space: the function space
        """
        # super
        AbstractExternalOperator.__init__(self, density, **kwargs)

        if 'operator_data' not in kwargs:
            raise ValueError("Missing kwarg 'operator_data'")
        operator_data = kwargs['operator_data']

        # Get connection & bound op and validate
        if 'connection' not in operator_data:
            raise ValueError("Missing operator_data 'connection'")
        connection = operator_data['connection']
        if not isinstance(connection, PotentialEvaluationLibraryConnection):
            raise TypeError("connection must be of type "
                            "PotentialEvaluationLibraryConnection, not %s."
                            % type(connection))
        if 'potential_operator' not in operator_data:
            raise ValueError("Missing operator_data 'potential_operator'")

        # Get function space and validate aginst bound op
        if 'function_space' not in kwargs:
            raise ValueError("Missing kwarg 'function_space'")

        function_space = kwargs['function_space']
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be of type WithGeometry, "
                            f"not {type(function_space)}")

        # Make sure density is a member of our function space, if it is
        if isinstance(density, Function):
            if density.function_space() != function_space:
                raise ValueError("density.function_space() must be the "
                                 "same as function_space")
        # save attributes
        self.density = density
        self.connection = connection
        self.potential_operator = operator_data['potential_operator']

    @cached_property
    def _evaluator(self):
        return PotentialEvaluator(self,
                                  self.density,
                                  self.connection,
                                  self.potential_operator,
                                  self.function_space())

    def _evaluate(self):
        return self._evaluator._evaluate()

    def _compute_derivatives(self):
        return self._evaluator._compute_derivatives()

    def _evaluate_action(self, *args):
        return self._evaluator._evaluate_action(*args)

    def _evaluate_adjoint_action(self, *args):
        return self._evaluator._evaluate_action(*args)

    def evaluate_adj_component_control(self, x, idx):
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        dN = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        return dN._evaluate_adjoint_action((x.function,)).vector()

    def evaluate_adj_component(self, x, idx):
        print(x, type(x))
        raise NotImplementedError


class PotentialEvaluator:
    """
    Evaluates a potential
    """
    def __init__(self, potential, density, connection, potential_operator, function_space):
        """
        :arg density: The UFL/firedrake density function
        :arg connection: A :class:`PotentialEvaluationLibraryConnection`
        :arg potential_operator: The external potential evaluation library
                                 bound operator.
        :arg function_space: the :mod:`firedrake` function space of this
            operator
        """
        self.potential = potential
        self.density = density
        self.connection = connection
        self.potential_operator = potential_operator
        self.function_space = function_space

        import matplotlib.pyplot as plt
        from meshmode.mesh.visualization import draw_2d_mesh, draw_curve
        src_discr = connection.get_source_discretization()
        tgt_discr = connection.get_target_discretization()
        draw_curve(src_discr.mesh)
        plt.title("SOURCE")
        plt.show()
        draw_2d_mesh(tgt_discr.mesh, set_bounding_box=True)
        plt.title("TARGET")
        plt.show()

    def _eval_potential_operator(self, density, out=None):
        """
        Evaluate the potential on the action coefficients and return.
        If *out* is not *None*, stores the result in *out*

        :arg density: the density
        :arg out: If not *None*, then a :class:`~firedrake.function.Function`
                  in which to store the evaluated potential
        :return: *out* if it is not *None*, otherwise a new
                 :class:`firedrake.function.Function` storing the evaluated
                 potential
        """
        density_discrete = interpolate(density, self.function_space)
        density = self.connection.from_firedrake(density_discrete)
        potential = self.potential_operator(density)
        return self.connection.to_firedrake(potential, out=out)

    def _evaluate(self):
        """
        Evaluate P(self.density) into self
        """
        return self._eval_potential_operator(self.density, out=self.potential)

    def _compute_derivatives(self):
        """
        Return a function
        Derivative(P, self.derivatives, self.density)
        """
        # FIXME : Assumes derivatives are Jacobian
        return self._eval_potential_operator

    def _evaluate_action(self, args):
        """
        Evaluate derivatives of layer potential at action coefficients.
        i.e. Derivative(P, self.derivatives, self.density) evaluated at
        the action coefficients and store into self
        """
        if len(args) == 0:
            return self._evaluate()

        # FIXME: Assumes just taking Jacobian
        assert len(args) == 1
        operator = self._compute_derivatives()
        return operator(*args, out=self.potential)


class PotentialEvaluationLibraryConnection:
    """
    A connection to an external library for potential evaluation
    """
    def __init__(self, function_space, potential_source_and_target,
                 warn_if_cg=True):
        """
        Initialize self to work on the function space

        :arg function_space: The :mod:`firedrake` function space
            (of type :class:`~firedrake.functionspaceimpl.WithGeometry`) on
            which to convert to/from. Must be a 'Lagrange' or
            'Discontinuous Lagrange' space.
        :arg potential_source_and_target: A :class:`PotentialSourceAndTarget`.
            mesh must match that of function_space.
        :arg warn_if_cg: If *True*, warn if the space is "CG"
        """
        # validate function space
        if not isinstance(function_space, WithGeometry):
            raise TypeError("function_space must be of type WithGeometry, not "
                            "%s" % type(function_space))

        family = function_space.ufl_element().family()
        acceptable_families = ['Discontinuous Lagrange', 'Lagrange']
        if family not in acceptable_families:
            raise ValueError("function_space.ufl_element().family() must be "
                             "one of %s, not '%s'" %
                             (acceptable_families, family))
        if family == 'Lagrange' and warn_if_cg:
            warn("Functions in continuous function space will be projected "
                 "to/from a 'Discontinuous Lagrange' space. Make sure "
                 "any operators evaluated are continuous. "
                 "Pass warn_if_cg=False to suppress this warning.")

        # validate potential_source_and_targets
        if not isinstance(potential_source_and_target,
                          PotentialSourceAndTarget):
            raise TypeError("potential_source_and_targets must be of type "
                            "PotentialSourceAndTarget, not '%s'."
                            % type(potential_source_and_target))

        # make sure meshes match
        if potential_source_and_target.mesh is not function_space.mesh():
            raise ValueError("function_space.mesh() and "
                             "potential_source_and_target.mesh must be the same"
                             " obejct.")

        # build DG space if necessary
        family = function_space.ufl_element().family()
        if family == 'Discontinuous Lagrange':
            dg_function_space = function_space
        elif family == 'Lagrange':
            mesh = function_space.mesh()
            degree = function_space.ufl_element().degree()
            shape = function_space.shape
            if shape is None or len(shape) == 0:
                dg_function_space = FunctionSpace(mesh, "DG", degree)
            elif len(shape) == 1:
                dg_function_space = VectorFunctionSpace(mesh, "DG", degree,
                                                        dim=shape)
            else:
                dg_function_space = TensorFunctionSpace(mesh, "DG", degree,
                                                        shape=shape)
        else:
            acceptable_families = ['Discontinuous Lagrange', 'Lagrange']
            raise ValueError("function_space.ufl_element().family() must be "
                             "one of %s, not '%s'" %
                             (acceptable_families, family))

        # store function space and dg function space
        self.function_space = function_space
        self.dg_function_space = dg_function_space
        self.is_dg = function_space == dg_function_space
        # store source and targets
        self.source_and_target = potential_source_and_target

    def from_firedrake(self, density):
        """
        Convert the density into a form acceptable by an bound operation
        in an external library

        :arg density: A :class:`~firedrake.function.Function` holding the
                      density.
        :returns: The converted density
        """
        raise NotImplementedError

    def to_firedrake(self, evaluated_potential, out=None):
        """
        Convert the evaluated potential from an external library
        into a firedrake function

        :arg evaluated_potential: the evaluated potential
        :arg out: If not *None*, store the converted potential into this
                  :class:`firedrake.function.Function`.
        :return: *out* if it is not *None*, otherwise a new
                 :class:`firedrake.function.Function` storing the evaluated
                 potential
        """
        raise NotImplementedError


class PotentialSourceAndTarget:
    """
    Holds the source and target for a layer or volume potential
    """
    def __init__(self, mesh,
                 source_region_dim=None,
                 source_region_id=None,
                 target_region_dim=None,
                 target_region_id=None):
        """
        Source and target of a layer or volume potential.
        The mesh must have co-dimension 0 or 1

        By region_dim, we mean the topological dimension of the
        region.
        Regions must have co-dimensions 0 or 1.
        They cannot have a higher dimension than the mesh.
        *None* indicates the topological dimension of the
        mesh.

        Region ids must be either a valid mesh subdomain id (an
        integer or tuple) or *None*. *None* indicates either
        the entire mesh, or the entire exterior boundary
        (as determined by the value of region).
        The string "on_boundary"/"everywhere" is also equivalent to a
        value of *None* (for the appropriate subdomain dimension)

        NOTE: For implementation reasons, if the target is
              a boundary, instead of just evaluating the
              target at DOFs on that boundary, it is instead
              evaluated at any DOF of a cell which has at least
              one vertex on the boundary.
        """
        # validate mesh
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("mesh must be of type MeshGeometry, not '%s'." %
                            type(mesh))

        # mesh must have co-dimension 1 or 0.
        valid_codims = [0, 1]
        codim = mesh.geometric_dimension() - mesh.topological_dimension()
        if codim not in valid_codims:
            raise ValueError("mesh has invalid co-dimension of %s. "
                             "co-dimension must be one of %s." %
                             (codim, valid_codims))

        # validate dims
        for dim in [source_region_dim, target_region_dim]:
            if dim is None:
                continue
            if not isinstance(dim, int):
                raise TypeError("source and target dimensions must be *None*"
                                " or an integer, not of type '%s'." % type(dim))
            valid_dims = set([mesh.topological_dimension(),
                              mesh.geometric_dimension()-1])
            if dim < mesh.geometric_dimension() - 1:
                raise ValueError("source and target dimensions must be "
                                 "one of %s or *None*, not '%s'." %
                                 (valid_dims, dim))
        # Get dims if *None*
        if source_region_dim is None:
            source_region_dim = mesh.topological_dimension()
        if target_region_dim is None:
            target_region_dim = mesh.topological_dimension()

        # Validate region ids type and validity
        for id_, dim in zip([source_region_id, target_region_id],
                            [source_region_dim, target_region_dim]):
            if id_ is None or id_ == "everywhere":
                continue
            if id_ == "everywhere":
                if dim != mesh.topological_dimension():
                    raise ValueError("\"everywhere\" only valid for subdomain"
                                     " of same dimension as mesh. Try "
                                     "\"on_boundary\" instead")
                continue
            if id_ == "on_boundary":
                if dim != mesh.topological_dimension()-1:
                    raise ValueError("\"on_boundary\" only valid for subdomain"
                                     " of dimension one less than the mesh. Try "
                                     "\"everywhere\" instead")
                continue
            if isinstance(id_, str):
                raise TypeError("Only valid strings are \"on_boundary\" and "
                                f"\"everywhere\", not '{id_}'.")
            # standardize id_ to a tuple
            if isinstance(id_, int):
                id_ = tuple([id_])
            # check type
            if not isinstance(id_, tuple):
                raise TypeError("source and target region ids must be "
                                "*None*, an int, or a tuple, not of type")
            # boundary case:
            if dim == mesh.topological_dimension() - 1:
                if not set(id_) <= set(mesh.exterior_facets.unique_markers):
                    raise ValueError(("boundary region ids %s are not a "
                                     + "subset of mesh.exterior_facets."
                                     + "unique_markers = %s.") % (id_,
                                     mesh.exterior_facets.unique_markers))
            else:
                # sanity check
                assert dim == mesh.topological_dimension()
                # this caches the cell subset on the mesh, which we'll
                # probably be doing anyway if we're targeting the region
                #
                # It also throws a ValueError if the id_ is invalid
                # FIXME: Figure out how to support multiple ids
                if len(id_) > 1:
                    raise NotImplementedError("Currently, any tuple of "
                                              "cell subdomain ids must be of "
                                              "length 1.")
                id_, = id_
                mesh.cell_subset(id_)

        # handle None case
        if source_region_id is None:
            if source_region_dim == mesh.topological_dimension():
                source_region_id = "everywhere"
            else:
                source_region_id = "on_boundary"
        if target_region_id is None:
            if target_region_dim == mesh.topological_dimension():
                target_region_id = "everywhere"
            else:
                target_region_id = "on_boundary"

        # store mesh, region dims, and region ids
        self.mesh = mesh
        self._source_region_dim = source_region_dim
        self._target_region_dim = target_region_dim
        self._source_region_id = source_region_id
        self._target_region_id = target_region_id

    def get_source_dimension(self):
        """
        Get the topological dimension of the source region
        """
        return self._source_region_dim

    def get_target_dimension(self):
        """
        Get the topological dimension of the target region
        """
        return self._target_region_dim

    def get_source_id(self):
        """
        Get the subdomain id of the source, or the string
        "everywhere" to represent the entire exterior boundary/mesh
        """
        return self._source_region_id

    def get_target_id(self):
        """
        Get the subdomain id of the target, or the string
        "everywhere" to represent the entire exterior boundary/mesh
        """
        return self._target_region_id
