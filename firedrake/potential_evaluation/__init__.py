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
    _external_operator_type = 'GLOBAL'

    def __init__(self, density, **kwargs):
        """
        :arg density: A :mod:`firedrake` :class:`firedrake.function.Function`
            or UFL expression which represents the density :math:`u`
        :kwarg connection: A :class:`PotentialEvaluationLibraryConnection`
        :kwarg potential_operator: The external potential evaluation library
                                 bound operator.
        """
        # super
        AbstractExternalOperator.__init__(self, density, **kwargs)

        # FIXME
        for order in self.derivatives:
            assert order == 1, "Assumes self.derivatives = (1,..,1)"

        # Get connection & bound op and validate
        connection = kwargs.get("connection", None)
        if connection is None:
            raise ValueError("Missing kwarg 'connection'")
        if not isinstance(connection, PotentialEvaluationLibraryConnection):
            raise TypeError("connection must be of type "
                            "PotentialEvaluationLibraryConnection, not %s."
                            % type(connection))
        self.connection = connection

        self.potential_operator = kwargs.get("potential_operator", None)
        if self.potential_operator is None:
            raise ValueError("Missing kwarg 'potential_operator'")

        # Get function space and validate aginst bound op
        function_space = self.ufl_function_space()
        assert isinstance(function_space, WithGeometry)  # sanity check
        if function_space is not self.potential_operator.function_space():
            raise ValueError("function_space must be same object as "
                             "potential_operator.function_space().")

        # Make sure density is a member of our function space, if it is
        if isinstance(density, Function):
            if density.function_space() is not function_space:
                raise ValueError("density.function_space() must be the "
                                 "same as function_space")
        # Make sure the shapes match, at least
        elif density.shape != function_space.shape:
            raise ValueError("Shape mismatch between function_space and "
                             "density. %s != %s." %
                             (density.shape, function_space.shape))

    @cached_property
    def _evaluator(self):
        return PotentialEvaluator(self,
                                  self.density,
                                  self.connection,
                                  self.potential_operator)

    def _evaluate(self):
        raise NotImplementedError

    def _compute_derivatives(self, continuity_tolerance=None):
        raise NotImplementedError

    def _evaluate_action(self, *args):
        return self._evaluator._evaluate_action()

    def _evaluate_adjoint_action(self, *args):
        return self._evaluator._evaluate_action()

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
    def __init__(self, density, connection, potential_operator):
        """
        :arg density: The UFL/firedrake density function
        :arg connection: A :class:`PotentialEvaluationLibraryConnection`
        :arg potential_operator: The external potential evaluation library
                                 bound operator.
        """
        self.density = density
        self.connection = connection
        self.potential_operator = potential_operator

    def _eval_potential_operator(self, action_coefficients, out=None):
        """
        Evaluate the potential on the action coefficients and return.
        If *out* is not *None*, stores the result in *out*

        :arg action_coefficients: A :class:`~firedrake.function.Function`
                                  to evaluate the potential at
        :arg out: If not *None*, then a :class:`~firedrake.function.Function`
                  in which to store the evaluated potential
        :return: *out* if it is not *None*, otherwise a new
                 :class:`firedrake.function.Function` storing the evaluated
                 potential
        """
        density = self.connection.from_firedrake(action_coefficients)
        potential = self.potential_operator(density)
        return self.connection.to_firedrake(potential, out=out)

    def _evaluate(self):
        """
        Evaluate P(self.density) into self
        """
        return self._eval_potential_operator(self.density, out=self)

    def _compute_derivatives(self):
        """
        Return a function
        Derivative(P, self.derivatives, self.density)
        """
        # FIXME : Assumes derivatives are Jacobian
        return self._eval_potential_operator

    def _evaluate_action(self, action_coefficients):
        """
        Evaluate derivatives of layer potential at action coefficients.
        i.e. Derivative(P, self.derivatives, self.density) evaluated at
        the action coefficients and store into self
        """
        operator = self._compute_derivatives()
        return operator(action_coefficients, out=self)


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
        The string "everywhere" is also equivalent to a
        value of *None*

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
            # standardize id_ to a tuple
            if isinstance(id_, int):
                id_ = tuple(id_)
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
                mesh.cell_subset(id_)

        # handle None case
        if source_region_id is None:
            source_region_id = "everywhere"
        if target_region_id is None:
            target_region_id = "everywhere"

        # store mesh, region dims, and region ids
        self.mesh = mesh
        self._source_region_dim = source_region_dim
        self._target_region_dim = source_region_dim
        self._source_region_id = source_region_id
        self._target_region_id = source_region_id

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
