"""
Fundamental classes for potential evaluation, including
Potential and a connection to potential-evaluaiton libraries
"""
from warnings import warn

from firedrake.functionspaceimpl import WithGeometry
from firedrake.pointwise_operators import AbstractExternalOperator
from firedrake.potential_evaluation import PotentialSourceAndTarget

from firedrake import Function, FunctionSpace, interpolate, \
    VectorFunctionSpace, TensorFunctionSpace


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
        density_discrete = interpolate(density, self.function_space())
        density = self.connection.from_firedrake(density_discrete)
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
        # FIXME : Assumes only one ufl operand, is that okay?
        assert len(self.ufl_operands) == 1
        assert self.ufl_operands[0] is self.density
        assert len(self.derivatives) == 1
        # Derivative(P, (0,), self.density)(density_star)
        #  = P(self.density)
        if self.derivatives == (0,):
            return lambda density_star, out=None: \
                self._eval_potential_operator(self.density, out=out)
        # P is linear, so the nth Gateaux derivative
        # Derivative(P, (n,), u)(u*) = P(u*)
        #
        # d/dx (P \circ u)(v)
        #  = lim_{t->0} (P(u(v+tx)) - P(u)) / t
        #  \approx lim_{t->0} (P(u(v) + t du/dx - P(u(v))) / t
        #  = P(du/dx)
        #
        # So d^n/dx^n( P \circ u ) = P(d^nu/dx^n)
        return self._eval_potential_operator

    def _evaluate_action(self, *args):
        """
        Evaluate derivatives of layer potential at action coefficients.
        i.e. Derivative(P, self.derivatives, self.density) evaluated at
        the action coefficients and store into self
        """
        assert len(args) == 1  # sanity check
        # Either () or (density_star,)
        density_star = args[0]
        assert len(density_star) in [0, 1]  # sanity check
        # Evaluate Derivative(P, self.derivatives, self.density) at density_star
        if len(density_star) == 0:
            return self._evaluate()

        # FIXME: Assumes just taking Jacobian
        density_star, = density_star
        operator = self._compute_derivatives()
        return operator(density_star, out=self)

    def _evaluate_adjoint_action(self, *args):
        """
        Operator is self-adjoint, so just evaluate action
        """
        return self._evaluate_action(*args)

    def evaluate_adj_component_control(self, x, idx):
        derivatives = tuple(dj + int(idx == j) for j, dj in enumerate(self.derivatives))
        dN = self._ufl_expr_reconstruct_(*self.ufl_operands, derivatives=derivatives)
        return dN._evaluate_adjoint_action((x.function,)).vector()

    def evaluate_adj_component(self, x, idx):
        print(x, type(x))
        raise NotImplementedError


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
        if not isinstance(potential_source_and_target, PotentialSourceAndTarget):
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
                dim = shape[0]
                dg_function_space = VectorFunctionSpace(mesh, "DG", degree,
                                                        dim=dim)
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
