from ufl.core.ufl_type import UFLType
from ufl.core.external_operator import ExternalOperator
from ufl.argument import BaseArgument

import firedrake.ufl_expr as ufl_expr
from firedrake.assemble import get_assembler
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.matrix import MatrixBase
from firedrake import functionspaceimpl


class AssemblyRegisterMetaClass(UFLType):
    """Metaclass registering assembly methods specified by external operator subclasses.

    This metaclass is used to register assembly methods specified by subclasses of :class:`~.AbstractExternalOperator`.
    For any new external operator subclass, :class:`AssemblyRegisterMetaClass` will collect all assembly methods specified by the
    subclass and construct a registry to map from assembly identifiers, specified via the `assemble_method` decorator
    to the corresponding assembly methods, and attach that registry to the subclass.

    Notes
    -----
    This metaclass subclasses `UFLType` to avoid metaclass conflict for :class:`~.AbstractExternalOperator`.
    """
    def __init__(cls, name, bases, attrs):
        cls._assembly_registry = {}
        # Collect assembly registries from parent classes
        for base in bases:
            cls._assembly_registry.update(getattr(base, '_assembly_registry', {}))
        # Update assembly registry with assembly methods from `cls`.
        for assembly_method in attrs.values():
            registry = getattr(assembly_method, '_registry', ())
            for assembly_id in registry:
                cls._assembly_registry.update({assembly_id: assembly_method})


class AbstractExternalOperator(ExternalOperator, metaclass=AssemblyRegisterMetaClass):

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=(), operator_data=None):
        """External operator base class providing the interface to build new external operators.

        The :class:`~.AbstractExternalOperator` encapsulates the external operator abstraction and is compatible
        with UFL symbolic operations, the Firedrake assembly, and the AD capabilities provided by `~.firedrake.adjoint`.
        The :class:`~.AbstractExternalOperator` class orchestrates the external operator assembly by linking the
        finite element assembly to the assembly implementations specified by the external operator subclasses.

        Parameters
        ----------
        *operands : ufl.core.expr.Expr or ufl.form.BaseForm
                    Operands of the external operator.
        function_space : firedrake.functionspaceimpl.WithGeometryBase
                         The function space the external operator is mapping to.
        derivatives : tuple
                      Tuple specifiying the derivative multiindex.
        *argument_slots : ufl.coefficient.BaseCoefficient or ufl.argument.BaseArgument
                          Tuple containing the arguments of the linear form associated with the external operator,
                          i.e. the arguments with respect to which the external operator is linear. Those arguments
                          can be ufl.Argument objects, as a result of differentiation, or ufl.Coefficient objects,
                          as a result of taking the action on a given function.
        operator_data : dict
                        Dictionary containing the data of the external operator, i.e. the external data
                        specific to the external operator subclass considered. This dictionary will be passed on
                        over the UFL symbolic reconstructions making the operator data accessible to the external operators
                        arising from symbolic operations on the original operator, such as the Jacobian of the external operator.
        """
        from firedrake_citations import Citations
        Citations().register("Bouziani2021")
        Citations().register("Bouziani2024")

        # Check function space
        if not isinstance(function_space, functionspaceimpl.WithGeometry):
            raise NotImplementedError("Can't make a Function defined on a " + str(type(function_space)))

        # -- ExternalOperator inheritance -- #
        ExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives,
                                  argument_slots=argument_slots)
        # Set function space
        self._function_space = function_space

        # -- Argument slots -- #
        if len(argument_slots) == 0:
            # Make v*
            v_star = ufl_expr.Argument(function_space.dual(), 0)
            argument_slots = (v_star,)
        self._argument_slots = argument_slots

        # -- Operator data -- #
        self.operator_data = operator_data

    def function_space(self):
        return self._function_space

    def assemble_method(derivs, args):
        """Decorator helper function to specify the type of external operator type associated with each assembly methods.

        The `assemble_method` decorator is used to specify the type of external operator associated with the assembly methods
        of the external operator subclass. Each assembly method must be decorated with `assemble_method`.
        The role of this decorator is to record the assembly methods of the subclass.
        The type of external operator is fully specified via the derivative multi-index and a tuple
        representing the argument slots of the external operator.

        Parameters
        ----------
        derivs: tuple
                Derivative multi-index of the external operator associated with the assembly method decorated.
        args: tuple
              Tuple representing the argument slots of the external operator, i.e. `self.argument_slots()`,
              in which integers stand for the numbers of the arguments of type :class:`~.firedrake.ufl_expr.Argument` or
              :class:`~.firedrake.ufl_expr.Coargument`, and `None` stands for arguments of type
              :class:`~.firedrake.function.Function` or :class:`~.firedrake.cofunction.Cofunction`.

        Notes
        -----
        More information can be found at `www.firedrakeproject.org/external_operators.html#build-your-own-external-operator`.
        """
        # Checks
        if not isinstance(derivs, (tuple, int)) or not isinstance(args, tuple):
            raise ValueError("Expecting `assemble_method` to take `(derivs, args)`, where `derivs` can be a derivative multi-index or an integer and `args` is a tuple")
        if isinstance(derivs, int):
            if derivs < 0:
                raise ValueError("Expecting a nonnegative integer and not %s" % str(derivs))
        else:
            if not all(isinstance(d, int) for d in derivs) or any(d < 0 for d in derivs):
                raise ValueError("Expecting a derivative multi-index with nonnegative indices and not %s" % str(derivs))
        if any((not isinstance(a, int) and a is not None) for a in args) or any(isinstance(a, int) and a < 0 for a in args):
            raise ValueError("Expecting an argument tuple with nonnegative integers or None objects and not %s" % str(args))

        # Set the registry
        registry = (derivs, args)

        # Set the decorator mechanism to record the available methods
        def decorator(assemble):
            if not hasattr(assemble, '_registry'):
                assemble._registry = ()
            assemble._registry += (registry,)
            return assemble
        return decorator

    def assemble(self, assembly_opts=None):
        """External operator assembly

        Parameters
        ----------
        assembly_opts: dict
                       Dictionary containing assembly options of the finite element assembly, which may
                       be of interest for the assembly methods of the external operator subclass.
                       These options are passed on to the assembly methods of the external operator subclass.

        Returns
        -------
        firedrake.function.Function or firedrake.cofunction.Cofunction or firedrake.matrix.MatrixBase
            The result of assembling the external operator.

        Notes
        -----
        More information can be found at `www.firedrakeproject.org/external_operators.html#assembly`.
        """

        # -- Checks -- #
        number_arguments = len(self.arguments())
        if number_arguments > 2:
            if sum(self.derivatives) > 2:
                err_msg = "Derivatives higher than 2 are not supported!"
            else:
                err_msg = "Cannot assemble external operators with more than 2 arguments! You need to take the action!"
            raise ValueError(err_msg)

        # -- Construct assembly identifier of the external operator `self` -- #

        derivs = self.derivatives
        arguments = tuple(arg.number() if isinstance(arg, BaseArgument) else None for arg in self.argument_slots())
        key = (derivs, arguments)

        # -- Get assembly methods -- #

        assembly_registry = self._assembly_registry
        try:
            assemble = assembly_registry[key]
        except KeyError:
            try:
                # User can provide the sum of derivatives instead of the multi-index
                #  => This is useful for arbitrary operators where the number of operators is unknwon a priori.
                assemble = assembly_registry[(sum(key[0]), key[1])]
            except KeyError:
                raise NotImplementedError(('The problem considered requires that your external operator class `%s`'
                                           + ' has an implementation for %s !') % (type(self).__name__, str(key)))

        # -- Assemble -- #
        result = assemble(self, assembly_opts=assembly_opts)

        # -- Compatibility check -- #
        if len(self.arguments()) == 1:
            # Will also catch the case where wrong fct space
            if not isinstance(result, (Function, Cofunction)):
                raise ValueError('External operators with one argument must result in a firedrake.Function or firedrake.Cofunction object!')
        elif len(self.arguments()) == 2:
            if not isinstance(result, MatrixBase):
                raise ValueError('External operators with two arguments must result in a firedrake.MatrixBase object!')
        return result

    def _matrix_builder(self, bcs, opts, integral_types):
        """Helper function for allocating a :class:`firedrake.matrix.MatrixBase` object.

        This helper function provides a way to allocate matrices that can then be populated
        in the assembly method(s) of the external operator subclass.

        Parameters
        ----------
        bcs: Tuple
             Tuple of boundary conditions.
        opts: dict
              Dictionary containing options for the matrix allocation.
        integral_types: set
                        Set of integral types.

        Returns
        -------
        firedrake.matrix.MatrixBase
            The allocated matrix.
        """

        # Remove `diagonal` keyword argument
        opts.pop('diagonal', None)
        # Allocate the matrix associated with `self`
        return get_assembler(self, bcs=bcs, allocation_integral_types=integral_types, **opts).allocate()

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None,
                               argument_slots=None, operator_data=None, add_kwargs={}):
        "Return a new object of the same type with new operands."
        return type(self)(*operands, function_space=function_space or self.function_space(),
                          derivatives=derivatives or self.derivatives,
                          argument_slots=argument_slots or self.argument_slots(),
                          operator_data=operator_data or self.operator_data,
                          **add_kwargs)

    def __hash__(self):
        "Hash code for use in dicts."
        hashdata = (type(self),
                    tuple(hash(op) for op in self.ufl_operands),
                    tuple(hash(arg) for arg in self._argument_slots),
                    self.derivatives,
                    hash(self.ufl_function_space()),
                    # Mutable objects are not hashable
                    id(self.operator_data))
        return hash(hashdata)

    def __eq__(self, other):
        if self is other:
            return True
        return (type(self) == type(other)
                # Operands' output spaces will be taken into account via Interp.__eq__
                # -> N(Interp(u, V1); v*) and N(Interp(u, V2); v*) will compare different.
                and all(a == b for a, b in zip(self.ufl_operands, other.ufl_operands))
                and all(a == b for a, b in zip(self._argument_slots, other._argument_slots))
                and self.derivatives == other.derivatives
                and self.ufl_function_space() == other.ufl_function_space()
                and self.operator_data == other.operator_data)

    def __repr__(self):
        "Default repr string construction for AbstractExternalOperator."
        r = "%s(%s; %s; %s; derivatives=%s; operator_data=%s)" % (type(self).__name__,
                                                                  ", ".join(repr(op) for op in self.ufl_operands),
                                                                  repr(self.ufl_function_space()),
                                                                  ", ".join(repr(arg) for arg in self.argument_slots()),
                                                                  repr(self.derivatives),
                                                                  repr(self.operator_data))
        return r


# Make a renamed public decorator function
assemble_method = AbstractExternalOperator.assemble_method
