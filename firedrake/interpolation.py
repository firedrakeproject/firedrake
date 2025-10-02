import numpy
import os
import tempfile
import abc
import warnings
from collections.abc import Iterable
from functools import partial, singledispatch
from typing import Hashable, Literal

import FIAT
import ufl
import finat.ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.domain import as_domain, extract_unique_domain

from pyop2 import op2
from pyop2.caching import memory_and_disk_cache

from finat.element_factory import create_element, as_fiat_cell
from tsfc import compile_expression_dual_evaluation
from tsfc.ufl_utils import extract_firedrake_constants, hash_expr

import gem
import finat

import firedrake
from firedrake import tsfc_interface, utils, functionspaceimpl
from firedrake.ufl_expr import Argument, Coargument, action, adjoint as expr_adjoint
from firedrake.cofunction import Cofunction
from firedrake.mesh import MissingPointsBehaviour, VertexOnlyMeshMissingPointsError, VertexOnlyMeshTopology
from firedrake.petsc import PETSc
from firedrake.halo import _get_mtype as get_dat_mpi_type
from mpi4py import MPI

from pyadjoint import stop_annotating, no_annotations

__all__ = (
    "interpolate",
    "Interpolator",
    "Interpolate",
    "DofNotDefinedError",
    "CrossMeshInterpolator",
    "SameMeshInterpolator",
)


class Interpolate(ufl.Interpolate):

    def __init__(self, expr, V,
                 subset=None,
                 access=None,
                 allow_missing_dofs=False,
                 default_missing_val=None,
                 matfree=True):
        """Symbolic representation of the interpolation operator.

        Parameters
        ----------
        expr : ufl.core.expr.Expr or ufl.BaseForm
               The UFL expression to interpolate.
        V : firedrake.functionspaceimpl.WithGeometryBase or firedrake.ufl_expr.Coargument
            The function space to interpolate into or the coargument defined
            on the dual of the function space to interpolate into.
        subset : pyop2.types.set.Subset
                 An optional subset to apply the interpolation over.
                 Cannot, at present, be used when interpolating across meshes unless
                 the target mesh is a :func:`.VertexOnlyMesh`.
        access : pyop2.types.access.Access
                 The pyop2 access descriptor for combining updates to shared
                 DoFs. Possible values include ``WRITE`` and ``INC``. Only ``WRITE`` is
                 supported at present when interpolating across meshes. See note in
                 :func:`.interpolate` if changing this from default.
        allow_missing_dofs : bool
                             For interpolation across meshes: allow degrees of freedom (aka DoFs/nodes)
                             in the target mesh that cannot be defined on the source mesh.
                             For example, where nodes are point evaluations, points in the target mesh
                             that are not in the source mesh. When ``False`` this raises a ``ValueError``
                             should this occur. When ``True`` the corresponding values are either
                             (a) unchanged if some ``output`` is given to the :meth:`interpolate` method
                             or (b) set to zero.
                             Can be overwritten with the ``default_missing_val`` kwarg of :meth:`interpolate`.
                             This does not affect adjoint interpolation. Ignored if interpolating within
                             the same mesh or onto a :func:`.VertexOnlyMesh` (the behaviour of a
                             :func:`.VertexOnlyMesh` in this scenario is, at present, set when it is created).
        default_missing_val : float
                              For interpolation across meshes: the optional value to assign to DoFs
                              in the target mesh that are outside the source mesh. If this is not set
                              then the values are either (a) unchanged if some ``output`` is given to
                              the :meth:`interpolate` method or (b) set to zero.
                              Ignored if interpolating within the same mesh or onto a :func:`.VertexOnlyMesh`.
        matfree : bool
                If ``False``, then construct the permutation matrix for interpolating
                between a VOM and its input ordering. Defaults to ``True`` which uses SF broadcast
                and reduce operations.
        """
        expr = ufl.as_ufl(expr)
        if isinstance(V, functionspaceimpl.WithGeometry):
            # Need to create a Firedrake Argument so that it has a .function_space() method
            expr_args = extract_arguments(expr)
            is_adjoint = len(expr_args) and expr_args[0].number() == 0
            V = Argument(V.dual(), 1 if is_adjoint else 0)

        target_shape = V.arguments()[0].function_space().value_shape
        if expr.ufl_shape != target_shape:
            raise ValueError(f"Shape mismatch: Expression shape {expr.ufl_shape}, FunctionSpace shape {target_shape}.")

        super().__init__(expr, V)

        # -- Interpolate data (e.g. `subset` or `access`) -- #
        self.interp_data = {"subset": subset,
                            "access": access,
                            "allow_missing_dofs": allow_missing_dofs,
                            "default_missing_val": default_missing_val,
                            "matfree": matfree}

    function_space = ufl.Interpolate.ufl_function_space

    def _ufl_expr_reconstruct_(self, expr, v=None, **interp_data):
        interp_data = interp_data or self.interp_data.copy()
        return ufl.Interpolate._ufl_expr_reconstruct_(self, expr, v=v, **interp_data)


@PETSc.Log.EventDecorator()
def interpolate(expr, V, subset=None, access=None, allow_missing_dofs=False, default_missing_val=None, matfree=True):
    """Returns a UFL expression for the interpolation operation of ``expr`` into ``V``.

    :arg expr: a UFL expression.
    :arg V: a :class:`.FunctionSpace` to interpolate into, or a :class:`.Cofunction`,
        or :class:`.Coargument`, or a :class:`ufl.form.Form` with one argument (a one-form).
        If a :class:`.Cofunction` or a one-form is provided, then we do adjoint interpolation.
    :kwarg subset: An optional :class:`pyop2.types.set.Subset` to apply the
        interpolation over. Cannot, at present, be used when interpolating
        across meshes unless the target mesh is a :func:`.VertexOnlyMesh`.
    :kwarg access: The pyop2 access descriptor for combining updates to shared
        DoFs. Possible values include ``WRITE`` and ``INC``. Only ``WRITE`` is
        supported at present when interpolating across meshes unless the target
        mesh is a :func:`.VertexOnlyMesh`. See note below.
    :kwarg allow_missing_dofs: For interpolation across meshes: allow
        degrees of freedom (aka DoFs/nodes) in the target mesh that cannot be
        defined on the source mesh. For example, where nodes are point
        evaluations, points in the target mesh that are not in the source mesh.
        When ``False`` this raises a ``ValueError`` should this occur. When
        ``True`` the corresponding values are either (a) unchanged if
        some ``output`` is given to the :meth:`interpolate` method or (b) set
        to zero. In either case, if ``default_missing_val`` is specified, that
        value is used. This does not affect adjoint interpolation. Ignored if
        interpolating within the same mesh or onto a :func:`.VertexOnlyMesh`
        (the behaviour of a :func:`.VertexOnlyMesh` in this scenario is, at
        present, set when it is created).
    :kwarg default_missing_val: For interpolation across meshes: the optional
        value to assign to DoFs in the target mesh that are outside the source
        mesh. If this is not set then the values are either (a) unchanged if
        some ``output`` is given to the :meth:`interpolate` method or (b) set
        to zero. Ignored if interpolating within the same mesh or onto a
        :func:`.VertexOnlyMesh`.
    :kwarg matfree: If ``False``, then construct the permutation matrix for interpolating
        between a VOM and its input ordering. Defaults to ``True`` which uses SF broadcast
        and reduce operations.
    :returns: A symbolic :class:`.Interpolate` object

    .. note::

       If you use an access descriptor other than ``WRITE``, the
       behaviour of interpolation changes if interpolating into a
       function space, or an existing function. If the former, then
       the newly allocated function will be initialised with
       appropriate values (e.g. for MIN access, it will be initialised
       with MAX_FLOAT). On the other hand, if you provide a function,
       then it is assumed that its values should take part in the
       reduction (hence using MIN will compute the MIN between the
       existing values and any new values).
    """
    return Interpolate(
        expr, V, subset=subset, access=access, allow_missing_dofs=allow_missing_dofs,
        default_missing_val=default_missing_val, matfree=matfree
    )


class Interpolator(abc.ABC):
    """A reusable interpolation object.

    This object can be used to carry out the same interpolation
    multiple times (for example in a timestepping loop).

    Parameters
    ----------
    expr
        The underlying ufl.Interpolate or the operand to the ufl.Interpolate.
    V
        The :class:`.FunctionSpace` or :class:`.Function` to
        interpolate into.
    subset
        An optional :class:`pyop2.types.set.Subset` to apply the
        interpolation over. Cannot, at present, be used when interpolating
        across meshes unless the target mesh is a :func:`.VertexOnlyMesh`.
    freeze_expr
        Set to True to prevent the expression being
        re-evaluated on each call. Cannot, at present, be used when
        interpolating across meshes unless the target mesh is a
        :func:`.VertexOnlyMesh`.
    access
        The pyop2 access descriptor for combining updates to shared DoFs.
        Only ``op2.WRITE`` is supported at present when interpolating across meshes.
        Only ``op2.INC`` is supported for the matrix-free adjoint interpolation.
        See note in :func:`.interpolate` if changing this from default.
    bcs
        An optional list of boundary conditions to zero-out in the
        output function space. Interpolator rows or columns which are
        associated with boundary condition nodes are zeroed out when this is
        specified.
    allow_missing_dofs
        For interpolation across meshes: allow
        degrees of freedom (aka DoFs/nodes) in the target mesh that cannot be
        defined on the source mesh. For example, where nodes are point
        evaluations, points in the target mesh that are not in the source mesh.
        When ``False`` this raises a ``ValueError`` should this occur. When
        ``True`` the corresponding values are either (a) unchanged if
        some ``output`` is given to the :meth:`interpolate` method or (b) set
        to zero. Can be overwritten with the ``default_missing_val`` kwarg
        of :meth:`interpolate`. This does not affect adjoint interpolation.
        Ignored if interpolating within the same mesh or onto a
        :func:`.VertexOnlyMesh` (the behaviour of a :func:`.VertexOnlyMesh` in
        this scenario is, at present, set when it is created).
    matfree
        If ``False``, then construct the permutation matrix for interpolating
        between a VOM and its input ordering. Defaults to ``True`` which uses SF broadcast
        and reduce operations.

    Notes
    -----

       The :class:`Interpolator` holds a reference to the provided
       arguments (such that they won't be collected until the
       :class:`Interpolator` is also collected).

    """

    def __new__(cls, expr, V, **kwargs):
        V_target = V if isinstance(V, ufl.FunctionSpace) else V.function_space()
        if not isinstance(expr, ufl.Interpolate):
            expr = interpolate(expr, V_target)

        arguments = expr.arguments()
        has_mixed_arguments = any(len(a.function_space()) > 1 for a in arguments)
        if len(arguments) == 2 and has_mixed_arguments:
            return object.__new__(MixedInterpolator)

        operand, = expr.ufl_operands
        target_mesh = as_domain(V)
        source_mesh = extract_unique_domain(operand) or target_mesh
        submesh_interp_implemented = \
            all(isinstance(m.topology, firedrake.mesh.MeshTopology) for m in [target_mesh, source_mesh]) and \
            target_mesh.submesh_ancesters[-1] is source_mesh.submesh_ancesters[-1] and \
            target_mesh.topological_dimension == source_mesh.topological_dimension
        if target_mesh is source_mesh or submesh_interp_implemented:
            return object.__new__(SameMeshInterpolator)
        else:
            if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
                return object.__new__(SameMeshInterpolator)
            elif has_mixed_arguments or len(V_target) > 1:
                return object.__new__(MixedInterpolator)
            else:
                return object.__new__(CrossMeshInterpolator)

    def __init__(
        self,
        expr: ufl.Interpolate | ufl.classes.Expr,
        V: ufl.FunctionSpace | firedrake.function.Function,
        subset: op2.Subset | None = None,
        freeze_expr: bool = False,
        access: Literal[op2.WRITE, op2.MIN, op2.MAX, op2.INC] | None = None,
        bcs: Iterable[firedrake.bcs.BCBase] | None = None,
        allow_missing_dofs: bool = False,
        matfree: bool = True
    ):
        if not isinstance(expr, ufl.Interpolate):
            expr = interpolate(expr, V if isinstance(V, ufl.FunctionSpace) else V.function_space())
        dual_arg, operand = expr.argument_slots()
        self.ufl_interpolate = expr
        self.expr = operand
        self.V = V
        self.subset = subset
        self.freeze_expr = freeze_expr
        self.bcs = bcs
        self._allow_missing_dofs = allow_missing_dofs
        self.matfree = matfree
        self.callable = None

        # TODO CrossMeshInterpolator and VomOntoVomXXX are not yet aware of
        # self.ufl_interpolate (which carries the dual argument).
        # See github issue https://github.com/firedrakeproject/firedrake/issues/4592
        target_mesh = as_domain(V)
        source_mesh = extract_unique_domain(operand) or target_mesh
        vom_onto_other_vom = ((source_mesh is not target_mesh)
                              and isinstance(self, SameMeshInterpolator)
                              and isinstance(source_mesh.topology, VertexOnlyMeshTopology)
                              and isinstance(target_mesh.topology, VertexOnlyMeshTopology))
        if isinstance(self, CrossMeshInterpolator) or vom_onto_other_vom:
            # For bespoke interpolation, we currently rely on different assembly procedures:
            # 1) Interpolate(Argument(V1, 1), Argument(V2.dual(), 0)) -> Forward operator (2-form)
            # 2) Interpolate(Argument(V1, 0), Argument(V2.dual(), 1)) -> Adjoint operator (2-form)
            # 3) Interpolate(Coefficient(V1), Argument(V2.dual(), 0)) -> Forward action (1-form)
            # 4) Interpolate(Argument(V1, 0), Cofunction(V2.dual()) -> Adjoint action (1-form)
            # 5) Interpolate(Coefficient(V1), Cofunction(V2.dual()) -> Double action (0-form)

            # CrossMeshInterpolator._interpolate only supports forward interpolation (cases 1 and 3).
            # For case 2, we first redundantly assemble case 1 and then construct the transpose.
            # For cases 4 and 5, we take the forward Interpolate that corresponds to dropping the Cofunction,
            # and we separately compute the action against the dropped Cofunction within assemble().
            if not isinstance(dual_arg, ufl.Coargument):
                # Drop the Cofunction
                expr = expr._ufl_expr_reconstruct_(operand, dual_arg.function_space().dual())
            expr_args = extract_arguments(operand)
            if expr_args and expr_args[0].number() == 0:
                # Construct the symbolic forward Interpolate
                v0, v1 = expr.arguments()
                expr = ufl.replace(expr, {v0: v0.reconstruct(number=v1.number()),
                                          v1: v1.reconstruct(number=v0.number())})

        dual_arg, operand = expr.argument_slots()
        self.expr_renumbered = operand
        self.ufl_interpolate_renumbered = expr

        if not isinstance(dual_arg, ufl.Coargument):
            # Matrix-free assembly of 0-form or 1-form requires INC access
            if access and access != op2.INC:
                raise ValueError("Matfree adjoint interpolation requires INC access")
            access = op2.INC
        elif access is None:
            # Default access for forward 1-form or 2-form (forward and adjoint)
            access = op2.WRITE
        self.access = access

    def interpolate(self, *function, transpose=None, adjoint=False, default_missing_val=None):
        """
        .. warning::

            This method has been removed. Use the function :func:`interpolate` to return a symbolic
            :class:`Interpolate` object.
        """
        raise FutureWarning(
            "The 'interpolate' method on `Interpolator` objects has been "
            "removed. Use the `interpolate` function instead."
        )

    @abc.abstractmethod
    def _interpolate(self, *args, **kwargs):
        """
        Compute the interpolation operation of interest.

        .. note::
            This method is called when an :class:`Interpolate` object is being assembled.
        """
        pass

    def assemble(self, tensor=None, default_missing_val=None):
        """Assemble the operator (or its action)."""
        from firedrake.assemble import assemble
        needs_adjoint = self.ufl_interpolate_renumbered != self.ufl_interpolate
        arguments = self.ufl_interpolate.arguments()
        if len(arguments) == 2:
            # Assembling the operator
            res = tensor.petscmat if tensor else PETSc.Mat()
            # Get the interpolation matrix
            op2mat = self.callable()
            petsc_mat = op2mat.handle
            if needs_adjoint:
                # Out-of-place Hermitian transpose
                petsc_mat.hermitianTranspose(out=res)
            elif tensor:
                petsc_mat.copy(tensor.petscmat)
            else:
                res = petsc_mat
            return tensor or firedrake.AssembledMatrix(arguments, self.bcs, res)
        else:
            # Assembling the action
            cofunctions = ()
            if needs_adjoint:
                # The renumbered Interpolate has dropped Cofunctions.
                # We need to explicitly operate on them.
                dual_arg, _ = self.ufl_interpolate.argument_slots()
                if not isinstance(dual_arg, ufl.Coargument):
                    cofunctions = (dual_arg,)

            if needs_adjoint and len(arguments) == 0:
                Iu = self._interpolate(default_missing_val=default_missing_val)
                return assemble(ufl.Action(*cofunctions, Iu), tensor=tensor)
            else:
                return self._interpolate(*cofunctions, output=tensor, adjoint=needs_adjoint,
                                         default_missing_val=default_missing_val)


class DofNotDefinedError(Exception):
    r"""Raised when attempting to interpolate across function spaces where the
    target function space contains degrees of freedom (i.e. nodes) which cannot
    be defined in the source function space. This typically occurs when the
    target mesh covers a larger domain than the source mesh.

    Attributes
    ----------
    src_mesh : :func:`.Mesh`
        The source mesh.
    dest_mesh : :func:`.Mesh`
        The destination mesh.

    """

    def __init__(self, src_mesh, dest_mesh):
        self.src_mesh = src_mesh
        self.dest_mesh = dest_mesh

    def __str__(self):
        return (
            f"The given target function space on domain {repr(self.dest_mesh)} "
            "contains degrees of freedom which cannot cannot be defined in the "
            f"source function space on domain {repr(self.src_mesh)}. "
            "This may be because the target mesh covers a larger domain than the "
            "source mesh. To disable this error, set allow_missing_dofs=True."
        )


class CrossMeshInterpolator(Interpolator):
    """
    Interpolate a function from one mesh and function space to another.

    For arguments, see :class:`.Interpolator`.
    """

    @no_annotations
    def __init__(
        self,
        expr,
        V,
        subset=None,
        freeze_expr=False,
        access=None,
        bcs=None,
        allow_missing_dofs=False,
        matfree=True
    ):
        if subset:
            raise NotImplementedError("subset not implemented")
        if freeze_expr:
            # Probably just need to pass freeze_expr to the various
            # interpolators for this to work.
            raise NotImplementedError("freeze_expr not implemented")
        if bcs:
            raise NotImplementedError("bcs not implemented")
        if V.ufl_element().mapping() != "identity":
            # Identity mapping between reference cell and physical coordinates
            # implies point evaluation nodes. A more general version would
            # require finding the global coordinates of all quadrature points
            # of the target function space in the source mesh.
            raise NotImplementedError(
                "Can only interpolate into spaces with point evaluation nodes."
            )
        super().__init__(expr, V, subset, freeze_expr, access, bcs, allow_missing_dofs, matfree)

        if self.access != op2.WRITE:
            raise NotImplementedError("access other than op2.WRITE not implemented")

        expr = self.expr_renumbered
        self.arguments = extract_arguments(expr)
        self.nargs = len(self.arguments)

        if self._allow_missing_dofs:
            missing_points_behaviour = MissingPointsBehaviour.IGNORE
        else:
            missing_points_behaviour = MissingPointsBehaviour.ERROR

        # setup
        V_dest = V.function_space() if isinstance(V, firedrake.Function) else V
        src_mesh = extract_unique_domain(expr)
        dest_mesh = as_domain(V_dest)
        src_mesh_gdim = src_mesh.geometric_dimension
        dest_mesh_gdim = dest_mesh.geometric_dimension
        if src_mesh_gdim != dest_mesh_gdim:
            raise ValueError(
                "geometric dimensions of source and destination meshes must match"
            )
        self.src_mesh = src_mesh
        self.dest_mesh = dest_mesh

        # Create a VOM at the nodes of V_dest in src_mesh. We don't include halo
        # node coordinates because interpolation doesn't usually include halos.
        # NOTE: it is very important to set redundant=False, otherwise the
        # input ordering VOM will only contain the points on rank 0!
        # QUESTION: Should any of the below have annotation turned off?
        ufl_scalar_element = V_dest.ufl_element()
        if isinstance(ufl_scalar_element, finat.ufl.MixedElement):
            if type(ufl_scalar_element) is finat.ufl.MixedElement:
                raise TypeError("Interpolation matrix with MixedFunctionSpace requires MixedInterpolator")

            # For a VectorElement or TensorElement the correct
            # VectorFunctionSpace equivalent is built from the scalar
            # sub-element.
            ufl_scalar_element, = set(ufl_scalar_element.sub_elements)
            if ufl_scalar_element.reference_value_shape != ():
                raise NotImplementedError(
                    "Can't yet cross-mesh interpolate onto function spaces made from VectorElements or TensorElements made from sub elements with value shape other than ()."
                )

        from firedrake.assemble import assemble
        V_dest_vec = firedrake.VectorFunctionSpace(dest_mesh, ufl_scalar_element)
        f_dest_node_coords = interpolate(dest_mesh.coordinates, V_dest_vec)
        f_dest_node_coords = assemble(f_dest_node_coords)
        dest_node_coords = f_dest_node_coords.dat.data_ro.reshape(-1, dest_mesh_gdim)
        try:
            self.vom_dest_node_coords_in_src_mesh = firedrake.VertexOnlyMesh(
                src_mesh,
                dest_node_coords,
                redundant=False,
                missing_points_behaviour=missing_points_behaviour,
            )
        except VertexOnlyMeshMissingPointsError:
            raise DofNotDefinedError(src_mesh, dest_mesh)
        # vom_dest_node_coords_in_src_mesh uses the parallel decomposition of
        # the global node coordinates of V_dest in the SOURCE mesh (src_mesh).
        # I first point evaluate my expression at these locations, giving a
        # P0DG function on the VOM. As described in the manual, this is an
        # interpolation operation.
        shape = V_dest.ufl_function_space().value_shape
        if len(shape) == 0:
            fs_type = firedrake.FunctionSpace
        elif len(shape) == 1:
            fs_type = partial(firedrake.VectorFunctionSpace, dim=shape[0])
        else:
            fs_type = partial(firedrake.TensorFunctionSpace, shape=shape)
        P0DG_vom = fs_type(self.vom_dest_node_coords_in_src_mesh, "DG", 0)
        self.point_eval_interpolate = interpolate(self.expr_renumbered, P0DG_vom)
        # The parallel decomposition of the nodes of V_dest in the DESTINATION
        # mesh (dest_mesh) is retrieved using the input_ordering attribute of the
        # VOM. This again is an interpolation operation, which, under the hood
        # is a PETSc SF reduce.
        P0DG_vom_i_o = fs_type(
            self.vom_dest_node_coords_in_src_mesh.input_ordering, "DG", 0
        )
        self.to_input_ordering_interpolate = interpolate(
            firedrake.TrialFunction(P0DG_vom), P0DG_vom_i_o
        )
        # The P0DG function outputted by the above interpolation has the
        # correct parallel decomposition for the nodes of V_dest in dest_mesh so
        # we can safely assign the dat values. This is all done in the actual
        # interpolation method below.

    @PETSc.Log.EventDecorator()
    def _interpolate(
        self,
        *function,
        output=None,
        transpose=None,
        adjoint=False,
        default_missing_val=None,
        **kwargs,
    ):
        """Compute the interpolation.

        For arguments, see :class:`.Interpolator`.
        """
        from firedrake.assemble import assemble

        if transpose is not None:
            warnings.warn("'transpose' argument is deprecated, use 'adjoint' instead", FutureWarning)
            adjoint = transpose or adjoint
        if adjoint and not self.nargs:
            raise ValueError(
                "Can currently only apply adjoint interpolation with arguments."
            )
        if self.nargs != len(function):
            raise ValueError(
                "Passed %d Functions to interpolate, expected %d"
                % (len(function), self.nargs)
            )

        if self.nargs:
            (f_src,) = function
            if not hasattr(f_src, "dat"):
                raise ValueError(
                    "The expression had arguments: we therefore need to be given a Function (not an expression) to interpolate!"
                )
        else:
            f_src = self.expr

        if adjoint:
            try:
                V_dest = self.expr.function_space().dual()
            except AttributeError:
                if self.nargs:
                    V_dest = self.arguments[-1].function_space().dual()
                else:
                    coeffs = extract_coefficients(self.expr)
                    if len(coeffs):
                        V_dest = coeffs[0].function_space().dual()
                    else:
                        raise ValueError(
                            "Can't adjoint interpolate an expression with no coefficients or arguments."
                        )
        else:
            if isinstance(self.V, (firedrake.Function, firedrake.Cofunction)):
                V_dest = self.V.function_space()
            else:
                V_dest = self.V
        if output:
            if output.function_space() != V_dest:
                raise ValueError("Given output has the wrong function space!")
        else:
            if isinstance(self.V, (firedrake.Function, firedrake.Cofunction)):
                output = self.V
            else:
                output = firedrake.Function(V_dest)

        if not adjoint:
            if f_src is self.expr:
                # f_src is already contained in self.point_eval_interpolate
                assert not self.nargs
                f_src_at_dest_node_coords_src_mesh_decomp = (
                    assemble(self.point_eval_interpolate)
                )
            else:
                f_src_at_dest_node_coords_src_mesh_decomp = (
                    assemble(action(self.point_eval_interpolate, f_src))
                )
            f_src_at_dest_node_coords_dest_mesh_decomp = firedrake.Function(
                self.to_input_ordering_interpolate.function_space()
            )
            # We have to create the Function before interpolating so we can
            # set default missing values (if requested).
            if default_missing_val is not None:
                f_src_at_dest_node_coords_dest_mesh_decomp.dat.data_wo[
                    :
                ] = default_missing_val
            elif self._allow_missing_dofs:
                # If we have allowed missing points we know we might end up
                # with points in the target mesh that are not in the source
                # mesh. However, since we haven't specified a default missing
                # value we expect the interpolation to leave these points
                # unchanged. By setting the dat values to NaN we can later
                # identify these points and skip over them when assigning to
                # the output function.
                f_src_at_dest_node_coords_dest_mesh_decomp.dat.data_wo[:] = numpy.nan

            interp = action(self.to_input_ordering_interpolate, f_src_at_dest_node_coords_src_mesh_decomp)
            assemble(interp, tensor=f_src_at_dest_node_coords_dest_mesh_decomp)

            # we can now confidently assign this to a function on V_dest
            if self._allow_missing_dofs and default_missing_val is None:
                indices = numpy.where(
                    ~numpy.isnan(f_src_at_dest_node_coords_dest_mesh_decomp.dat.data_ro)
                )[0]
                output.dat.data_wo[
                    indices
                ] = f_src_at_dest_node_coords_dest_mesh_decomp.dat.data_ro[indices]
            else:
                output.dat.data_wo[
                    :
                ] = f_src_at_dest_node_coords_dest_mesh_decomp.dat.data_ro[:]

        else:
            # adjoint interpolation

            # f_src is a cofunction on V_dest.dual as originally specified when
            # creating the interpolator. Our first adjoint operation is to
            # assign the dat values to a P0DG cofunction on our input ordering
            # VOM. This has the parallel decomposition V_dest on our orinally
            # specified dest_mesh. We can therefore safely create a P0DG
            # cofunction on the input-ordering VOM (which has this parallel
            # decomposition and ordering) and assign the dat values.
            f_src_at_dest_node_coords_dest_mesh_decomp = firedrake.Cofunction(
                self.to_input_ordering_interpolate.function_space().dual()
            )
            f_src_at_dest_node_coords_dest_mesh_decomp.dat.data_wo[
                :
            ] = f_src.dat.data_ro[:]

            # The rest of the adjoint interpolation is merely the composition
            # of the adjoint interpolators in the reverse direction. NOTE: I
            # don't have to worry about skipping over missing points here
            # because I'm going from the input ordering VOM to the original VOM
            # and all points from the input ordering VOM are in the original.
            interp = action(expr_adjoint(self.to_input_ordering_interpolate), f_src_at_dest_node_coords_dest_mesh_decomp)
            f_src_at_src_node_coords = assemble(interp)
            # NOTE: if I wanted the default missing value to be applied to
            # adjoint interpolation I would have to do it here. However,
            # this would require me to implement default missing values for
            # adjoint interpolation from a point evaluation interpolator
            # which I haven't done. I wonder if it is necessary - perhaps the
            # adjoint operator always sets all the values of the resulting
            # cofunction? My initial attempt to insert setting the dat values
            # prior to performing the multHermitian operation in
            # SameMeshInterpolator.interpolate did not effect the result. For
            # now, I say in the docstring that it only applies to forward
            # interpolation.
            interp = action(expr_adjoint(self.point_eval_interpolate), f_src_at_src_node_coords)
            assemble(interp, tensor=output)

        return output


class SameMeshInterpolator(Interpolator):
    """
    An interpolator for interpolation within the same mesh or onto a validly-
    defined :func:`.VertexOnlyMesh`.

    For arguments, see :class:`.Interpolator`.
    """

    @no_annotations
    def __init__(self, expr, V, subset=None, freeze_expr=False, access=None,
                 bcs=None, matfree=True, allow_missing_dofs=False, **kwargs):
        if subset is None:
            if isinstance(expr, ufl.Interpolate):
                operand, = expr.ufl_operands
            else:
                operand = expr
            target_mesh = as_domain(V)
            source_mesh = extract_unique_domain(operand) or target_mesh
            target = target_mesh.topology
            source = source_mesh.topology
            if all(isinstance(m, firedrake.mesh.MeshTopology) for m in [target, source]) and target is not source:
                composed_map, result_integral_type = source.trans_mesh_entity_map(target, "cell", "everywhere", None)
                if result_integral_type != "cell":
                    raise AssertionError("Only cell-cell interpolation supported")
                indices_active = composed_map.indices_active_with_halo
                make_subset = not indices_active.all()
                make_subset = target.comm.allreduce(make_subset, op=MPI.LOR)
                if make_subset:
                    if not allow_missing_dofs:
                        raise ValueError("iteration (sub)set unclear: run with `allow_missing_dofs=True`")
                    subset = op2.Subset(target.cell_set, numpy.where(indices_active))
                else:
                    # Do not need subset as target <= source.
                    pass
        super().__init__(expr, V, subset=subset, freeze_expr=freeze_expr,
                         access=access, bcs=bcs, matfree=matfree, allow_missing_dofs=allow_missing_dofs)
        expr = self.ufl_interpolate_renumbered
        try:
            self.callable = make_interpolator(expr, V, subset, self.access, bcs=bcs, matfree=matfree)
        except FIAT.hdiv_trace.TraceError:
            raise NotImplementedError("Can't interpolate onto traces sorry")
        self.arguments = expr.arguments()

    @PETSc.Log.EventDecorator()
    def _interpolate(self, *function, output=None, transpose=None, adjoint=False, **kwargs):
        """Compute the interpolation.

        For arguments, see :class:`.Interpolator`.
        """

        if transpose is not None:
            warnings.warn("'transpose' argument is deprecated, use 'adjoint' instead", FutureWarning)
            adjoint = transpose or adjoint
        try:
            assembled_interpolator = self.frozen_assembled_interpolator
            copy_required = True
        except AttributeError:
            assembled_interpolator = self.callable()
            copy_required = False  # Return the original
            if self.freeze_expr:
                if len(self.arguments) == 2:
                    # Interpolation operator
                    self.frozen_assembled_interpolator = assembled_interpolator
                else:
                    # Interpolation action
                    self.frozen_assembled_interpolator = assembled_interpolator.copy()

        if len(self.arguments) == 2 and len(function) > 0:
            function, = function
            if not hasattr(function, "dat"):
                raise ValueError("The expression had arguments: we therefore need to be given a Function (not an expression) to interpolate!")
            if adjoint:
                mul = assembled_interpolator.handle.multHermitian
                col, row = self.arguments
            else:
                mul = assembled_interpolator.handle.mult
                row, col = self.arguments
            V = row.function_space().dual()
            assert function.function_space() == col.function_space()

            result = output or firedrake.Function(V)
            with function.dat.vec_ro as x, result.dat.vec_wo as out:
                if x is not out:
                    mul(x, out)
                else:
                    out_ = out.duplicate()
                    mul(x, out_)
                    out_.copy(result=out)
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
                if len(self.arguments) == 0:
                    return assembled_interpolator.dat.data.item()
                elif copy_required:
                    return assembled_interpolator.copy()
                else:
                    return assembled_interpolator


@PETSc.Log.EventDecorator()
def make_interpolator(expr, V, subset, access, bcs=None, matfree=True):
    if not isinstance(expr, ufl.Interpolate):
        raise ValueError(f"Expecting to interpolate a ufl.Interpolate, got {type(expr).__name__}.")
    dual_arg, operand = expr.argument_slots()
    target_mesh = as_domain(dual_arg)
    source_mesh = extract_unique_domain(operand) or target_mesh
    vom_onto_other_vom = ((source_mesh is not target_mesh)
                          and isinstance(source_mesh.topology, VertexOnlyMeshTopology)
                          and isinstance(target_mesh.topology, VertexOnlyMeshTopology))

    arguments = expr.arguments()
    rank = len(arguments)
    if rank <= 1:
        if rank == 0:
            R = firedrake.FunctionSpace(target_mesh, "Real", 0)
            f = firedrake.Function(R, dtype=utils.ScalarType)
        elif isinstance(V, firedrake.Function):
            f = V
            V = f.function_space()
        else:
            V_dest = arguments[0].function_space().dual()
            f = firedrake.Function(V_dest)
            if access in {firedrake.MIN, firedrake.MAX}:
                finfo = numpy.finfo(f.dat.dtype)
                if access == firedrake.MIN:
                    val = firedrake.Constant(finfo.max)
                else:
                    val = firedrake.Constant(finfo.min)
                f.assign(val)
        tensor = f.dat
    elif rank == 2:
        if isinstance(V, firedrake.Function):
            raise ValueError("Cannot interpolate an expression with an argument into a Function")
        Vrow = arguments[0].function_space()
        Vcol = arguments[1].function_space()
        if len(Vrow) > 1 or len(Vcol) > 1:
            raise TypeError("Interpolation matrix with MixedFunctionSpace requires MixedInterpolator")
        if isinstance(target_mesh.topology, VertexOnlyMeshTopology) and target_mesh is not source_mesh and not vom_onto_other_vom:
            if not isinstance(target_mesh.topology, VertexOnlyMeshTopology):
                raise NotImplementedError("Can only interpolate onto a VertexOnlyMesh")
            if target_mesh.geometric_dimension != source_mesh.geometric_dimension:
                raise ValueError("Cannot interpolate onto a mesh of a different geometric dimension")
            if not hasattr(target_mesh, "_parent_mesh") or target_mesh._parent_mesh is not source_mesh:
                raise ValueError("Can only interpolate across meshes where the source mesh is the parent of the target")

        if vom_onto_other_vom:
            # We make our own linear operator for this case using PETSc SFs
            tensor = None
        else:
            Vrow_map = get_interp_node_map(source_mesh, target_mesh, Vrow)
            Vcol_map = get_interp_node_map(source_mesh, target_mesh, Vcol)
            sparsity = op2.Sparsity((Vrow.dof_dset, Vcol.dof_dset),
                                    [(Vrow_map, Vcol_map, None)],  # non-mixed
                                    name="%s_%s_sparsity" % (Vrow.name, Vcol.name),
                                    nest=False,
                                    block_sparse=True)
            tensor = op2.Mat(sparsity)
        f = tensor
    else:
        raise ValueError(f"Cannot interpolate an expression with {rank} arguments")

    if vom_onto_other_vom:
        wrapper = VomOntoVomWrapper(V, source_mesh, target_mesh, operand, matfree)
        # NOTE: get_dat_mpi_type ensures we get the correct MPI type for the
        # data, including the correct data size and dimensional information
        # (so for vector function spaces in 2 dimensions we might need a
        # concatenation of 2 MPI.DOUBLE types when we are in real mode)
        if tensor is not None:
            # Callable will do interpolation into our pre-supplied function f
            # when it is called.
            assert f.dat is tensor
            wrapper.mpi_type, _ = get_dat_mpi_type(f.dat)
            assert len(arguments) == 1

            def callable():
                wrapper.forward_operation(f.dat)
                return f
        else:
            assert len(arguments) == 2
            assert tensor is None
            # we know we will be outputting either a function or a cofunction,
            # both of which will use a dat as a data carrier. At present, the
            # data type does not depend on function space dimension, so we can
            # safely use the argument function space. NOTE: If this changes
            # after cofunctions are fully implemented, this will need to be
            # reconsidered.
            temp_source_func = firedrake.Function(Vcol)
            wrapper.mpi_type, _ = get_dat_mpi_type(temp_source_func.dat)

            # Leave wrapper inside a callable so we can access the handle
            # property. If matfree is True, then the handle is a PETSc SF
            # pretending to be a PETSc Mat. If matfree is False, then this
            # will be a PETSc Mat representing the equivalent permutation
            # matrix
            def callable():
                return wrapper

        return callable
    else:
        loops = []
        # Initialise to zero if needed
        if access is op2.INC:
            loops.append(tensor.zero)

        # Arguments in the operand are allowed to be from a MixedFunctionSpace
        # We need to split the target space V and generate separate kernels
        if len(arguments) == 2:
            # Matrix case assumes that the spaces are not mixed
            expressions = {(0,): expr}
        elif isinstance(dual_arg, Coargument):
            # Split in the coargument
            expressions = dict(firedrake.formmanipulation.split_form(expr))
        else:
            # Split in the cofunction: split_form can only split in the coargument
            # Replace the cofunction with a coargument to construct the Jacobian
            interp = expr._ufl_expr_reconstruct_(operand, V)
            # Split the Jacobian into blocks
            interp_split = dict(firedrake.formmanipulation.split_form(interp))
            # Split the cofunction
            dual_split = dict(firedrake.formmanipulation.split_form(dual_arg))
            # Combine the splits by taking their action
            expressions = {i: action(interp_split[i], dual_split[i[-1:]]) for i in interp_split}

        # Interpolate each sub expression into each function space
        for indices, sub_expr in expressions.items():
            sub_tensor = tensor[indices[0]] if rank == 1 else tensor
            loops.extend(_interpolator(sub_tensor, sub_expr, subset, access, bcs=bcs))
        # Apply bcs
        if bcs and rank == 1:
            loops.extend(partial(bc.apply, f) for bc in bcs)

        def callable(loops, f):
            for l in loops:
                l()
            return f

        return partial(callable, loops, f)


@utils.known_pyop2_safe
def _interpolator(tensor, expr, subset, access, bcs=None):
    if isinstance(expr, ufl.ZeroBaseForm):
        # Zero simplification, avoid code-generation
        if access is op2.INC:
            return ()
        elif access is op2.WRITE:
            return (partial(tensor.zero, subset=subset),)
        # Unclear how to avoid codegen for MIN and MAX
        # Reconstruct the expression as an Interpolate
        V = expr.arguments()[-1].function_space().dual()
        expr = interpolate(ufl.zero(V.value_shape), V)

    if not isinstance(expr, ufl.Interpolate):
        raise ValueError("Expecting to interpolate a ufl.Interpolate")

    arguments = expr.arguments()
    dual_arg, operand = expr.argument_slots()
    V = dual_arg.arguments()[0].function_space()

    try:
        to_element = create_element(V.ufl_element())
    except KeyError:
        # FInAT only elements
        raise NotImplementedError("Don't know how to create FIAT element for %s" % V.ufl_element())

    if access is op2.READ:
        raise ValueError("Can't have READ access for output function")

    # NOTE: The par_loop is always over the target mesh cells.
    target_mesh = as_domain(V)
    source_mesh = extract_unique_domain(operand) or target_mesh
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        if target_mesh is not source_mesh:
            if not isinstance(target_mesh.topology, VertexOnlyMeshTopology):
                raise NotImplementedError("Can only interpolate onto a Vertex Only Mesh")
            if target_mesh.geometric_dimension != source_mesh.geometric_dimension:
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
            try:
                cell = operand.ufl_element().ufl_cell()
            except AttributeError:
                # expression must be pure function of spatial coordinates so
                # domain has correct ufl cell
                cell = source_mesh.ufl_cell()
            to_element = rebuild(to_element, cell, rt_var_name)

    cell_set = target_mesh.cell_set
    if subset is not None:
        assert subset.superset == cell_set
        cell_set = subset

    parameters = {}
    parameters['scalar_type'] = utils.ScalarType

    copyin = ()
    copyout = ()

    # For the matfree adjoint 1-form and the 0-form, the cellwise kernel will add multiple
    # contributions from the facet DOFs of the dual argument.
    # The incoming Cofunction needs to be weighted by the reciprocal of the DOF multiplicity.
    needs_weight = isinstance(dual_arg, ufl.Cofunction) and not to_element.is_dg()
    if needs_weight:
        # Create a buffer for the weighted Cofunction
        W = dual_arg.function_space()
        v = firedrake.Function(W)
        expr = expr._ufl_expr_reconstruct_(operand, v=v)
        copyin += (partial(dual_arg.dat.copy, v.dat),)

        # Compute the reciprocal of the DOF multiplicity
        wdat = W.make_dat()
        m_ = get_interp_node_map(source_mesh, target_mesh, W)
        wsize = W.finat_element.space_dimension() * W.block_size
        kernel_code = f"""
        void multiplicity(PetscScalar *restrict w) {{
            for (PetscInt i=0; i<{wsize}; i++) w[i] += 1;
        }}"""
        kernel = op2.Kernel(kernel_code, "multiplicity")
        op2.par_loop(kernel, cell_set, wdat(op2.INC, m_))
        with wdat.vec as w:
            w.reciprocal()

        # Create a callable to apply the weight
        with wdat.vec_ro as w, v.dat.vec as y:
            copyin += (partial(y.pointwiseMult, y, w),)

    # We need to pass both the ufl element and the finat element
    # because the finat elements might not have the right mapping
    # (e.g. L2 Piola, or tensor element with symmetries)
    # FIXME: for the runtime unknown point set (for cross-mesh
    # interpolation) we have to pass the finat element we construct
    # here. Ideally we would only pass the UFL element through.
    kernel = compile_expression(cell_set.comm, expr, to_element, V.ufl_element(),
                                domain=source_mesh, parameters=parameters)
    ast = kernel.ast
    oriented = kernel.oriented
    needs_cell_sizes = kernel.needs_cell_sizes
    coefficient_numbers = kernel.coefficient_numbers
    needs_external_coords = kernel.needs_external_coords
    name = kernel.name
    kernel = op2.Kernel(ast, name, requires_zeroed_output_arguments=(access is not op2.INC),
                        flop_count=kernel.flop_count, events=(kernel.event,))

    parloop_args = [kernel, cell_set]

    coefficients = tsfc_interface.extract_numbered_coefficients(expr, coefficient_numbers)
    if needs_external_coords:
        coefficients = [source_mesh.coordinates] + coefficients

    if any(c.dat == tensor for c in coefficients):
        output = tensor
        tensor = op2.Dat(tensor.dataset)
        if access is not op2.WRITE:
            copyin += (partial(output.copy, tensor), )
        copyout += (partial(tensor.copy, output), )
    if isinstance(tensor, op2.Global):
        parloop_args.append(tensor(access))
    elif isinstance(tensor, op2.Dat):
        V_dest = arguments[-1].function_space()
        m_ = get_interp_node_map(source_mesh, target_mesh, V_dest)
        parloop_args.append(tensor(access, m_))
    else:
        assert access == op2.WRITE  # Other access descriptors not done for Matrices.
        Vrow = arguments[0].function_space()
        Vcol = arguments[1].function_space()
        assert tensor.handle.getSize() == (Vrow.dim(), Vcol.dim())
        rows_map = get_interp_node_map(source_mesh, target_mesh, Vrow)
        columns_map = get_interp_node_map(source_mesh, target_mesh, Vcol)

        lgmaps = None
        if bcs:
            if ufl.duals.is_dual(Vrow):
                Vrow = Vrow.dual()
            if ufl.duals.is_dual(Vcol):
                Vcol = Vcol.dual()
            bc_rows = [bc for bc in bcs if bc.function_space() == Vrow]
            bc_cols = [bc for bc in bcs if bc.function_space() == Vcol]
            lgmaps = [(Vrow.local_to_global_map(bc_rows), Vcol.local_to_global_map(bc_cols))]
        parloop_args.append(tensor(access, (rows_map, columns_map), lgmaps=lgmaps))
    if oriented:
        co = target_mesh.cell_orientations()
        parloop_args.append(co.dat(op2.READ, co.cell_node_map()))
    if needs_cell_sizes:
        cs = source_mesh.cell_sizes
        parloop_args.append(cs.dat(op2.READ, cs.cell_node_map()))

    for coefficient in coefficients:
        m_ = get_interp_node_map(source_mesh, target_mesh, coefficient.function_space())
        parloop_args.append(coefficient.dat(op2.READ, m_))

    for const in extract_firedrake_constants(expr):
        parloop_args.append(const.dat(op2.READ))

    # Finally, add the target mesh reference coordinates if they appear in the kernel
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
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
            if any(arg.name == rt_var_name for arg in kernel.code[name].args):
                # Add the coordinates of the target mesh quadrature points in the
                # source mesh's reference cell as an extra argument for the inner
                # loop. (With a vertex only mesh this is a single point for each
                # vertex cell.)
                target_ref_coords = target_mesh.reference_coordinates
                m_ = target_ref_coords.cell_node_map()
                parloop_args.append(target_ref_coords.dat(op2.READ, m_))

    parloop = op2.ParLoop(*parloop_args)
    if isinstance(tensor, op2.Mat):
        return parloop, tensor.assemble
    else:
        return copyin + (parloop, ) + copyout


def get_interp_node_map(source_mesh, target_mesh, fs):
    """Return the map between cells of the target mesh and nodes of the function space.

    If the function space is defined on the source mesh then the node map is composed
    with a map between target and source cells.
    """
    if isinstance(target_mesh.topology, VertexOnlyMeshTopology):
        coeff_mesh = fs.mesh()
        m_ = fs.cell_node_map()
        if coeff_mesh is target_mesh or not coeff_mesh:
            # NOTE: coeff_mesh is None is allowed e.g. when interpolating from
            # a Real space
            pass
        elif coeff_mesh is source_mesh:
            if m_:
                # Since the par_loop is over the target mesh cells we need to
                # compose a map that takes us from target mesh cells to the
                # function space nodes on the source mesh.
                if source_mesh.extruded:
                    # ExtrudedSet cannot be a map target so we need to build
                    # this ourselves
                    m_ = vom_cell_parent_node_map_extruded(target_mesh, m_)
                else:
                    m_ = compose_map_and_cache(target_mesh.cell_parent_cell_map, m_)
            else:
                # m_ is allowed to be None when interpolating from a Real space,
                # even in the trans-mesh case.
                pass
        else:
            raise ValueError("Have coefficient with unexpected mesh")
    else:
        m_ = fs.entity_node_map(target_mesh.topology, "cell", None, None)
    return m_


try:
    _expr_cachedir = os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"]
except KeyError:
    _expr_cachedir = os.path.join(tempfile.gettempdir(),
                                  f"firedrake-tsfc-expression-kernel-cache-uid{os.getuid()}")


def _compile_expression_key(comm, expr, to_element, ufl_element, domain, parameters) -> tuple[Hashable, ...]:
    """Generate a cache key suitable for :func:`tsfc.compile_expression_dual_evaluation`."""
    dual_arg, operand = expr.argument_slots()
    return (hash_expr(operand), type(dual_arg), hash(ufl_element), utils.tuplify(parameters))


@memory_and_disk_cache(
    hashkey=_compile_expression_key,
    cachedir=tsfc_interface._cachedir
)
@PETSc.Log.EventDecorator()
def compile_expression(comm, *args, **kwargs):
    return compile_expression_dual_evaluation(*args, **kwargs)


@singledispatch
def rebuild(element, expr_cell, rt_var_name):
    raise NotImplementedError(f"Cross mesh interpolation not implemented for a {element} element.")


@rebuild.register(finat.fiat_elements.ScalarFiatElement)
def rebuild_dg(element, expr_cell, rt_var_name):
    # To tabulate on the given element (which is on a different mesh to the
    # expression) we must do so at runtime. We therefore create a quadrature
    # element with runtime points to evaluate for each point in the element's
    # dual basis. This exists on the same reference cell as the input element
    # and we can interpolate onto it before mapping the result back onto the
    # target space.
    expr_tdim = expr_cell.topological_dimension
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
    rule = finat.quadrature.QuadratureRule(rule_pointset, weights=weights)
    return finat.QuadratureElement(as_fiat_cell(expr_cell), rule)


@rebuild.register(finat.TensorFiniteElement)
def rebuild_te(element, expr_cell, rt_var_name):
    return finat.TensorFiniteElement(rebuild(element.base_element,
                                             expr_cell, rt_var_name),
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
    if not isinstance(vertex_only_mesh.topology, VertexOnlyMeshTopology):
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
    matfree : bool
        If ``False``, the matrix representating the permutation of the points is
        constructed and used to perform the interpolation. If ``True``, then the
        interpolation is performed using the broadcast and reduce operations on the
        PETSc Star Forest.
    """

    def __init__(self, V, source_vom, target_vom, expr, matfree):
        arguments = extract_arguments(expr)
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
        self.dummy_mat = VomOntoVomDummyMat(
            original_vom.input_ordering_without_halos_sf, reduce, V, source_vom, expr, arguments
        )
        if matfree:
            # If matfree, we use the SF to perform the interpolation
            self.handle = self.dummy_mat._wrap_dummy_mat()
        else:
            # Otherwise we create the permutation matrix
            self.handle = self.dummy_mat._create_permutation_mat()

    @property
    def mpi_type(self):
        """
        The MPI type to use for the PETSc SF.

        Should correspond to the underlying data type of the PETSc Vec.
        """
        return self.handle.mpi_type

    @mpi_type.setter
    def mpi_type(self, val):
        self.dummy_mat.mpi_type = val

    def forward_operation(self, target_dat):
        coeff = self.dummy_mat.expr_as_coeff()
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
        vec, whilst the adjoint action (accessed via the `multHermitian`
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
        # Calculate correct local and global sizes for the matrix
        nroots, leaves, _ = sf.getGraph()
        self.nleaves = len(leaves)
        self._local_sizes = V.comm.allgather(nroots)
        self.source_size = (self.V.block_size * nroots, self.V.block_size * sum(self._local_sizes))
        self.target_size = (
            self.V.block_size * self.nleaves,
            self.V.block_size * V.comm.allreduce(self.nleaves, op=MPI.SUM),
        )

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
                arg_coeff.dat.data_wo[:] = source_vec.getArray(readonly=True).reshape(
                    arg_coeff.dat.data_wo.shape
                )
                coeff_expr = ufl.replace(self.expr, {arg: arg_coeff})
            coeff = firedrake.Function(P0DG).interpolate(coeff_expr)
        return coeff

    def reduce(self, source_vec, target_vec):
        source_arr = source_vec.getArray(readonly=True)
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
        source_arr = source_vec.getArray(readonly=True)
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

    def mult(self, mat, source_vec, target_vec):
        # need to evaluate expression before doing mult
        coeff = self.expr_as_coeff(source_vec)
        with coeff.dat.vec_ro as coeff_vec:
            if self.forward_reduce:
                self.reduce(coeff_vec, target_vec)
            else:
                self.broadcast(coeff_vec, target_vec)

    def multHermitian(self, mat, source_vec, target_vec):
        self.multTranspose(mat, source_vec, target_vec)

    def multTranspose(self, mat, source_vec, target_vec):
        # can only do adjoint if our expression exclusively contains a
        # single argument, making the application of the adjoint operator
        # straightforward (haven't worked out how to do this otherwise!)
        if not len(self.arguments) == 1:
            raise NotImplementedError(
                "Can only apply adjoint to expressions with one argument!"
            )
        if self.arguments[0] is not self.expr:
            raise NotImplementedError(
                "Can only apply adjoint to expressions consisting of a single argument at the moment."
            )
        if self.forward_reduce:
            self.broadcast(source_vec, target_vec)
        else:
            # We need to ensure the target vec is zeroed for SF Reduce to
            # represent multHermitian in case the interpolation matrix is not
            # square (in which case it will have columns which are zero). This
            # happens when we interpolate from an input-ordering vertex-only
            # mesh to an immersed vertex-only mesh where the input ordering
            # contains points that are not in the immersed mesh. The resulting
            # interpolation matrix will have columns of zeros for the points
            # that are not in the immersed mesh. The adjoint interpolation
            # matrix will then have rows of zeros for those points.
            target_vec.zeroEntries()
            self.reduce(source_vec, target_vec)

    def _create_permutation_mat(self):
        """Creates the PETSc matrix that represents the interpolation operator from a vertex-only mesh to
        its input ordering vertex-only mesh"""
        mat = PETSc.Mat().createAIJ((self.target_size, self.source_size), nnz=1, comm=self.V.comm)
        mat.setUp()
        start = sum(self._local_sizes[:self.V.comm.rank])
        end = start + self.source_size[0]
        contiguous_indices = numpy.arange(start, end, dtype=utils.IntType)
        perm = numpy.zeros(self.nleaves, dtype=utils.IntType)
        self.sf.bcastBegin(MPI.INT, contiguous_indices, perm, MPI.REPLACE)
        self.sf.bcastEnd(MPI.INT, contiguous_indices, perm, MPI.REPLACE)
        rows = numpy.arange(self.target_size[0] + 1, dtype=utils.IntType)
        cols = (self.V.block_size * perm[:, None] + numpy.arange(self.V.block_size, dtype=utils.IntType)[None, :]).reshape(-1)
        mat.setValuesCSR(rows, cols, numpy.ones_like(cols, dtype=utils.IntType))
        mat.assemble()
        if self.forward_reduce:
            mat.transpose()
        return mat

    def _wrap_dummy_mat(self):
        mat = PETSc.Mat().create(comm=self.V.comm)
        if self.forward_reduce:
            mat_size = (self.source_size, self.target_size)
        else:
            mat_size = (self.target_size, self.source_size)
        mat.setSizes(mat_size)
        mat.setType(mat.Type.PYTHON)
        mat.setPythonContext(self)
        mat.setUp()
        return mat

    def duplicate(self, mat=None, op=None):
        return self._wrap_dummy_mat()


class MixedInterpolator(Interpolator):
    """A reusable interpolation object between MixedFunctionSpaces.

    Parameters
    ----------
    expr
        The underlying ufl.Interpolate or the operand to the ufl.Interpolate.
    V
        The :class:`.FunctionSpace` or :class:`.Function` to
        interpolate into.
    bcs
        A list of boundary conditions.
    **kwargs
        Any extra kwargs are passed on to the sub Interpolators.
        For details see :class:`firedrake.interpolation.Interpolator`.
    """
    def __init__(self, expr, V, bcs=None, **kwargs):
        super(MixedInterpolator, self).__init__(expr, V, bcs=bcs, **kwargs)
        expr = self.ufl_interpolate
        self.arguments = expr.arguments()
        # Get the primal spaces
        spaces = tuple(a.function_space().dual() if isinstance(a, Coargument) else a.function_space()
                       for a in self.arguments)
        # TODO consider a stricter equality test for indexed MixedFunctionSpace
        # See https://github.com/firedrakeproject/firedrake/issues/4668
        space_equals = lambda V1, V2: V1 == V2 and V1.parent == V2.parent and V1.index == V2.index

        # We need a Coargument in order to split the Interpolate
        needs_action = len([a for a in self.arguments if isinstance(a, Coargument)]) == 0
        if needs_action:
            dual_arg, operand = expr.argument_slots()
            # Split the dual argument
            dual_split = dict(firedrake.formmanipulation.split_form(dual_arg))
            # Create the Jacobian to be split into blocks
            expr = expr._ufl_expr_reconstruct_(operand, V)

        Isub = {}
        # Split in the arguments of the Interpolate
        for indices, form in firedrake.formmanipulation.split_form(expr):
            if isinstance(form, ufl.ZeroBaseForm):
                # Ensure block sparsity
                continue
            vi, _ = form.argument_slots()
            Vtarget = vi.function_space().dual()
            sub_bcs = []
            for space, index in zip(spaces, indices):
                subspace = space.sub(index)
                sub_bcs.extend(bc for bc in bcs if space_equals(bc.function_space(), subspace))
            if needs_action:
                # Take the action of each sub-cofunction against each block
                form = action(form, dual_split[indices[-1:]])

            Isub[indices] = Interpolator(form, Vtarget, bcs=sub_bcs, **kwargs)

        self._sub_interpolators = Isub
        self.callable = self._assemble_matnest

    def __getitem__(self, item):
        return self._sub_interpolators[item]

    def __iter__(self):
        return iter(self._sub_interpolators)

    def _assemble_matnest(self):
        """Assemble the operator."""
        shape = tuple(len(a.function_space()) for a in self.arguments)
        blocks = numpy.full(shape, PETSc.Mat(), dtype=object)
        # Assemble the sparse block matrix
        for i in self:
            blocks[i] = self[i].callable().handle
        petscmat = PETSc.Mat().createNest(blocks)
        tensor = firedrake.AssembledMatrix(self.arguments, self.bcs, petscmat)
        return tensor.M

    def _interpolate(self, *function, output=None, adjoint=False, **kwargs):
        """Assemble the action."""
        rank = len(self.arguments)
        if rank == 0:
            result = sum(self[i].assemble(**kwargs) for i in self)
            return output.assign(result) if output else result

        if output is None:
            output = firedrake.Function(self.arguments[-1].function_space().dual())

        if rank == 1:
            for k, sub_tensor in enumerate(output.subfunctions):
                sub_tensor.assign(sum(self[i].assemble(**kwargs) for i in self if i[0] == k))
        elif rank == 2:
            for k, sub_tensor in enumerate(output.subfunctions):
                sub_tensor.assign(sum(self[i]._interpolate(*function, adjoint=adjoint, **kwargs)
                                      for i in self if i[0] == k))
        return output
