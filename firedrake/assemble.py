import abc
from collections import OrderedDict
import functools
import itertools
from itertools import product
import operator

import cachetools
import finat
import firedrake
import numpy
from tsfc import kernel_args
from tsfc.finatinterface import create_element
import ufl
from firedrake import (assemble_expressions, extrusion_utils as eutils, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.functionspacedata import entity_dofs_key, entity_permutations_key
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.slate.slac.kernel_builder import CellFacetKernelArg, LayerCountKernelArg
from firedrake.utils import ScalarType, tuplify
from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError
from pyop2.utils import cached_property


__all__ = "assemble",


_FORM_CACHE_KEY = "firedrake.assemble.FormAssembler"
"""Entry used in form cache to try and reuse assemblers where possible."""


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, *args, **kwargs):
    r"""Evaluate expr.

    :arg expr: a :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or
        a :class:`~slate.TensorBase` expression.
    :arg tensor: Existing tensor object to place the result in.
    :arg bcs: Iterable of boundary conditions to apply.
    :kwarg diagonal: If assembling a matrix is it diagonal?
    :kwarg form_compiler_parameters: Dictionary of parameters to pass to
        the form compiler. Ignored if not assembling a :class:`~ufl.classes.Form`.
        Any parameters provided here will be overridden by parameters set on the
        :class:`~ufl.classes.Measure` in the form. For example, if a
        ``quadrature_degree`` of 4 is specified in this argument, but a degree of
        3 is requested in the measure, the latter will be used.
    :kwarg mat_type: String indicating how a 2-form (matrix) should be
        assembled -- either as a monolithic matrix (``"aij"`` or ``"baij"``),
        a block matrix (``"nest"``), or left as a :class:`.ImplicitMatrix` giving
        matrix-free actions (``'matfree'``). If not supplied, the default value in
        ``parameters["default_matrix_type"]`` is used.  BAIJ differs
        from AIJ in that only the block sparsity rather than the dof
        sparsity is constructed.  This can result in some memory
        savings, but does not work with all PETSc preconditioners.
        BAIJ matrices only make sense for non-mixed matrices.
    :kwarg sub_mat_type: String indicating the matrix type to
        use *inside* a nested block matrix.  Only makes sense if
        ``mat_type`` is ``nest``.  May be one of ``"aij"`` or ``"baij"``.  If
        not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    :kwarg appctx: Additional information to hang on the assembled
        matrix if an implicit matrix is requested (mat_type ``"matfree"``).
    :kwarg options_prefix: PETSc options prefix to apply to matrices.

    :returns: See below.

    If expr is a :class:`~ufl.classes.Form` or Slate tensor expression then
    this evaluates the corresponding integral(s) and returns a :class:`float`
    for 0-forms, a :class:`.Function` for 1-forms and a :class:`.Matrix` or
    :class:`.ImplicitMatrix` for 2-forms. In the case of 2-forms the rows
    correspond to the test functions and the columns to the trial functions.

    If expr is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``expr`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``expr`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.

    .. note::
        For 1-form assembly, the resulting object should in fact be a *cofunction*
        instead of a :class:`.Function`. However, since cofunctions are not
        currently supported in UFL, functions are used instead.
    """
    if isinstance(expr, (ufl.form.Form, slate.TensorBase)):
        return _assemble_form(expr, *args, **kwargs)
    elif isinstance(expr, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(expr)
    else:
        raise TypeError(f"Unable to assemble: {expr}")


@PETSc.Log.EventDecorator()
def allocate_matrix(expr, bcs=None, *, mat_type=None, sub_mat_type=None,
                    appctx=None, form_compiler_parameters=None, options_prefix=None):
    r"""Allocate a matrix given an expression.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    bcs = bcs or ()
    appctx = appctx or {}

    matfree = mat_type == "matfree"
    arguments = expr.arguments()
    if bcs is None:
        bcs = ()
    else:
        if any(isinstance(bc, EquationBC) for bc in bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
    if matfree:
        return matrix.ImplicitMatrix(expr, bcs,
                                     appctx=appctx,
                                     fc_params=form_compiler_parameters,
                                     options_prefix=options_prefix)

    integral_types = set(i.integral_type() for i in expr.integrals())
    for bc in bcs:
        integral_types.update(integral.integral_type()
                              for integral in bc.integrals())
    nest = mat_type == "nest"
    if nest:
        baij = sub_mat_type == "baij"
    else:
        baij = mat_type == "baij"

    if any(len(a.function_space()) > 1 for a in arguments) and mat_type == "baij":
        raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")

    get_cell_map = operator.methodcaller("cell_node_map")
    get_extf_map = operator.methodcaller("exterior_facet_node_map")
    get_intf_map = operator.methodcaller("interior_facet_node_map")
    domains = OrderedDict((k, set()) for k in (get_cell_map,
                                               get_extf_map,
                                               get_intf_map))
    mapping = {"cell": (get_cell_map, op2.ALL),
               "exterior_facet_bottom": (get_cell_map, op2.ON_BOTTOM),
               "exterior_facet_top": (get_cell_map, op2.ON_TOP),
               "interior_facet_horiz": (get_cell_map, op2.ON_INTERIOR_FACETS),
               "exterior_facet": (get_extf_map, op2.ALL),
               "exterior_facet_vert": (get_extf_map, op2.ALL),
               "interior_facet": (get_intf_map, op2.ALL),
               "interior_facet_vert": (get_intf_map, op2.ALL)}
    for integral_type in integral_types:
        try:
            get_map, region = mapping[integral_type]
        except KeyError:
            raise ValueError(f"Unknown integral type '{integral_type}'")
        domains[get_map].add(region)

    test, trial = arguments
    map_pairs, iteration_regions = zip(*(((get_map(test), get_map(trial)),
                                          tuple(sorted(regions)))
                                         for get_map, regions in domains.items()
                                         if regions))
    try:
        sparsity = op2.Sparsity((test.function_space().dof_dset,
                                 trial.function_space().dof_dset),
                                tuple(map_pairs),
                                iteration_regions=tuple(iteration_regions),
                                nest=nest,
                                block_sparse=baij)
    except SparsityFormatError:
        raise ValueError("Monolithic matrix assembly not supported for systems "
                         "with R-space blocks")

    return matrix.Matrix(expr, bcs, mat_type, sparsity, ScalarType,
                         options_prefix=options_prefix)


@PETSc.Log.EventDecorator()
def create_assembly_callable(expr, tensor=None, bcs=None, form_compiler_parameters=None,
                             mat_type=None, sub_mat_type=None, diagonal=False):
    r"""Create a callable object than be used to assemble expr into a tensor.

    This is really only designed to be used inside residual and
    jacobian callbacks, since it always assembles back into the
    initially provided tensor.  See also :func:`allocate_matrix`.

    .. warning::

        This function is now deprecated.

    .. warning::

       Really do not use this function unless you know what you're doing.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("once", DeprecationWarning)
        warnings.warn("create_assembly_callable is now deprecated. Please use assemble or FormAssembler instead.",
                      DeprecationWarning)

    if tensor is None:
        raise ValueError("Have to provide tensor to write to")

    rank = len(expr.arguments())
    if rank == 0:
        return ZeroFormAssembler(expr, tensor, form_compiler_parameters).assemble
    elif rank == 1 or (rank == 2 and diagonal):
        return OneFormAssembler(expr, tensor, bcs, diagonal=diagonal,
                                form_compiler_parameters=form_compiler_parameters).assemble
    elif rank == 2:
        return TwoFormAssembler(expr, tensor, bcs, form_compiler_parameters).assemble
    else:
        raise AssertionError


def _assemble_form(form, tensor=None, bcs=None, *,
                   diagonal=False,
                   mat_type=None,
                   sub_mat_type=None,
                   appctx=None,
                   options_prefix=None,
                   form_compiler_parameters=None):
    """Assemble a form.

    See :func:`assemble` for a description of the arguments to this function.
    """
    bcs = solving._extract_bcs(bcs)

    _check_inputs(form, tensor, bcs, diagonal)

    if tensor is not None:
        _zero_tensor(tensor, form, diagonal)
    else:
        tensor = _make_tensor(form, bcs, diagonal=diagonal, mat_type=mat_type,
                              sub_mat_type=sub_mat_type, appctx=appctx,
                              form_compiler_parameters=form_compiler_parameters,
                              options_prefix=options_prefix)

    # It is expensive to construct new assemblers because extracting the data
    # from the form is slow. Since all of the data structures in the assembler
    # are persistent apart from the output tensor, we stash the assembler on the
    # form and swap out the tensor if needed.
    # The cache key only needs to contain the boundary conditions, diagonal and
    # form compiler parameters since all other assemble kwargs are only used for
    # creating the tensor which is handled above and has no bearing on the assembler.
    # Note: This technically creates a memory leak since bcs are 'heavy' and so
    # repeated assembly of the same form but with different boundary conditions
    # will lead to old bcs getting stored along with old tensors.

    # FIXME This only works for 1-forms at the moment
    is_cacheable = len(form.arguments()) == 1
    if is_cacheable:
        try:
            key = tuple(bcs), diagonal, tuplify(form_compiler_parameters)
            assembler = form._cache[_FORM_CACHE_KEY][key]
            assembler.replace_tensor(tensor)
            return assembler.assemble()
        except KeyError:
            pass

    rank = len(form.arguments())
    if rank == 0:
        assembler = ZeroFormAssembler(form, tensor, form_compiler_parameters)
    elif rank == 1 or (rank == 2 and diagonal):
        assembler = OneFormAssembler(form, tensor, bcs, diagonal=diagonal,
                                     form_compiler_parameters=form_compiler_parameters)
    elif rank == 2:
        assembler = TwoFormAssembler(form, tensor, bcs, form_compiler_parameters)
    else:
        raise AssertionError

    if is_cacheable:
        if _FORM_CACHE_KEY not in form._cache:
            form._cache[_FORM_CACHE_KEY] = {}
        form._cache[_FORM_CACHE_KEY][key] = assembler

    return assembler.assemble()


def _check_inputs(form, tensor, bcs, diagonal):
    # Ensure mesh is 'initialised' as we could have got here without building a
    # function space (e.g. if integrating a constant).
    for mesh in form.ufl_domains():
        mesh.init()

    if diagonal and any(isinstance(bc, EquationBCSplit) for bc in bcs):
        raise NotImplementedError("Diagonal assembly and EquationBC not supported")

    rank = len(form.arguments())
    if rank == 0:
        assert tensor is None
        assert not bcs
    elif rank == 1:
        test, = form.arguments()

        if tensor is not None and test.function_space() != tensor.function_space():
            raise ValueError("Form's argument does not match provided result tensor")
    elif rank == 2 and diagonal:
        test, trial = form.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Can only assemble the diagonal of 2-form if the function spaces match")
    elif rank == 2:
        if tensor is not None and tensor.a.arguments() != form.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")
    else:
        raise AssertionError


def _zero_tensor(tensor, form, diagonal):
    rank = len(form.arguments())
    assert rank != 0
    if rank == 1 or (rank == 2 and diagonal):
        tensor.dat.zero()
    elif rank == 2:
        if not isinstance(tensor, matrix.ImplicitMatrix):
            tensor.M.zero()
    else:
        raise AssertionError


def _make_tensor(form, bcs, *, diagonal, mat_type, sub_mat_type, appctx,
                 form_compiler_parameters, options_prefix):
    rank = len(form.arguments())
    if rank == 0:
        return op2.Global(1, [0.0], dtype=utils.ScalarType)
    elif rank == 1:
        test, = form.arguments()
        return firedrake.Function(test.function_space())
    elif rank == 2 and diagonal:
        test, _ = form.arguments()
        return firedrake.Function(test.function_space())
    elif rank == 2:
        mat_type, sub_mat_type = _get_mat_type(mat_type, sub_mat_type, form.arguments())
        return allocate_matrix(form, bcs, mat_type=mat_type, sub_mat_type=sub_mat_type,
                               appctx=appctx, form_compiler_parameters=form_compiler_parameters,
                               options_prefix=options_prefix)
    else:
        raise AssertionError


class FormAssembler(abc.ABC):
    """Abstract base class for assembling forms.

    :param form: The variational form to be assembled.
    :param tensor: The output tensor to store the result.
    :param bcs: Iterable of boundary conditions to apply.
    :param form_compiler_parameters: Optional parameters to pass to the
        TSFC and/or Slate compilers.
    :param needs_zeroing: Should ``tensor`` be zeroed before assembling?
    """

    def __init__(self, form, tensor, bcs=(), form_compiler_parameters=None, needs_zeroing=True):
        assert tensor is not None

        bcs = solving._extract_bcs(bcs)

        self._form = form
        self._tensor = tensor
        self._bcs = bcs
        self._form_compiler_params = form_compiler_parameters or {}
        self._needs_zeroing = needs_zeroing

    @property
    @abc.abstractmethod
    def result(self):
        """The result of the assembly operation."""

    @property
    @abc.abstractmethod
    def diagonal(self):
        """Are we assembling the diagonal of a 2-form?"""

    def assemble(self):
        """Perform the assembly.

        :returns: The assembled object.
        """
        if self._needs_zeroing:
            self._as_pyop2_type(self._tensor).zero()

        for parloop in self.parloops:
            parloop()

        for bc in self._bcs:
            if isinstance(bc, EquationBC):  # can this be lifted?
                bc = bc.extract_form("F")
            self._apply_bc(bc)

        return self.result

    def replace_tensor(self, tensor):
        if tensor is self._tensor:
            return

        # TODO We should have some proper checks here
        for lknl, parloop in zip(self.local_kernels, self.parloops):
            data = _FormHandler.index_tensor(tensor, self._form, lknl.indices, self.diagonal)
            parloop.arguments[0].data = data
        self._tensor = tensor

    @cached_property
    def local_kernels(self):
        try:
            topology, = set(d.topology for d in self._form.ufl_domains())
        except ValueError:
            raise NotImplementedError("All integration domains must share a mesh topology")

        for o in itertools.chain(self._form.arguments(), self._form.coefficients()):
            domain = o.ufl_domain()
            if domain is not None and domain.topology != topology:
                raise NotImplementedError("Assembly with multiple meshes is not supported")

        if isinstance(self._form, ufl.Form):
            return tsfc_interface.compile_form(self._form, "form", diagonal=self.diagonal,
                                               parameters=self._form_compiler_params)
        elif isinstance(self._form, slate.TensorBase):
            return slac.compile_expression(self._form, compiler_parameters=self._form_compiler_params)
        else:
            raise AssertionError

    @cached_property
    def all_integer_subdomain_ids(self):
        return tsfc_interface.gather_integer_subdomain_ids(self.local_kernels)

    @cached_property
    def global_kernels(self):
        return tuple(_make_global_kernel(self._form, tsfc_knl, self.all_integer_subdomain_ids,
                                         diagonal=self.diagonal,
                                         unroll=self.needs_unrolling(tsfc_knl, self._bcs))
                     for tsfc_knl in self.local_kernels)

    @cached_property
    def parloops(self):
        return tuple(ParloopBuilder(self._form, lknl, gknl, self._tensor,
                                    self.all_integer_subdomain_ids, diagonal=self.diagonal,
                                    lgmaps=self.collect_lgmaps(lknl, self._bcs)).build()
                     for lknl, gknl in zip(self.local_kernels, self.global_kernels))

    def needs_unrolling(self, local_knl, bcs):
        """Do we need to address matrix elements directly rather than in
        a blocked fashion?

        This is slower but required for the application of some boundary conditions
        to 2-forms.

        :param local_knl: A :class:`tsfc_interface.SplitKernel`.
        :param bcs: Iterable of boundary conditions.
        """
        return False

    def collect_lgmaps(self, local_knl, bcs):
        """Return any local-to-global maps that need to be swapped out.

        This is only needed when applying boundary conditions to 2-forms.

        :param local_knl: A :class:`tsfc_interface.SplitKernel`.
        :param bcs: Iterable of boundary conditions.
        """
        return None

    @staticmethod
    def _as_pyop2_type(tensor):
        if isinstance(tensor, op2.Global):
            return tensor
        elif isinstance(tensor, firedrake.Function):
            return tensor.dat
        elif isinstance(tensor, matrix.Matrix):
            return tensor.M
        else:
            raise AssertionError


class ZeroFormAssembler(FormAssembler):
    """Class for assembling a 0-form."""

    diagonal = False
    """Diagonal assembly not possible for zero forms."""

    def __init__(self, form, tensor, form_compiler_parameters=None):
        super().__init__(form, tensor, (), form_compiler_parameters)

    @property
    def result(self):
        return self._tensor.data[0]


class OneFormAssembler(FormAssembler):
    """Class for assembling a 1-form.

    :param diagonal: Are we actually assembling the diagonal of a 2-form?
    :param zero_bc_nodes: If ``True``, set the boundary condition nodes in the
        output tensor to zero rather than to the values prescribed by the
        boundary condition.

    For all other arguments see :class:`FormAssembler` for more information.
    """

    def __init__(self, form, tensor, bcs=(), diagonal=False, zero_bc_nodes=False,
                 form_compiler_parameters=None, needs_zeroing=True):
        super().__init__(form, tensor, bcs, form_compiler_parameters, needs_zeroing)
        self._diagonal = diagonal
        self._zero_bc_nodes = zero_bc_nodes

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def result(self):
        return self._tensor

    def _apply_bc(self, bc):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            bc.zero(self._tensor)

            type(self)(bc.f, self._tensor, bc.bcs, self._diagonal, self._zero_bc_nodes,
                       self._form_compiler_params, needs_zeroing=False).assemble()
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, bc):
        if not self._zero_bc_nodes:
            if self._diagonal:
                bc.set(self._tensor, 1)
            else:
                bc.apply(self._tensor)
        else:
            bc.zero(self._tensor)


def TwoFormAssembler(form, tensor, *args, **kwargs):
    if isinstance(tensor, matrix.ImplicitMatrix):
        return MatrixFreeAssembler(tensor)
    else:
        return ExplicitMatrixAssembler(form, tensor, *args, **kwargs)


class ExplicitMatrixAssembler(FormAssembler):
    """Class for assembling a 2-form."""

    diagonal = False
    """Diagonal assembly not possible for two forms."""

    @property
    def test_function_space(self):
        test, _ = self._form.arguments()
        return test.function_space()

    @property
    def trial_function_space(self):
        _, trial = self._form.arguments()
        return trial.function_space()

    def get_indicess(self, knl):
        if all(i is None for i in knl.indices):
            return numpy.ndindex(self._tensor.block_shape)
        else:
            assert all(i is not None for i in knl.indices)
            return knl.indices,

    @property
    def result(self):
        self._tensor.M.assemble()
        return self._tensor

    def needs_unrolling(self, knl, bcs):
        for i, j in self.get_indicess(knl):
            for bc in itertools.chain(*self._filter_bcs(bcs, i, j)):
                if bc.function_space().component is not None:
                    return True
        return False

    def collect_lgmaps(self, knl, bcs):
        lgmaps = []
        for i, j in self.get_indicess(knl):
            row_bcs, col_bcs = self._filter_bcs(bcs, i, j)
            rlgmap, clgmap = self._tensor.M[i, j].local_to_global_maps
            rlgmap = self.test_function_space[i].local_to_global_map(row_bcs, rlgmap)
            clgmap = self.trial_function_space[j].local_to_global_map(col_bcs, clgmap)
            lgmaps.append((rlgmap, clgmap))
        return tuple(lgmaps)

    def _filter_bcs(self, bcs, row, col):
        if len(self.test_function_space) > 1:
            bcrow = tuple(bc for bc in bcs
                          if bc.function_space_index() == row)
        else:
            bcrow = bcs

        if len(self.trial_function_space) > 1:
            bccol = tuple(bc for bc in bcs
                          if bc.function_space_index() == col
                          and isinstance(bc, DirichletBC))
        else:
            bccol = tuple(bc for bc in bcs if isinstance(bc, DirichletBC))
        return bcrow, bccol

    def _apply_bc(self, bc):
        op2tensor = self._tensor.M
        spaces = tuple(a.function_space() for a in self._tensor.a.arguments())
        V = bc.function_space()
        component = V.component
        if component is not None:
            V = V.parent
        index = 0 if V.index is None else V.index
        space = V if V.parent is None else V.parent
        if isinstance(bc, DirichletBC):
            if space != spaces[0]:
                raise TypeError("bc space does not match the test function space")
            elif space != spaces[1]:
                raise TypeError("bc space does not match the trial function space")

            # Set diagonal entries on bc nodes to 1 if the current
            # block is on the matrix diagonal and its index matches the
            # index of the function space the bc is defined on.
            op2tensor[index, index].set_local_diagonal_entries(bc.nodes, idx=component)

            # Handle off-diagonal block involving real function space.
            # "lgmaps" is correctly constructed in _matrix_arg, but
            # is ignored by PyOP2 in this case.
            # Walk through row blocks associated with index.
            for j, s in enumerate(space):
                if j != index and s.ufl_element().family() == "Real":
                    self._apply_bcs_mat_real_block(op2tensor, index, j, component, bc.node_set)
            # Walk through col blocks associated with index.
            for i, s in enumerate(space):
                if i != index and s.ufl_element().family() == "Real":
                    self._apply_bcs_mat_real_block(op2tensor, i, index, component, bc.node_set)
        elif isinstance(bc, EquationBCSplit):
            for j, s in enumerate(spaces[1]):
                if s.ufl_element().family() == "Real":
                    self._apply_bcs_mat_real_block(op2tensor, index, j, component, bc.node_set)
            type(self)(bc.f, self._tensor, bc.bcs, self._form_compiler_params,
                       needs_zeroing=False).assemble()
        else:
            raise AssertionError

    @staticmethod
    def _apply_bcs_mat_real_block(op2tensor, i, j, component, node_set):
        dat = op2tensor[i, j].handle.getPythonContext().dat
        if component is not None:
            dat = op2.DatView(dat, component)
        dat.zero(subset=node_set)


class MatrixFreeAssembler:
    """Stub class wrapping matrix-free assembly."""

    def __init__(self, tensor):
        self._tensor = tensor

    def assemble(self):
        self._tensor.assemble()
        return self._tensor


def _global_kernel_cache_key(form, local_knl, all_integer_subdomain_ids, **kwargs):
    # N.B. Generating the global kernel is not a collective operation so the
    # communicator does not need to be a part of this cache key.

    if isinstance(form, ufl.Form):
        sig = form.signature()
    elif isinstance(form, slate.TensorBase):
        sig = form.expression_hash

    # The form signature does not store this information. This should be accessible from
    # the UFL so we don't need this nasty hack.
    subdomain_key = []
    for val in form.subdomain_data().values():
        for k, v in val.items():
            if v is not None:
                extruded = v._extruded
                constant_layers = extruded and v.constant_layers
                subset = isinstance(v, op2.Subset)
                subdomain_key.append((k, extruded, constant_layers, subset))
            else:
                subdomain_key.append((k,))

    return ((sig,)
            + tuple(subdomain_key)
            + tuplify(all_integer_subdomain_ids)
            + cachetools.keys.hashkey(local_knl, **kwargs))


@cachetools.cached(cache={}, key=_global_kernel_cache_key)
def _make_global_kernel(*args, **kwargs):
    return _GlobalKernelBuilder(*args, **kwargs).build()


class _GlobalKernelBuilder:
    """Class that builds a :class:`op2.GlobalKernel`.

    :param form: The variational form.
    :param local_knl: :class:`tsfc_interface.SplitKernel` compiled by either
        TSFC or Slate.
    :param all_integer_subdomain_ids: See :func:`tsfc_interface.gather_integer_subdomain_ids`.
    :param diagonal: Are we assembling the diagonal of a 2-form?
    :param unroll: If ``True``, address matrix elements directly rather than in
        a blocked fashion. This is slower but required for the application of
        some boundary conditions.

    .. note::
        One should be able to generate a global kernel without needing to
        use any data structures (i.e. a stripped form should be sufficient).
    """

    def __init__(self, form, local_knl, all_integer_subdomain_ids, diagonal=False, unroll=False):
        self._form = form
        self._indices, self._kinfo = local_knl
        self._all_integer_subdomain_ids = all_integer_subdomain_ids.get(self._kinfo.integral_type, None)
        self._diagonal = diagonal
        self._unroll = unroll

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)

        self._map_arg_cache = {}
        # Cache for holding :class:`op2.MapKernelArg` instances.
        # This is required to ensure that we use the same map argument when the
        # data objects in the parloop would be using the same map. This is to avoid
        # unnecessary packing in the global kernel.

    def build(self):
        """Build the global kernel."""
        kernel_args = [self._as_global_kernel_arg(arg)
                       for arg in self._kinfo.arguments]

        iteration_regions = {"exterior_facet_top": op2.ON_TOP,
                             "exterior_facet_bottom": op2.ON_BOTTOM,
                             "interior_facet_horiz": op2.ON_INTERIOR_FACETS}
        iteration_region = iteration_regions.get(self._integral_type, None)
        extruded = self._mesh.extruded
        constant_layers = extruded and not self._mesh.variable_layers

        return op2.GlobalKernel(self._kinfo.kernel,
                                kernel_args,
                                iteration_region=iteration_region,
                                pass_layer_arg=self._kinfo.pass_layer_arg,
                                extruded=extruded,
                                constant_layers=constant_layers,
                                subset=self._needs_subset)

    @property
    def _integral_type(self):
        return self._kinfo.integral_type

    @cached_property
    def _mesh(self):
        return self._form.ufl_domains()[self._kinfo.domain_number]

    @cached_property
    def _needs_subset(self):
        subdomain_data = self._form.subdomain_data()[self._mesh]
        if subdomain_data.get(self._integral_type, None) is not None:
            return True

        if self._kinfo.subdomain_id == "everywhere":
            return False
        elif self._kinfo.subdomain_id == "otherwise":
            return self._all_integer_subdomain_ids is not None
        else:
            return True

    @property
    def _indexed_function_spaces(self):
        return _FormHandler.index_function_spaces(self._form, self._indices)

    def _as_global_kernel_arg(self, tsfc_arg):
        # TODO Make singledispatchmethod with Python 3.8
        return _as_global_kernel_arg(tsfc_arg, self)

    def _get_map_arg(self, finat_element):
        """Get the appropriate map argument for the given FInAT element.

        :arg finat_element: A FInAT element.
        :returns: A :class:`op2.MapKernelArg` instance corresponding to
            the given FInAT element. This function uses a cache to ensure
            that PyOP2 knows when it can reuse maps.
        """
        key = self._get_map_id(finat_element)

        try:
            return self._map_arg_cache[key]
        except KeyError:
            pass

        shape = finat_element.index_shape
        if isinstance(finat_element, finat.TensorFiniteElement):
            shape = shape[:-len(finat_element._shape)]
        arity = numpy.prod(shape, dtype=int)
        if self._integral_type in {"interior_facet", "interior_facet_vert"}:
            arity *= 2

        if self._mesh.extruded:
            offset = tuple(eutils.calculate_dof_offset(finat_element))
            # for interior facet integrals we double the size of the offset array
            if self._integral_type in {"interior_facet", "interior_facet_vert"}:
                offset += offset
        else:
            offset = None

        map_arg = op2.MapKernelArg(arity, offset)
        self._map_arg_cache[key] = map_arg
        return map_arg

    def _get_dim(self, finat_element):
        if isinstance(finat_element, finat.TensorFiniteElement):
            return finat_element._shape
        else:
            return (1,)

    def _make_dat_global_kernel_arg(self, finat_element, index=None):
        if isinstance(finat_element, finat.EnrichedElement) and finat_element.is_mixed:
            assert index is None
            subargs = tuple(self._make_dat_global_kernel_arg(subelem.element)
                            for subelem in finat_element.elements)
            return op2.MixedDatKernelArg(subargs)
        else:
            dim = self._get_dim(finat_element)
            map_arg = self._get_map_arg(finat_element)
            return op2.DatKernelArg(dim, map_arg, index)

    def _make_mat_global_kernel_arg(self, relem, celem):
        if any(isinstance(e, finat.EnrichedElement) and e.is_mixed for e in {relem, celem}):
            subargs = tuple(self._make_mat_global_kernel_arg(rel.element, cel.element)
                            for rel, cel in product(relem.elements, celem.elements))
            shape = len(relem.elements), len(celem.elements)
            return op2.MixedMatKernelArg(subargs, shape)
        else:
            # PyOP2 matrix objects have scalar dims so we flatten them here
            rdim = numpy.prod(self._get_dim(relem), dtype=int)
            cdim = numpy.prod(self._get_dim(celem), dtype=int)
            map_args = self._get_map_arg(relem), self._get_map_arg(celem)
            return op2.MatKernelArg((((rdim, cdim),),), map_args, unroll=self._unroll)

    @staticmethod
    def _get_map_id(finat_element):
        """Return a key that is used to check if we reuse maps.

        This mirrors firedrake.functionspacedata.
        """
        if isinstance(finat_element, finat.TensorFiniteElement):
            finat_element = finat_element.base_element

        real_tensorproduct = eutils.is_real_tensor_product_element(finat_element)
        try:
            eperm_key = entity_permutations_key(finat_element.entity_permutations)
        except NotImplementedError:
            eperm_key = None
        return entity_dofs_key(finat_element.entity_dofs()), real_tensorproduct, eperm_key


@functools.singledispatch
def _as_global_kernel_arg(tsfc_arg, self):
    raise NotImplementedError


@_as_global_kernel_arg.register(kernel_args.OutputKernelArg)
def _as_global_kernel_arg_output(_, self):
    rank = len(self._form.arguments())
    Vs = self._indexed_function_spaces

    if rank == 0:
        return op2.GlobalKernelArg((1,))
    elif rank == 1 or rank == 2 and self._diagonal:
        V, = Vs
        if V.ufl_element().family() == "Real":
            return op2.GlobalKernelArg((1,))
        else:
            return self._make_dat_global_kernel_arg(create_element(V.ufl_element()))
    elif rank == 2:
        if all(V.ufl_element().family() == "Real" for V in Vs):
            return op2.GlobalKernelArg((1,))
        elif any(V.ufl_element().family() == "Real" for V in Vs):
            el, = (create_element(V.ufl_element()) for V in Vs
                   if V.ufl_element().family() != "Real")
            return self._make_dat_global_kernel_arg(el)
        else:
            rel, cel = (create_element(V.ufl_element()) for V in Vs)
            return self._make_mat_global_kernel_arg(rel, cel)
    else:
        raise AssertionError


@_as_global_kernel_arg.register(kernel_args.CoordinatesKernelArg)
def _as_global_kernel_arg_coordinates(_, self):
    finat_element = create_element(self._mesh.ufl_coordinate_element())
    return self._make_dat_global_kernel_arg(finat_element)


@_as_global_kernel_arg.register(kernel_args.CoefficientKernelArg)
def _as_global_kernel_arg_coefficient(_, self):
    coeff = next(self._active_coefficients)
    V = coeff.ufl_function_space()
    if hasattr(V, "component") and V.component is not None:
        index = V.component,
        V = V.parent
    else:
        index = None

    ufl_element = V.ufl_element()
    if ufl_element.family() == "Real":
        return op2.GlobalKernelArg((ufl_element.value_size(),))
    else:
        finat_element = create_element(ufl_element)
        return self._make_dat_global_kernel_arg(finat_element, index)


@_as_global_kernel_arg.register(kernel_args.CellSizesKernelArg)
def _as_global_kernel_arg_cell_sizes(_, self):
    # this mirrors tsfc.kernel_interface.firedrake_loopy.KernelBuilder.set_cell_sizes
    ufl_element = ufl.FiniteElement("P", self._mesh.ufl_cell(), 1)
    finat_element = create_element(ufl_element)
    return self._make_dat_global_kernel_arg(finat_element)


@_as_global_kernel_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_global_kernel_arg_exterior_facet(_, self):
    return op2.DatKernelArg((1,))


@_as_global_kernel_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_global_kernel_arg_interior_facet(_, self):
    return op2.DatKernelArg((2,))


@_as_global_kernel_arg.register(CellFacetKernelArg)
def _as_global_kernel_arg_cell_facet(_, self):
    if self._mesh.extruded:
        num_facets = self._mesh._base_mesh.ufl_cell().num_facets()
    else:
        num_facets = self._mesh.ufl_cell().num_facets()
    return op2.DatKernelArg((num_facets, 2))


@_as_global_kernel_arg.register(kernel_args.CellOrientationsKernelArg)
def _as_global_kernel_arg_cell_orientations(_, self):
    # this mirrors firedrake.mesh.MeshGeometry.init_cell_orientations
    ufl_element = ufl.FiniteElement("DG", cell=self._form.ufl_domain().ufl_cell(), degree=0)
    finat_element = create_element(ufl_element)
    return self._make_dat_global_kernel_arg(finat_element)


@_as_global_kernel_arg.register(LayerCountKernelArg)
def _as_global_kernel_arg_layer_count(_, self):
    return op2.GlobalKernelArg((1,))


class ParloopBuilder:
    """Class that builds a :class:`op2.Parloop`.

    :param form: The variational form.
    :param local_knl: :class:`tsfc_interface.SplitKernel` compiled by either
        TSFC or Slate.
    :param global_knl: A :class:`pyop2.GlobalKernel` instance.
    :param tensor: The output tensor to write to (cannot be ``None``).
    :param all_integer_subdomain_ids: See :func:`tsfc_interface.gather_integer_subdomain_ids`.
    :param diagonal: Are we assembling the diagonal of a 2-form?
    :param lgmaps: Optional iterable of local-to-global maps needed for applying
        boundary conditions to 2-forms.
    """

    def __init__(self, form, local_knl, global_knl, tensor,
                 all_integer_subdomain_ids, diagonal=False, lgmaps=None):
        self._form = form
        self._local_knl = local_knl
        self._global_knl = global_knl
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._tensor = tensor
        self._diagonal = diagonal
        self._lgmaps = lgmaps

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)

    def build(self):
        """Construct the parloop."""
        parloop_args = [self._as_parloop_arg(tsfc_arg)
                        for tsfc_arg in self._kinfo.arguments]
        try:
            return op2.Parloop(self._global_knl, self._iterset, parloop_args)
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all "
                               "coefficients/arguments")

    @property
    def _indices(self):
        return self._local_knl.indices

    @property
    def _kinfo(self):
        return self._local_knl.kinfo

    @property
    def _integral_type(self):
        return self._kinfo.integral_type

    @property
    def _indexed_function_spaces(self):
        return _FormHandler.index_function_spaces(self._form, self._indices)

    @property
    def _indexed_tensor(self):
        return _FormHandler.index_tensor(self._tensor, self._form, self._indices, self._diagonal)

    @cached_property
    def _mesh(self):
        return self._form.ufl_domains()[self._kinfo.domain_number]

    @cached_property
    def _iterset(self):
        try:
            subdomain_data = self._form.subdomain_data()[self._mesh][self._integral_type]
        except KeyError:
            subdomain_data = None

        if subdomain_data is not None:
            if self._integral_type != "cell":
                raise NotImplementedError("subdomain_data only supported with cell integrals")
            if self._kinfo.subdomain_id not in ["everywhere", "otherwise"]:
                raise ValueError("Cannot use subdomain data and subdomain_id")
            return subdomain_data
        else:
            return self._mesh.measure_set(self._integral_type, self._kinfo.subdomain_id,
                                          self._all_integer_subdomain_ids)

    def _get_map(self, V):
        """Return the appropriate PyOP2 map for a given function space."""
        assert isinstance(V, ufl.FunctionSpace)

        if self._integral_type in {"cell", "exterior_facet_top",
                                   "exterior_facet_bottom", "interior_facet_horiz"}:
            return V.cell_node_map()
        elif self._integral_type in {"exterior_facet", "exterior_facet_vert"}:
            return V.exterior_facet_node_map()
        elif self._integral_type in {"interior_facet", "interior_facet_vert"}:
            return V.interior_facet_node_map()
        else:
            raise AssertionError

    def _as_parloop_arg(self, tsfc_arg):
        """Return a :class:`op2.ParloopArg` corresponding to the provided
        :class:`tsfc.KernelArg`.
        """
        # TODO Make singledispatchmethod with Python 3.8
        return _as_parloop_arg(tsfc_arg, self)


@functools.singledispatch
def _as_parloop_arg(tsfc_arg, self):
    raise NotImplementedError


@_as_parloop_arg.register(kernel_args.OutputKernelArg)
def _as_parloop_arg_output(_, self):
    rank = len(self._form.arguments())
    tensor = self._indexed_tensor
    Vs = self._indexed_function_spaces

    if rank == 0:
        return op2.GlobalParloopArg(tensor)
    elif rank == 1 or rank == 2 and self._diagonal:
        V, = Vs
        if V.ufl_element().family() == "Real":
            return op2.GlobalParloopArg(tensor)
        else:
            return op2.DatParloopArg(tensor, self._get_map(V))
    elif rank == 2:
        rmap, cmap = [self._get_map(V) for V in Vs]

        if all(V.ufl_element().family() == "Real" for V in Vs):
            assert rmap is None and cmap is None
            return op2.GlobalParloopArg(tensor.handle.getPythonContext().global_)
        elif any(V.ufl_element().family() == "Real" for V in Vs):
            m = rmap or cmap
            return op2.DatParloopArg(tensor.handle.getPythonContext().dat, m)
        else:
            return op2.MatParloopArg(tensor, (rmap, cmap), lgmaps=self._lgmaps)
    else:
        raise AssertionError


@_as_parloop_arg.register(kernel_args.CoordinatesKernelArg)
def _as_parloop_arg_coordinates(_, self):
    func = self._mesh.coordinates
    map_ = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, map_)


@_as_parloop_arg.register(kernel_args.CoefficientKernelArg)
def _as_parloop_arg_coefficient(arg, self):
    coeff = next(self._active_coefficients)
    if coeff.ufl_element().family() == "Real":
        return op2.GlobalParloopArg(coeff.dat)
    else:
        m = self._get_map(coeff.function_space())
        return op2.DatParloopArg(coeff.dat, m)


@_as_parloop_arg.register(kernel_args.CellOrientationsKernelArg)
def _as_parloop_arg_cell_orientations(_, self):
    func = self._mesh.cell_orientations()
    m = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, m)


@_as_parloop_arg.register(kernel_args.CellSizesKernelArg)
def _as_parloop_arg_cell_sizes(_, self):
    func = self._mesh.cell_sizes
    m = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, m)


@_as_parloop_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_parloop_arg_exterior_facet(_, self):
    return op2.DatParloopArg(self._mesh.exterior_facets.local_facet_dat)


@_as_parloop_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_parloop_arg_interior_facet(_, self):
    return op2.DatParloopArg(self._mesh.interior_facets.local_facet_dat)


@_as_parloop_arg.register(CellFacetKernelArg)
def _as_parloop_arg_cell_facet(_, self):
    return op2.DatParloopArg(self._mesh.cell_to_facets)


@_as_parloop_arg.register(LayerCountKernelArg)
def _as_parloop_arg_layer_count(_, self):
    glob = op2.Global((1,), self._iterset.layers-2, dtype=numpy.int32)
    return op2.GlobalParloopArg(glob)


class _FormHandler:
    """Utility class for inspecting forms and local kernels."""

    @staticmethod
    def iter_active_coefficients(form, kinfo):
        """Yield the form coefficients referenced in ``kinfo``."""
        for idx, subidxs in kinfo.coefficient_map:
            for subidx in subidxs:
                yield form.coefficients()[idx].split()[subidx]

    @staticmethod
    def index_function_spaces(form, indices):
        """Return the function spaces of the form's arguments, indexed
        if necessary.
        """
        if all(i is None for i in indices):
            return tuple(a.ufl_function_space() for a in form.arguments())
        elif all(i is not None for i in indices):
            return tuple(a.ufl_function_space()[i] for i, a in zip(indices, form.arguments()))
        else:
            raise AssertionError

    @staticmethod
    def index_tensor(tensor, form, indices, diagonal):
        """Return the PyOP2 data structure tied to ``tensor``, indexed
        if necessary.
        """
        rank = len(form.arguments())
        is_indexed = any(i is not None for i in indices)

        if rank == 0:
            return tensor
        elif rank == 1 or rank == 2 and diagonal:
            i, = indices
            return tensor.dat[i] if is_indexed else tensor.dat
        elif rank == 2:
            i, j = indices
            return tensor.M[i, j] if is_indexed else tensor.M
        else:
            raise AssertionError


def _get_mat_type(mat_type, sub_mat_type, arguments):
    """Validate the matrix types provided by the user and set any that are
    undefined to default values.

    :arg mat_type: (:class:`str`) PETSc matrix type for the assembled matrix.
    :arg sub_mat_type: (:class:`str`) PETSc matrix type for blocks if
        ``mat_type`` is ``"nest"``.
    :arg arguments: The test and trial functions of the expression being assembled.
    :raises ValueError: On bad arguments.
    :returns: 2-:class:`tuple` of validated/default ``mat_type`` and ``sub_mat_type``.
    """
    if mat_type is None:
        mat_type = parameters.parameters["default_matrix_type"]
        if any(V.ufl_element().family() == "Real"
               for arg in arguments
               for V in arg.function_space()):
            mat_type = "nest"
    if mat_type not in {"matfree", "aij", "baij", "nest", "dense"}:
        raise ValueError(f"Unrecognised matrix type, '{mat_type}'")
    if sub_mat_type is None:
        sub_mat_type = parameters.parameters["default_sub_matrix_type"]
    if sub_mat_type not in {"aij", "baij"}:
        raise ValueError(f"Invalid submatrix type, '{sub_mat_type}' (not 'aij' or 'baij')")
    return mat_type, sub_mat_type
