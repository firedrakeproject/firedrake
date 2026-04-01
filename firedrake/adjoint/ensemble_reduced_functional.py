from functools import cached_property
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint import Control, AdjFloat, no_annotations, OverloadedType
from pyadjoint.enlisting import Enlist
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from .ensemble_adjvec import EnsembleAdjVec
from firedrake.ensemble import (
    Ensemble, EnsembleFunctionSpace, EnsembleFunction)
from firedrake.ensemble.ensemble_function import EnsembleFunctionBase

__all__ = (
    "EnsembleReducedFunctional",
    "EnsembleBcastReducedFunctional",
    "EnsembleReduceReducedFunctional",
    "EnsembleAllgatherReducedFunctional",
    "EnsembleTransformReducedFunctional",
)


# utility functions to hide API differences between EnsembleFunction and EnsembleAdjVec


def _local_subs(val):
    if isinstance(val, EnsembleFunctionBase):
        return val.subfunctions
    elif isinstance(val, EnsembleAdjVec):
        return val.subvec
    elif isinstance(val, (list, tuple)):
        return val
    else:
        raise TypeError(
            f"Cannot use {type(val).__name__} as an ensemble overloaded type.")


def _global_size(val):
    if isinstance(val, EnsembleFunctionBase):
        return val.function_space().nglobal_spaces
    elif isinstance(val, EnsembleAdjVec):
        return val.global_size
    else:
        raise TypeError(
            f"Cannot use {type(val).__name__} as an ensemble overloaded type.")


def _local_size(val):
    return len(_local_subs(val))


def _set_local_subs(dst, src):
    assert _local_size(dst) == _local_size(src)
    dst_subs = _local_subs(dst)
    for i, s in enumerate(src):
        if hasattr(dst_subs[i], 'assign'):
            dst_subs[i].assign(s)
        else:
            dst_subs[i] = s
    return dst


def _ensemble(val):
    if isinstance(val, EnsembleFunctionBase):
        return val.function_space().ensemble
    elif isinstance(val, EnsembleAdjVec):
        return val.ensemble


def _make_ensemble_obj(local_vals, ensemble):
    if all(isinstance(val, float) for val in local_vals):
        return EnsembleAdjVec(local_vals, ensemble)
    elif all(isinstance(val, (Function, Cofunction)) for val in local_vals):
        ensemble_space = EnsembleFunctionSpace(
            [val.function_space() for val in local_vals], ensemble)
        ensemble_function = EnsembleFunction(ensemble_space)
        _set_local_subs(ensemble_function, local_vals)
        return ensemble_function
    else:
        raise TypeError("All local values must be of same type,"
                        " either AdjFloat or Function or Cofunction.")


class EnsembleReduceReducedFunctional(AbstractReducedFunctional):
    """
    A parallel reduction from all components of an :class:`.EnsembleFunction`
    or :class:`.EnsembleAdjVec` distributed over an :class:`.Ensemble` into a
    :class:`.Function` or :class:`pyadjoint.AdjFloat` on each ensemble member.

    Currently the only reduction operation implemented is a sum. The adjoint
    operation to a sum is a broadcast, so the ``derivative`` function will
    return an ensemble object with a copy of the ``adj_input`` in all
    components.

    The ``functional`` must be suitable to reduce each component of ``control``
    into. The acceptable combinations are shown in the table below. For an
    :class:`.EnsembleFunction` or :class:`.EnsembleCofunction` ``control``
    there is an additional restriction that the ``functional`` and all
    components of ``control`` must have the same :func:`.FunctionSpace`.

    .. list-table::
      :header-rows: 1

      * - ``control``
        - ``functional``
      * -  :class:`.EnsembleFunction`
        -  :class:`.Function`
      * -  :class:`.EnsembleCofunction`
        -  :class:`.Cofunction`
      * -  :class:`.EnsembleAdjVec`
        -  :class:`~pyadjoint.AdjFloat`

    Parameters
    ----------
    functional :
        The result of the reduction, i.e. the sum of all components of the
        ``control`` from all ensemble members. Must be identical on all
        members.
    control :
        An object with several components to sum, distributed over an
        :class:`.Ensemble`.  Must be a single :class:`~pyadjoint.Control`
        rather than a list.

    Notes
    -----
    Unlike most ``ReducedFunctional`` classes, this one does not require any
    operations to be taped before creating it. The ``functional`` and
    ``control`` arguments are just to specify the source and destination
    spaces.

    This class is primarily intended as a component for building larger
    ``ReducedFunctional`` classes over an :class:`.Ensemble`, for example the
    :class:`.EnsembleReducedFunctional`.

    Developer Notes
    ---------------
    The "hidden" parameter ``_only_forward`` exists because bcast and reduce
    are adjoint to each other so we want to implement the derivative of each
    using the other. ``_only_forward`` is used to do this without infinite
    recursion.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    :class:`~.Ensemble`
    :class:`~.EnsembleFunction`
    :class:`~.EnsembleCofunction`
    :class:`~.EnsembleAdjVec`
    :class:`~.EnsembleBcastReducedFunctional`
    :class:`~.EnsembleTransformReducedFunctional`
    :class:`~.EnsembleAllgatherReducedFunctional`
    :class:`~.EnsembleReducedFunctional`
    """
    def __init__(self, functional: OverloadedType, control: Control,
                 _only_forward=False):
        self.functional = functional
        self._controls = Enlist(control)

        if isinstance(functional, AdjFloat):
            if not isinstance(control.control, EnsembleAdjVec):
                raise TypeError(
                    f"Control for {type(self).__name__} must be an"
                    " EnsembleAdjVec if using an AdjFloat functional.")
        elif isinstance(functional, (Function, Cofunction)):
            if not isinstance(control.control, EnsembleFunctionBase):
                raise TypeError(
                    f"Control for {type(self).__name__} must be an"
                    " EnsembleFunction or EnsembleCofunction if using"
                    " a Function or Cofunction functional.")
            if not all([c.function_space() == functional.function_space()
                        for c in control.subfunctions]):
                raise ValueError(
                    f"All subfunctions of the {type(self).__name__} control"
                    " must have the same function space as the functional.")
        else:
            raise ValueError(
                f"Do not know how to handle a {type(functional).__name__}"
                f" control for {type(self).__name__}.")

        # Adjoint action is a bcast so we just piggyback.
        # Possibly don't do this if we're being created by the
        # bcast rf to avoid infinite recursion.
        if not _only_forward:
            self._bcast = EnsembleBcastReducedFunctional(
                functional=control.control._ad_init_zero(dual=True),
                control=Control(functional._ad_init_zero(dual=True)),
                _only_forward=True
            )

    @property
    def controls(self):
        return self._controls

    @property
    def ensemble(self):
        """The :class:`.Ensemble` that the control is defined over."""
        return _ensemble(self.controls[0].control)

    @no_annotations
    def __call__(self, values):
        for c, v in zip(self.controls, Enlist(values)):
            c.update(v)
        return self.tlm(values)

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        dJ = self._bcast(adj_input)
        if apply_riesz:
            return self.controls[0]._ad_convert_riesz(
                dJ, riesz_map=self.controls[0].riesz_map)
        return dJ

    @no_annotations
    def tlm(self, m_dot):
        if isinstance(m_dot, (list, tuple)):
            m_dot = m_dot[0]
        vals = _local_subs(m_dot)
        local_sum = vals[0]._ad_init_zero()
        for v in vals:
            local_sum = local_sum._ad_add(v)
        return self.ensemble.allreduce(local_sum)

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True,
                apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)
        if hessian_input is None:
            hessian_input = self.functional._ad_init_zero(dual=True)
        return self.derivative(hessian_input, apply_riesz=apply_riesz)


class EnsembleBcastReducedFunctional(AbstractReducedFunctional):
    """
    A parallel broadcast from a :class:`.Function` or
    :class:`pyadjoint.AdjFloat` into all components of an
    :class:`.EnsembleFunction` or :class:`.EnsembleAdjVec` distributed over an
    :class:`.Ensemble`.

    The adjoint operation to a broadcast is a sum, so the ``derivative``
    function will return an object on each ensemble member with the sum of
    all components of the ``adj_input``.

    The ``functional`` must be suitable to broadcast the ``control`` into.
    The acceptable combinations are shown in the table below. For a
    :class:`.Function` or :class:`.Cofunction` ``control`` there is an
    additional restriction that the ``control`` and all components of
    ``functional`` must have the same :func:`.FunctionSpace`.

    .. list-table::
      :header-rows: 1

      * - ``control``
        - ``functional``
      * -  :class:`.Function`
        -  :class:`.EnsembleFunction`
      * -  :class:`.Cofunction`
        -  :class:`.EnsembleCofunction`
      * -  :class:`~pyadjoint.AdjFloat`
        -  :class:`.EnsembleAdjVec`

    Parameters
    ----------
    functional :
        The result of the broadcast, i.e. all components of ``functional``
        will be copies of ``control``.
    control :
        The object to broadcast. Must be identical on each ensemble member.
        Must be a single :class:`~pyadjoint.Control` rather than a list.
    root :
        If ``None`` then the argument to ``__call__`` or ``tlm`` is assumed to
        be the same on all ensemble members, and the "broadcast" only requires
        local copies rather than global communications.
        If ``root`` is an ``int`` then this ensemble member is used as the
        root for the broadcast and ``__call__`` and ``tlm`` will carry out
        global communication.

    Notes
    -----
    Unlike most ``ReducedFunctional`` classes, this one does not require any
    operations to be taped before creating it. The ``functional`` and
    ``control`` arguments are just to specify the source and destination
    spaces.

    This class is primarily intended as a component for building larger
    ``ReducedFunctional`` classes over an :class:`.Ensemble`, for example the
    :class:`.EnsembleReducedFunctional`.

    Developer Notes
    ---------------
    The "hidden" parameter ``_only_forward`` exists because bcast and reduce
    are adjoint to each other so we want to implement the derivative of each
    using the other. ``_only_forward`` is used to do this without infinite
    recursion.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    :class:`~.Ensemble`
    :class:`~.EnsembleFunction`
    :class:`~.EnsembleCofunction`
    :class:`~.EnsembleAdjVec`
    :class:`~.EnsembleReduceReducedFunctional`
    :class:`~.EnsembleTransformReducedFunctional`
    :class:`~.EnsembleAllgatherReducedFunctional`
    :class:`~.EnsembleReducedFunctional`
    """
    def __init__(self, functional: OverloadedType,
                 control: Control, root: int | None = None,
                 _only_forward=False):
        self.functional = functional
        self._controls = Enlist(control)
        self.root = root

        if isinstance(control.control, AdjFloat):
            if not isinstance(functional, EnsembleAdjVec):
                raise TypeError(
                    f"Functional for {type(self).__name__} must be an"
                    " EnsembleAdjVec if using an AdjFloat control.")
        elif isinstance(control.control, (Function, Cofunction)):
            if not isinstance(functional, EnsembleFunctionBase):
                raise TypeError(
                    f"Functional for {type(self).__name__} must be an"
                    " EnsembleFunction or EnsembleCofunction if using"
                    " a Function or Cofunction control.")
            if not all([f.function_space() == control.function_space()
                        for f in functional.subfunctions]):
                raise ValueError(
                    f"All subfunctions of the {type(self).__name__} functional"
                    " must have the same function space as the control.")
        else:
            raise ValueError(
                f"Do not know how to handle a {type(control).__name__}"
                f" control for {type(self).__name__}.")

        # Adjoint action is a reduction so we just piggyback.
        # Possibly don't do this if we're being created by the
        # reduction rf to avoid infinite recursion.
        if not _only_forward:
            self._reduce = EnsembleReduceReducedFunctional(
                functional=control.control._ad_init_zero(dual=True),
                control=Control(functional._ad_init_zero(dual=True)),
                _only_forward=True
            )

    @property
    def controls(self):
        return self._controls

    @property
    def ensemble(self):
        """The :class:`.Ensemble` that the functional is defined over."""
        return _ensemble(self.functional)

    @no_annotations
    def __call__(self, values):
        for c, v in zip(self.controls, Enlist(values)):
            c.update(v)
        return self.tlm(values)

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        dJ = self._reduce(adj_input)
        if apply_riesz:
            dJ = self.controls[0]._ad_convert_riesz(
                dJ, riesz_map=self.controls[0].riesz_map)
        return dJ

    @no_annotations
    def tlm(self, m_dot):
        if self.root is None:
            m_dot = m_dot
        else:
            m_dot = self.ensemble.bcast(m_dot, root=self.root)
        tlv = self.functional._ad_init_zero()
        _set_local_subs(
            tlv, [m_dot for _ in range(_local_size(self.functional))])
        return tlv

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True,
                apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)
        return self.derivative(hessian_input, apply_riesz=apply_riesz)


class EnsembleAllgatherReducedFunctional(AbstractReducedFunctional):
    """
    A parallel allgather of all components of an :class:`.EnsembleFunction`,
    :class:`.EnsembleCofunction`, or :class:`.EnsembleAdjVec` onto all ensemble
    members. The result is placed in an :class:`.EnsembleFunction`,
    :class:`.EnsembleCofunction`, or :class:`.EnsembleAdjVec` with enough
    components locally on each ensemble member to fit all components from every
    ensemble member of the control.

    For example in the code below, the control is a :class:`.EnsembleFunction`
    with two components on the first ensemble member and one on the second
    member, so the functional must be a :class:`.EnsembleFunction` with three
    components on each ensemble member with matching spaces.

    .. code-block:: python3

        V0 = FunctionSpace(mesh, "DG", 0)
        V1 = FunctionSpace(mesh, "CG", 1)

        if ensemble.ensemble_rank == 0:
            local_controls = [V0, V1]
        elif ensemble.ensemble_rank == 1:
            local_controls = [V0]

        local_functionals = [V0, V1, V0]

        control_space = EnsembleFunctionSpace(local_controls, ensemble)
        functional_space = EnsembleFunctionSpace(local_functionals, ensemble)

    Parameters
    ----------
    functional :
        The result of the allgather.
    control :
        The object to allgather.
        Must be a single :class:`~pyadjoint.Control` rather than a list.

    Notes
    -----
    Unlike most ``ReducedFunctional`` classes, this one does not require any
    operations to be taped before creating it. The ``functional`` and
    ``control`` arguments are just to specify the source and destination
    spaces.

    This class is primarily intended as a component for building larger
    ``ReducedFunctional`` classes over an :class:`.Ensemble`, for example the
    :class:`.EnsembleReducedFunctional`.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    :class:`~.Ensemble`
    :class:`~.EnsembleFunction`
    :class:`~.EnsembleCofunction`
    :class:`~.EnsembleAdjVec`
    :class:`~.EnsembleReduceReducedFunctional`
    :class:`~.EnsembleBcastReducedFunctional`
    :class:`~.EnsembleTransformReducedFunctional`
    :class:`~.EnsembleReducedFunctional`
    """
    def __init__(self, functional: OverloadedType, control: Control):
        assert _local_size(functional) == _global_size(control.control)
        self._controls = Enlist(control)
        self.functional = functional

    @property
    def controls(self):
        return self._controls

    @property
    def ensemble(self):
        """The :class:`.Ensemble` that the control and functional are defined over."""
        return _ensemble(self.functional)

    @no_annotations
    def __call__(self, values):
        for c, v in zip(self.controls, Enlist(values)):
            c.update(v)
        return self.tlm(values)

    @no_annotations
    def tlm(self, m_dot):
        if isinstance(m_dot, (list, tuple)):
            m_dot = m_dot[0]
        tlv = self.functional._ad_init_zero()
        # Ensemble.allgather has slightly different signature to Comm.allgather
        # so we actually have to distinguish between the two cases here.
        if isinstance(self.functional, EnsembleFunctionBase):
            self.ensemble.allgather(
                _local_subs(m_dot), _local_subs(tlv))
        elif isinstance(self.functional, EnsembleAdjVec):
            local_mdot_lists = self.ensemble.allgather(_local_subs(m_dot))
            # allgather will return a list of lists that we need to flatten
            global_mdot = [md
                           for local_mdots in local_mdot_lists
                           for md in local_mdots]
            _set_local_subs(tlv, global_mdot)
        else:
            raise TypeError(
                f"Cannot use {type(m_dot).__name__}"
                " as an ensemble overloaded type.")
        return tlv

    @cached_property
    def _local_indices(self):
        local_size = _local_size(self.controls[0].control)
        local_offset = self.ensemble.ensemble_comm.exscan(local_size) or 0
        return [local_offset + i for i in range(local_size)]

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        dJ_global = _local_subs(adj_input)
        dJ_local = [dJ_global[i] for i in self._local_indices]

        dJ = self.controls[0].control._ad_init_zero(dual=True)
        _set_local_subs(dJ, dJ_local)
        if apply_riesz:
            dJ = dJ._ad_convert_riesz(
                dJ, riesz_map=self.controls[0].riesz_map)
        return dJ

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True,
                apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)
        return self.derivative(hessian_input, apply_riesz=apply_riesz)


class EnsembleTransformReducedFunctional(AbstractReducedFunctional):
    """
    A parallel transformation from a set of :class:`.EnsembleFunction`,
    :class:`.EnsembleCofunction`, or :class:`.EnsembleAdjVec` controls to
    a functional which is also an ensemble object (i.e. one of these classes).

    The functional and all controls must have the same number of components on
    each ensemble member, and the transformation maps from components in the
    same position in each control to the corresponding component in the
    functional. This is explained in more detail below.
    The transform for each component position is provided by an
    :class:`~pyadjoint.reduced_functional.AbstractReducedFunctional` defined on
    the local ensemble member, whose ``controls`` and ``functional`` must match
    the corresponding components of the ``controls`` and ``functional`` of the
    ``EnsembleTransformReducedFunctional``.

    For example (ignoring the ensemble parallel partition for now), if we have
    two controls :math:`u` and :math:`v`, and a functional :math:`w`, which
    each have three components:

    .. math::

      u \\in U=U_{0} \\times U_{1} \\times U_{2},

      v \\in V=V_{0} \\times V_{1} \\times V_{2},

      w \\in W=W_{0} \\times W_{1} \\times W_{2},

    Then the ``EnsembleTransformReducedFunctional`` maps
    :math:`\\hat{J} : U \\times V \\to W`, and we need a
    :class:`~pyadjoint.ReducedFunctional` for each component:

    .. math::

      \\hat{J}_{0} : U_{0} \\times V_{0} \\to W_{0},

      \\hat{J}_{1} : U_{1} \\times V_{1} \\to W_{1},

      \\hat{J}_{2} : U_{2} \\times V_{2} \\to W_{2}.

    If :math:`u`, :math:`v`, and :math:`w` are each an
    :class:`.EnsembleFunction`, :class:`.EnsembleCofunction` or
    :class:`.EnsembleAdjVec`, then the components are distributed over the
    ensemble members. The corresponding :class:`~.pyadjoint.ReducedFunctional`
    for each :math:`\\hat{J}_{i}` are defined locally on each ensemble member.
    For example:

    .. code-block:: python3

      if ensemble.ensemble_rank == 0:
          Ulocals = [U0, U1]
          Vlocals = [V0, V1]
      elif ensemble.ensemble_rank == 1:
          Ulocals = [U2]
        Vlocals = [V2]

      U = EnsembleFunctionSpace(Ulocals, ensemble)
      V = EnsembleFunctionSpace(Vlocals, ensemble)

      u = EnsembleFunction(U)
      v = EnsembleFunction(V)

      Jlocals = []
      wlocals = []
      for ui, vi in zip(u.subfunctions, v.subfunctions):
          with set_working_tape() as tape:
              wi = assemble(ui*vi*dx)
              Ji = ReducedFunctional(wi, [Control(ui), Control(vi)], tape=tape)

          wlocals.append(wi)
          Jlocals.append(Ji)

      w = EnsembleAdjVec(wlocals, ensemble)

      Jhat = EnsembleTransformReducedFunctional(
          w, [Control(u), Control(v)], Jlocals)

    Note that by using :func:`~pyadjoint.set_working_tape` we ensure that
    each local :class:`~pyadjoint.ReducedFunctional` has its own tape.
    For such a simple example this is unlikely to make a difference, but
    for more complex operations this will ensure that each local
    ``ReducedFunctional`` does not interfere with the others.

    Parameters
    ----------
    functional :
        The result of the transform.
    control :
        The inputs to the transform.

    Notes
    -----
    Unlike most ``ReducedFunctional`` classes, this one does not require any
    operations on the ``control`` and ``functional`` to be taped before
    creating it.  The ``functional`` and ``control`` arguments are just to
    specify the source and destination spaces.

    This class is primarily intended as a component for building larger
    ``ReducedFunctional`` classes over an :class:`.Ensemble`, for example the
    :class:`.EnsembleReducedFunctional`.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    :class:`~.Ensemble`
    :class:`~.EnsembleFunction`
    :class:`~.EnsembleCofunction`
    :class:`~.EnsembleAdjVec`
    :class:`~.EnsembleReduceReducedFunctional`
    :class:`~.EnsembleBcastReducedFunctional`
    :class:`~.EnsembleAllgatherReducedFunctional`
    :class:`~.EnsembleReducedFunctional`
    """
    def __init__(self, functional: OverloadedType, control: Control | list[Control],
                 rfs: AbstractReducedFunctional | list[AbstractReducedFunctional]):
        self.rfs = Enlist(rfs)
        self.functional = functional
        self._controls = Enlist(control)

        EnsembleTypes = (EnsembleFunctionBase, EnsembleAdjVec)

        if not isinstance(functional, EnsembleTypes):
            raise TypeError(
                f"Functional for {type(self).__name__} must be either an"
                f" EnsembleFunction or EnsembleAdjVec, not {type(functional)}"
            )
        for c in self.controls:
            if not isinstance(c.control, EnsembleTypes):
                raise TypeError(
                    f"Controls for {type(self).__name__} must be either an "
                    f"EnsembleFunction or EnsembleAdjVec not {type(c.control)}"
                )

        clens = set(len(_local_subs(c.control)) for c in self.controls)
        flen = len(_local_subs(functional))
        rlen = len(self.rfs)
        if len(clens) != 1:
            raise ValueError(
                f"All Controls for {type(self).__name__} must have"
                " the same number of components on each ensemble rank"
            )
        clen = clens.pop()
        if clen != flen:
            raise ValueError(
                f"Control of with {clen} local components for"
                f" {type(self).__name__} must have the same number of local"
                f" components as the functional ({flen})"
            )
        if clen != rlen:
            raise ValueError(
                f"{type(self).__name__} given {rlen} local ReducedFunctionals,"
                f" but needs one for each local component of Control with"
                f" length {clen}")

    @property
    def controls(self):
        return self._controls

    @property
    def ensemble(self):
        """The :class:`.Ensemble` that the control and functional are defined
        over."""
        return _ensemble(self.functional)

    @no_annotations
    def __call__(self, values):
        for c, v in zip(self.controls, Enlist(values)):
            c.update(v)

        local_vals = self._global_to_local_data(values)
        local_Js = [rf(v) for rf, v in zip(self.rfs, local_vals)]

        J = self.functional._ad_init_zero()
        self._local_to_global_data(local_Js, J)

        return J

    @no_annotations
    def tlm(self, m_dot):
        local_mdot = self._global_to_local_data(m_dot)
        local_tlm = [rf.tlm(md) for rf, md in zip(self.rfs, local_mdot)]

        tlm = self.functional._ad_init_zero()
        self._local_to_global_data(local_tlm, tlm)

        return tlm

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        local_adj = self._global_to_local_data(adj_input)
        local_dJ = [rf.derivative(adj_input=adj[0], apply_riesz=apply_riesz)
                    for rf, adj in zip(self.rfs, local_adj)]

        dJ = self.controls.delist(
            [c.control._ad_init_zero(dual=not apply_riesz)
             for c in self.controls])

        self._local_to_global_data(local_dJ, dJ)

        return dJ

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True,
                apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)

        local_hin = self._global_to_local_data(hessian_input)
        local_hess = [rf.hessian(m_dot=None, evaluate_tlm=False,
                                 hessian_input=hin[0],
                                 apply_riesz=apply_riesz)
                      for rf, hin in zip(self.rfs, local_hin)]

        hessian = self.controls.delist(
            [c.control._ad_init_zero(dual=not apply_riesz)
             for c in self.controls])

        self._local_to_global_data(local_hess, hessian)

        return hessian

    def _local_to_global_data(self, local_data, global_data):
        # N local lists of length n -> n global lists of length N
        # [(1,), (2,), (3,)]->  [(1, 2, 3)]
        # [(1, 11), (2, 12), (3, 13)] -> [(1, 2, 3), (11, 12, 13)]

        for j, global_group in enumerate(Enlist(global_data)):
            local_group = [Enlist(local_group)[j]
                           for local_group in local_data]
            _set_local_subs(global_group, local_group)

        return global_data

    def _global_to_local_data(self, global_data):
        # n global lists of length N -> N local lists of length n
        # [(1, 2, 3)] -> [(1,), (2,), (3,)]
        # [(1, 2, 3), (11, 12, 13)] -> [(1, 11), (2, 12), (3, 13)]

        local_groups = [
            ld for ld in zip(*map(_local_subs, Enlist(global_data)))]
        return local_groups


class EnsembleReducedFunctional(AbstractReducedFunctional):
    """
    A reduced functional where multiple independent terms of the
    objective functional are calculated in parallel.

    This class covers two cases:

    1. The terms all depend on the same controls.

    2. The terms each depend on different controls.

    In the first case, if we have a functional :math:`J(m)` where :math:`m` is
    the control parameter, then we assume that :math:`J` is the sum of
    :math:`N` functionals :math:`J_i(m)`, which all depend on :math:`m` but are
    independent of each other, i.e.

    .. math::

        J(m) = \\sum_{i=1}^{N} J_i(m).

    The gradient over a summation is a linear operation. Therefore, we can
    write the gradient :math:`\\dfrac{dJ}{dm}` as

    .. math::

        \\frac{dJ}{dm} = \\sum_{i=1}^{N} \\frac{dJ_i}{dm}.

    In this case both the ``controls`` and ``functional`` are distributed only
    in space, e.g. are a :class:`.Function`, :class:`~pyadjoint.AdjFloat` etc.

    In the second case, we assume that the control :math:`m` now has :math:`N`
    components :math:`m_i`, and that each term :math:`J_i` depends on a
    different component of :math:`m`, i.e.

    .. math::

        J(m) = \\sum_{i=1}^{N} J_i(m_i).

    Now we can write the gradient :math:`\\dfrac{dJ}{dm}` component-wise:

    .. math::

        \\frac{dJ}{dm_i} = \\frac{dJ_i}{dm_i}.

    In this case the ``functional`` is distributed only in space, e.g.
    a :class:`.Function`, :class:`~pyadjoint.AdjFloat` etc, but the
    ``controls`` are distributed also over the ``Ensemble``, for example
    an :class:`~.EnsembleFunction` or :class:`~.EnsembleAdjVec`.

    In both cases, :class:`EnsembleReducedFunctional` allows simultaneous
    evaluation of :math:`J_i` and either :math:`\\dfrac{dJ_i}{dm}` or
    :math:`\\dfrac{dJ_i}{dm_i}` (as well as the tangent linear model and
    Hessian actions) by different spatial communicators on the ``ensemble``.
    All the required communication is handled by the :class:`~.Ensemble`.

    The terms :math:`J_i` on each spatial communicator are provided as a list
    of :class:`pyadjoint.ReducedFunctional` for the local terms. For the
    second case the number of terms on each spatial comm must match the number
    of local components of the controls (e.g.  :meth:`.EnsembleAdjVec.subvec`).
    It is essential that each :math:`J_i` has its own :class:`pyadjoint.Tape`
    so that it can maintain its own state independent of the other terms.

    If the ``gather_rf`` :class:`~pyadjoint.ReducedFunctional` is then, instead
    of just summing the values of :math:`J_i`, they are all-gathered onto every
    spatial comm and passed as the controls to ``gather_rf``. :math:`J` is then
    the output of the ``gather_rf``.

    If :math:`G` is the ``gather_rf`` then for case 1 this is equivalent to

    .. math::

        J(m) = G(J_1(m), J_2(m), \\dots, J_N(m))

    and for case 2 this is equivalent to

    .. math::

        J(m) = G(J_1(m_1), J_2(m_2), \\dots, J_N(m_N))

    Parameters
    ----------
    functional :
        An instance of an OverloadedType, usually :class:`~pyadjoint.AdjFloat`.
        This should be the functional that we want to calculate.
    control :
        A single or a list of :class:`pyadjoint.Control` instances, which you
        want to map to the functional.
    rfs :
        The :class:`~pyadjoint.ReducedFunctional` for each term :math:`J_{i}`.
        It is essential that each ``rf`` has its own :class:`pyadjoint.Tape`
        so that it can maintain its own state independent of the other terms.
    gather_rf :
        The reduced functional to map all ensemble components to the
        functional. Requires the functional to be a non-ensemble type (e.g. an
        :class:`~pyadjoint.AdjFloat` or :class:`~.Function`)
    ensemble :
        An instance of the :class:`~.ensemble.Ensemble`. It is used to
        communicate the functionals and their derivatives between the ensemble
        members. If either the functional or controls are ensemble types (e.g.
        :class:`.EnsembleFunction` or :class:`.EnsembleAdjVec`) then the
        ``ensemble`` is accessed from them, and this argument is ignored.

    Notes
    -----
    Each :class:`~pyadjoint.ReducedFunctional` in ``rfs`` that define the
    functionals :math:`J_i` must be defined over a single ``ensemble.comm``
    communicator.

    To understand more about ensemble parallelism, please refer to the
    `Firedrake manual <https://www.firedrakeproject.org/ensemble_parallelism.html>`_.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    :class:`~.Ensemble`
    :class:`~.EnsembleFunction`
    :class:`~.EnsembleCofunction`
    :class:`~.EnsembleAdjVec`
    :class:`~.EnsembleReduceReducedFunctional`
    :class:`~.EnsembleBcastReducedFunctional`
    :class:`~.EnsembleTransformReducedFunctional`
    :class:`~.EnsembleAllgatherReducedFunctional`
    """
    def __init__(self, functional: OverloadedType,
                 control: Control | list[Control],
                 rfs: list[AbstractReducedFunctional],
                 gather_rf: AbstractReducedFunctional | None = None,
                 ensemble: Ensemble | None = None):
        self._local_rfs = Enlist(rfs)
        self._controls = Enlist(control)
        self.functional = functional
        self._ensemble = ensemble
        self._gather_rf = gather_rf

        # total number of ensemble components
        local_size = len(self._local_rfs)
        global_size = self.ensemble.allreduce(local_size)

        # Case 1: Standard summation reduction
        #                    -------                           -----------                             --------
        # outer_controls -> | bcast | -> ensemble_controls -> | transform | -> ensemble_functional -> | reduce |  -> outer_functional
        #                    -------                           -----------                             --------
        #
        # Case 2: Non-summation reduction requiring gather
        #                    -------                           -----------                             --------                           -----------
        # outer_controls -> | bcast | -> ensemble_controls -> | transform | -> ensemble_functional -> | gather | -> gather_functional -> | gather_rf | -> outer_functional
        #                    -------                           -----------                             --------                           -----------

        ensemble_types = (EnsembleFunctionBase, EnsembleAdjVec)

        # Do we need to broadcast the controls?
        is_ensemble_control = set(isinstance(c.control, ensemble_types)
                                  for c in self.controls)
        if len(is_ensemble_control) != 1:
            raise TypeError(
                "Either all or none of the controls must be ensemble types")

        if is_ensemble_control.pop():
            self._input_op = 'none'
        else:
            self._input_op = 'bcast'

        # Do we need to reduce or gather the functional?
        if isinstance(functional, ensemble_types):
            self._output_op = 'none'
        elif gather_rf is None:
            self._output_op = 'reduce'
        else:
            self._output_op = 'gather'

        # build ensemble_controls
        if self._input_op == 'bcast':
            ensemble_controls = [
                Control(_make_ensemble_obj(
                    [control.control for _ in range(local_size)],
                    ensemble=self.ensemble))
                for control in self.controls
            ]
        elif self._input_op == 'none':
            ensemble_controls = self.controls

        # build ensemble_functional
        if self._output_op in ('reduce', 'gather'):
            ensemble_functional = _make_ensemble_obj(
                [rf.functional for rf in self._local_rfs],
                ensemble=self.ensemble)
        elif self._output_op == 'none':
            ensemble_functional = self.functional

        # build the transform
        self._ensemble_transform = EnsembleTransformReducedFunctional(
            ensemble_functional,
            self.controls.delist(ensemble_controls),
            self._local_rfs)

        # build the input operation
        if self._input_op == 'bcast':
            # controls are Functions or AdjFloats, so need to bcast to
            # EnsembleFunctions or EnsembleAdjVecs for EnsembleTransform input.
            self._ensemble_bcast = [
                EnsembleBcastReducedFunctional(
                    ec.control._ad_init_zero(),
                    Control(c.control._ad_init_zero()))
                for ec, c in zip(ensemble_controls, self.controls)]

        # build the output operation
        if self._output_op == 'reduce':
            # functional is Function or AdjFloat, so need to reduce from
            # EnsembleFunction or EnsembleAdjVec from EnsembleTransform output.
            self._ensemble_reduce = EnsembleReduceReducedFunctional(
                functional._ad_init_zero(),
                Control(ensemble_functional._ad_init_zero()))

        if self._output_op == 'gather':
            # Need to gather all components of the ensemble_functional onto
            # each rank to pipe through gather_rf before returning result.
            # check gather takes right type of arguments
            if len(gather_rf.controls) != global_size:
                raise ValueError("gather_rf must have one control for"
                                 " each component on all ensemble members")

            # check gather takes correct type of arguments
            gather_type = type(self._local_rfs[0].functional)
            if not all(isinstance(c.control, gather_type)
                       for c in gather_rf.controls):
                raise ValueError(
                    "gather_rf.controls must match types of rf.functional")

            # Now make the massive output type for the gather
            gather_functional = _make_ensemble_obj(
                [c.control._ad_init_zero() for c in gather_rf.controls],
                ensemble=ensemble)

            self._ensemble_gather = EnsembleAllgatherReducedFunctional(
                gather_functional,
                Control(ensemble_functional._ad_init_zero()))

    @property
    def controls(self):
        return self._controls

    @property
    def ensemble(self):
        """The :class:`.Ensemble` that the reduced functional is
        defined over."""
        return self._ensemble

    @no_annotations
    def __call__(self, values):
        vals = Enlist(values)
        for c, v in zip(self.controls, vals):
            c.update(v)

        if self._input_op == 'bcast':
            vals = [bcast(val)
                    for bcast, val in zip(self._ensemble_bcast, vals)]

        J = self._ensemble_transform(vals)

        if self._output_op == 'reduce':
            # just sum the transform results
            J = self._ensemble_reduce(J)
        elif self._output_op == 'gather':
            # pipe the gathered transform results through the gather_rf
            gathered_Js = self._ensemble_gather(J)
            J = self._gather_rf(_local_subs(gathered_Js))

        return J

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        if self._output_op == 'reduce':
            # just broadcast the adj_input
            adj_input = self._ensemble_reduce.derivative(adj_input,
                                                         apply_riesz=False)

        elif self._output_op == 'gather':
            # pipe the adj_input through the gather_rf before broadcasting
            local_input = self._gather_rf.derivative(adj_input,
                                                     apply_riesz=False)

            global_input = _make_ensemble_obj(local_input, self.ensemble)

            adj_input = self._ensemble_gather.derivative(global_input,
                                                         apply_riesz=False)

        transform_riesz = apply_riesz if self._input_op == 'none' else False

        dJ = self._ensemble_transform.derivative(
            adj_input=adj_input, apply_riesz=transform_riesz)

        if self._input_op == 'bcast':
            dJ = self.controls.delist(
                [bcast.derivative(adj_input=dj, apply_riesz=apply_riesz)
                 for bcast, dj in zip(self._ensemble_bcast, Enlist(dJ))])

        return dJ

    @no_annotations
    def tlm(self, m_dot):
        if self._input_op == 'bcast':
            m_dot = [bcast(md)
                     for bcast, md in zip(self._ensemble_bcast, Enlist(m_dot))]

        tlv = self._ensemble_transform.tlm(m_dot)

        if self._output_op == 'reduce':
            # just sum the transform results
            tlv = self._ensemble_reduce.tlm(tlv)
        elif self._output_op == 'gather':
            # pipe the gathered transform results through the gather_rf
            gathered_tlvs = self._ensemble_gather.tlm(tlv)
            tlv = self._gather_rf.tlm(_local_subs(gathered_tlvs))

        return tlv

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)

        hkwargs = {'m_dot': None, 'evaluate_tlm': False}

        if self._output_op == 'reduce':
            # just broadcast the hessian_input
            hessian_input = self._ensemble_reduce.hessian(
                **hkwargs, hessian_input=hessian_input, apply_riesz=False)

        elif self._output_op == 'gather':
            # pipe the hessian_input through the gather_rf before broadcasting
            local_input = self._gather_rf.hessian(
                **hkwargs, hessian_input=hessian_input, apply_riesz=False)

            global_input = _make_ensemble_obj(local_input, self.ensemble)

            hessian_input = self._ensemble_gather.hessian(
                **hkwargs, hessian_input=global_input, apply_riesz=False)

        transform_riesz = apply_riesz if self._input_op == 'none' else False

        hessian = self._ensemble_transform.hessian(
            **hkwargs, hessian_input=hessian_input, apply_riesz=transform_riesz)

        if self._input_op == 'bcast':
            hessian = self.controls.delist(
                [bcast.hessian(**hkwargs, hessian_input=hess, apply_riesz=apply_riesz)
                 for bcast, hess in zip(self._ensemble_bcast, Enlist(hessian))])

        return hessian
