from pyadjoint import stop_annotating, get_working_tape, OverloadedType, Control, Tape, ReducedFunctional
from pyadjoint.enlisting import Enlist
from typing import Optional


def intermediate_options(options: dict):
    """
    Options set for the intermediate stages of a chain of ReducedFunctionals

    Takes all elements of the options except riesz_representation, which
    is set to None to prevent returning derivatives to the primal space.

    Parameters
    ----------
    options
        The dictionary of options provided by the user

    Returns
    -------
    dict
        The options for ReducedFunctionals at intermediate stages

    """
    return {
        **{k: v for k, v in (options or {}).items()
           if k != 'riesz_representation'},
        'riesz_representation': None
    }


def compute_tlm(J: OverloadedType,
                m: Control,
                m_dot: OverloadedType,
                options: Optional[dict] = None,
                tape: Optional[Tape] = None):
    """
    Compute the tangent linear model of J in a direction m_dot at the current value of m

    Parameters
    ----------

    J
        The objective functional.
    m
        The (list of) :class:`pyadjoint.Control` for the functional.
    m_dot
        The direction in which to compute the Hessian.
        Must be a (list of) :class:`pyadjoint.OverloadedType`.
    options
        A dictionary of options. To find a list of available options
        have a look at the specific control type.
    tape
        The tape to use. Default is the current tape.

    Returns
    -------
    pyadjoint.OverloadedType
        The tangent linear with respect to the control in direction m_dot.
        Should be an instance of the same type as the control.

    """
    tape = tape or get_working_tape()

    # reset tlm values
    tape.reset_tlm_values()

    m = Enlist(m)
    m_dot = Enlist(m_dot)

    # set initial tlm values
    for mi, mdi in zip(m, m_dot):
        mi.tlm_value = mdi

    # evaluate tlm
    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_tlm(markings=True)

    # return functional's tlm
    return J._ad_convert_type(J.block_variable.tlm_value,
                              options=options or {})


def compute_hessian(J: OverloadedType,
                    m: Control,
                    options: Optional[dict] = None,
                    tape: Optional[Tape] = None,
                    hessian_value: Optional[OverloadedType] = 0.):
    """
    Compute the Hessian of J at the current value of m with the current tlm values on the tape.

    Parameters
    ----------
    J
        The objective functional.
    m
        The (list of) :class:`pyadjoint.Control` for the functional.
    options
        A dictionary of options. To find a list of available options
        have a look at the specific control type.
    tape
        The tape to use. Default is the current tape.
    hessian_value
        The initial hessian_value to start accumulating from.

    Returns
    -------
    pyadjoint.OverloadedType
        The second derivative with respect to the control in direction m_dot.
        Should be an instance of the same type as the control.

    """
    tape = tape or get_working_tape()

    # reset hessian values
    tape.reset_hessian_values()

    m = Enlist(m)

    # set initial hessian_value
    J.block_variable.hessian_value = J._ad_convert_type(
        hessian_value, options=intermediate_options(options))

    # evaluate hessian
    with stop_annotating():
        with tape.marked_nodes(m):
            tape.evaluate_hessian(markings=True)

    # return controls' hessian values
    return m.delist([v.get_hessian(options=options or {}) for v in m])


def tlm(rf: ReducedFunctional,
        m_dot: OverloadedType,
        options: Optional[dict] = None):
    """Returns the action of the tangent linear model of the functional w.r.t. the control on a vector m_dot.

    Parameters
    ----------
    rf
        The :class:`pyadjoint.ReducedFunctional` to evaluate the tlm of.
    m_dot
        The direction in which to compute the action of the tangent linear model.
    options
        A dictionary of options. To find a list of available options
        have a look at the specific control type.

    Returns
    -------
    pyadjoint.OverloadedType
        The action of the tangent linear model in the direction m_dot.
        Should be an instance of the same type as the control.

    """
    return compute_tlm(rf.functional, rf.controls, m_dot,
                       tape=rf.tape, options=options)


def hessian(rf: ReducedFunctional,
            options: Optional[dict] = None,
            hessian_value: Optional[OverloadedType] = 0.):
    """Returns the action of the Hessian of the functional w.r.t. the control.

    Using the second-order adjoint method, the action of the Hessian of the
    functional with respect to the control, around the last supplied value
    of the control and the last tlm values, is computed and returned.

    Parameters
    ----------
    rf
        The :class:`pyadjoint.ReducedFunctional` to evaluate the tlm of.
    options
        A dictionary of options. To find a list of available options
        have a look at the specific control type.
    hessian_value
        The initial hessian_value to start accumulating from.

    Returns
    -------
    pyadjoint.OverloadedType
        The action of the Hessian. Should be an instance of the same type as the control.

    """
    return rf.controls.delist(
        compute_hessian(rf.functional, rf.controls,
                        tape=rf.tape, options=options,
                        hessian_value=hessian_value))


class CompositeReducedFunctional:
    """Class representing the composition of two reduced functionals.

    For two reduced functionals J1: X->Y and J2: Y->Z, this is a convenience
    class representing the composition J12: X->Z = J2(J1(x)) and providing
    methods for the evaluation, derivative, tlm, and hessian action of J12.

    Parameters
    ----------
    rf1
        The first :class:`pyadjoint.ReducedFunctional` in the composition.
    rf2
        The second :class:`pyadjoint.ReducedFunctional` in the composition.
        The control for rf2 must have the same type as the functional of rf1.

    """
    def __init__(self, rf1, rf2):
        self.rf1 = rf1
        self.rf2 = rf2

    def __call__(self, values: OverloadedType):
        """Computes the reduced functional with supplied control value.

        Parameters
        ----------

        values
            If you have multiple controls this should be a list of new values
            for each control in the order you listed the controls to the constructor.
            If you have a single control it can either be a list or a single object.
            Each new value should have the same type as the corresponding control.

        Returns
        -------
        pyadjoint.OverloadedType
            The computed value. Typically of instance of :class:`pyadjoint.AdjFloat`.

        """
        return self.rf2(self.rf1(values))

    def derivative(self, adj_input: Optional[float] = 1.0, options: Optional[dict] = None):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input
            The adjoint input.

        options
            Additional options for the derivative computation.

        Returns
        -------
        pyadjoint.OverloadedType
            The derivative with respect to the control.
            Should be an instance of the same type as the control.

        """
        deriv2 = self.rf2.derivative(
            adj_input=adj_input, options=intermediate_options(options))
        deriv1 = self.rf1.derivative(
            adj_input=deriv2, options=options or {})
        return deriv1

    def tlm(self, m_dot: OverloadedType, options: Optional[dict] = None):
        """Returns the action of the tangent linear model of the functional w.r.t. the control on a vector m_dot.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        options
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.

        """
        tlm1 = self._eval_tlm(
            self.rf1, m_dot, intermediate_options(options)),
        tlm2 = self._eval_tlm(
            self.rf2, tlm1, options)
        return tlm2

    def hessian(self, m_dot: OverloadedType,
                options: Optional[dict] = None,
                evaluate_tlm: Optional[bool] = True):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        options
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

        evaluate_tlm
            If True, the tlm values on the tape will be reset and evaluated before
            the Hessian action is evaluated. If False, the existing tlm values on
            the tape will be used.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.

        """
        if evaluate_tlm:
            self.tlm(m_dot, options=intermediate_options(options))
        hess2 = self._eval_hessian(
            self.rf2, 0., intermediate_options(options))
        hess1 = self._eval_hessian(
            self.rf1, hess2, options or {})
        return hess1

    def _eval_tlm(self, rf, m_dot, options):
        if isinstance(rf, CompositeReducedFunctional):
            return rf.tlm(m_dot, options=options)
        else:
            return tlm(rf, m_dot=m_dot, options=options)

    def _eval_hessian(self, rf, hessian_value, options):
        if isinstance(rf, CompositeReducedFunctional):
            return rf.hessian(None, options, evaluate_tlm=False)
        else:
            return hessian(rf, hessian_value=hessian_value, options=options)
