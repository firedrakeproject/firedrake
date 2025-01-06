from firedrake.adjoint import stop_annotating, get_working_tape
from pyadjoint.enlisting import Enlist


def intermediate_options(options):
    """
    Options set for the intermediate stages of a chain of ReducedFunctionals

    Takes all elements of the options except riesz_representation, which
    is set to None to prevent returning derivatives to the primal space.
    """
    return {
        **{k: v for k, v in (options or {}).items()
           if k != 'riesz_representation'},
        'riesz_representation': None
    }


def compute_tlm(J, m, m_dot, options=None, tape=None):
    """
    Compute the tangent linear model of J in a direction m_dot at the current value of m

    Args:
        J (OverloadedType):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        m_dot (list or instance of the control type): The direction in which to compute the Hessian.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The tangent linear with respect to the control in direction m_dot.
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


def compute_hessian(J, m, options=None, tape=None, hessian_value=0.):
    """
    Compute the Hessian of J at the current value of m with the current tlm values on the tape.

    Args:
        J (OverloadedType):  The objective functional.
        m (list or instance of Control): The (list of) controls.
        options (dict): A dictionary of options. To find a list of available options
            have a look at the specific control type.
        tape: The tape to use. Default is the current tape.

    Returns:
        OverloadedType: The second derivative with respect to the control in direction m_dot.
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


def tlm(rf, m_dot, options=None):
    """Returns the action of the tangent linear model of the functional w.r.t. the control on a vector m_dot.

    Args:
        m_dot ([OverloadedType]): The direction in which to compute the
            action of the tangent linear model.
        options (dict): A dictionary of options. To find a list of
            available options have a look at the specific control type.

    Returns:
        OverloadedType: The action of the tangent linear model in the direction m_dot.
            Should be an instance of the same type as the control.
    """
    return compute_tlm(rf.functional, rf.controls, m_dot,
                       tape=rf.tape, options=options)


def hessian(rf, options=None, hessian_value=0.):
    """Returns the action of the Hessian of the functional w.r.t. the control.

    Using the second-order adjoint method, the action of the Hessian of the
    functional with respect to the control, around the last supplied value
    of the control and the last tlm values, is computed and returned.

    Args:
        options (dict): A dictionary of options. To find a list of
            available options have a look at the specific control type.
        hessian_value: The Hessian value to initialise the accumulation
            from the functional block variable.

    Returns:
        OverloadedType: The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.
    """
    return rf.controls.delist(
        compute_hessian(rf.functional, rf.controls,
                        tape=rf.tape, options=options,
                        hessian_value=hessian_value))


class CompositeReducedFunctional:
    def __init__(self, rf1, rf2):
        self.rf1 = rf1
        self.rf2 = rf2

    def __call__(self, values):
        return self.rf2(self.rf1(values))

    def derivative(self, adj_input=1.0, options=None):
        deriv2 = self.rf2.derivative(
            adj_input=adj_input, options=intermediate_options(options))
        deriv1 = self.rf1.derivative(
            adj_input=deriv2, options=options or {})
        return deriv1

    def tlm(self, m_dot, options=None):
        tlm1 = self._eval_tlm(
            self.rf1, m_dot, intermediate_options(options)),
        tlm2 = self._eval_tlm(
            self.rf2, tlm1, options)
        return tlm2

    def hessian(self, m_dot, options=None, evaluate_tlm=True):
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
