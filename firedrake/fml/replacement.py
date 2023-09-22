"""
Generic routines for replacing functions using FML.
"""

import ufl
from .form_manipulation_language import Term, subject
from firedrake import split, MixedElement

__all__ = ["replace_test_function", "replace_trial_function", "replace_subject"]


# ---------------------------------------------------------------------------- #
# A general routine for building the replacement dictionary
# ---------------------------------------------------------------------------- #
def _replace_dict(old, new, old_idx, new_idx, replace_type):
    """
    Build a dictionary to pass to the ufl.replace routine. The dictionary
    matches variables in the old term with those in the new.

    Does not check types unless indexing is required (leave type-checking to
    ufl.replace).

    Parameters
    ----------
    old : :class:`Function` or :class:`TestFunction` or :class:`TrialFunction`
        The old variable to be replaced.
    new : :class:`Function` or :class:`TestFunction` or :class:`TrialFunction`
        The new variable to be replace with.
    old_idx : int
        The index of the old variable to be replaced. If the old variable is not
        indexable then this should be None.
    new_idx : int
        The index of the new variable to replace with. If the new variable is
        not indexable then this should be None.
    replace_type : str
        A string to use in error messages, describing the type of replacement
        that is happening.

    Returns
    -------
    dict
        A dictionary pairing the variables in the old term to be replaced with
        the new variables to replace them.

    Raises
    ------
    ValueError
        If the old_idx argument is not provided when an indexable variable is to
        be replaced by something not of the same shape.
    ValueError
        If the new_idx argument is not provided when an indexable variable is to
        be replace something not of the same shape.
    """

    mixed_old = type(old.ufl_element()) is MixedElement
    mixed_new = hasattr(new, "ufl_element") and type(new.ufl_element()) is MixedElement

    indexable_old = mixed_old
    indexable_new = mixed_new or type(new) is tuple

    if mixed_old:
        split_old = split(old)
    if indexable_new:
        split_new = new if type(new) is tuple else split(new)

    # check indices arguments are valid
    if not indexable_old and old_idx is not None:
        raise ValueError(f"old_idx should not be specified to replace_{replace_type}"
                         + f" when replaced {replace_type} of type {old} is not mixed.")

    if not indexable_new and new_idx is not None:
        raise ValueError(f"new_idx should not be specified to replace_{replace_type} when"
                         + f" new {replace_type} of type {new} is not mixed or indexable.")

    if indexable_old and not indexable_new:
        if old_idx is None:
            raise ValueError(f"old_idx must be specified to replace_{replace_type} when replaced"
                             + f" {replace_type} of type {old} is mixed and new {replace_type}"
                             + f" of type {new} is not mixed or indexable.")

    if indexable_new and not indexable_old:
        if new_idx is None:
            raise ValueError(f"new_idx must be specified to replace_{replace_type} when new"
                             + f" {replace_type} of type {new} is mixed or indexable and"
                             + f" old {replace_type} of type {old} is not mixed.")

    if indexable_old and indexable_new:
        # must be both True or both False
        if (old_idx is None) ^ (new_idx is None):
            raise ValueError("both or neither old_idx and new_idx must be specified to"
                             + f" replace_{replace_type} when old {replace_type} of type"
                             + f" {old} is mixed and new {replace_type} of type {new} is"
                             + " mixed or indexable.")
        if old_idx is None:  # both indexes are none
            if len(split_old) != len(split_new):
                raise ValueError(f"if neither index is specified to replace_{replace_type}"
                                 + f" and both old {replace_type} of type {old} and new"
                                 + f" {replace_type} of type {new} are mixed or indexable"
                                 + f" then old of length {len(split_old)} and new of length {len(split_new)}"
                                 + " must be the same length.")

    # make the replace_dict

    replace_dict = {}

    if not indexable_old and not indexable_new:
        replace_dict[old] = new

    elif not indexable_old and indexable_new:
        replace_dict[old] = split_new[new_idx]

    elif indexable_old and not indexable_new:
        replace_dict[split_old[old_idx]] = new

    elif indexable_old and indexable_new:
        if old_idx is None:  # replace everything
            for k, v in zip(split_old, split_new):
                replace_dict[k] = v
        else:  # idxs are given
            replace_dict[split_old[old_idx]] = split_new[new_idx]

    return replace_dict


# ---------------------------------------------------------------------------- #
# Replacement routines
# ---------------------------------------------------------------------------- #
def replace_test_function(new_test, old_idx=None, new_idx=None):
    """
    A routine to replace the test function in a term with a new test function.

    Parameters
    ----------
    new_test : :func:`.TestFunction`
        The new test function.

    Returns
    -------
    :obj:`callable`
        A function that takes in t, a :class:`.Term`, and returns a new
        :class:`.Term` with form containing the ``new_test`` and
        ``labels=t.labels``
    """

    def repl(t):
        """
        Replaces the test function in a term with a new expression. This is
        built around the ufl replace routine.

        Parameters
        ----------
        t : :class:`.Term`
            The original term.

        Returns
        -------
        :class:`.Term`
            The new term.
        """
        old_test = t.form.arguments()[0]
        replace_dict = _replace_dict(old_test, new_test,
                                     old_idx=old_idx, new_idx=new_idx,
                                     replace_type='test')

        try:
            new_form = ufl.replace(t.form, replace_dict)
        except Exception as err:
            error_message = f"{type(err)} raised by ufl.replace when trying to" \
                            + f" replace_test_function with {new_test}"
            raise type(err)(error_message) from err

        return Term(new_form, t.labels)

    return repl


def replace_trial_function(new_trial, old_idx=None, new_idx=None):
    """
    A routine to replace the trial function in a term with a new expression.

    Parameters
    ----------
    new : :func:`.TrialFunction` or :class:`.Function`
        The new function.

    Returns
    -------
    :obj:`callable`
        A function that takes in t, a :class:`.Term`, and returns a new
        :class:`.Term` with form containing the ``new_test`` and
        ``labels=t.labels``
    """

    def repl(t):
        """
        Replaces the trial function in a term with a new expression. This is
        built around the ufl replace routine.

        Parameters
        ----------
        t (:class:`.Term`)
            The original term.

        Raises
        ------
        TypeError
            If the form is not linear.

        Returns
        -------
        :class:`.Term`
            The new term.
        """
        if len(t.form.arguments()) != 2:
            raise TypeError('Trying to replace trial function of a form that is not linear')
        old_trial = t.form.arguments()[1]
        replace_dict = _replace_dict(old_trial, new_trial,
                                     old_idx=old_idx, new_idx=new_idx,
                                     replace_type='trial')

        try:
            new_form = ufl.replace(t.form, replace_dict)
        except Exception as err:
            error_message = f"{type(err)} raised by ufl.replace when trying to" \
                            + f" replace_trial_function with {new_trial}"
            raise type(err)(error_message) from err

        return Term(new_form, t.labels)

    return repl


def replace_subject(new_subj, old_idx=None, new_idx=None):
    """
    A routine to replace the subject in a term with a new variable.

    Parameters
    ----------
    new : :class:`ufl.Expr`
        The new expression to replace the subject.
    idx : :obj:`int`, optional
        Index of the subject in the equation's :class:`.MixedFunctionSpace`.
        Defaults to None.
    """
    def repl(t):
        """
        Replaces the subject in a term with a new expression. This is built
        around the ufl replace routine.

        Parameters
        ----------
        t : :class:`.Term`
            The original term.

        Raises
        ------
        ValueError
            When the new expression and subject are not of compatible sizes
            (e.g. a mixed function vs a non-mixed function)

        Returns
        -------
        :class:`.Term`:
            The new term.
        """

        old_subj = t.get(subject)
        replace_dict = _replace_dict(old_subj, new_subj,
                                     old_idx=old_idx, new_idx=new_idx,
                                     replace_type='subject')

        try:
            new_form = ufl.replace(t.form, replace_dict)
        except Exception as err:
            error_message = f"{type(err)} raised by ufl.replace when trying to" \
                            + f" replace_subject with {new_subj}"
            raise type(err)(error_message) from err

        return Term(new_form, t.labels)

    return repl
