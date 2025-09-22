"""
Tests FML's Term objects. A term contains a form and labels.
"""

from firedrake import (
    IntervalMesh, FunctionSpace, Function, TestFunction, dx, Constant
)
from firedrake.fml import Label, Term, LabelledForm
import pytest


# Two methods of making a Term with Labels. Either pass them as a dict
# at the initialisation of the Term, or apply them afterwards
@pytest.mark.parametrize("initialise", ["from_dicts", "apply_labels"])
def test_term(initialise):

    # ------------------------------------------------------------------------ #
    # Set up terms
    # ------------------------------------------------------------------------ #

    # Some basic labels
    foo_label = Label("foo", validator=lambda value: type(value) == bool)
    lorem_label = Label("lorem", validator=lambda value: type(value) == str)
    ipsum_label = Label("ipsum", validator=lambda value: type(value) == int)

    # Dict for matching the label names to the label objects
    all_labels = [foo_label, lorem_label, ipsum_label]
    all_label_dict = {label.label: label for label in all_labels}

    # Create mesh, function space and forms
    L = 3.0
    n = 3
    mesh = IntervalMesh(n, L)
    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V)
    g = Function(V)
    h = Function(V)
    test = TestFunction(V)
    form = f*test*dx

    # Declare what the labels will be
    label_dict = {'foo': True, 'lorem': 'etc', 'ipsum': 1}

    # Make terms
    if initialise == "from_dicts":
        term = Term(form, label_dict)
    else:
        term = Term(form)

        # Apply labels
        for label_name, value in label_dict.items():
            term = all_label_dict[label_name](term, value)

    # ------------------------------------------------------------------------ #
    # Test Term.get routine
    # ------------------------------------------------------------------------ #

    for label in all_labels:
        if label.label in label_dict.keys():
            # Check if label is attached to Term and it has correct value
            assert term.get(label) == label_dict[label.label], \
                f'term should have label {label.label} with value equal ' + \
                f'to {label_dict[label.label]} and not {term.get(label)}'
        else:
            # Labelled shouldn't be attached to Term so this should return None
            assert term.get(label) is None, 'term should not have ' + \
                f'label {label.label} but term.get(label) returns ' + \
                f'{term.get(label)}'

    # ------------------------------------------------------------------------ #
    # Test Term.has_label routine
    # ------------------------------------------------------------------------ #

    # Test has_label for each label one by one
    for label in all_labels:
        assert term.has_label(label) == (label.label in label_dict.keys()), \
            f'term.has_label giving incorrect value for {label.label}'

    # Test has_labels by passing all labels at once
    has_labels = term.has_label(*all_labels, return_tuple=True)
    for i, label in enumerate(all_labels):
        assert has_labels[i] == (label.label in label_dict.keys()), \
            f'has_label for label {label.label} returning wrong value'

    # Check the return_tuple option is correct when only one label is passed
    has_labels = term.has_label(*[foo_label], return_tuple=True)
    assert len(has_labels) == 1, 'Length returned by has_label is ' + \
        f'incorrect, it is {len(has_labels)} but should be 1'
    assert has_labels[0] == (label.label in label_dict.keys()), \
        f'has_label for label {label.label} returning wrong value'

    # ------------------------------------------------------------------------ #
    # Test Term addition and subtraction
    # ------------------------------------------------------------------------ #

    form_2 = g*test*dx
    term_2 = ipsum_label(Term(form_2), 2)

    labelled_form_1 = term_2 + term
    labelled_form_2 = term + term_2

    # Adding two Terms should return a LabelledForm containing the Terms
    assert type(labelled_form_1) is LabelledForm, 'The sum of two Terms ' + \
        f'should be a LabelledForm, not {type(labelled_form_1)}'
    assert type(labelled_form_2) is LabelledForm, 'The sum of two Terms ' + \
        f'should be a LabelledForm, not {type(labelled_form_1)}'

    # Adding a LabelledForm to a Term should return a LabelledForm
    labelled_form_3 = term + labelled_form_2
    assert type(labelled_form_3) is LabelledForm, 'The sum of a Term and ' + \
        f'Labelled Form should be a LabelledForm, not {type(labelled_form_3)}'

    labelled_form_1 = term_2 - term
    labelled_form_2 = term - term_2

    # Subtracting two Terms should return a LabelledForm containing the Terms
    assert type(labelled_form_1) is LabelledForm, 'The difference of two ' + \
        f'Terms should be a LabelledForm, not {type(labelled_form_1)}'
    assert type(labelled_form_2) is LabelledForm, 'The difference of two ' + \
        f'Terms should be a LabelledForm, not {type(labelled_form_1)}'

    # Subtracting a LabelledForm from a Term should return a LabelledForm
    labelled_form_3 = term - labelled_form_2
    assert type(labelled_form_3) is LabelledForm, 'The differnce of a Term ' + \
        f'and a Labelled Form should be a LabelledForm, not {type(labelled_form_3)}'

    # Adding None to a Term should return the Term
    new_term = term + None
    assert term == new_term, 'Adding None to a Term should give the same Term'

    # ------------------------------------------------------------------------ #
    # Test Term multiplication and division
    # ------------------------------------------------------------------------ #

    # Multiplying a term by an integer should give a Term
    new_term = term*3
    assert type(new_term) is Term, 'Multiplying a Term by an integer ' + \
        f'give a Term, not a {type(new_term)}'

    # Multiplying a term by a float should give a Term
    new_term = term*19.0
    assert type(new_term) is Term, 'Multiplying a Term by a float ' + \
        f'give a Term, not a {type(new_term)}'

    # Multiplying a term by a Constant should give a Term
    new_term = term*Constant(-4.0)
    assert type(new_term) is Term, 'Multiplying a Term by a Constant ' + \
        f'give a Term, not a {type(new_term)}'

    # Dividing a term by an integer should give a Term
    new_term = term/3
    assert type(new_term) is Term, 'Dividing a Term by an integer ' + \
        f'give a Term, not a {type(new_term)}'

    # Dividing a term by a float should give a Term
    new_term = term/19.0
    assert type(new_term) is Term, 'Dividing a Term by a float ' + \
        f'give a Term, not a {type(new_term)}'

    # Dividing a term by a Constant should give a Term
    new_term = term/Constant(-4.0)
    assert type(new_term) is Term, 'Dividing a Term by a Constant ' + \
        f'give a Term, not a {type(new_term)}'

    # Multiplying a term by a Function should fail
    try:
        new_term = term*h
        # If we get here we have failed
        assert False, 'Multiplying a Term by a Function should fail'
    except TypeError:
        pass
