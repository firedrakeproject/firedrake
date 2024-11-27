"""
Tests FML's LabelledForm objects.
"""

from firedrake import (
    IntervalMesh, FunctionSpace, Function, TestFunction, dx, Constant
)
from firedrake.fml import Label, Term, LabelledForm
from ufl import Form


def test_labelled_form():

    # ------------------------------------------------------------------------ #
    # Set up labelled forms
    # ------------------------------------------------------------------------ #

    # Some basic labels
    lorem_label = Label("lorem", validator=lambda value: type(value) == str)
    ipsum_label = Label("ipsum", validator=lambda value: type(value) == int)

    # Create mesh, function space and forms
    L = 3.0
    n = 3
    mesh = IntervalMesh(n, L)
    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V)
    g = Function(V)
    test = TestFunction(V)
    form_1 = f*test*dx
    form_2 = g*test*dx
    term_1 = lorem_label(Term(form_1), 'i_have_lorem')
    term_2 = ipsum_label(Term(form_2), 5)

    # ------------------------------------------------------------------------ #
    # Test labelled forms have the correct number of terms
    # ------------------------------------------------------------------------ #

    # Create from a single term
    labelled_form_1 = LabelledForm(term_1)
    assert len(labelled_form_1) == 1, 'LabelledForm should have 1 term'

    # Create from multiple terms
    labelled_form_2 = LabelledForm(*[term_1, term_2])
    assert len(labelled_form_2) == 2, 'LabelledForm should have 2 terms'

    # Trying to create from two LabelledForms should give an error
    try:
        labelled_form_3 = LabelledForm(labelled_form_1, labelled_form_2)
        # If we get here something has gone wrong
        assert False, 'We should not be able to create LabelledForm ' + \
            'from two LabelledForms'
    except TypeError:
        pass

    # Create from a single LabelledForm
    labelled_form_3 = LabelledForm(labelled_form_1)
    assert len(labelled_form_3) == 1, 'LabelledForm should have 1 term'

    # ------------------------------------------------------------------------ #
    # Test getting form
    # ------------------------------------------------------------------------ #

    assert type(labelled_form_1.form) is Form, 'The form belonging to the ' + \
        f'LabelledForm must be a Form, and not {type(labelled_form_1.form)}'

    assert type(labelled_form_2.form) is Form, 'The form belonging to the ' + \
        f'LabelledForm must be a Form, and not {type(labelled_form_2.form)}'

    assert type(labelled_form_3.form) is Form, 'The form belonging to the ' + \
        f'LabelledForm must be a Form, and not {type(labelled_form_3.form)}'

    # ------------------------------------------------------------------------ #
    # Test addition and subtraction of labelled forms
    # ------------------------------------------------------------------------ #

    # Add a Form to a LabelledForm
    new_labelled_form = labelled_form_1 + form_2
    assert len(new_labelled_form) == 2, 'LabelledForm should have 2 terms'

    # Add a Term to a LabelledForm
    new_labelled_form = labelled_form_1 + term_2
    assert len(new_labelled_form) == 2, 'LabelledForm should have 2 terms'

    # Add a LabelledForm to a LabelledForm
    new_labelled_form = labelled_form_1 + labelled_form_2
    assert len(new_labelled_form) == 3, 'LabelledForm should have 3 terms'

    # Adding None to a LabelledForm should give the same LabelledForm
    new_labelled_form = labelled_form_1 + None
    assert new_labelled_form == labelled_form_1, 'Two LabelledForms should be equal'

    # Subtract a Form from a LabelledForm
    new_labelled_form = labelled_form_1 - form_2
    assert len(new_labelled_form) == 2, 'LabelledForm should have 2 terms'

    # Subtract a Term from a LabelledForm
    new_labelled_form = labelled_form_1 - term_2
    assert len(new_labelled_form) == 2, 'LabelledForm should have 2 terms'

    # Subtract a LabelledForm from a LabelledForm
    new_labelled_form = labelled_form_1 - labelled_form_2
    assert len(new_labelled_form) == 3, 'LabelledForm should have 3 terms'

    # Subtracting None from a LabelledForm should give the same LabelledForm
    new_labelled_form = labelled_form_1 - None
    assert new_labelled_form == labelled_form_1, 'Two LabelledForms should be equal'

    # ------------------------------------------------------------------------ #
    # Test multiplication and division of labelled forms
    # ------------------------------------------------------------------------ #

    # Multiply by integer
    new_labelled_form = labelled_form_1 * -4
    assert len(new_labelled_form) == 1, 'LabelledForm should have 1 term'

    # Multiply by float
    new_labelled_form = labelled_form_1 * 12.4
    assert len(new_labelled_form) == 1, 'LabelledForm should have 1 term'

    # Multiply by Constant
    new_labelled_form = labelled_form_1 * Constant(5.0)
    assert len(new_labelled_form) == 1, 'LabelledForm should have 1 term'

    # Divide by integer
    new_labelled_form = labelled_form_1 / (-8)
    assert len(new_labelled_form) == 1, 'LabelledForm should have 1 term'

    # Divide by float
    new_labelled_form = labelled_form_1 / (-6.2)
    assert len(new_labelled_form) == 1, 'LabelledForm should have 1 term'

    # Divide by Constant
    new_labelled_form = labelled_form_1 / Constant(0.01)
    assert len(new_labelled_form) == 1, 'LabelledForm should have 1 term'
