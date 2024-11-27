"""
Tests FML's Label objects.
"""

from firedrake import (
    IntervalMesh, FunctionSpace, Function, TestFunction, dx
)
from firedrake.fml import Label, LabelledForm, Term
from ufl import Form
import pytest


@pytest.fixture
def label_and_values(label_type):
    # Returns labels with different value validation

    bad_value = "bar"

    if label_type == "boolean":
        # A label that is simply a string, whose value is Boolean
        this_label = Label("foo")
        good_value = True
        new_value = False

    elif label_type == "integer":
        # A label whose value is an integer
        this_label = Label("foo", validator=lambda value: (type(value) == int and value < 9))
        good_value = 5
        bad_value = 10
        new_value = 7

    elif label_type == "function":
        # A label whose value is an Function
        this_label = Label("foo", validator=lambda value: type(value) == Function)
        good_value, _ = setup_form()
        new_value = Function(good_value.function_space())

    return this_label, good_value, bad_value, new_value


def setup_form():
    # Create mesh and function space
    L = 3.0
    n = 3
    mesh = IntervalMesh(n, L)
    V = FunctionSpace(mesh, "DG", 0)
    f = Function(V)
    g = TestFunction(V)
    form = f*g*dx

    return f, form


@pytest.fixture
def object_to_label(object_type):
    # A series of different objects to be labelled

    if object_type == int:
        return 10

    else:
        _, form = setup_form()
        term = Term(form)

        if object_type == Form:
            return form

        elif object_type == Term:
            return term

        elif object_type == LabelledForm:
            return LabelledForm(term)

        else:
            raise ValueError(f'object_type {object_type} not implemented')


@pytest.mark.parametrize("label_type", ["boolean", "integer", "function"])
@pytest.mark.parametrize("object_type", [LabelledForm, Term, Form, int])
def test_label(label_type, object_type, label_and_values, object_to_label):

    label, good_value, bad_value, new_value = label_and_values

    # ------------------------------------------------------------------------ #
    # Check label has correct name
    # ------------------------------------------------------------------------ #

    assert label.label == "foo", "Label has incorrect name"

    # ------------------------------------------------------------------------ #
    # Check we can't label unsupported objects
    # ------------------------------------------------------------------------ #

    if object_type == int:
        # Can't label integers, so check this fails and force end
        try:
            labelled_object = label(object_to_label)
        except ValueError:
            # Appropriate error has been returned so end the test
            return

        # If we get here there has been an error
        assert False, "Labelling an integer should throw an error"

    # ------------------------------------------------------------------------ #
    # Test application of labels
    # ------------------------------------------------------------------------ #

    if label_type == "boolean":
        labelled_object = label(object_to_label)

    else:
        # Check that passing an inappropriate label gives the correct error
        try:
            labelled_object = label(object_to_label, bad_value)
            # If we get here the validator has not worked
            assert False, 'The labelling validator has not worked for ' \
                + f'label_type {label_type} and object_type {object_type}'

        except AssertionError:
            # Now label object properly
            labelled_object = label(object_to_label, good_value)

    # ------------------------------------------------------------------------ #
    # Check labelled form or term has been returned
    # ------------------------------------------------------------------------ #

    if object_type == Term:
        assert type(labelled_object) == Term, 'Labelled Term should be a ' \
            + f'be a Term and not type {type(labelled_object)}'
    else:
        assert type(labelled_object) == LabelledForm, 'Labelled Form should ' \
            + f'be a Labelled Form and not type {type(labelled_object)}'

    # ------------------------------------------------------------------------ #
    # Test that the values are correct
    # ------------------------------------------------------------------------ #

    if object_type == Term:
        assert labelled_object.get(label) == good_value, 'Value of label ' \
            + f'should be {good_value} and not {labelled_object.get(label)}'
    else:
        assert labelled_object.terms[0].get(label) == good_value, 'Value of ' \
            + f'label should be {good_value} and not ' \
            + f'{labelled_object.terms[0].get(label)}'

    # ------------------------------------------------------------------------ #
    # Test updating of values
    # ------------------------------------------------------------------------ #

    # Check that passing an inappropriate label gives the correct error
    try:
        labelled_object = label.update_value(labelled_object, bad_value)
        # If we get here the validator has not worked
        assert False, 'The validator has not worked for updating label of ' \
            + f'label_type {label_type} and object_type {object_type}'
    except AssertionError:
        # Update new value
        labelled_object = label.update_value(labelled_object, new_value)

    # Check that new value is correct
    if object_type == Term:
        assert labelled_object.get(label) == new_value, 'Updated value of ' \
            + f'label should be {new_value} and not {labelled_object.get(label)}'
    else:
        assert labelled_object.terms[0].get(label) == new_value, 'Updated ' \
            + f'value of label should be {new_value} and not ' \
            + f'{labelled_object.terms[0].get(label)}'

    # ------------------------------------------------------------------------ #
    # Test removal of values
    # ------------------------------------------------------------------------ #

    labelled_object = label.remove(labelled_object)

    # Try to see if object still has that label
    if object_type == Term:
        label_value = labelled_object.get(label)
    else:
        label_value = labelled_object.terms[0].get(label)

    # If we get here then the label has been extracted but it shouldn't have
    assert label_value is None, f'The label {label_type} appears has not to ' \
        + f'have been removed for object_type {object_type}'
